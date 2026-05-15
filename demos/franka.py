from __future__ import annotations

import argparse
from dataclasses import dataclass

from .common import *
import numpy as np
import torch

from boed import BOED, FisherEstimator, ParameterDistribution, QOED, QOED_AGNOSTIC

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


@dataclass(frozen=True)
class Config:
    baseline: str = QOED
    seed: int | None = None
    steps: int = 200
    obs_noise: float = 0.05
    log_every: int = 25
    fisher_weight: float = 1.0
    num_iterations: int = 1
    num_samples: int = 1024
    eig_ratio_thresh: float = 0.01
    dist_threshold: float = 0.05
    smallest_eigval_threshold: float = 0.1

    dt: float = 1.0 / 10.0
    max_accel: float = 8.0
    max_pos_offset: float = 0.2
    max_z_lift: float = 0.05
    max_z_press: float = -0.02
    cmd_vel_damp: float = 0.90

    plan_horizon: int = 20
    action_noise_std: float = 2.5

    est_window: int = 20

    gravity: float = 9.81
    v0: float = 0.02

    var_small_thresh: float = 1e-4
    param_contrib_ratio: float = 1e-3

    theta_low: tuple[float, ...] = (0.05, 0.05, 0.0)
    theta_high: tuple[float, ...] = (5.0, 2.0, 10.0)
    theta_mean0: tuple[float, ...] = (0.50, 0.60, 1.00)
    theta_std0: tuple[float, ...] = (0.40, 0.40, 0.80)
    theta_true: tuple[float, ...] = (0.218, 0.4, 3.0)
    candidate_mask: tuple[bool, ...] = (True, True, True)

    all_baselines: tuple[str, ...] = (QOED, QOED_AGNOSTIC, BOED)
    summary_metrics: tuple[str, ...] = ("param_rmse", "pred_rmse", "std_norm", "fisher_trace")


DEFAULT_CONFIG = Config()


@dataclass
class Summary:
    baseline: str
    device: str
    param_rmse: float
    pred_rmse: float
    std_norm: float
    fisher_trace: float
    final_mean: np.ndarray
    final_std: np.ndarray


def _cube_step_batch(state, action, params, cfg: Config):
    pos, vel = state[:, :3], state[:, 3:6]
    vel_next = (vel + action * cfg.dt) * cfg.cmd_vel_damp
    pos_next = pos + vel_next * cfg.dt

    xy = pos_next[:, :2]
    radius = torch.linalg.norm(xy, dim=1, keepdim=True) + 1e-9
    xy = xy * torch.clamp(cfg.max_pos_offset / radius, max=1.0)
    z = torch.clamp(pos_next[:, 2:3], cfg.max_z_press, cfg.max_z_lift)
    pos_next = torch.cat([xy, z], -1)

    mass, mu, drag = params.T
    in_contact = (pos_next[:, 2] < 0.005).to(state.dtype)[:, None]
    friction_dir = torch.tanh(vel_next / cfg.v0)
    force_inertial = -(mass[:, None] * action)
    force_friction = -((mu * mass * cfg.gravity)[:, None] * friction_dir) * in_contact
    force_drag = -(drag[:, None] * vel_next)
    force_total = force_inertial + force_friction + force_drag

    vel_norm = torch.linalg.norm(vel_next, dim=1, keepdim=True) + 1e-6
    vel_hat = vel_next / vel_norm
    y = (force_total * vel_hat).sum(1, keepdim=True)

    friction_proj = (friction_dir * vel_hat).sum(1) * in_contact[:, 0]
    action_proj = (action * vel_hat).sum(1)
    jac = torch.stack(
        [
            -action_proj - (mu * cfg.gravity) * friction_proj,
            -(mass * cfg.gravity) * friction_proj,
            -vel_norm[:, 0],
        ],
        -1,
    )
    return torch.cat([pos_next, vel_next], -1), y, jac


def cube_step(state, action, params, cfg: Config):
    squeeze = as_tensor(state).ndim == 1
    device = as_tensor(state).device
    state = batch_if_vector(state, device)
    action = batch_if_vector(action, device)
    params = broadcast_rows(batch_if_vector(params, device), state.shape[0])
    out = _cube_step_batch(state, action, params, cfg)
    return tuple(x[0] for x in out) if squeeze else out


def _outer_fisher(jac, obs_var_inv):
    weighted = jac * torch.sqrt(torch.as_tensor(obs_var_inv, device=jac.device, dtype=jac.dtype))
    return weighted[:, :, None] * weighted[:, None, :]


def _project_plan_actions(action, cfg: Config):
    action = clip_action_norm(action, cfg.max_accel).clone()
    action[..., 2] = 0.0
    return action


def _history_fisher_core(states, actions, theta, obs_var_inv, cfg: Config):
    with torch.no_grad():
        states = batch_if_vector(states, states.device)
        actions = batch_if_vector(actions, states.device)
        params = broadcast_rows(batch_if_vector(theta, states.device), states.shape[0])
        _, _, jac = _cube_step_batch(states, actions, params, cfg)
        f = _outer_fisher(jac, obs_var_inv).sum(0)
        f = torch.nan_to_num(f, nan=0.0, posinf=1e6, neginf=-1e6)
        return 0.5 * (f + f.T)


class FrankaDynamics:
    def __init__(self, cfg: Config, device: torch.device | None = None):
        self.cfg = cfg
        self.device = torch.device("cpu") if device is None else device

    def step(self, state, action, theta):
        return cube_step(state, action, theta, self.cfg)[0]

    def force_observation(self, state, action, theta):
        return cube_step(state, action, theta, self.cfg)[1]


class FrankaForceObservationDynamics(FrankaDynamics):
    def step(self, state, action, theta):
        return self.force_observation(state, action, theta)


class FrankaFisherEstimator(FisherEstimator):
    """Franka-specialized analytic Fisher on top of the generic BOED estimator."""

    def __init__(self, dynamics: FrankaDynamics, cfg: Config, dist: ParameterDistribution, key=None):
        super().__init__(
            dynamics,
            dist,
            key,
            obs_noise_std=cfg.obs_noise,
            obs_dim=1,
            max_history=cfg.est_window,
            qoed=cfg.baseline.startswith("qoed"),
            var_threshold_for_update=cfg.var_small_thresh,
            eig_ratio_thresh=cfg.eig_ratio_thresh,
            dist_threshold=cfg.dist_threshold,
            param_contrib_ratio=cfg.param_contrib_ratio,
            smallest_eigval_threshold=cfg.smallest_eigval_threshold,
            param_min=cfg.theta_low,
            param_max=cfg.theta_high,
        )
        self.cfg = cfg
        self.candidate_mask = as_tensor(cfg.candidate_mask, self.device, torch.bool)

    def compute_fisher(self, use_cache: bool = True):
        if self._fisher_cache is not None and use_cache:
            return self._fisher_cache
        if not self.history:
            p = self.dist.mean.numel()
            return torch.zeros((p, p), device=self.device, dtype=self.dist.mean.dtype)
        states, actions, *_ = self._stack(self.history)
        f = _history_fisher_core(
            states,
            actions,
            self.dist.as_theta().reshape(-1),
            self.R_inv.reshape(-1)[0],
            self.cfg,
        )
        if use_cache:
            self._fisher_cache = f
        return f


class MPPIController:
    def __init__(
        self,
        dynamics: FrankaDynamics,
        key,
        fisher: FrankaFisherEstimator,
        num_samples: int | None = None,
        num_iterations: int | None = None,
        baseline: str = QOED,
        fisher_weight: float = 1.0,
    ):
        self.dyn = dynamics
        self.cfg = dynamics.cfg
        self.device = fisher.device
        self.generator = make_generator(key, self.device)
        self.T = self.cfg.plan_horizon
        self.K = int(self.cfg.num_samples if num_samples is None else num_samples)
        self.num_iterations = int(self.cfg.num_iterations if num_iterations is None else num_iterations)
        self.baseline = baseline
        self.fisher_estimator = fisher
        self.fisher_weight = float(fisher_weight)
        self.U = torch.zeros((self.T, 3), device=self.device, dtype=torch.float32)

    def control(self, state, theta):
        with torch.no_grad():
            state = as_tensor(state, self.device).reshape(1, -1)
            theta = as_tensor(theta, self.device).reshape(-1)
            self.U = shift_control_sequence(self.U)
            if float(self.U.abs().max().detach().cpu()) < 0.1:
                random_u = torch.randn(self.U.shape, generator=self.generator, device=self.device, dtype=self.U.dtype) * 0.5
                self.U = _project_plan_actions(random_u, self.cfg)
            else:
                self.U = _project_plan_actions(self.U, self.cfg)
            if self.num_iterations <= 0:
                return self.U[0].detach().clone()

            est = self.fisher_estimator

            def sample_population(generator, u, num_samples):
                return sample_gaussian_control_population(
                    generator,
                    u,
                    num_samples,
                    self.cfg.action_noise_std,
                    postprocess=lambda x: _project_plan_actions(x, self.cfg),
                )

            def evaluate_population(population, noise):
                del noise
                params = theta[None].expand(population.shape[0], theta.numel())
                state_batch = state.expand(population.shape[0], state.shape[-1])
                f_cur = torch.zeros((population.shape[0], theta.numel(), theta.numel()), device=self.device, dtype=self.U.dtype)
                energy = torch.zeros((population.shape[0],), device=self.device, dtype=self.U.dtype)

                for t in range(population.shape[1]):
                    action_t = population[:, t]
                    state_batch, _, jac = _cube_step_batch(state_batch, action_t, params, self.cfg)
                    f_cur = f_cur + _outer_fisher(jac, est.R_inv.reshape(-1)[0])
                    energy = energy + action_t.square().sum(1)

                boed_trace = f_cur.diagonal(dim1=-2, dim2=-1).sum(-1)
                costs = 0.05 * energy
                return costs, f_cur, boed_trace

            self.U, est.F_cur, est.eig_mask, _ = mppi_info_gain_refine(
                generator=self.generator,
                u=self.U,
                estimator=est,
                baseline=self.baseline,
                num_samples=self.K,
                num_iterations=self.num_iterations,
                sample_population=sample_population,
                evaluate_population=evaluate_population,
                fisher_weight=self.fisher_weight,
                temperature=1e-3,
                candidate_mask=est.candidate_mask,
                fallback_to_all_mask=True,
                postprocess_update=lambda u: _project_plan_actions(u, self.cfg),
            )
            return self.U[0].detach().clone()


class SimulatedFrankaInterface:
    def __init__(self, cfg: Config, seed: int | None, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.generator = make_generator(seed, device)
        self.true_params = as_tensor(cfg.theta_true, device).reshape(1, 3)
        self.true_bias = 0.1
        self.tare_offset = 0.0
        self.state = torch.zeros((1, 6), device=device, dtype=torch.float32)

    def tare_sensor(self):
        self.tare_offset = self.true_bias

    def get_state(self):
        return self.state[0].detach().clone()

    def step(self, accel_cmd):
        with torch.no_grad():
            action = as_tensor(accel_cmd, self.device).reshape(1, 3)
            self.state, y_clean, _ = _cube_step_batch(self.state, action, self.true_params, self.cfg)
            noise = torch.randn((), generator=self.generator, device=self.device) * self.cfg.obs_noise
            y_obs = y_clean[0, 0] + self.true_bias - self.tare_offset + noise
            return float(y_obs.detach().cpu())


def _update_fisher_history_core(f_hist, state, action, params, cfg: Config):
    with torch.no_grad():
        fisher_step = _history_fisher_core(
            batch_if_vector(state, f_hist.device),
            batch_if_vector(action, f_hist.device),
            as_tensor(params, f_hist.device),
            1.0 / (cfg.obs_noise**2),
            cfg,
        )
        fisher_trace = torch.trace(fisher_step)
        return f_hist + fisher_step, fisher_trace


def prediction_rmse(states, actions, y_obs, theta, cfg: Config):
    if len(actions) == 0:
        return 0.0
    states_t = as_tensor(np.asarray(states, np.float32))
    actions_t = as_tensor(np.asarray(actions, np.float32))
    params = as_tensor(theta).reshape(1, 3).expand(actions_t.shape[0], 3)
    _, y_pred, _ = _cube_step_batch(states_t, actions_t, params, cfg)
    err = y_pred[:, 0] - as_tensor(np.asarray(y_obs, np.float32))
    return float(torch.sqrt(torch.mean(err.square())).detach().cpu())


def run_simulation(cfg: Config | None = None, *, verbose: bool = True, device: str = "auto", **kwargs):
    updates = {
        (
            "obs_noise"
            if name in {"noise", "obs_noise_std"}
            else "num_samples"
            if name == "plan_samples"
            else name
        ): value
        for name, value in kwargs.items()
    }
    cfg = Config(**({} if cfg is None else cfg.__dict__) | updates)
    seed = 0 if cfg.seed is None else cfg.seed
    device_obj = resolve_device(device)

    env = SimulatedFrankaInterface(cfg, seed + 1, device_obj)
    env.tare_sensor()
    dyn = FrankaDynamics(cfg, device_obj)
    est_dyn = FrankaForceObservationDynamics(cfg, device_obj)
    theta_mean = as_tensor(cfg.theta_mean0, device_obj)
    theta_std = as_tensor(cfg.theta_std0, device_obj)
    theta_true = as_tensor(cfg.theta_true, device_obj)
    dist = ParameterDistribution(theta_mean.clone(), torch.diag(theta_std.square()))
    estimator = FrankaFisherEstimator(est_dyn, cfg, dist, seed + 2)
    controller = MPPIController(
        dyn,
        seed + 3,
        estimator,
        num_samples=cfg.num_samples,
        num_iterations=cfg.num_iterations,
        baseline=cfg.baseline,
        fisher_weight=cfg.fisher_weight,
    )

    f_hist = torch.zeros((3, 3), device=device_obj, dtype=torch.float32)
    states, actions, y_obs_hist = [], [], []

    for step in range(cfg.steps):
        state = env.get_state()
        action = controller.control(state, estimator.dist.as_theta())
        y_obs = env.step(action)
        estimator.add_sample(state, action, torch.as_tensor([y_obs], device=device_obj, dtype=torch.float32))
        f_hist, _ = _update_fisher_history_core(
            f_hist,
            state,
            as_tensor(action, device_obj),
            theta_true,
            cfg,
        )

        param_rmse = float(torch.sqrt(torch.mean((estimator.dist.mean - theta_true).square())).detach().cpu())
        belief_std = torch.sqrt(torch.clamp(torch.diagonal(estimator.dist.cov), min=0.0))
        states.append(to_numpy(state))
        actions.append(to_numpy(action))
        y_obs_hist.append(y_obs)

        if verbose and ((step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps):
            fisher_trace = float(torch.trace(f_hist).detach().cpu())
            print(
                f"[{cfg.baseline}] step={step + 1:03d} "
                f"param_rmse={param_rmse:.3f} "
                f"std_norm={float(torch.linalg.norm(belief_std).detach().cpu()):.3f} "
                f"fisher_trace={fisher_trace:.1f}"
            )

    final_mean = to_numpy(estimator.dist.mean)
    final_std = to_numpy(torch.sqrt(torch.clamp(torch.diagonal(estimator.dist.cov), min=0.0)))
    pred_rmse = prediction_rmse(states, actions, y_obs_hist, final_mean, cfg)
    fisher_trace = float(torch.trace(f_hist).detach().cpu())
    std_norm = float(np.linalg.norm(final_std))
    summary = Summary(
        baseline=cfg.baseline,
        device=str(device_obj),
        param_rmse=float(np.sqrt(np.mean((final_mean - np.asarray(cfg.theta_true, np.float32)) ** 2))),
        pred_rmse=pred_rmse,
        std_norm=std_norm,
        fisher_trace=fisher_trace,
        final_mean=final_mean.copy(),
        final_std=final_std.copy(),
    )

    if verbose:
        with np.printoptions(precision=3, suppress=True):
            print("\ntrue_theta:", np.asarray(cfg.theta_true, np.float32))
            print("std:       ", summary.final_std)
            print(f"prediction_rmse={summary.pred_rmse:.4f}")
            print("\n", "-" * 30, "\n")
    return summary


def _config_from_args(args, baseline: str, seed: int | None):
    return Config(
        baseline=baseline,
        seed=seed,
        **{
            name: getattr(args, name)
            for name in (
                "steps",
                "obs_noise",
                "log_every",
                "num_samples",
                "num_iterations",
                "fisher_weight",
                "eig_ratio_thresh",
                "dist_threshold",
                "smallest_eigval_threshold",
            )
        },
    )


def run_demo(args):
    run_baseline_demo(args, DEFAULT_CONFIG.all_baselines, run_simulation, _config_from_args, DEFAULT_CONFIG.summary_metrics)


def parse(argv=None):
    defaults = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(description="PyTorch Franka cube-slide BOED/QOED demo.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--obs-noise", "--obs_noise", type=float, default=defaults.obs_noise)
    parser.add_argument("--num-iterations", "--num_iterations", type=int, default=defaults.num_iterations)
    parser.add_argument("--baseline", choices=["all", *defaults.all_baselines], default="all")
    parser.add_argument("--fisher-weight", "--fisher_weight", type=float, default=defaults.fisher_weight)
    parser.add_argument("--eig-ratio-thresh", "--eig_ratio_thresh", type=float, default=defaults.eig_ratio_thresh)
    parser.add_argument("--dist-threshold", "--dist_threshold", type=float, default=defaults.dist_threshold)
    parser.add_argument("--smallest-eigval-threshold", "--smallest_eigval_threshold", type=float, default=defaults.smallest_eigval_threshold)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-seeds", "--sweep_seeds", type=int, default=10)
    parser.add_argument("--log-every", "--log_every", type=int, default=defaults.log_every)
    parser.add_argument("--num-samples", "--num_samples", type=int, default=defaults.num_samples)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse(argv)
    run_demo(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
