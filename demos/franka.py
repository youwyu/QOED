from __future__ import annotations

from dataclasses import dataclass

from .common import (
    as_tensor,
    batch_if_vector,
    broadcast_rows,
    make_generator,
    mppi_info_gain_refine,
    resolve_device,
    run_demo,
    shift_control_sequence,
    stack_to_numpy,
    to_numpy,
)

import numpy as np
import torch

from boed import BOED, QOED, QOED_AGNOSTIC, FisherEstimator, ParameterDistribution

BASELINES = (QOED, QOED_AGNOSTIC, BOED)
METRICS = ("param_rmse", "pred_rmse", "std_norm", "fisher_trace")


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


DEFAULT_CONFIG = Config()


@dataclass
class Summary:
    baseline: str
    device: str
    param_rmse: float
    pred_rmse: float
    std_norm: float
    fisher_trace: float


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


def _outer_fisher(jac, obs_var_inv):
    weighted = jac * torch.sqrt(torch.as_tensor(obs_var_inv, device=jac.device, dtype=jac.dtype))
    return weighted[:, :, None] * weighted[:, None, :]


def _project_plan_actions(action, cfg: Config):
    norm = torch.linalg.norm(action, dim=-1, keepdim=True) + 1e-6
    action = action * torch.clamp(float(cfg.max_accel) / norm, max=1.0)
    action[..., 2] = 0.0
    return action


def _history_fisher_core(states, actions, theta, obs_var_inv, cfg: Config):
    with torch.no_grad():
        states = batch_if_vector(states, states.device)
        actions = batch_if_vector(actions, states.device)
        params = broadcast_rows(batch_if_vector(theta, states.device), states.shape[0])
        _, _, jac = _cube_step_batch(states, actions, params, cfg)
        f = torch.nan_to_num(_outer_fisher(jac, obs_var_inv).sum(0), nan=0.0, posinf=1e6, neginf=-1e6)
        return 0.5 * (f + f.T)


class CubeForceModel:

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def step(self, state, action, theta):
        state = as_tensor(state)
        squeeze = state.ndim == 1
        state = batch_if_vector(state, state.device)
        action = batch_if_vector(action, state.device)
        params = broadcast_rows(batch_if_vector(theta, state.device), state.shape[0])
        y = _cube_step_batch(state, action, params, self.cfg)[1]
        return y[0] if squeeze else y


class FrankaFisherEstimator(FisherEstimator):
    """Franka-specialized analytic Fisher on top of the generic BOED estimator."""

    def __init__(self, cfg: Config, dist: ParameterDistribution, seed: int):
        super().__init__(
            CubeForceModel(cfg),
            dist,
            seed,
            obs_noise_std=cfg.obs_noise,
            obs_dim=1,
            max_history=cfg.est_window,
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
        states, actions, *_ = self.stack(self.history)
        f = _history_fisher_core(states, actions, self.dist.mean.reshape(-1), self.R_inv.reshape(-1)[0], self.cfg)
        if use_cache:
            self._fisher_cache = f
        return f


class MPPIController:
    def __init__(self, cfg: Config, seed: int, fisher: FrankaFisherEstimator):
        self.cfg = cfg
        self.device = fisher.device
        self.generator = make_generator(seed, self.device)
        self.K = int(cfg.num_samples)
        self.num_iterations = int(cfg.num_iterations)
        self.baseline = cfg.baseline
        self.fisher_estimator = fisher
        self.fisher_weight = float(cfg.fisher_weight)
        self.U = torch.zeros((cfg.plan_horizon, 3), device=self.device, dtype=torch.float32)

    def _project(self, action):
        return _project_plan_actions(action, self.cfg)

    def _sample(self, u):
        noise = torch.randn((self.K, *u.shape), generator=self.generator, device=u.device, dtype=u.dtype)
        noise = noise * float(self.cfg.action_noise_std)
        return self._project(u[None] + noise), noise

    def control(self, state, theta):
        with torch.no_grad():
            state = as_tensor(state, self.device).reshape(1, -1)
            theta = as_tensor(theta, self.device).reshape(-1)
            self.U = shift_control_sequence(self.U)
            if float(self.U.abs().max()) < 0.1:
                self.U = torch.randn(self.U.shape, generator=self.generator, device=self.device, dtype=self.U.dtype) * 0.5
            self.U = self._project(self.U)
            if self.num_iterations <= 0:
                return self.U[0].clone()
            r_inv = self.fisher_estimator.R_inv.reshape(-1)[0]

            def evaluate(population, noise):
                k = population.shape[0]
                params = theta[None].expand(k, theta.numel())
                state_batch = state.expand(k, state.shape[-1])
                f_cur = torch.zeros((k, theta.numel(), theta.numel()), device=self.device, dtype=self.U.dtype)
                energy = torch.zeros((k,), device=self.device, dtype=self.U.dtype)
                for t in range(population.shape[1]):
                    action_t = population[:, t]
                    state_batch, _, jac = _cube_step_batch(state_batch, action_t, params, self.cfg)
                    f_cur = f_cur + _outer_fisher(jac, r_inv)
                    energy = energy + action_t.square().sum(1)
                return 0.05 * energy, f_cur, f_cur.diagonal(dim1=-2, dim2=-1).sum(-1)

            self.U = mppi_info_gain_refine(
                u=self.U,
                estimator=self.fisher_estimator,
                baseline=self.baseline,
                num_iterations=self.num_iterations,
                sample_population=self._sample,
                evaluate_population=evaluate,
                fisher_weight=self.fisher_weight,
                temperature=1e-3,
                candidate_mask=self.fisher_estimator.candidate_mask,
                fallback_to_all_mask=True,
                postprocess=self._project,
            )
            return self.U[0].clone()


class SimulatedFrankaInterface:

    def __init__(self, cfg: Config, seed: int, device: torch.device):
        self.cfg = cfg
        self.device = device
        self.generator = make_generator(seed, device)
        self.true_params = as_tensor(cfg.theta_true, device).reshape(1, 3)
        self.true_bias = 0.1
        self.tare_offset = self.true_bias
        self.state = torch.zeros((1, 6), device=device, dtype=torch.float32)

    def step(self, accel_cmd) -> float:
        with torch.no_grad():
            action = as_tensor(accel_cmd, self.device).reshape(1, 3)
            self.state, y_clean, _ = _cube_step_batch(self.state, action, self.true_params, self.cfg)
            noise = torch.randn((), generator=self.generator, device=self.device) * self.cfg.obs_noise
            return float(y_clean[0, 0] + self.true_bias - self.tare_offset + noise)


def prediction_rmse(states, actions, y_obs, theta, cfg: Config):
    if len(actions) == 0:
        return 0.0
    states_t = torch.from_numpy(np.asarray(states, np.float32))
    actions_t = torch.from_numpy(np.asarray(actions, np.float32))
    params = as_tensor(theta).reshape(1, 3).expand(actions_t.shape[0], 3)
    _, y_pred, _ = _cube_step_batch(states_t, actions_t, params, cfg)
    err = y_pred[:, 0] - torch.from_numpy(np.asarray(y_obs, np.float32))
    return float(torch.sqrt(torch.mean(err.square())))


def run_simulation(cfg: Config = DEFAULT_CONFIG, *, verbose: bool = True, device: str = "auto") -> Summary:
    seed = 0 if cfg.seed is None else cfg.seed
    device = resolve_device(device)

    env = SimulatedFrankaInterface(cfg, seed + 1, device)
    theta_true = as_tensor(cfg.theta_true, device)
    dist = ParameterDistribution(as_tensor(cfg.theta_mean0, device), torch.diag(as_tensor(cfg.theta_std0, device).square()))
    estimator = FrankaFisherEstimator(cfg, dist, seed + 2)
    controller = MPPIController(cfg, seed + 3, estimator)

    f_hist = torch.zeros((3, 3), device=device, dtype=torch.float32)
    obs_var_inv = 1.0 / (cfg.obs_noise**2)
    states, actions, y_obs_hist = [], [], []
    for step in range(cfg.steps):
        state = env.state[0].clone()
        action = controller.control(state, dist.mean)
        y_obs = env.step(action)
        estimator.add_sample(state, action, torch.as_tensor([y_obs], device=device, dtype=torch.float32))
        f_hist = f_hist + _history_fisher_core(state[None], action[None], theta_true, obs_var_inv, cfg)
        states.append(state)
        actions.append(action)
        y_obs_hist.append(y_obs)

        if verbose and ((step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps):
            belief_std = torch.sqrt(torch.clamp(torch.diagonal(dist.cov), min=0.0))
            print(
                f"[{cfg.baseline}] step={step + 1:03d} "
                f"param_rmse={float(torch.sqrt(torch.mean((dist.mean - theta_true).square()))):.3f} "
                f"std_norm={float(torch.linalg.norm(belief_std)):.3f} "
                f"fisher_trace={float(torch.trace(f_hist)):.1f}"
            )

    final_mean = to_numpy(dist.mean)
    final_std = to_numpy(torch.sqrt(torch.clamp(torch.diagonal(dist.cov), min=0.0)))
    summary = Summary(
        baseline=cfg.baseline,
        device=str(device),
        param_rmse=float(np.sqrt(np.mean((final_mean - np.asarray(cfg.theta_true, np.float32)) ** 2))),
        pred_rmse=prediction_rmse(stack_to_numpy(states, (0, 6)), stack_to_numpy(actions, (0, 3)), y_obs_hist, final_mean, cfg),
        std_norm=float(np.linalg.norm(final_std)),
        fisher_trace=float(torch.trace(f_hist)),
    )
    if verbose:
        with np.printoptions(precision=3, suppress=True):
            print("\ntrue_theta:", np.asarray(cfg.theta_true, np.float32))
            print("std:       ", final_std)
            print(f"prediction_rmse={summary.pred_rmse:.4f}")
            print("\n", "-" * 30, "\n")
    return summary


def main(argv=None):
    return run_demo("PyTorch Franka cube-slide BOED/QOED demo.", Config, run_simulation, BASELINES, METRICS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
