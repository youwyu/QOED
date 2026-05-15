from __future__ import annotations

import argparse
from dataclasses import dataclass

from .common import *
import numpy as np
import torch

from boed import *

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

    dt: float = 0.1
    plan_dt: float = 0.02
    wheel_base: float = 0.37559
    init_state: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    goal: tuple[float, ...] = (5.0, 5.0, 0.0)
    theta_base: tuple[float, ...] = (20.0, 1.0, 23.5, 23.5, 23.5, 23.5, 0.0, 0.0)
    theta_std: tuple[float, ...] = (10.0, 0.5, 2.0, 2.0, 2.0, 2.0, 50.0, 50.0)
    theta_min: tuple[float, ...] = (10.0, 0.4, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0)
    u_min: tuple[float, ...] = (-1.0, -float(np.pi / 2))
    u_max: tuple[float, ...] = (1.0, float(np.pi / 2))
    noise_cov_diag: tuple[float, ...] = (0.4, 0.4)
    plan_steps: int = 50

    all_baselines: tuple[str, ...] = (QOED, QOED_AGNOSTIC, BOED)
    summary_metrics: tuple[str, ...] = ("param_rmse", "pred_rmse", "goal_dist", "fisher_trace")


DEFAULT_CONFIG = Config()


def _wrap_angle(x):
    return torch.remainder(x + torch.pi, 2.0 * torch.pi) - torch.pi


def _jackal_gain(theta):
    return 0.5 * theta[:, 2:6].sum(1)


def _jackal_inertia(mass, wheel_base):
    return torch.clamp(0.5 * mass * wheel_base * wheel_base, min=1e-3)


def _jackal_step_batch(state, action, theta, dt, wheel_base):
    x, y, yaw, v, omega = state.T
    v_cmd, omega_cmd = action[:, 0], action[:, 1]
    mass = torch.clamp(theta[:, 0], min=0.1)
    friction = torch.clamp(theta[:, 1], min=0.0)
    gain = _jackal_gain(theta)
    inertia = _jackal_inertia(mass, wheel_base)
    v_next = v + dt * (gain * v_cmd - friction * v - torch.clamp(theta[:, 6], min=0.0)) / mass
    omega_next = omega + dt * (gain * omega_cmd * wheel_base * 0.5 + theta[:, 7] * wheel_base * 0.5) / inertia
    x_next = x + dt * v_next * torch.cos(yaw)
    y_next = y + dt * v_next * torch.sin(yaw)
    yaw_next = _wrap_angle(yaw + dt * omega_next)
    return torch.stack([x_next, y_next, yaw_next, v_next, omega_next], -1)


def jackal_step(state, action, theta, dt=DEFAULT_CONFIG.dt, wheel_base=DEFAULT_CONFIG.wheel_base):
    state_t = as_tensor(state)
    squeeze = state_t.ndim == 1
    device = state_t.device
    state_t = batch_if_vector(state_t, device)
    action_t = batch_if_vector(action, device)
    theta_t = broadcast_rows(batch_if_vector(theta, device), state_t.shape[0])
    out = _jackal_step_batch(state_t, action_t, theta_t, dt, wheel_base)
    return out[0] if squeeze else out


def _jackal_obs_batch(state):
    x, y, yaw, v, omega = state.T
    return torch.stack([x, y, yaw, v * torch.cos(yaw), v * torch.sin(yaw), omega], -1)


def jackal_obs(state):
    state_t = as_tensor(state)
    squeeze = state_t.ndim == 1
    obs = _jackal_obs_batch(batch_if_vector(state_t, state_t.device))
    return obs[0] if squeeze else obs


def _mppi_stage_cost(state, action, noise, goal):
    pos_err = state[:, :2] - goal[:2]
    goal_cost = 2.5 * pos_err.square().sum(-1)
    action_cost = 0.5 * action.square().sum(-1)
    noise_cost = 0.1 * noise.square().sum(-1)
    return goal_cost + action_cost + noise_cost


def _jackal_step_sens(state, sens, action, theta, dt, wheel_base):
    k, p = theta.shape
    x, y, yaw, v, omega = state.T
    v_cmd, omega_cmd = action[:, 0], action[:, 1]
    mass = torch.clamp(theta[:, 0], min=0.1)
    friction = torch.clamp(theta[:, 1], min=0.0)
    gain = _jackal_gain(theta)
    fx = torch.clamp(theta[:, 6], min=0.0)
    inertia_raw = 0.5 * mass * wheel_base * wheel_base
    inertia = _jackal_inertia(mass, wheel_base)
    force = gain * v_cmd - friction * v - fx
    turn = 0.5 * wheel_base * (gain * omega_cmd + theta[:, 7])
    v_next = v + dt * force / mass
    omega_next = omega + dt * turn / inertia
    c, s = torch.cos(yaw), torch.sin(yaw)
    x_next = x + dt * v_next * c
    y_next = y + dt * v_next * s
    yaw_next = _wrap_angle(yaw + dt * omega_next)
    state_next = torch.stack([x_next, y_next, yaw_next, v_next, omega_next], -1)

    zeros = torch.zeros((k, p), device=theta.device, dtype=theta.dtype)
    dmass = (theta[:, 0] > 0.1).to(theta.dtype)
    dfriction = (theta[:, 1] > 0.0).to(theta.dtype) + 0.5 * (theta[:, 1] == 0.0).to(theta.dtype)
    dfx = (theta[:, 6] > 0.0).to(theta.dtype) + 0.5 * (theta[:, 6] == 0.0).to(theta.dtype)
    dinertia = (inertia_raw > 1e-3).to(theta.dtype) * 0.5 * wheel_base * wheel_base * dmass
    dv_theta = zeros.clone()
    dv_theta[:, 0] = dt * (-force / (mass * mass)) * dmass
    dv_theta[:, 1] = dt * (-v / mass) * dfriction
    dv_theta[:, 2:6] = (dt * 0.5 * v_cmd / mass)[:, None].expand(k, 4)
    dv_theta[:, 6] = dt * (-dfx / mass)

    domega_theta = zeros.clone()
    domega_theta[:, 0] = dt * (-turn / (inertia * inertia)) * dinertia
    domega_theta[:, 2:6] = (dt * 0.25 * wheel_base * omega_cmd / inertia)[:, None].expand(k, 4)
    domega_theta[:, 7] = dt * 0.5 * wheel_base / inertia

    sx, sy, syaw, sv, somega = sens.transpose(0, 1)
    sv_next = (1.0 - dt * friction / mass)[:, None] * sv + dv_theta
    somega_next = somega + domega_theta
    sx_next = sx + dt * (c[:, None] * sv_next - (v_next * s)[:, None] * syaw)
    sy_next = sy + dt * (s[:, None] * sv_next + (v_next * c)[:, None] * syaw)
    syaw_next = syaw + dt * somega_next
    sens_next = torch.stack([sx_next, sy_next, syaw_next, sv_next, somega_next], 1)

    cn, sn, vn = torch.cos(state_next[:, 2]), torch.sin(state_next[:, 2]), state_next[:, 3]
    jac = torch.stack(
        [
            sx_next,
            sy_next,
            syaw_next,
            cn[:, None] * sv_next - (vn * sn)[:, None] * syaw_next,
            sn[:, None] * sv_next + (vn * cn)[:, None] * syaw_next,
            somega_next,
        ],
        1,
    )
    return state_next, sens_next, jac


def _history_fisher_core(s_seq, u_seq, theta, r_inv, fd_eps, valid, dt, wheel_base):
    with torch.no_grad():
        t_len, p = s_seq.shape[0], theta.numel()
        eps = fd_eps * (theta.abs() + 1.0)
        eye = torch.eye(p, device=theta.device, dtype=theta.dtype)
        theta_p = theta[None] + eye * eps[:, None]
        theta_m = theta[None] - eye * eps[:, None]
        theta_combined = torch.cat([theta_p.repeat_interleave(t_len, 0), theta_m.repeat_interleave(t_len, 0)], 0)
        s_combined = s_seq.repeat((2 * p, 1))
        u_combined = u_seq.repeat((2 * p, 1))
        y = _jackal_obs_batch(_jackal_step_batch(s_combined, u_combined, theta_combined, dt, wheel_base))
        split = p * t_len
        y_p = y[:split].reshape(p, t_len, -1)
        y_m = y[split:].reshape(p, t_len, -1)
        jac = (y_p - y_m).permute(1, 2, 0) / (2.0 * eps.reshape(1, 1, p))
        f_t = jac.transpose(1, 2) @ r_inv @ jac
        f = torch.nan_to_num((f_t * valid[:, None, None].to(f_t.dtype)).sum(0), nan=0.0, posinf=1e6, neginf=-1e6)
        return 0.5 * (f + f.T)


class JackalDynamics:
    def __init__(self, dt: float = DEFAULT_CONFIG.dt, wheel_base: float = DEFAULT_CONFIG.wheel_base, device: torch.device | None = None):
        self.dt = dt
        self.wheel_base = wheel_base
        self.device = torch.device("cpu") if device is None else device

    def step(self, state, action, theta):
        return jackal_step(state, action, theta, self.dt, self.wheel_base)

    def observe(self, state):
        return jackal_obs(state)

    def state_to_observation(self, state, goal):
        obs = self.observe(state)
        squeeze = obs.ndim == 1
        obs = batch_if_vector(obs, obs.device)
        goal = batch_if_vector(goal, obs.device)
        if goal.shape[-1] == 2:
            goal = torch.cat([goal, torch.zeros((*goal.shape[:-1], 1), device=goal.device, dtype=goal.dtype)], -1)
        goal = broadcast_rows(goal, obs.shape[0])
        out = torch.cat([obs, goal[:, :3]], -1)
        return out[0] if squeeze else out


class JackalFisherEstimator(FisherEstimator):
    """Jackal-specialized fast kernels on top of the generic BOED estimator."""

    def compute_fisher(self, use_cache: bool = True):
        if self._fisher_cache is not None and use_cache:
            return self._fisher_cache
        if not self.history:
            p = self.dist.mean.numel()
            return torch.zeros((p, p), device=self.device, dtype=self.dist.mean.dtype)
        s_seq, u_seq, *_ = stack_rows(self.history)
        n = s_seq.shape[0]
        if self.max_history is None:
            valid = torch.ones((n,), device=self.device, dtype=torch.bool)
        else:
            cap = max(n, self.max_history + 1)
            valid = torch.arange(cap, device=self.device) < n
            if n < cap:
                s_pad = torch.zeros((cap - n, s_seq.shape[-1]), device=self.device, dtype=s_seq.dtype)
                u_pad = torch.zeros((cap - n, u_seq.shape[-1]), device=self.device, dtype=u_seq.dtype)
                s_seq = torch.cat([s_seq, s_pad], 0)
                u_seq = torch.cat([u_seq, u_pad], 0)
        f = _history_fisher_core(
            s_seq,
            u_seq,
            self.dist.as_theta().reshape(-1),
            self.R_inv,
            self.fd_eps,
            valid,
            self.dyn.dt,
            self.dyn.wheel_base,
        )
        if use_cache:
            self._fisher_cache = f
        return f

    def score_paths(self, boed_trace, baseline: str = QOED):
        scores, self.eig_mask = score_paths(
            self.F_cur,
            boed_trace,
            self.compute_fisher(),
            self.dist.cov,
            baseline,
            bool(self.history),
            self.eig_ratio_thresh,
            self.dist_threshold,
            self.param_contrib_ratio,
            self.var_threshold_for_update,
            self.smallest_eigval_threshold,
        )
        return scores


class MPPIController:
    def __init__(
        self,
        dynamics: JackalDynamics,
        key,
        fisher: FisherEstimator,
        num_samples: int = 1024,
        plan_steps: int = DEFAULT_CONFIG.plan_steps,
        noise_cov_diag=None,
        u_min=None,
        u_max=None,
        num_iterations: int = 1,
        baseline: str = QOED,
        fisher_weight: float = 1.0,
    ):
        self.dyn = dynamics
        self.device = fisher.device
        self.generator = make_generator(key, self.device)
        self.T = int(plan_steps)
        self.K = int(num_samples)
        self.noise_cov_diag = as_tensor(DEFAULT_CONFIG.noise_cov_diag if noise_cov_diag is None else noise_cov_diag, self.device)
        self.u_min = as_tensor(DEFAULT_CONFIG.u_min if u_min is None else u_min, self.device).reshape(1, 2)
        self.u_max = as_tensor(DEFAULT_CONFIG.u_max if u_max is None else u_max, self.device).reshape(1, 2)
        self.num_iterations = int(num_iterations)
        self.baseline = baseline
        self.fisher_estimator = fisher
        self.fisher_weight = float(fisher_weight)
        self.U = torch.zeros((self.T, 2), device=self.device, dtype=torch.float32)

    def control(self, state, goal, theta):
        with torch.no_grad():
            state = as_tensor(state, self.device).reshape(1, -1)
            goal = as_tensor(goal, self.device).reshape(-1)
            theta = as_tensor(theta, self.device).reshape(-1)
            self.U = shift_control_sequence(self.U)
            if self.num_iterations <= 0:
                return self.U[0].detach().clone()

            est = self.fisher_estimator

            def sample_population(generator, u, num_samples):
                return sample_bounded_control_population(
                    generator,
                    u,
                    self.u_min,
                    self.u_max,
                    self.noise_cov_diag,
                    num_samples,
                )

            def evaluate_population(population, noise):
                theta_batch = broadcast_rows(theta.reshape(1, -1), population.shape[0])
                state_t = state.expand(population.shape[0], state.shape[-1])
                sens_t = torch.zeros((population.shape[0], 5, theta.shape[-1]), device=self.device, dtype=self.U.dtype)
                f_cur = torch.zeros((population.shape[0], theta.shape[-1], theta.shape[-1]), device=self.device, dtype=self.U.dtype)
                costs = torch.zeros((population.shape[0],), device=self.device, dtype=self.U.dtype)
                boed_trace = torch.zeros((population.shape[0],), device=self.device, dtype=self.U.dtype)

                for t in range(population.shape[1]):
                    action_t = population[:, t]
                    noise_t = noise[:, t]
                    state_t, sens_t, jac = _jackal_step_sens(state_t, sens_t, action_t, theta_batch, self.dyn.dt, self.dyn.wheel_base)
                    costs = costs + _mppi_stage_cost(state_t, action_t, noise_t, goal)
                    f_t = jac.transpose(1, 2) @ est.R_inv @ jac
                    f_cur = f_cur + f_t
                    boed_trace = boed_trace + f_t.diagonal(dim1=-2, dim2=-1).sum(-1)
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
                cost_scale=self.dyn.dt,
                temperature=1e-3,
                u_min=self.u_min,
                u_max=self.u_max,
            )
            return self.U[0].detach().clone()


@dataclass
class Summary:
    baseline: str
    device: str
    param_rmse: float
    pred_rmse: float
    goal_dist: float
    fisher_trace: float
    goal: np.ndarray
    state_hist: np.ndarray
    goal_dist_hist: np.ndarray
    theta_rmse_hist: np.ndarray
    fisher_trace_hist: np.ndarray


def prediction_rmse(dynamics: JackalDynamics, states, actions, theta):
    if len(actions) == 0:
        return 0.0
    device = theta.device if isinstance(theta, torch.Tensor) else dynamics.device
    states_t = as_tensor(states, device)
    actions_t = as_tensor(actions, device)
    theta_t = as_tensor(theta, device)
    errs = torch.linalg.norm(dynamics.step(states_t[:-1], actions_t, theta_t) - states_t[1:], dim=-1)
    return float(torch.sqrt(torch.mean(errs.square())).detach().cpu())


def _stack_to_numpy(values, empty_shape):
    if not values:
        return np.empty(empty_shape, np.float32)
    return to_numpy(torch.stack([as_tensor(v) for v in values]))


def run_simulation(cfg: Config | None = None, *, verbose: bool = True, device: str = "auto", **kwargs):
    updates = {("obs_noise" if k == "noise" else k): v for k, v in kwargs.items()}
    cfg = Config(**({} if cfg is None else cfg.__dict__) | updates)
    seed = 0 if cfg.seed is None else cfg.seed
    device_obj = resolve_device(device)
    sim_generator = make_generator(seed, device_obj)

    dyn = JackalDynamics(cfg.dt, cfg.wheel_base, device_obj)
    plan_dyn = JackalDynamics(cfg.plan_dt, cfg.wheel_base, device_obj)
    theta_base = as_tensor(cfg.theta_base, device_obj)
    theta_std = as_tensor(cfg.theta_std, device_obj)
    theta_min = as_tensor(cfg.theta_min, device_obj)
    theta_true = theta_base.clone()
    if cfg.seed is not None:
        theta_true = torch.maximum(theta_base + torch.randn(theta_base.shape, generator=sim_generator, device=device_obj) * theta_std, theta_min)
    dist = ParameterDistribution(theta_base.clone(), torch.diag(theta_std.square()))
    estimator = JackalFisherEstimator(
        dyn,
        dist,
        seed + 1,
        obs_noise_std=cfg.obs_noise,
        fd_epsilon=0.01,
        max_history=20,
        qoed=cfg.baseline.startswith("qoed"),
        var_threshold_for_update=0.05 * 0.05,
        eig_ratio_thresh=cfg.eig_ratio_thresh,
        dist_threshold=cfg.dist_threshold,
        smallest_eigval_threshold=cfg.smallest_eigval_threshold,
        obs_dim=6,
    )
    controller = MPPIController(
        plan_dyn,
        seed + 2,
        num_samples=cfg.num_samples,
        plan_steps=cfg.plan_steps,
        noise_cov_diag=cfg.noise_cov_diag,
        u_min=cfg.u_min,
        u_max=cfg.u_max,
        num_iterations=cfg.num_iterations,
        fisher=estimator,
        baseline=cfg.baseline,
        fisher_weight=cfg.fisher_weight,
    )
    state, goal = as_tensor(cfg.init_state, device_obj), as_tensor(cfg.goal, device_obj)
    state_hist, action_hist, goal_dist_hist, theta_rmse_hist, fisher_trace_hist = [], [], [], [], []
    f_hist = torch.zeros((theta_true.numel(), theta_true.numel()), device=device_obj, dtype=torch.float32)
    for step in range(cfg.steps):
        state_hist.append(state.detach().clone())
        action = controller.control(state, goal, dist.as_theta())
        next_state = dyn.step(state, action, theta_true)
        obs_next = dyn.state_to_observation(next_state, goal).clone()
        obs_next[:6] = obs_next[:6] + torch.randn((6,), generator=sim_generator, device=device_obj) * estimator.obs_noise_std
        estimator.add_sample(state, action, obs_next, goal)
        f_hist = f_hist + _history_fisher_core(
            state[None],
            action[None],
            theta_true.reshape(-1),
            estimator.R_inv,
            estimator.fd_eps,
            torch.ones((1,), device=device_obj, dtype=torch.bool),
            dyn.dt,
            dyn.wheel_base,
        )
        action_hist.append(action.detach().clone())
        state = next_state
        dist_to_goal = torch.linalg.norm(goal[:2] - state[:2])
        theta_rmse = torch.sqrt(torch.mean((estimator.dist.mean - theta_true).square()))
        fisher_trace = torch.trace(f_hist)
        goal_dist_hist.append(dist_to_goal.detach().clone())
        theta_rmse_hist.append(theta_rmse.detach().clone())
        fisher_trace_hist.append(fisher_trace.detach().clone())
        if verbose and ((step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps):
            print(
                f"[{cfg.baseline}] step={step + 1:03d} "
                f"goal_dist={float(dist_to_goal.detach().cpu()):.3f} "
                f"theta_rmse={float(theta_rmse.detach().cpu()):.3f} "
                f"fisher_trace={float(fisher_trace.detach().cpu()):.1f}"
            )
    state_hist_np = _stack_to_numpy([*state_hist, state], (0, 5))
    action_hist_np = _stack_to_numpy(action_hist, (0, 2))
    goal_dist_hist_np = _stack_to_numpy(goal_dist_hist, (0,))
    theta_rmse_hist_np = _stack_to_numpy(theta_rmse_hist, (0,))
    fisher_trace_hist_np = _stack_to_numpy(fisher_trace_hist, (0,))
    pred_rmse = prediction_rmse(dyn, state_hist_np, action_hist_np, estimator.dist.mean)
    err = float(torch.sqrt(torch.mean((estimator.dist.mean - theta_true).square())).detach().cpu())
    if verbose:
        with np.printoptions(precision=3, suppress=True):
            print("\ntrue_theta:", to_numpy(theta_true))
            print("var_diag:  ", to_numpy(torch.diagonal(estimator.dist.cov)))
            print(f"prediction_rmse={pred_rmse:.4f}")
            print("\n", "-" * 30, "\n")
    return Summary(
        baseline=cfg.baseline,
        device=str(device_obj),
        fisher_trace=float(fisher_trace_hist_np[-1]) if fisher_trace_hist_np.size else float(torch.trace(f_hist).detach().cpu()),
        param_rmse=err,
        pred_rmse=pred_rmse,
        goal_dist=float(goal_dist_hist_np[-1]) if goal_dist_hist_np.size else float(torch.linalg.norm(goal[:2] - state[:2]).detach().cpu()),
        goal=to_numpy(goal),
        state_hist=state_hist_np,
        goal_dist_hist=goal_dist_hist_np,
        theta_rmse_hist=theta_rmse_hist_np,
        fisher_trace_hist=fisher_trace_hist_np,
    )


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
                "fisher_weight",
                "num_iterations",
                "num_samples",
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
    parser = argparse.ArgumentParser(description="PyTorch Jackal parameter-estimation demo.")
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
