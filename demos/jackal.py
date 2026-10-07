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
    to_numpy,
)

import numpy as np
import torch

from boed import BOED, QOED, QOED_AGNOSTIC, FisherEstimator, ParameterDistribution

BASELINES = (QOED, QOED_AGNOSTIC, BOED)
METRICS = ("param_rmse", "pred_rmse", "goal_dist", "fisher_trace")


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
    wheel_base: float = 0.37559
    init_state: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    goal: tuple[float, ...] = (5.0, 5.0, 0.0)
    theta_base: tuple[float, ...] = (20.0, 1.0, 23.5, 23.5, 23.5, 23.5, 0.0, 0.0)
    theta_std: tuple[float, ...] = (10.0, 0.5, 2.0, 2.0, 2.0, 2.0, 50.0, 50.0)
    theta_min: tuple[float, ...] = (10.0, 0.4, 2.0, 2.0, 2.0, 2.0, 0.0, 0.0)
    u_min: tuple[float, ...] = (-1.0, -float(np.pi / 2))
    u_max: tuple[float, ...] = (1.0, float(np.pi / 2))
    noise_cov_diag: tuple[float, ...] = (0.4, 0.4)
    plan_steps: int = 10


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


def _jackal_obs_batch(state):
    x, y, yaw, v, omega = state.T
    return torch.stack([x, y, yaw, v * torch.cos(yaw), v * torch.sin(yaw), omega], -1)


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

    dmass = (theta[:, 0] > 0.1).to(theta.dtype)
    dfriction = (theta[:, 1] > 0.0).to(theta.dtype) + 0.5 * (theta[:, 1] == 0.0).to(theta.dtype)
    dfx = (theta[:, 6] > 0.0).to(theta.dtype) + 0.5 * (theta[:, 6] == 0.0).to(theta.dtype)
    dinertia = (inertia_raw > 1e-3).to(theta.dtype) * 0.5 * wheel_base * wheel_base * dmass
    dv_theta = torch.zeros((k, p), device=theta.device, dtype=theta.dtype)
    dv_theta[:, 0] = dt * (-force / (mass * mass)) * dmass
    dv_theta[:, 1] = dt * (-v / mass) * dfriction
    dv_theta[:, 2:6] = (dt * 0.5 * v_cmd / mass)[:, None].expand(k, 4)
    dv_theta[:, 6] = dt * (-dfx / mass)

    domega_theta = torch.zeros((k, p), device=theta.device, dtype=theta.dtype)
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


def _history_fisher_core(s_seq, u_seq, theta, r_inv, fd_eps, dt, wheel_base):
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
        f = torch.nan_to_num(f_t.sum(0), nan=0.0, posinf=1e6, neginf=-1e6)
        return 0.5 * (f + f.T)


class JackalDynamics:
    def __init__(self, dt: float, wheel_base: float):
        self.dt = dt
        self.wheel_base = wheel_base

    def step(self, state, action, theta):
        state = as_tensor(state)
        squeeze = state.ndim == 1
        state = batch_if_vector(state, state.device)
        action = batch_if_vector(action, state.device)
        theta = broadcast_rows(batch_if_vector(theta, state.device), state.shape[0])
        out = _jackal_step_batch(state, action, theta, self.dt, self.wheel_base)
        return out[0] if squeeze else out

    def observe(self, state):
        state = as_tensor(state)
        obs = _jackal_obs_batch(batch_if_vector(state, state.device))
        return obs[0] if state.ndim == 1 else obs

    def state_to_observation(self, state, goal):
        obs = self.observe(state)
        squeeze = obs.ndim == 1
        obs = batch_if_vector(obs, obs.device)
        goal = batch_if_vector(goal, obs.device)
        if goal.shape[-1] == 2:
            goal = torch.cat([goal, torch.zeros((*goal.shape[:-1], 1), device=goal.device, dtype=goal.dtype)], -1)
        out = torch.cat([obs, broadcast_rows(goal, obs.shape[0])[:, :3]], -1)
        return out[0] if squeeze else out


class JackalFisherEstimator(FisherEstimator):
    """Jackal-specialized fast kernels on top of the generic BOED estimator."""

    def compute_fisher(self, use_cache: bool = True):
        if self._fisher_cache is not None and use_cache:
            return self._fisher_cache
        if not self.history:
            p = self.dist.mean.numel()
            return torch.zeros((p, p), device=self.device, dtype=self.dist.mean.dtype)
        s_seq, u_seq, *_ = self.stack(self.history)
        f = _history_fisher_core(s_seq, u_seq, self.dist.mean.reshape(-1), self.R_inv, self.fd_eps, self.dyn.dt, self.dyn.wheel_base)
        if use_cache:
            self._fisher_cache = f
        return f


class MPPIController:
    def __init__(self, dynamics: JackalDynamics, seed: int, fisher: FisherEstimator, cfg: Config):
        self.dyn = dynamics
        self.device = fisher.device
        self.generator = make_generator(seed, self.device)
        self.K = int(cfg.num_samples)
        self.noise_cov_diag = as_tensor(cfg.noise_cov_diag, self.device)
        self.u_min = as_tensor(cfg.u_min, self.device).reshape(1, 2)
        self.u_max = as_tensor(cfg.u_max, self.device).reshape(1, 2)
        self.num_iterations = int(cfg.num_iterations)
        self.baseline = cfg.baseline
        self.fisher_estimator = fisher
        self.fisher_weight = float(cfg.fisher_weight)
        self.U = torch.zeros((int(cfg.plan_steps), 2), device=self.device, dtype=torch.float32)

    def _sample(self, u):
        bound_std = torch.minimum((0.5 * (u - self.u_min)).square(), (0.5 * (self.u_max - u)).square())
        std = torch.sqrt(torch.clamp(torch.minimum(bound_std, self.noise_cov_diag[None]), min=1e-6))
        noise = torch.randn((self.K, *u.shape), generator=self.generator, device=u.device, dtype=u.dtype) * std[None]
        return torch.clamp(u[None] + noise, min=self.u_min, max=self.u_max), noise

    def control(self, state, goal, theta):
        with torch.no_grad():
            state = as_tensor(state, self.device).reshape(1, -1)
            goal = as_tensor(goal, self.device).reshape(-1)
            theta = as_tensor(theta, self.device).reshape(-1)
            self.U = shift_control_sequence(self.U)
            if self.num_iterations <= 0:
                return self.U[0].clone()
            r_inv = self.fisher_estimator.R_inv

            def evaluate(population, noise):
                k, p = population.shape[0], theta.shape[-1]
                theta_batch = broadcast_rows(theta.reshape(1, -1), k)
                state_t = state.expand(k, state.shape[-1])
                sens_t = torch.zeros((k, 5, p), device=self.device, dtype=self.U.dtype)
                f_cur = torch.zeros((k, p, p), device=self.device, dtype=self.U.dtype)
                costs = torch.zeros((k,), device=self.device, dtype=self.U.dtype)
                boed_trace = torch.zeros((k,), device=self.device, dtype=self.U.dtype)
                for t in range(population.shape[1]):
                    action_t = population[:, t]
                    state_t, sens_t, jac = _jackal_step_sens(state_t, sens_t, action_t, theta_batch, self.dyn.dt, self.dyn.wheel_base)
                    costs = costs + _mppi_stage_cost(state_t, action_t, noise[:, t], goal)
                    f_t = jac.transpose(1, 2) @ r_inv @ jac
                    f_cur = f_cur + f_t
                    boed_trace = boed_trace + f_t.diagonal(dim1=-2, dim2=-1).sum(-1)
                return costs, f_cur, boed_trace

            self.U = mppi_info_gain_refine(
                u=self.U,
                estimator=self.fisher_estimator,
                baseline=self.baseline,
                num_iterations=self.num_iterations,
                sample_population=self._sample,
                evaluate_population=evaluate,
                fisher_weight=self.fisher_weight,
                cost_scale=self.dyn.dt,
                temperature=1e-3,
                u_min=self.u_min,
                u_max=self.u_max,
            )
            return self.U[0].clone()


@dataclass
class Summary:
    baseline: str
    device: str
    param_rmse: float
    pred_rmse: float
    goal_dist: float
    fisher_trace: float


def prediction_rmse(dynamics: JackalDynamics, states, actions, theta):
    if len(actions) == 0:
        return 0.0
    errs = torch.linalg.norm(dynamics.step(states[:-1], actions, theta) - states[1:], dim=-1)
    return float(torch.sqrt(torch.mean(errs.square())))


def run_simulation(cfg: Config = DEFAULT_CONFIG, *, verbose: bool = True, device: str = "auto") -> Summary:
    seed = 0 if cfg.seed is None else cfg.seed
    device = resolve_device(device)
    sim_generator = make_generator(seed, device)

    dyn = JackalDynamics(cfg.dt, cfg.wheel_base)
    theta_base = as_tensor(cfg.theta_base, device)
    theta_std = as_tensor(cfg.theta_std, device)
    theta_true = theta_base.clone()
    if cfg.seed is not None:
        noise = torch.randn(theta_base.shape, generator=sim_generator, device=device)
        theta_true = torch.maximum(theta_base + noise * theta_std, as_tensor(cfg.theta_min, device))
    estimator = JackalFisherEstimator(
        dyn,
        ParameterDistribution(theta_base.clone(), torch.diag(theta_std.square())),
        seed + 1,
        obs_noise_std=cfg.obs_noise,
        fd_epsilon=0.01,
        max_history=20,
        var_threshold_for_update=0.05 * 0.05,
        eig_ratio_thresh=cfg.eig_ratio_thresh,
        dist_threshold=cfg.dist_threshold,
        smallest_eigval_threshold=cfg.smallest_eigval_threshold,
        obs_dim=6,
    )
    dist = estimator.dist
    controller = MPPIController(dyn, seed + 2, estimator, cfg)
    state, goal = as_tensor(cfg.init_state, device), as_tensor(cfg.goal, device)
    state_hist, action_hist = [], []
    f_hist = torch.zeros((theta_true.numel(), theta_true.numel()), device=device, dtype=torch.float32)
    for step in range(cfg.steps):
        state_hist.append(state)
        action = controller.control(state, goal, dist.mean)
        next_state = dyn.step(state, action, theta_true)
        obs_next = dyn.state_to_observation(next_state, goal)
        obs_next[:6] = obs_next[:6] + torch.randn((6,), generator=sim_generator, device=device) * estimator.obs_noise_std
        estimator.add_sample(state, action, obs_next, goal)
        f_hist = f_hist + _history_fisher_core(
            state[None], action[None], theta_true.reshape(-1), estimator.R_inv, estimator.fd_eps, dyn.dt, dyn.wheel_base
        )
        action_hist.append(action)
        state = next_state
        if verbose and ((step + 1) % cfg.log_every == 0 or step + 1 == cfg.steps):
            print(
                f"[{cfg.baseline}] step={step + 1:03d} "
                f"goal_dist={float(torch.linalg.norm(goal[:2] - state[:2])):.3f} "
                f"theta_rmse={float(torch.sqrt(torch.mean((dist.mean - theta_true).square()))):.3f} "
                f"fisher_trace={float(torch.trace(f_hist)):.1f}"
            )
    state_hist.append(state)
    pred_rmse = prediction_rmse(dyn, torch.stack(state_hist), torch.stack(action_hist) if action_hist else [], dist.mean)
    if verbose:
        with np.printoptions(precision=3, suppress=True):
            print("\ntrue_theta:", to_numpy(theta_true))
            print("var_diag:  ", to_numpy(torch.diagonal(dist.cov)))
            print(f"prediction_rmse={pred_rmse:.4f}")
            print("\n", "-" * 30, "\n")
    return Summary(
        baseline=cfg.baseline,
        device=str(device),
        param_rmse=float(torch.sqrt(torch.mean((dist.mean - theta_true).square()))),
        pred_rmse=pred_rmse,
        goal_dist=float(torch.linalg.norm(goal[:2] - state[:2])),
        fisher_trace=float(torch.trace(f_hist)),
    )


def main(argv=None):
    return run_demo("PyTorch Jackal parameter-estimation demo.", Config, run_simulation, BASELINES, METRICS, argv)


if __name__ == "__main__":
    raise SystemExit(main())
