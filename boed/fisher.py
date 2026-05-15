from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch

from .objectives import QOED as QOED_MODE, _as_tensor, _finite, score_paths


def _make_generator(seed_or_generator=None, device=None) -> torch.Generator:
    if isinstance(seed_or_generator, torch.Generator):
        return seed_or_generator
    seed = 0
    if seed_or_generator is not None:
        try:
            seed = int(np.asarray(seed_or_generator).reshape(-1)[-1])
        except Exception:
            seed = int(seed_or_generator)
    device = torch.device("cpu" if device is None else device)
    try:
        gen = torch.Generator(device=device)
    except RuntimeError:
        gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


@dataclass
class ParameterDistribution:
    mean: torch.Tensor
    cov: torch.Tensor

    def __post_init__(self):
        mean = _as_tensor(self.mean)
        if not mean.is_floating_point():
            mean = mean.float()
        self.mean = mean.detach().clone()
        self.cov = _as_tensor(self.cov, device=self.mean.device, dtype=self.mean.dtype).detach().clone()

    def as_theta(self):
        return self.mean


class FisherEstimator:
    """Finite-difference Fisher estimator for arbitrary PyTorch dynamics.

    Dynamics must provide ``step(state, action, theta)``. Observations are
    produced by ``observation_fn(state, context)`` when supplied, otherwise by
    ``dynamics.state_to_observation(state, context)``, ``dynamics.observe``, or
    the state itself.
    """

    def __init__(
        self,
        dynamics,
        dist: ParameterDistribution | None = None,
        key=None,
        *,
        param_dist: ParameterDistribution | None = None,
        obs_noise_std: float = 0.05,
        obs_noise_cov=None,
        fd_epsilon: float = 0.01,
        fd_eps: float | None = None,
        fd_delta_floor: float = 1.0,
        max_history: int | None = 20,
        qoed: bool | None = None,
        QOED: bool | None = None,
        var_threshold_for_update: float = 0.0025,
        eig_ratio_thresh: float = 0.01,
        dist_threshold: float = 0.05,
        param_contrib_ratio: float = 1e-3,
        smallest_eigval_threshold: float = 0.1,
        obs_dim: int | None = None,
        observation_fn: Callable | None = None,
        cem_samples: int = 2048,
        cem_elites: int | None = None,
        cem_iters: int = 5,
        param_min: float | torch.Tensor = 1e-6,
        param_max: float | torch.Tensor = 100.0,
    ):
        self.dyn = dynamics
        self.dist = param_dist or dist
        if self.dist is None:
            raise ValueError("Parameter distribution is required.")
        if not isinstance(self.dist, ParameterDistribution):
            self.dist = ParameterDistribution(self.dist.mean, self.dist.cov)

        self.device = self.dist.mean.device
        self.generator = _make_generator(key, self.device)
        self.key = self.generator
        self.obs_noise_std = float(obs_noise_std)
        self.obs_noise_cov = None if obs_noise_cov is None else _as_tensor(obs_noise_cov, device=self.device, dtype=self.dist.mean.dtype)
        self.fd_eps = float(fd_epsilon if fd_eps is None else fd_eps)
        self.fd_delta_floor = float(fd_delta_floor)
        self.max_history = max_history
        self.QOED = bool(True if qoed is None and QOED is None else (qoed if qoed is not None else QOED))
        self.var_threshold_for_update = float(var_threshold_for_update)
        self.eig_ratio_thresh = float(eig_ratio_thresh)
        self.dist_threshold = float(dist_threshold)
        self.param_contrib_ratio = float(param_contrib_ratio)
        self.smallest_eigval_threshold = float(smallest_eigval_threshold)
        self.obs_dim = int(obs_dim) if obs_dim is not None else None
        self.observation_fn = observation_fn
        self.cem_samples = int(cem_samples)
        self.cem_elites = min(int(cem_elites or max(1, self.cem_samples // 8)), self.cem_samples)
        self.cem_iters = int(cem_iters)
        p = self.dist.mean.numel()
        self.param_min = self._parameter_bound(param_min, p, "param_min")
        self.param_max = self._parameter_bound(param_max, p, "param_max")

        self.R_inv = None
        if self.obs_noise_cov is not None:
            self.obs_dim = int(self.obs_noise_cov.shape[0])
            self.R_inv = torch.linalg.pinv(self.obs_noise_cov)
        elif self.obs_dim is not None:
            self.R_inv = (1.0 / (self.obs_noise_std**2)) * torch.eye(
                self.obs_dim,
                device=self.device,
                dtype=self.dist.mean.dtype,
            )

        self.history = []
        self.est_history = []
        self.active_mask = None
        self.score_mask = None
        self.eig_mask = None
        p = self.dist.mean.numel()
        self.F_cur = torch.zeros((1, p, p), device=self.device, dtype=self.dist.mean.dtype)
        self.F_history = torch.zeros((p, p), device=self.device, dtype=self.dist.mean.dtype)
        self.min_idx = 0
        self._fisher_cache = None

    def _parameter_bound(self, value, dim: int, name: str) -> torch.Tensor:
        bound = _as_tensor(value, device=self.device, dtype=self.dist.mean.dtype).reshape(-1)
        if bound.numel() == 1:
            return bound.expand(dim).detach().clone()
        if bound.numel() != dim:
            raise ValueError(f"Expected {name} dim {dim}, got {bound.numel()}")
        return bound.detach().clone()

    def _ensure_r_inv(self, obs_dim: int | None = None, dtype=None):
        if obs_dim is not None and self.obs_dim is None:
            self.obs_dim = int(obs_dim)
        if self.R_inv is None:
            if self.obs_dim is None:
                raise RuntimeError("obs_dim could not be inferred before Fisher computation")
            self.R_inv = (1.0 / (self.obs_noise_std**2)) * torch.eye(
                self.obs_dim,
                device=self.device,
                dtype=dtype or self.dist.mean.dtype,
            )
        return self.R_inv

    def _observed(self, obs):
        obs = _as_tensor(obs, device=self.device)
        return obs if self.obs_dim is None else obs[..., : self.obs_dim]

    def _context_or_none(self, context):
        if context is None:
            return None
        context = _as_tensor(context, device=self.device)
        return None if context.shape[-1] == 0 else context

    def _predict_observation(self, state, context=None):
        context = self._context_or_none(context)
        if self.observation_fn is not None:
            return self.observation_fn(state, context)
        if context is not None and hasattr(self.dyn, "state_to_observation"):
            return self.dyn.state_to_observation(state, context)
        if hasattr(self.dyn, "observe"):
            return self.dyn.observe(state)
        return state

    def _as_context(self, context, like):
        like = _as_tensor(like, device=self.device)
        if context is None:
            return torch.zeros((0,), device=self.device, dtype=like.dtype)
        return _as_tensor(context, device=self.device, dtype=like.dtype).reshape(-1)

    def _batch_context(self, context, n: int, dtype):
        if context is None:
            return torch.zeros((n, 0), device=self.device, dtype=dtype)
        context = _as_tensor(context, device=self.device, dtype=dtype)
        context = context[None] if context.ndim == 1 else context
        return context.expand(n, context.shape[-1]) if context.shape[0] == 1 and n > 1 else context

    @staticmethod
    def _stack(rows):
        return tuple(torch.stack(items) for items in zip(*rows))

    def add_sample(self, state, control, obs_next, context=None):
        state = _as_tensor(state, device=self.device, dtype=self.dist.mean.dtype).detach().reshape(-1)
        row = (
            state,
            _as_tensor(control, device=self.device, dtype=state.dtype).detach().reshape(-1),
            _as_tensor(obs_next, device=self.device, dtype=state.dtype).detach().reshape(-1),
            self._as_context(context, state).detach(),
        )
        if self.obs_dim is None:
            self.obs_dim = int(row[2].numel())
            self._ensure_r_inv(self.obs_dim, row[2].dtype)
        self.history.append(row)
        self.est_history.append(row)
        self._fisher_cache = None
        if self.max_history is not None and len(self.history) > self.max_history:
            self.update_posterior()
            self.history.pop(0)
            self.est_history.pop(0)
            self._fisher_cache = None

    def _traj_jacobian(self, states, actions, contexts, theta):
        t_len, p = states.shape[0], theta.numel()
        eps = self.fd_eps * (theta.abs() + self.fd_delta_floor)
        eye = torch.eye(p, device=theta.device, dtype=theta.dtype)
        theta_p = theta[None] + eye * eps[:, None]
        theta_m = theta[None] - eye * eps[:, None]
        theta_batch = torch.cat([theta_p.repeat_interleave(t_len, 0), theta_m.repeat_interleave(t_len, 0)], 0)
        s_batch = states.repeat((2 * p, 1))
        u_batch = actions.repeat((2 * p, 1))
        c_batch = None if contexts.shape[-1] == 0 else contexts.repeat((2 * p, 1))
        obs = self._observed(self._predict_observation(self.dyn.step(s_batch, u_batch, theta_batch), c_batch))
        self._ensure_r_inv(obs.shape[-1], obs.dtype)
        split = p * t_len
        obs_p = obs[:split].reshape(p, t_len, -1)
        obs_m = obs[split:].reshape(p, t_len, -1)
        return (obs_p - obs_m).permute(1, 2, 0) / (2.0 * eps.reshape(1, 1, p))

    def compute_fisher(self, use_cache: bool = True):
        if self._fisher_cache is not None and use_cache:
            return self._fisher_cache
        if not self.history:
            p = self.dist.mean.numel()
            return torch.zeros((p, p), device=self.device, dtype=self.dist.mean.dtype)

        states, actions, _, contexts = self._stack(self.history)
        with torch.no_grad():
            jac = self._traj_jacobian(states, actions, contexts, self.dist.as_theta().reshape(-1))
            r_inv = self._ensure_r_inv(jac.shape[1], jac.dtype)
            f_t = jac.transpose(1, 2) @ r_inv @ jac
            f = _finite(f_t.sum(0))
            f = 0.5 * (f + f.T)
        if use_cache:
            self._fisher_cache = f
        return f

    def fisher_trace_value(self):
        return torch.trace(self.compute_fisher())

    def fisher_trace(self) -> float:
        return float(self.fisher_trace_value().detach().cpu())

    def update_posterior(self, search_sigma_window: float = 3.0):
        if not self.est_history:
            return

        with torch.no_grad():
            mu0 = self.dist.as_theta().reshape(-1)
            p = mu0.numel()
            sigma0 = self.dist.cov.reshape(p, p)
            var_diag = torch.diagonal(sigma0)
            active = var_diag >= self.var_threshold_for_update
            self.active_mask = active

            states, actions, obs, contexts = self._stack(self.est_history)
            t_len = states.shape[0]
            r_inv = self._ensure_r_inv(self.obs_dim, obs.dtype)
            lambda0 = torch.linalg.pinv(sigma0)
            std0 = torch.sqrt(torch.diagonal(sigma0 + 1e-6 * torch.eye(p, device=self.device, dtype=mu0.dtype)))
            target = self._observed(obs)
            mu = mu0
            std = torch.clamp(std0 * search_sigma_window, 0.001, 25.0)
            inactive = ~active
            n, n_elite = self.cem_samples, self.cem_elites

            for _ in range(self.cem_iters):
                noise = torch.randn((n, p), generator=self.generator, device=self.device, dtype=mu0.dtype)
                theta = torch.clamp(mu + noise * std, self.param_min, self.param_max)
                theta = torch.where(inactive[None], mu0[None], theta)
                theta_batch = theta[None].expand(t_len, n, p).reshape(-1, p)
                s_batch = states[:, None].expand(t_len, n, states.shape[-1]).reshape(-1, states.shape[-1])
                u_batch = actions[:, None].expand(t_len, n, actions.shape[-1]).reshape(-1, actions.shape[-1])
                c_batch = None
                if contexts.shape[-1] != 0:
                    c_batch = contexts[:, None].expand(t_len, n, contexts.shape[-1]).reshape(-1, contexts.shape[-1])
                pred = self._observed(self._predict_observation(self.dyn.step(s_batch, u_batch, theta_batch), c_batch))
                pred = pred.reshape(t_len, n, -1)
                residual = target[:, None] - pred
                obs_loss = torch.einsum("tni,ij,tnj->n", residual, r_inv, residual)
                delta = theta - mu0
                cost = torch.nan_to_num(obs_loss + (delta @ lambda0 * delta).sum(1), nan=torch.inf, posinf=torch.inf)
                elite_idx = torch.topk(-cost, n_elite).indices
                elites = theta[elite_idx]
                mu = elites.mean(0)
                std = elites.std(0, unbiased=False) + 1e-9

            mu = torch.where(inactive, mu0, mu)
            old = self.dist.mean
            self.dist.mean = mu
            f_opt = torch.diag(torch.clamp(torch.diagonal(self.compute_fisher(use_cache=False)), min=0.0))
            self.dist.mean = old
            eye = torch.eye(p, device=self.device, dtype=mu0.dtype)
            sigma_post = torch.clamp(torch.linalg.inv(lambda0 + f_opt + 1e-9 * eye), min=1e-10)
            if bool(inactive.any()):
                sigma_post[inactive, :] = sigma0[inactive, :]
                sigma_post[:, inactive] = sigma0[:, inactive]
                sigma_post = 0.5 * (sigma_post + sigma_post.T)
            self.dist.mean = mu.detach()
            self.dist.cov = sigma_post.detach()
            self._fisher_cache = None

    def local_fisher_trace_path(self, states_seq, controls_seq, context=None, baseline: str = QOED_MODE):
        controls_seq = _as_tensor(controls_seq, device=self.device, dtype=self.dist.mean.dtype)
        k, horizon, p = controls_seq.shape[0], controls_seq.shape[1], self.dist.mean.numel()
        states_seq = _as_tensor(states_seq, device=self.device, dtype=controls_seq.dtype)
        state = states_seq[:, 0] if states_seq.ndim == 3 else self._batch_context(states_seq, k, controls_seq.dtype)
        context = self._batch_context(context, k, controls_seq.dtype)
        theta = self.dist.as_theta().reshape(-1)
        eps = self.fd_eps * (theta.abs() + self.fd_delta_floor)
        eye = torch.eye(p, device=self.device, dtype=theta.dtype)
        theta_p = theta[None] + eye * eps[:, None]
        theta_m = theta[None] - eye * eps[:, None]
        theta_batch = torch.cat(
            [
                theta_p[None].expand(k, p, p).reshape(k * p, p),
                theta_m[None].expand(k, p, p).reshape(k * p, p),
            ],
            0,
        )
        init = state[:, None].expand(k, p, state.shape[-1]).reshape(k * p, state.shape[-1])
        curr = init.repeat((2, 1))
        c_path = None
        if context.shape[-1] != 0:
            c_init = context[:, None].expand(k, p, context.shape[-1]).reshape(k * p, context.shape[-1])
            c_path = c_init.repeat((2, 1))

        f_cur = torch.zeros((k, p, p), device=self.device, dtype=theta.dtype)
        boed_trace = torch.zeros((k,), device=self.device, dtype=theta.dtype)
        with torch.no_grad():
            for t in range(horizon):
                action = controls_seq[:, t, None].expand(k, p, controls_seq.shape[-1]).reshape(k * p, controls_seq.shape[-1])
                action = action.repeat((2, 1))
                curr = self.dyn.step(curr, action, theta_batch)
                obs = self._observed(self._predict_observation(curr, c_path))
                self._ensure_r_inv(obs.shape[-1], obs.dtype)
                obs_p = obs[: k * p].reshape(k, p, -1)
                obs_m = obs[k * p :].reshape(k, p, -1)
                jac = (obs_p - obs_m).permute(0, 2, 1) / (2.0 * eps.reshape(1, 1, p))
                f_t = jac.transpose(1, 2) @ self.R_inv @ jac
                f_cur = f_cur + f_t
                boed_trace = boed_trace + f_t.diagonal(dim1=-2, dim2=-1).sum(-1)
        self.F_cur = f_cur
        return self.score_paths(boed_trace, baseline=baseline)

    def score_paths(self, boed_trace, baseline: str = QOED_MODE):
        scores, mask = score_paths(
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
        self.score_mask = mask
        self.eig_mask = mask
        return scores


FisherParameterEstimator = FisherEstimator
