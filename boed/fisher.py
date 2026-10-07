from __future__ import annotations

from dataclasses import dataclass

import torch

from .objectives import QOED, _as_tensor, _finite, score_from_mask, score_mask


@dataclass
class ParameterDistribution:
    mean: torch.Tensor
    cov: torch.Tensor

    def __post_init__(self):
        mean = _as_tensor(self.mean)
        self.mean = (mean if mean.is_floating_point() else mean.float()).detach().clone()
        self.cov = _as_tensor(self.cov, device=self.mean.device, dtype=self.mean.dtype).detach().clone()


class FisherEstimator:
    """Finite-difference Fisher estimator for arbitrary PyTorch dynamics.

    Dynamics must provide ``step(state, action, theta)``. Observations are
    produced by ``dynamics.state_to_observation(state, context)``,
    ``dynamics.observe``, or the state itself.
    """

    def __init__(
        self,
        dynamics,
        dist: ParameterDistribution,
        seed: int | None = None,
        *,
        obs_noise_std: float = 0.05,
        fd_epsilon: float = 0.01,
        fd_delta_floor: float = 1.0,
        max_history: int | None = 20,
        var_threshold_for_update: float = 0.0025,
        eig_ratio_thresh: float = 0.01,
        dist_threshold: float = 0.05,
        param_contrib_ratio: float = 1e-3,
        smallest_eigval_threshold: float = 0.1,
        obs_dim: int | None = None,
        cem_samples: int = 2048,
        cem_iters: int = 5,
        param_min: float | torch.Tensor = 1e-6,
        param_max: float | torch.Tensor = 100.0,
    ):
        self.dyn = dynamics
        self.dist = dist
        self.device = dist.mean.device
        self.generator = torch.Generator(device=self.device).manual_seed(0 if seed is None else int(seed))
        self.obs_noise_std = float(obs_noise_std)
        self.fd_eps = float(fd_epsilon)
        self.fd_delta_floor = float(fd_delta_floor)
        self.max_history = max_history
        self.var_threshold_for_update = float(var_threshold_for_update)
        self.eig_ratio_thresh = float(eig_ratio_thresh)
        self.dist_threshold = float(dist_threshold)
        self.param_contrib_ratio = float(param_contrib_ratio)
        self.smallest_eigval_threshold = float(smallest_eigval_threshold)
        self.cem_samples = int(cem_samples)
        self.cem_elites = max(1, self.cem_samples // 8)
        self.cem_iters = int(cem_iters)
        p = dist.mean.numel()
        self.param_min = _as_tensor(param_min, device=self.device, dtype=dist.mean.dtype).reshape(-1).expand(p).clone()
        self.param_max = _as_tensor(param_max, device=self.device, dtype=dist.mean.dtype).reshape(-1).expand(p).clone()

        self.obs_dim = obs_dim
        self.R_inv = None
        if obs_dim is not None:
            self._ensure_r_inv(obs_dim)
        self.history = []
        self.est_history = []
        self._fisher_cache = None
        self._mask_cache = None

    def _ensure_r_inv(self, obs_dim: int):
        if self.obs_dim is None:
            self.obs_dim = int(obs_dim)
        if self.R_inv is None:
            eye = torch.eye(self.obs_dim, device=self.device, dtype=self.dist.mean.dtype)
            self.R_inv = (1.0 / (self.obs_noise_std**2)) * eye
        return self.R_inv

    def _observe(self, state, context=None):
        if context is not None and context.shape[-1] > 0 and hasattr(self.dyn, "state_to_observation"):
            obs = self.dyn.state_to_observation(state, context)
        elif hasattr(self.dyn, "observe"):
            obs = self.dyn.observe(state)
        else:
            obs = state
        return obs if self.obs_dim is None else obs[..., : self.obs_dim]

    @staticmethod
    def stack(rows):
        return tuple(torch.stack(items) for items in zip(*rows))

    def add_sample(self, state, control, obs_next, context=None):
        def row(x):
            return _as_tensor(x, device=self.device, dtype=self.dist.mean.dtype).detach().reshape(-1)

        sample = (
            row(state),
            row(control),
            row(obs_next),
            torch.zeros((0,), device=self.device, dtype=self.dist.mean.dtype) if context is None else row(context),
        )
        if self.obs_dim is None:
            self._ensure_r_inv(sample[2].numel())
        self.history.append(sample)
        self.est_history.append(sample)
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
        c_batch = contexts.repeat((2 * p, 1))
        obs = self._observe(self.dyn.step(s_batch, u_batch, theta_batch), c_batch)
        self._ensure_r_inv(obs.shape[-1])
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

        states, actions, _, contexts = self.stack(self.history)
        with torch.no_grad():
            jac = self._traj_jacobian(states, actions, contexts, self.dist.mean.reshape(-1))
            f = _finite((jac.transpose(1, 2) @ self.R_inv @ jac).sum(0))
            f = 0.5 * (f + f.T)
        if use_cache:
            self._fisher_cache = f
        return f

    def mask(self, baseline: str, candidate_mask=None):
        fisher, cov = self.compute_fisher(), self.dist.cov
        cache = self._mask_cache
        if cache is not None and cache[0] is fisher and cache[1] is cov and cache[2] == baseline and cache[3] is candidate_mask:
            return cache[4]
        mask = score_mask(
            fisher,
            cov,
            baseline,
            bool(self.history),
            self.eig_ratio_thresh,
            self.dist_threshold,
            self.param_contrib_ratio,
            self.var_threshold_for_update,
            self.smallest_eigval_threshold,
            candidate_mask,
        )
        self._mask_cache = (fisher, cov, baseline, candidate_mask, mask)
        return mask

    def update_posterior(self):
        if not self.est_history:
            return

        with torch.no_grad():
            mu0 = self.dist.mean.reshape(-1)
            p = mu0.numel()
            sigma0 = self.dist.cov.reshape(p, p)
            inactive = ~(torch.diagonal(sigma0) >= self.var_threshold_for_update)

            states, actions, obs, contexts = self.stack(self.est_history)
            t_len = states.shape[0]
            r_inv = self._ensure_r_inv(self.obs_dim)
            lambda0 = torch.linalg.pinv(sigma0)
            std0 = torch.sqrt(torch.diagonal(sigma0 + 1e-6 * torch.eye(p, device=self.device, dtype=mu0.dtype)))
            target = obs if self.obs_dim is None else obs[..., : self.obs_dim]
            mu = mu0
            std = torch.clamp(std0 * 3.0, 0.001, 25.0)
            n = self.cem_samples

            def tile(x):
                return x[:, None].expand(t_len, n, x.shape[-1]).reshape(t_len * n, x.shape[-1])

            for _ in range(self.cem_iters):
                noise = torch.randn((n, p), generator=self.generator, device=self.device, dtype=mu0.dtype)
                theta = torch.clamp(mu + noise * std, self.param_min, self.param_max)
                theta = torch.where(inactive[None], mu0[None], theta)
                theta_batch = theta[None].expand(t_len, n, p).reshape(-1, p)
                pred = self._observe(self.dyn.step(tile(states), tile(actions), theta_batch), tile(contexts))
                residual = target[:, None] - pred.reshape(t_len, n, -1)
                obs_loss = torch.einsum("tni,ij,tnj->n", residual, r_inv, residual)
                delta = theta - mu0
                cost = torch.nan_to_num(obs_loss + (delta @ lambda0 * delta).sum(1), nan=torch.inf, posinf=torch.inf)
                elites = theta[torch.topk(-cost, self.cem_elites).indices]
                mu = elites.mean(0)
                std = elites.std(0, unbiased=False) + 1e-9

            mu = torch.where(inactive, mu0, mu)
            old = self.dist.mean
            self.dist.mean = mu
            f_opt = torch.diag(torch.clamp(torch.diagonal(self.compute_fisher(use_cache=False)), min=0.0))
            self.dist.mean = old
            eye = torch.eye(p, device=self.device, dtype=mu0.dtype)
            sigma_post = torch.clamp(torch.linalg.inv(lambda0 + f_opt + 1e-9 * eye), min=1e-10)
            if inactive.any():
                sigma_post[inactive, :] = sigma0[inactive, :]
                sigma_post[:, inactive] = sigma0[:, inactive]
                sigma_post = 0.5 * (sigma_post + sigma_post.T)
            self.dist.mean = mu.detach()
            self.dist.cov = sigma_post.detach()
            self._fisher_cache = None

    def local_fisher_trace_path(self, state, controls_seq, baseline: str = QOED):
        controls_seq = _as_tensor(controls_seq, device=self.device, dtype=self.dist.mean.dtype)
        k, horizon, p = controls_seq.shape[0], controls_seq.shape[1], self.dist.mean.numel()
        state = _as_tensor(state, device=self.device, dtype=controls_seq.dtype)
        state = state[None] if state.ndim == 1 else state
        state = state.expand(k, state.shape[-1]) if state.shape[0] == 1 and k > 1 else state
        theta = self.dist.mean.reshape(-1)
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
        curr = state[:, None].expand(k, p, state.shape[-1]).reshape(k * p, state.shape[-1]).repeat((2, 1))

        f_cur = torch.zeros((k, p, p), device=self.device, dtype=theta.dtype)
        boed_trace = torch.zeros((k,), device=self.device, dtype=theta.dtype)
        with torch.no_grad():
            for t in range(horizon):
                action = controls_seq[:, t, None].expand(k, p, controls_seq.shape[-1]).reshape(k * p, controls_seq.shape[-1])
                curr = self.dyn.step(curr, action.repeat((2, 1)), theta_batch)
                obs = self._observe(curr)
                r_inv = self._ensure_r_inv(obs.shape[-1])
                obs_p = obs[: k * p].reshape(k, p, -1)
                obs_m = obs[k * p :].reshape(k, p, -1)
                jac = (obs_p - obs_m).permute(0, 2, 1) / (2.0 * eps.reshape(1, 1, p))
                f_t = jac.transpose(1, 2) @ r_inv @ jac
                f_cur = f_cur + f_t
                boed_trace = boed_trace + f_t.diagonal(dim1=-2, dim2=-1).sum(-1)
        return score_from_mask(f_cur, boed_trace, baseline, self.mask(baseline))
