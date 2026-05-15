from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import jvp


def uses_flow(backbone: str) -> bool:
    return backbone in {"flow", "shortcut"}


def _exists(x) -> bool:
    return x is not None


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        half = max(dim // 2, 1)
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * (-math.log(10000.0) / max(half - 1, 1)))
        self.register_buffer("freqs", freqs)
        self.dim = int(dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = t.reshape(-1, 1) * self.freqs.reshape(1, -1)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[..., : self.dim]


class TimestepEncoder(nn.Module):
    def __init__(self, embed_dim: int, *, dual: bool, mlp_ratio: float = 4.0):
        super().__init__()
        self.dual = bool(dual)
        self.pos = SinusoidalPosEmb(embed_dim)
        in_dim = 2 * embed_dim if dual else embed_dim
        hidden = int(embed_dim * mlp_ratio)
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Mish(), nn.Linear(hidden, embed_dim))

    def forward(self, t1: torch.Tensor, t2: Optional[torch.Tensor] = None) -> torch.Tensor:
        emb = self.pos(t1)
        if self.dual:
            emb = torch.cat([emb, self.pos(t1 if t2 is None else t2)], dim=-1)
        return self.net(emb)


class AdaLNBlock(nn.Module):
    def __init__(self, dim: int, cond_dim: int, num_heads: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * dim))

    @staticmethod
    def _modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = self.mod(cond).chunk(6, dim=-1)
        y = self._modulate(self.norm1(x), shift_a, scale_a)
        attn, _ = self.attn(y, y, y, need_weights=False)
        x = x + gate_a.unsqueeze(1) * attn
        y = self._modulate(self.norm2(x), shift_m, scale_m)
        return x + gate_m.unsqueeze(1) * self.mlp(y)


class AdaLNFinal(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.mod = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 2 * dim))
        self.linear = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        shift, scale = self.mod(cond).chunk(2, dim=-1)
        return self.linear(self.norm(x) * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1))


class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        privilege_dim: int,
        *,
        embed_dim: int,
        timestep_embed_dim: int,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_registers: int = 8,
        use_shortcut: bool = True,
    ):
        super().__init__()
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.privilege_dim = int(privilege_dim)
        self.embed_dim = int(embed_dim)
        self.num_registers = int(num_registers)
        self.use_shortcut = bool(use_shortcut)
        hidden = int(max(self.state_dim, self.embed_dim) * mlp_ratio)
        self.state_encoder = (
            nn.Sequential(nn.Linear(self.state_dim, hidden), nn.Mish(), nn.Linear(hidden, self.embed_dim))
            if self.state_dim != self.embed_dim
            else nn.Identity()
        )
        self.state_decoder = (
            nn.Sequential(nn.Linear(self.embed_dim, hidden), nn.Mish(), nn.Linear(hidden, self.state_dim))
            if self.state_dim != self.embed_dim
            else nn.Identity()
        )
        if self.action_dim > 0:
            hidden = int(max(self.action_dim, self.embed_dim) * mlp_ratio)
            self.action_encoder = nn.Sequential(nn.Linear(self.action_dim, hidden), nn.Mish(), nn.Linear(hidden, self.embed_dim))
        if self.privilege_dim > 0:
            hidden = int(max(self.privilege_dim, self.embed_dim) * mlp_ratio)
            self.privilege_encoder = nn.Sequential(nn.Linear(self.privilege_dim, hidden), nn.Mish(), nn.Linear(hidden, self.embed_dim))

        self.timestep_embedding = TimestepEncoder(timestep_embed_dim, dual=True, mlp_ratio=mlp_ratio)
        if self.use_shortcut:
            self.d_timestep_embedding = TimestepEncoder(timestep_embed_dim, dual=True, mlp_ratio=mlp_ratio)
        cond_dim = (
            self.embed_dim
            + (self.embed_dim if self.action_dim > 0 else 0)
            + (self.embed_dim if self.privilege_dim > 0 else 0)
            + timestep_embed_dim * (2 if self.use_shortcut else 1)
        )
        self.blocks = nn.ModuleList([AdaLNBlock(self.embed_dim, cond_dim, num_heads, mlp_ratio) for _ in range(depth)])
        self.final = AdaLNFinal(self.embed_dim, cond_dim)
        self.registers = nn.Parameter(torch.empty(1, self.num_registers, self.embed_dim).normal_(std=0.02))
        self.pos_embed = nn.Parameter(torch.empty(1, 1 + self.num_registers, self.embed_dim).normal_(std=0.02))
        self.apply(self._init)
        nn.init.zeros_(self.final.linear.weight)
        nn.init.zeros_(self.final.linear.bias)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _feature(self, x: torch.Tensor | None, batch: int, dim: int, device, dtype) -> torch.Tensor:
        if dim <= 0:
            return torch.zeros(batch, 0, device=device, dtype=dtype)
        if x is None or x.numel() == 0:
            return torch.zeros(batch, dim, device=device, dtype=dtype)
        return x.reshape(batch, dim).to(device=device, dtype=dtype)

    @staticmethod
    def _time(x: torch.Tensor | float | None, batch: int, device, dtype) -> torch.Tensor:
        if x is None:
            return torch.zeros(batch, device=device, dtype=dtype)
        x = torch.as_tensor(x, device=device, dtype=dtype)
        return x.expand(batch) if x.ndim == 0 else x.reshape(batch)

    def forward(
        self,
        next_state: torch.Tensor,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        privilege: torch.Tensor | None = None,
        next_state_t: torch.Tensor | None = None,
        privilege_t: torch.Tensor | None = None,
        next_state_dt: torch.Tensor | None = None,
        privilege_dt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, device, dtype = state.shape[0], state.device, state.dtype
        next_is_flat = next_state.ndim == 2
        next_seq = next_state.unsqueeze(1) if next_is_flat else next_state
        cond = [self.state_encoder(state)]
        next_embed = self.state_encoder(next_seq).reshape(batch, 1, self.embed_dim)
        if self.action_dim > 0:
            cond.append(self.action_encoder(self._feature(action, batch, self.action_dim, device, dtype)))
        if self.privilege_dim > 0:
            cond.append(self.privilege_encoder(self._feature(privilege, batch, self.privilege_dim, device, dtype)))
        t = self._time(next_state_t, batch, device, dtype)
        pt = self._time(privilege_t if privilege_t is not None else next_state_t, batch, device, dtype)
        cond.append(self.timestep_embedding(t, pt))
        if self.use_shortcut:
            dt = self._time(next_state_dt, batch, device, dtype)
            pdt = self._time(privilege_dt if privilege_dt is not None else next_state_dt, batch, device, dtype)
            cond.append(self.d_timestep_embedding(dt, pdt))
        cond_t = torch.cat(cond, dim=-1)
        x = torch.cat([next_embed, self.registers.expand(batch, -1, -1)], dim=1) + self.pos_embed
        for block in self.blocks:
            x = block(x, cond_t)
        out = self.state_decoder(self.final(x, cond_t)[:, :1])
        return out[:, 0] if next_is_flat else out


class TimestepSampler:
    def __init__(self, rate_self_consistency: float = 0.25, min_dt: float = 0.0078125):
        if not 0.0 <= rate_self_consistency <= 1.0:
            raise ValueError("rate_self_consistency must be in [0, 1]")
        self.rate_sc = float(rate_self_consistency)
        self.min_dt = float(min_dt)

    def sample_t(self, num: int, device) -> tuple[torch.Tensor, torch.Tensor, int]:
        num_sc = round(num * self.rate_sc)
        num_fm = num - num_sc
        t_fm = torch.rand(num_fm, device=device)
        dt_fm = torch.zeros(num_fm, device=device)
        t_sc = torch.rand(num_sc, device=device) * (1.0 - self.min_dt)
        dt_sc = self.min_dt + torch.rand(num_sc, device=device) * (1.0 - t_sc - self.min_dt)
        return torch.cat([t_sc, t_fm]), torch.cat([dt_sc, dt_fm]), num_sc


class ShortcutModel(nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        privilege_dim: int,
        *,
        embed_dim: int,
        timestep_embed_dim: int,
        use_shortcut: bool,
        shortcut_self_consistency: float,
        shortcut_min_dt: float,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_registers: int = 8,
    ):
        super().__init__()
        self.use_shortcut = bool(use_shortcut)
        self.flow = DiffusionTransformer(
            state_dim,
            action_dim,
            privilege_dim,
            embed_dim=embed_dim,
            timestep_embed_dim=timestep_embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_registers=num_registers,
            use_shortcut=use_shortcut,
        )
        self.ts_sampler = TimestepSampler(shortcut_self_consistency, shortcut_min_dt)
        self.prior_loc = nn.Parameter(torch.zeros(state_dim))
        self.prior_scale = nn.Parameter(torch.ones(state_dim))

    @property
    def prior(self):
        scale = self.prior_scale.clamp_min(1e-6)
        return torch.distributions.Independent(torch.distributions.Normal(self.prior_loc, scale), 1)

    def loss(self, next_state: torch.Tensor, state: torch.Tensor, action: torch.Tensor | None = None, privilege: torch.Tensor | None = None) -> torch.Tensor:
        t, dt, n_sc = self.ts_sampler.sample_t(next_state.shape[0], next_state.device)
        x1 = next_state.unsqueeze(1) if next_state.ndim == 2 else next_state
        x0 = torch.randn_like(x1)
        x_t = (1.0 - t[:, None, None]) * x0 + t[:, None, None] * x1
        target = x1 - x0
        if n_sc:
            with torch.no_grad():
                half = dt[:n_sc] * 0.5
                v1 = self.flow(x_t[:n_sc], state[:n_sc], None if action is None else action[:n_sc], None if privilege is None else privilege[:n_sc], t[:n_sc], t[:n_sc], half, half)
                v2 = self.flow(x_t[:n_sc] + half[:, None, None] * v1, state[:n_sc], None if action is None else action[:n_sc], None if privilege is None else privilege[:n_sc], t[:n_sc] + half, t[:n_sc] + half, half, half)
                target = target.clone()
                target[:n_sc] = 0.5 * (v1 + v2)
        dt = dt.clone()
        dt[n_sc:] = 0.0
        pred = self.flow(x_t, state, action, privilege, t, t, dt, dt)
        loss = torch.zeros((), device=next_state.device, dtype=next_state.dtype)
        if n_sc < next_state.shape[0]:
            loss = loss + F.mse_loss(target[n_sc:], pred[n_sc:])
        if n_sc:
            loss = loss + F.mse_loss(target[:n_sc], pred[:n_sc])
        return loss

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None, privilege: torch.Tensor | None = None, n_step: int = 1, dt_list: Optional[list[float]] = None) -> torch.Tensor:
        if not _exists(dt_list):
            if not _exists(n_step):
                raise ValueError("n_step or dt_list is required")
            dt_list = [1.0 / max(int(n_step), 1)] * max(int(n_step), 1)
        x = torch.randn_like(state)
        squeeze = x.ndim == 2
        if squeeze:
            x = x.unsqueeze(1)
        t_cur = torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        for dt_val in dt_list:
            dt = torch.full((state.shape[0],), float(dt_val), device=state.device, dtype=state.dtype)
            dt_in = dt if self.use_shortcut else torch.zeros_like(dt)
            x = x + self.flow(x, state, action, privilege, t_cur, t_cur, dt_in, dt_in) * dt[:, None, None]
            t_cur = t_cur + dt
        return x[:, 0] if squeeze else x

    def sample_and_log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        privilege: torch.Tensor | None = None,
        n_step: int = 1,
        dt_list: Optional[list[float]] = None,
        num_trace_samples: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        def fn(s):
            return self.forward(s, action, privilege, n_step, dt_list)

        trace = torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        out = None
        for _ in range(num_trace_samples):
            probe = torch.empty_like(state).bernoulli_(0.5).mul_(2).sub_(1)
            out, jvp_out = jvp(fn, (state,), (probe,))
            trace = trace + (jvp_out * probe).sum(dim=-1)
        trace = trace / float(num_trace_samples)
        return out.detach(), self.prior.log_prob(state) - trace

    @staticmethod
    def _rank1_eigh(grad: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch, p = grad.shape
        vals = torch.zeros(batch, p, device=grad.device, dtype=grad.dtype)
        vecs = torch.eye(p, device=grad.device, dtype=grad.dtype).expand(batch, -1, -1).clone()
        norm = grad.norm(dim=-1, keepdim=True)
        valid = norm.squeeze(-1) > 1e-12
        if valid.any():
            vals[valid, -1] = norm[valid, 0].square()
            vecs[valid, :, -1] = grad[valid] / norm[valid]
        return vals, vecs

    def fisher(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        privilege: torch.Tensor | None = None,
        n_step: int = 1,
        dt_list: Optional[list[float]] = None,
        max_microbatch: int = 512,
        compute_eigendecomp: bool = True,
        num_trace_samples: int = 1,
        eig_on_cpu: bool = True,
    ):
        if privilege is None:
            raise ValueError("privilege is required for Fisher estimation")
        if privilege.ndim != 3 or privilege.shape[1] != 1:
            raise ValueError("expected privilege with shape [B, 1, P]")
        batch, _, p = privilege.shape
        fisher_vals = torch.zeros(batch, 1, device=state.device, dtype=state.dtype)
        grads = torch.zeros(batch, p, device=state.device, dtype=state.dtype) if compute_eigendecomp else None
        flat_priv = privilege[:, 0]
        with torch.enable_grad():
            for offset in range(0, batch, max_microbatch):
                end = min(offset + max_microbatch, batch)
                priv = flat_priv[offset:end].detach().clone().requires_grad_(True)
                _, log_q = self.sample_and_log_prob(
                    state[offset:end].detach(),
                    None if action is None else action[offset:end].detach(),
                    priv,
                    n_step,
                    dt_list,
                    num_trace_samples,
                )
                (grad,) = torch.autograd.grad(log_q.sum(), priv, retain_graph=False, create_graph=False)
                fisher_vals[offset:end] = grad.square().sum(dim=-1, keepdim=True)
                if grads is not None:
                    grads[offset:end] = grad
        if not compute_eigendecomp:
            return fisher_vals, None, None
        if eig_on_cpu:
            vals, vecs = self._rank1_eigh(grads.detach().cpu())
            return fisher_vals, vals.to(state.device, state.dtype), vecs.to(state.device, state.dtype)
        vals, vecs = self._rank1_eigh(grads)
        return fisher_vals, vals, vecs


class FlowDynamics(nn.Module):
    """Flow-matching dynamics used by the PyTorch MBPO world model."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        privilege_dim: int,
        *,
        latent_dim: int | None = None,
        timestep_embed_dim: int | None = None,
        use_shortcut: bool = True,
        shortcut_self_consistency: float = 0.25,
        shortcut_min_dt: float = 0.0078125,
        depth: int = 6,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        num_registers: int = 8,
    ):
        super().__init__()
        latent_dim = int(latent_dim or state_dim)
        timestep_embed_dim = int(timestep_embed_dim or max(latent_dim // 2, 1))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.privilege_dim = int(privilege_dim)
        self.model = ShortcutModel(
            self.state_dim,
            self.action_dim,
            self.privilege_dim,
            embed_dim=latent_dim,
            timestep_embed_dim=timestep_embed_dim,
            use_shortcut=use_shortcut,
            shortcut_self_consistency=shortcut_self_consistency,
            shortcut_min_dt=shortcut_min_dt,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            num_registers=num_registers,
        )

    @staticmethod
    def _feature(x: torch.Tensor | None, batch: int, dim: int, device, dtype) -> torch.Tensor | None:
        if dim <= 0:
            return None
        if x is None or x.numel() == 0:
            return torch.zeros(batch, dim, device=device, dtype=dtype)
        return x.reshape(batch, dim).to(device=device, dtype=dtype)

    def forward(self, state: torch.Tensor, action: torch.Tensor | None = None, privilege: torch.Tensor | None = None, n_step: int = 1) -> torch.Tensor:
        batch = state.shape[0]
        state = state.reshape(batch, self.state_dim)
        action = self._feature(action, batch, self.action_dim, state.device, state.dtype)
        privilege = self._feature(privilege, batch, self.privilege_dim, state.device, state.dtype)
        return self.model(state, action, privilege, n_step=n_step)

    def loss(self, next_state: torch.Tensor, state: torch.Tensor, action: torch.Tensor | None = None, privilege: torch.Tensor | None = None) -> torch.Tensor:
        batch = state.shape[0]
        state = state.reshape(batch, self.state_dim)
        next_state = next_state.reshape(batch, self.state_dim).contiguous()
        action = self._feature(action, batch, self.action_dim, state.device, state.dtype)
        privilege = self._feature(privilege, batch, self.privilege_dim, state.device, state.dtype)
        return self.model.loss(next_state, state, action, privilege)

    def fisher(
        self,
        state: torch.Tensor,
        action: torch.Tensor | None = None,
        privilege: torch.Tensor | None = None,
        *,
        compute_eigendecomp: bool = True,
        max_microbatch: int = 512,
        num_trace_samples: int = 1,
    ):
        if self.privilege_dim <= 0 or privilege is None or privilege.numel() == 0:
            batch = state.shape[0]
            zeros = torch.zeros(batch, 1, device=state.device, dtype=state.dtype)
            return zeros, None, None
        batch = state.shape[0]
        action = self._feature(action, batch, self.action_dim, state.device, state.dtype)
        privilege = privilege.reshape(batch, 1, self.privilege_dim).to(state.device, state.dtype)
        return self.model.fisher(
            state,
            action,
            privilege,
            compute_eigendecomp=compute_eigendecomp,
            max_microbatch=max_microbatch,
            num_trace_samples=num_trace_samples,
            eig_on_cpu=True,
        )
