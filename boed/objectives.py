from __future__ import annotations

from dataclasses import dataclass

import torch

BOED = "boed"
QOED_AGNOSTIC = "qoed-agnostic"
QOED = "qoed"
VALID_MODES = (BOED, QOED_AGNOSTIC, QOED)


def _as_tensor(x, *, device=None, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        out = x
        if device is not None or dtype is not None:
            out = out.to(device=device if device is not None else out.device, dtype=dtype if dtype is not None else out.dtype)
        return out
    return torch.as_tensor(x, device=device, dtype=dtype)


def _finite(x: torch.Tensor, *, posinf: float = 1e6, neginf: float = -1e6) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=posinf, neginf=neginf)


def _sym_finite(fim) -> torch.Tensor:
    fim = _as_tensor(fim)
    return _finite(0.5 * (fim + fim.transpose(-1, -2)))


def trace_objective(fim, mask=None):
    fim = _as_tensor(fim)
    diag = torch.diagonal(fim, dim1=-2, dim2=-1)
    if mask is None:
        return diag.sum(-1)
    mask = _as_tensor(mask, device=fim.device, dtype=diag.dtype)
    return (diag * mask).sum(-1)


def schur_objective(fim, mask):
    fim = _sym_finite(fim)
    mask = _as_tensor(mask, device=fim.device).to(torch.bool).reshape(-1)
    if not bool(mask.any()):
        return torch.zeros(fim.shape[:-2], device=fim.device, dtype=fim.dtype)
    if bool(mask.all()):
        return trace_objective(fim)

    keep = mask.nonzero(as_tuple=False).flatten()
    drop = (~mask).nonzero(as_tuple=False).flatten()
    fkk = fim[..., keep[:, None], keep]
    if drop.numel() == 0:
        return trace_objective(fkk)
    fbb = fim[..., drop[:, None], drop]
    fkb = fim[..., keep[:, None], drop]
    fbk = fim[..., drop[:, None], keep]
    eye = torch.eye(drop.numel(), device=fim.device, dtype=fim.dtype)
    fbb = 0.5 * (fbb + fbb.transpose(-1, -2))
    schur = fkk - fkb @ torch.linalg.pinv(fbb + 1e-3 * eye) @ fbk
    return trace_objective(_finite(schur))


masked_schur_trace = schur_objective


def active_mask(covariance, var_threshold: float):
    covariance = _as_tensor(covariance)
    var = covariance if covariance.ndim == 1 else torch.diagonal(covariance)
    return var >= var_threshold


def identifiable_mask(
    fim,
    covariance=None,
    eig_ratio: float = 0.01,
    dist_threshold: float = 0.05,
    param_contrib_ratio: float = 1e-3,
    var_threshold: float = 0.0025,
    eig_floor: float = 0.1,
    candidate_mask=None,
):
    fim = _as_tensor(fim)
    p = fim.shape[-1]
    active = torch.ones((p,), device=fim.device, dtype=torch.bool) if covariance is None else active_mask(covariance, var_threshold).to(fim.device)
    if candidate_mask is not None:
        active = active & _as_tensor(candidate_mask, device=fim.device).to(torch.bool)

    fim = 0.5 * (fim + fim.T) * active[:, None].to(fim.dtype) * active[None].to(fim.dtype)
    vals, vecs = torch.linalg.eigh(fim)
    order = torch.argsort(vals, descending=True)
    vals, vecs = vals[order], vecs[:, order]
    selected = torch.zeros((p,), device=fim.device, dtype=torch.bool)
    if vals.numel() == 0 or float(vals[0].detach().cpu()) <= eig_floor:
        return selected

    keep = vals >= eig_ratio * vals[0]
    if not bool(keep.any()):
        return selected
    observed = vecs[:, keep]
    scores = observed.square().sum(1)
    if not bool((scores > 0).any()):
        return selected

    candidates = active & (scores >= param_contrib_ratio * scores.max())
    if not bool(candidates.any()):
        return selected
    w_norm = observed / torch.clamp(scores.sqrt(), min=1e-9)[:, None]
    pool = candidates.clone()
    count, n_obs = 0, int(keep.sum().detach().cpu())
    for idx in torch.argsort(scores, descending=True).detach().cpu().tolist():
        if count >= n_obs:
            break
        if not bool(pool[idx]):
            continue
        selected[idx] = True
        count += 1
        pool = pool & (1.0 - torch.abs(w_norm @ w_norm[idx]) >= dist_threshold)
    return selected


def score_mask(
    history_fim,
    cov,
    baseline,
    has_history,
    eig_ratio,
    dist_threshold,
    param_contrib_ratio,
    var_threshold,
    eig_floor,
    candidate_mask=None,
):
    history_fim = _as_tensor(history_fim)
    if baseline == BOED or not has_history:
        return torch.ones((history_fim.shape[-1],), device=history_fim.device, dtype=torch.bool)
    return identifiable_mask(
        history_fim,
        cov,
        eig_ratio,
        dist_threshold,
        param_contrib_ratio,
        var_threshold,
        eig_floor,
        candidate_mask,
    )


def score_from_mask(f_cur, boed_trace, baseline, mask):
    f_cur = _as_tensor(f_cur)
    boed_trace = _as_tensor(boed_trace, device=f_cur.device, dtype=f_cur.dtype)
    mask = _as_tensor(mask, device=f_cur.device).to(torch.bool)
    if baseline == BOED:
        return boed_trace
    if baseline == QOED_AGNOSTIC:
        return (f_cur.diagonal(dim1=-2, dim2=-1) * mask.to(f_cur.dtype)).sum(-1)
    return schur_objective(f_cur, mask)


def score_paths(
    f_cur,
    boed_trace,
    history_fim,
    cov,
    baseline,
    has_history,
    eig_ratio,
    dist_threshold,
    param_contrib_ratio,
    var_threshold,
    eig_floor,
    candidate_mask=None,
):
    mask = score_mask(
        history_fim,
        cov,
        baseline,
        has_history,
        eig_ratio,
        dist_threshold,
        param_contrib_ratio,
        var_threshold,
        eig_floor,
        candidate_mask,
    )
    return score_from_mask(f_cur, boed_trace, baseline, mask), mask


@dataclass(frozen=True)
class BOEDObjective:
    mode: str = QOED
    eig_ratio: float = 0.01
    dist_threshold: float = 0.05
    param_contrib_ratio: float = 1e-3
    var_threshold: float = 0.0025
    eig_floor: float = 0.1

    def __post_init__(self):
        if self.mode not in VALID_MODES:
            raise ValueError(f"invalid BOED objective mode: {self.mode}")

    @property
    def uses_fisher(self) -> bool:
        return self.mode in VALID_MODES

    def active_mask(self, covariance):
        return active_mask(covariance, self.var_threshold)

    def mask(self, fim, covariance=None, has_history: bool = True, candidate_mask=None):
        fim = _as_tensor(fim)
        if self.mode == BOED or not has_history or covariance is None:
            return torch.ones((fim.shape[-1],), device=fim.device, dtype=torch.bool)
        return self.identifiable_mask(fim, covariance, candidate_mask)

    def identifiable_mask(self, fim, covariance=None, candidate_mask=None):
        return identifiable_mask(
            fim,
            covariance,
            self.eig_ratio,
            self.dist_threshold,
            self.param_contrib_ratio,
            self.var_threshold,
            self.eig_floor,
            candidate_mask,
        )

    def value(self, fim, mask=None):
        fim = _as_tensor(fim)
        if self.mode == BOED:
            return trace_objective(fim)
        if mask is None:
            mask = torch.ones((fim.shape[-1],), device=fim.device, dtype=torch.bool)
        if self.mode == QOED_AGNOSTIC:
            return trace_objective(fim, mask)
        return schur_objective(fim, mask)

    def __call__(self, fim, covariance=None, history_fim=None, has_history: bool = True, mask=None):
        if mask is None:
            mask = self.mask(fim if history_fim is None else history_fim, covariance, has_history)
        return self.value(fim, mask), mask

    def bonus(self, trace, grad):
        if isinstance(trace, torch.Tensor):
            trace_t = trace
        elif isinstance(grad, torch.Tensor):
            trace_t = _as_tensor(trace, device=grad.device, dtype=grad.dtype)
        else:
            trace_t = _as_tensor(trace)
        grad_t = _as_tensor(grad, device=trace_t.device, dtype=trace_t.dtype)
        if self.mode == BOED or grad_t.numel() == 0:
            return trace_t
        norm_sq = grad_t.sum(-1, keepdim=True)
        max_sq = grad_t.max(-1, keepdim=True).values
        identifiable = (norm_sq > 0.1).to(trace_t.dtype)
        if self.mode == QOED_AGNOSTIC:
            return max_sq * identifiable
        nuisance = torch.clamp(norm_sq - max_sq, min=0.0)
        return max_sq * (1e-3 / (1e-3 + nuisance)) * identifiable


def uses_fisher(mode: str) -> bool:
    return mode in VALID_MODES


def fim_objective(fim, mask):
    return schur_objective(fim, mask)


def analyze_fim_identifiability(
    fim,
    covariance=None,
    eig_ratio: float = 0.01,
    param_contrib_ratio: float = 1e-3,
    var_threshold: float = 0.0025,
    eig_floor: float = 0.1,
    dist_threshold: float = 0.05,
    var_diag=None,
    eig_ratio_thresh=None,
    var_small_thresh=None,
    smallest_eigval_threshold=None,
):
    if covariance is None:
        covariance = var_diag
    objective = BOEDObjective(
        QOED,
        eig_ratio=eig_ratio if eig_ratio_thresh is None else eig_ratio_thresh,
        dist_threshold=dist_threshold,
        param_contrib_ratio=param_contrib_ratio,
        var_threshold=var_threshold if var_small_thresh is None else var_small_thresh,
        eig_floor=eig_floor if smallest_eigval_threshold is None else smallest_eigval_threshold,
    )
    mask = objective.identifiable_mask(fim, covariance)
    fim = _as_tensor(fim)
    active = torch.ones((fim.shape[-1],), device=fim.device, dtype=torch.bool) if covariance is None else objective.active_mask(covariance)
    return {"param_ident_mask": mask, "active_param_mask": active}


def mask_from_fisher(mode: str, fim, covariance, has_history: bool):
    return BOEDObjective(mode).mask(fim, covariance, has_history)


def fisher_bonus(mode: str, trace, grad):
    return BOEDObjective(mode).bonus(trace, grad)
