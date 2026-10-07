from __future__ import annotations

import torch

BOED = "boed"
QOED_AGNOSTIC = "qoed-agnostic"
QOED = "qoed"
VALID_MODES = (BOED, QOED_AGNOSTIC, QOED)


def _as_tensor(x, *, device=None, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def _finite(x: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)


def trace_objective(fim, mask=None):
    diag = torch.diagonal(_as_tensor(fim), dim1=-2, dim2=-1)
    if mask is None:
        return diag.sum(-1)
    return (diag * _as_tensor(mask, device=diag.device, dtype=diag.dtype)).sum(-1)


def schur_objective(fim, mask):
    fim = _as_tensor(fim)
    fim = _finite(0.5 * (fim + fim.transpose(-1, -2)))
    mask = _as_tensor(mask, device=fim.device).to(torch.bool).reshape(-1)
    if not mask.any():
        return torch.zeros(fim.shape[:-2], device=fim.device, dtype=fim.dtype)
    if mask.all():
        return trace_objective(fim)

    keep = mask.nonzero().flatten()
    drop = (~mask).nonzero().flatten()
    fkk = fim[..., keep[:, None], keep]
    fbb = fim[..., drop[:, None], drop]
    fkb = fim[..., keep[:, None], drop]
    fbk = fim[..., drop[:, None], keep]
    eye = torch.eye(drop.numel(), device=fim.device, dtype=fim.dtype)
    schur = fkk - fkb @ torch.linalg.pinv(fbb + 1e-3 * eye) @ fbk
    return trace_objective(_finite(schur))


def identifiable_mask(
    fim,
    covariance,
    eig_ratio: float = 0.01,
    dist_threshold: float = 0.05,
    param_contrib_ratio: float = 1e-3,
    var_threshold: float = 0.0025,
    eig_floor: float = 0.1,
    candidate_mask=None,
):
    fim = _as_tensor(fim)
    p = fim.shape[-1]
    active = (torch.diagonal(_as_tensor(covariance)) >= var_threshold).to(fim.device)
    if candidate_mask is not None:
        active = active & _as_tensor(candidate_mask, device=fim.device).to(torch.bool)

    fim = 0.5 * (fim + fim.T) * active[:, None].to(fim.dtype) * active[None].to(fim.dtype)
    vals, vecs = torch.linalg.eigh(fim)
    order = torch.argsort(vals, descending=True)
    vals, vecs = vals[order], vecs[:, order]
    selected = torch.zeros((p,), device=fim.device, dtype=torch.bool)
    if p == 0 or float(vals[0]) <= eig_floor:
        return selected

    keep = vals >= eig_ratio * vals[0]
    observed = vecs[:, keep]
    scores = observed.square().sum(1)
    if not (scores > 0).any():
        return selected

    pool = (active & (scores >= param_contrib_ratio * scores.max())).cpu()
    w_norm = observed / torch.clamp(scores.sqrt(), min=1e-9)[:, None]
    distinct = torch.stack([1.0 - torch.abs(w_norm @ w) >= dist_threshold for w in w_norm]).cpu()
    n_obs, chosen = int(keep.sum()), []
    for idx in torch.argsort(scores, descending=True).tolist():
        if len(chosen) >= n_obs:
            break
        if pool[idx]:
            chosen.append(idx)
            pool &= distinct[idx]
    selected[chosen] = True
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
    if baseline == BOED:
        return boed_trace
    if baseline == QOED_AGNOSTIC:
        return trace_objective(f_cur, mask)
    return schur_objective(f_cur, mask)
