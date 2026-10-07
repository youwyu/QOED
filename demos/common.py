from __future__ import annotations

import os
import sys
import warnings


def _restart_without_system_cuda() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"CUDA initialization: CUDA driver initialization failed.*",
        category=UserWarning,
        module=r"torch\.cuda",
    )
    parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    kept = [part for part in parts if not part.startswith("/usr/local/cuda")]
    cuda_visible = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if kept == parts and not cuda_visible:
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
    spec = sys.modules["__main__"].__spec__
    entry = ["-m", spec.name] if spec is not None and spec.name.startswith("demos.") else [sys.argv[0]]
    os.execve(sys.executable, [sys.executable, *entry, *sys.argv[1:]], os.environ)


_restart_without_system_cuda()

import argparse
from dataclasses import fields

import numpy as np
import torch

from boed import BOED, QOED, QOED_AGNOSTIC, score_from_mask

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

LABELS = {QOED: "QOED", QOED_AGNOSTIC: "QOED-Agnostic", BOED: "BOED"}


def resolve_device(name: str | None) -> torch.device:
    name = (name or "auto").lower()
    index = int(name.split(":", 1)[1]) if name.startswith("cuda:") else 0
    if name == "cpu" or not torch.cuda.is_available() or index >= torch.cuda.device_count():
        return torch.device("cpu")
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    return device


def make_generator(seed: int | None, device: torch.device) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(0 if seed is None else int(seed))


def as_tensor(x, device: torch.device | None = None, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device or x.device, dtype=dtype or x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def stack_to_numpy(values, empty_shape):
    return to_numpy(torch.stack(values)) if values else np.empty(empty_shape, np.float32)


def batch_if_vector(x, device: torch.device | None = None):
    x = as_tensor(x, device=device)
    return x[None] if x.ndim == 1 else x


def broadcast_rows(x, n: int):
    return x.expand(n, x.shape[-1]) if x.shape[0] == 1 and n > 1 else x


def shift_control_sequence(u):
    return torch.cat([u[1:], torch.zeros_like(u[:1])], 0)


def mppi_info_gain_refine(
    *,
    u,
    estimator,
    baseline: str,
    num_iterations: int,
    sample_population,
    evaluate_population,
    fisher_weight: float = 1.0,
    cost_scale: float = 1.0,
    temperature: float = 1.0,
    candidate_mask=None,
    fallback_to_all_mask: bool = False,
    u_min=None,
    u_max=None,
    postprocess=None,
):
    mask = estimator.mask(baseline, candidate_mask)
    if fallback_to_all_mask and not mask.any():
        mask = torch.ones_like(mask)
    temperature = max(float(temperature), 1e-9)
    for _ in range(int(num_iterations)):
        population, noise = sample_population(u)
        costs, f_cur, boed_trace = evaluate_population(population, noise)
        info = score_from_mask(f_cur, boed_trace, baseline, mask)
        costs = torch.nan_to_num((costs - fisher_weight * info) * cost_scale, nan=1e9, posinf=1e9, neginf=-1e9)
        weights = torch.softmax(-(costs - costs.min()) / temperature, dim=0)
        u = torch.sum(weights[:, None, None] * population, dim=0)
        if u_min is not None and u_max is not None:
            u = torch.clamp(u, min=u_min, max=u_max)
        if postprocess is not None:
            u = postprocess(u)
    return u


def _metric_values(summary, metrics: tuple[str, ...]) -> str:
    return " ".join(f"{name}={getattr(summary, name):.4f}" for name in metrics)


def _mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), 0.0 if arr.size == 1 else float(arr.std(ddof=1))


def run_demo(description: str, config_cls, run_simulation, baselines, metrics, argv=None) -> int:
    defaults = config_cls()
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--steps", type=int, default=defaults.steps)
    parser.add_argument("--obs-noise", "--obs_noise", type=float, default=defaults.obs_noise)
    parser.add_argument("--num-iterations", "--num_iterations", type=int, default=defaults.num_iterations)
    parser.add_argument("--baseline", choices=["all", *baselines], default="all")
    parser.add_argument("--fisher-weight", "--fisher_weight", type=float, default=defaults.fisher_weight)
    parser.add_argument("--eig-ratio-thresh", "--eig_ratio_thresh", type=float, default=defaults.eig_ratio_thresh)
    parser.add_argument("--dist-threshold", "--dist_threshold", type=float, default=defaults.dist_threshold)
    parser.add_argument(
        "--smallest-eigval-threshold",
        "--smallest_eigval_threshold",
        type=float,
        default=defaults.smallest_eigval_threshold,
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--sweep-seeds", "--sweep_seeds", type=int, default=10)
    parser.add_argument("--log-every", "--log_every", type=int, default=defaults.log_every)
    parser.add_argument("--num-samples", "--num_samples", type=int, default=defaults.num_samples)
    args = parser.parse_args(argv)

    overrides = {f.name for f in fields(config_cls)} & vars(args).keys() - {"baseline", "seed"}

    def config(baseline: str, seed: int | None):
        return config_cls(baseline=baseline, seed=seed, **{name: getattr(args, name) for name in overrides})

    baselines = list(baselines) if args.baseline == "all" else [args.baseline]
    if not args.sweep:
        results = [run_simulation(config(b, args.seed), device=args.device, verbose=not args.quiet) for b in baselines]
        print("\nBaseline Summary")
        for summary in results:
            print(f"{LABELS[summary.baseline]}: {_metric_values(summary, metrics)} device={summary.device}")
        return 0

    grouped = {baseline: [] for baseline in baselines}
    for baseline in baselines:
        for seed in range(args.sweep_seeds):
            summary = run_simulation(config(baseline, seed), device=args.device, verbose=False)
            grouped[baseline].append(summary)
            print(f"[{LABELS[baseline]}][seed {seed:02d}] {_metric_values(summary, metrics)}")

    print("\nPaper Summary")
    for baseline, runs in grouped.items():
        stats = {name: _mean_std([float(getattr(run, name)) for run in runs]) for name in metrics}
        values = " ".join(f"{name}={mean:.4f}+-{std:.4f}" for name, (mean, std) in stats.items())
        print(f"{LABELS[baseline]}: {values} n={len(runs)}")

    return 0
