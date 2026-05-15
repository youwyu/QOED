from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Callable


_CUDA_INIT_WARNING = r"CUDA initialization: CUDA driver initialization failed.*"


def _preparse_requested_device() -> None:
    args = sys.argv[1:]
    for i, arg in enumerate(args):
        value = None
        if arg == "--device" and i + 1 < len(args):
            value = args[i + 1]
        elif arg.startswith("--device="):
            value = arg.split("=", 1)[1]
        if value is None:
            continue
        value = value.strip().lower()
        if value == "cuda":
            os.environ["QOED_REQUESTED_CUDA_DEVICE"] = "0"
        elif value.startswith("cuda:"):
            os.environ["QOED_REQUESTED_CUDA_DEVICE"] = value.split(":", 1)[1]
        return


def prepare_demo_cuda_environment() -> None:
    warnings.filterwarnings(
        "ignore",
        message=_CUDA_INIT_WARNING,
        category=UserWarning,
        module=r"torch\.cuda",
    )
    _preparse_requested_device()

    raw_path = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in raw_path.split(os.pathsep) if part]
    kept = [part for part in parts if not part.startswith("/usr/local/cuda")]
    restart = kept != parts

    cuda_visible = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        os.environ["QOED_REQUESTED_CUDA_VISIBLE_DEVICES"] = cuda_visible
        restart = True
    if not restart:
        return

    removed = [part for part in parts if part not in kept]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
    if removed:
        print(f"[INFO] Restarting with system CUDA library paths removed: {removed}", flush=True)
    elif cuda_visible:
        print(f"[INFO] Restarting with CUDA_VISIBLE_DEVICES cleared: {cuda_visible}", flush=True)

    main_spec = getattr(sys.modules.get("__main__"), "__spec__", None)
    module_name = getattr(main_spec, "name", None)
    if module_name and module_name.startswith("demos."):
        args = [sys.executable, "-m", module_name, *sys.argv[1:]]
    else:
        args = [sys.executable, *sys.argv]
    os.execvpe(sys.executable, args, os.environ)


prepare_demo_cuda_environment()

import numpy as np
import torch

from boed import BOED, QOED, QOED_AGNOSTIC, score_from_mask, score_mask

__all__ = [
    "resolve_device",
    "make_generator",
    "as_tensor",
    "to_numpy",
    "batch_if_vector",
    "broadcast_rows",
    "stack_rows",
    "shift_control_sequence",
    "finite_cost",
    "clip_action_norm",
    "sample_bounded_control_population",
    "sample_gaussian_control_population",
    "mppi_weighted_update",
    "mppi_info_gain_refine",
    "baseline_label",
    "metric_stats",
    "summary_line",
    "print_paper_summary",
    "print_baseline_summary",
    "run_baseline_demo",
]


def resolve_device(name: str | None):
    name = (name or "auto").lower()
    if name == "cpu":
        return torch.device("cpu")

    explicit_cuda = name.startswith("cuda")
    requested = 0
    if explicit_cuda and ":" in name:
        requested = int(name.split(":", 1)[1])

    if not torch.cuda.is_available():
        if explicit_cuda:
            print("[WARN] CUDA was requested but failed to initialize; falling back to CPU.")
        return torch.device("cpu")

    visible = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    requested_env = os.environ.get("QOED_REQUESTED_CUDA_DEVICE")
    if requested_env and requested_env in visible:
        requested = visible.index(requested_env)
    elif str(requested) in visible:
        requested = visible.index(str(requested))

    if requested >= torch.cuda.device_count():
        if explicit_cuda:
            print(f"[WARN] Requested CUDA device {requested}, but only {torch.cuda.device_count()} device(s) are visible; falling back to CPU.")
        return torch.device("cpu")

    device = torch.device(f"cuda:{requested}")
    try:
        torch.cuda.set_device(device)
        torch.empty(1, device=device)
    except Exception as exc:
        if explicit_cuda:
            print(f"[WARN] CUDA was requested but failed to initialize; falling back to CPU. Error: {exc}")
        return torch.device("cpu")
    return device


def make_generator(seed: int | None, device: torch.device):
    try:
        gen = torch.Generator(device=device)
    except RuntimeError:
        gen = torch.Generator()
    gen.manual_seed(0 if seed is None else int(seed))
    return gen


def as_tensor(x, device: torch.device | None = None, dtype=torch.float32):
    if isinstance(x, torch.Tensor):
        return x.to(device=device if device is not None else x.device, dtype=dtype if dtype is not None else x.dtype)
    return torch.as_tensor(x, device=device, dtype=dtype)


def to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def batch_if_vector(x, device: torch.device | None = None):
    x = as_tensor(x, device=device, dtype=torch.float32)
    return x[None] if x.ndim == 1 else x


def broadcast_rows(x, n: int):
    return x.expand(n, x.shape[-1]) if x.shape[0] == 1 and n > 1 else x


def stack_rows(rows):
    return tuple(torch.stack(items) for items in zip(*rows))


def shift_control_sequence(u):
    return torch.cat([u[1:], torch.zeros_like(u[:1])], 0)


def finite_cost(cost):
    return torch.nan_to_num(cost, nan=1e9, posinf=1e9, neginf=-1e9)


def clip_action_norm(action, max_norm: float):
    norm = torch.linalg.norm(action, dim=-1, keepdim=True) + 1e-6
    return action * torch.clamp(float(max_norm) / norm, max=1.0)


def sample_bounded_control_population(generator, u, u_min, u_max, noise_cov_diag, num_samples: int):
    lower_slack = u - u_min
    upper_slack = u_max - u
    bound_std = torch.minimum((0.5 * lower_slack).square(), (0.5 * upper_slack).square())
    std = torch.sqrt(torch.clamp(torch.minimum(bound_std, noise_cov_diag[None]), min=1e-6))
    noise = torch.randn(
        (int(num_samples), *u.shape),
        generator=generator,
        device=u.device,
        dtype=u.dtype,
    ) * std[None]
    return torch.clamp(u[None] + noise, min=u_min, max=u_max), noise


def sample_gaussian_control_population(
    generator,
    u,
    num_samples: int,
    noise_std: float,
    postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    noise = torch.randn(
        (int(num_samples), *u.shape),
        generator=generator,
        device=u.device,
        dtype=u.dtype,
    ) * float(noise_std)
    population = u[None] + noise
    if postprocess is not None:
        population = postprocess(population)
    return population, noise


def mppi_weighted_update(
    population,
    costs,
    *,
    temperature: float = 1.0,
    u_min=None,
    u_max=None,
    postprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    temperature = max(float(temperature), 1e-9)
    weights = torch.softmax(-(costs - costs.min()) / temperature, dim=0)
    u = torch.sum(weights[:, None, None] * population, dim=0)
    if u_min is not None and u_max is not None:
        u = torch.clamp(u, min=u_min, max=u_max)
    if postprocess is not None:
        u = postprocess(u)
    return u, weights


def mppi_info_gain_refine(
    *,
    generator,
    u,
    estimator,
    baseline: str,
    num_samples: int,
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
    postprocess_update: Callable[[torch.Tensor], torch.Tensor] | None = None,
):
    p = estimator.dist.mean.numel()
    f_cur = torch.zeros((int(num_samples), p, p), device=u.device, dtype=u.dtype)
    mask = score_mask(
        estimator.compute_fisher(),
        estimator.dist.cov,
        baseline,
        bool(estimator.history),
        estimator.eig_ratio_thresh,
        estimator.dist_threshold,
        estimator.param_contrib_ratio,
        estimator.var_threshold_for_update,
        estimator.smallest_eigval_threshold,
        candidate_mask,
    )
    if fallback_to_all_mask and not bool(mask.any()):
        mask = torch.ones_like(mask)

    idx = torch.zeros((), device=u.device, dtype=torch.long)
    for _ in range(int(num_iterations)):
        population, noise = sample_population(generator, u, int(num_samples))
        costs, f_cur, boed_trace = evaluate_population(population, noise)
        info = score_from_mask(f_cur, boed_trace, baseline, mask)
        costs = finite_cost((costs - float(fisher_weight) * info) * float(cost_scale))
        idx = torch.argmin(costs)
        u, _ = mppi_weighted_update(
            population,
            costs,
            temperature=temperature,
            u_min=u_min,
            u_max=u_max,
            postprocess=postprocess_update,
        )
    return u, f_cur, mask, idx


def baseline_label(name: str):
    return {QOED: "QOED", QOED_AGNOSTIC: "QOED-Agnostic", BOED: "BOED"}[name]


def metric_stats(values):
    arr = np.asarray(values, np.float64)
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), 0.0 if arr.size == 1 else float(arr.std(ddof=1))


def summary_line(summary, metrics: tuple[str, ...], seed: int | None = None):
    seed_part = "" if seed is None else f"[seed {seed:02d}]"
    values = " ".join(f"{name}={getattr(summary, name):.4f}" for name in metrics)
    return f"[{baseline_label(summary.baseline)}]{seed_part} {values}"


def print_paper_summary(results_by_baseline, metrics: tuple[str, ...]):
    print("\nPaper Summary")
    for baseline, runs in results_by_baseline.items():
        stats = {
            name: metric_stats([float(getattr(run, name)) for run in runs])
            for name in metrics
        }
        values = " ".join(f"{name}={mean:.4f}+-{std:.4f}" for name, (mean, std) in stats.items())
        print(f"{baseline_label(baseline)}: {values} n={len(runs)}")


def print_baseline_summary(results, metrics: tuple[str, ...]):
    print("\nBaseline Summary")
    for summary in results:
        values = " ".join(f"{name}={getattr(summary, name):.4f}" for name in metrics)
        print(f"{baseline_label(summary.baseline)}: {values} device={summary.device}")


def run_baseline_demo(args, all_baselines, run_simulation, config_from_args, metrics: tuple[str, ...]):
    baselines = list(all_baselines) if args.baseline == "all" else [args.baseline]
    if args.sweep:
        grouped = {baseline: [] for baseline in baselines}
        for baseline in baselines:
            for seed in range(args.sweep_seeds):
                summary = run_simulation(config_from_args(args, baseline, seed), device=args.device, verbose=False)
                grouped[baseline].append(summary)
                print(summary_line(summary, metrics, seed))
        print_paper_summary(grouped, metrics)
        return

    results = [
        run_simulation(config_from_args(args, baseline, args.seed), device=args.device, verbose=not args.quiet)
        for baseline in baselines
    ]
    print_baseline_summary(results, metrics)
