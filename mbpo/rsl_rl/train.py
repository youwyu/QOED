"""MJLab RSL-RL training entry point."""

from __future__ import annotations

import os
import re
import sys
import warnings
from dataclasses import asdict
from pathlib import Path


warnings.filterwarnings(
    "ignore",
    message=r"CUDA initialization: CUDA driver initialization failed.*",
    category=UserWarning,
    module=r"torch\.cuda",
)


def _restart_without_system_cuda_library_paths() -> None:
    raw_path = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [part for part in raw_path.split(os.pathsep) if part]
    kept = [part for part in parts if not part.startswith("/usr/local/cuda")]
    restart = kept != parts
    cuda_visible = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        os.environ["QOED_TRAIN_REQUESTED_CUDA_VISIBLE_DEVICES"] = cuda_visible
        restart = True
    if not restart:
        return

    removed = [part for part in parts if part not in kept]
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
    if removed:
        print(f"[INFO] Restarting with system CUDA library paths removed: {removed}", flush=True)
    elif cuda_visible:
        print(f"[INFO] Restarting with CUDA_VISIBLE_DEVICES cleared: {cuda_visible}", flush=True)
    os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)


_restart_without_system_cuda_library_paths()


def _preparse_domain_randomization_flag() -> None:
    """Expose the legacy DR flag before local Go1 tasks are registered."""
    enabled = False
    for arg in sys.argv[1:]:
        if arg in {"--domain_randomization", "--domain-randomization"}:
            enabled = True
        elif arg.startswith("--domain_randomization=") or arg.startswith("--domain-randomization="):
            value = arg.split("=", 1)[1].lower()
            enabled = value not in {"0", "false", "no", "off"}
    os.environ["QOED_GO1_DOMAIN_RANDOMIZATION"] = "1" if enabled else "0"


_preparse_domain_randomization_flag()

import torch
from go1_tasks import GO1_MODE_TASKS, normalize_mjlab_rsl_rl_cfg
from mjlab.rl.runner import MjlabOnPolicyRunner

_BASE_GO1_TASKS = {"go1", "go1_velocity_flat", "Mjlab-Velocity-Flat-Unitree-Go1"}
_INFO_GAIN_CHOICES = {
    "boed": "boed",
    "qoed": "qoed",
    "q-oed": "qoed",
    "qoed-agnostic": "qoed-agnostic",
    "qoed_agnostic": "qoed-agnostic",
    "q-oed-agnostic": "qoed-agnostic",
    "q-oed_agnostic": "qoed-agnostic",
    "agnostic": "qoed-agnostic",
    "nothing": "nothing",
    "none": "nothing",
    "off": "nothing",
    "false": "nothing",
    "0": "nothing",
}


def _latest_go1_pretrain_checkpoint() -> Path | None:
    roots = {Path.cwd(), Path(__file__).resolve().parents[2]}
    checkpoints = [
        path
        for root in roots
        for path in (root / "logs/rsl_rl/go1_velocity").glob("*_pretrain/model_*.pt")
    ]
    if not checkpoints:
        return None

    nonzero_checkpoints = [
        path for path in checkpoints if re.search(r"model_([1-9]\d*)\.pt$", path.name)
    ]
    checkpoints = nonzero_checkpoints or checkpoints

    def sort_key(path: Path):
        match = re.search(r"model_(\d+)\.pt$", path.name)
        return path.parent.name, int(match.group(1)) if match else -1, path.stat().st_mtime

    return max(checkpoints, key=sort_key)


def _normalize_info_gain_mode(value: str) -> str:
    key = value.strip().lower()
    try:
        return _INFO_GAIN_CHOICES[key]
    except KeyError as exc:
        valid = "boed, qoed, qoed-agnostic, nothing"
        raise SystemExit(f"Unknown --info-gain mode '{value}'. Available modes: {valid}") from exc


def _record_requested_gpu_ids(value: str) -> None:
    ids = re.findall(r"\d+", value)
    if ids:
        os.environ["QOED_TRAIN_REQUESTED_CUDA_VISIBLE_DEVICES"] = ",".join(ids)
    elif value.strip().lower() == "all":
        os.environ["QOED_TRAIN_REQUESTED_CUDA_VISIBLE_DEVICES"] = "all"


def _rewrite_legacy_go1_args() -> None:
    args = sys.argv[1:]
    mode = None
    task = None
    viewer = None
    info_gain = None
    follow_camera = False
    gpu_ids_value = None
    gpu_ids_args = []
    remaining = []
    i = 0

    while i < len(args):
        arg = args[i]
        if arg == "--headless":
            i += 1
        elif arg in {"--domain_randomization", "--domain-randomization"}:
            i += 1
        elif arg.startswith("--domain_randomization=") or arg.startswith("--domain-randomization="):
            i += 1
        elif arg in {"--info-gain", "--info_gain"}:
            if i + 1 < len(args) and not args[i + 1].startswith("-"):
                info_gain = _normalize_info_gain_mode(args[i + 1])
                i += 2
            else:
                info_gain = "qoed"
                i += 1
        elif arg.startswith("--info-gain=") or arg.startswith("--info_gain="):
            info_gain = _normalize_info_gain_mode(arg.split("=", 1)[1])
            i += 1
        elif arg in {"--gpu-ids", "--gpu_ids"}:
            if i + 1 >= len(args) or args[i + 1].startswith("-"):
                raise SystemExit(f"{arg} expects a GPU id list")
            gpu_ids_value = args[i + 1]
            gpu_ids_args = [arg, args[i + 1]]
            i += 2
        elif arg.startswith("--gpu-ids=") or arg.startswith("--gpu_ids="):
            gpu_ids_value = arg.split("=", 1)[1]
            gpu_ids_args = [arg]
            i += 1
        elif arg == "--viewer":
            if i + 1 >= len(args) or args[i + 1].startswith("-"):
                raise SystemExit("--viewer expects one of: none, auto, native")
            viewer = args[i + 1].lower()
            i += 2
        elif arg.startswith("--viewer="):
            viewer = arg.split("=", 1)[1].lower()
            i += 1
        elif arg in {"--follow-camera", "--follow_camera"}:
            follow_camera = True
            i += 1
        elif arg in {"--system_dynamics_load_path", "--system-dynamics-load-path"}:
            if i + 1 >= len(args) or args[i + 1].startswith("-"):
                raise SystemExit(f"{arg} expects a checkpoint path")
            os.environ["QOED_SYSTEM_DYNAMICS_LOAD_PATH"] = args[i + 1]
            i += 2
        elif arg.startswith("--system_dynamics_load_path=") or arg.startswith("--system-dynamics-load-path="):
            os.environ["QOED_SYSTEM_DYNAMICS_LOAD_PATH"] = arg.split("=", 1)[1]
            i += 1
        elif arg in {"--checkpoint", "--checkpoint-file", "--checkpoint_file"}:
            if i + 1 >= len(args) or args[i + 1].startswith("-"):
                raise SystemExit(f"{arg} expects a checkpoint path")
            os.environ["QOED_POLICY_CHECKPOINT"] = args[i + 1]
            i += 2
        elif (
            arg.startswith("--checkpoint=")
            or arg.startswith("--checkpoint-file=")
            or arg.startswith("--checkpoint_file=")
        ):
            os.environ["QOED_POLICY_CHECKPOINT"] = arg.split("=", 1)[1]
            i += 1
        elif arg == "--mode":
            if i + 1 >= len(args) or args[i + 1].startswith("-"):
                raise SystemExit("--mode expects one of: " + ", ".join(sorted(GO1_MODE_TASKS)))
            mode = args[i + 1].lower()
            i += 2
        elif arg.startswith("--mode="):
            mode = arg.split("=", 1)[1].lower()
            i += 1
        elif arg == "--task":
            if i + 1 >= len(args) or args[i + 1].startswith("-"):
                raise SystemExit("--task expects a task id")
            task = args[i + 1]
            i += 2
        elif arg.startswith("--task="):
            task = arg.split("=", 1)[1]
            i += 1
        else:
            remaining.append(arg)
            i += 1

    if viewer is not None:
        if viewer not in {"none", "auto", "native"}:
            raise SystemExit(f"Unknown training viewer '{viewer}'. Available viewers: none, auto, native")
        os.environ["QOED_TRAIN_VIEWER"] = viewer
    os.environ["QOED_GO1_INFO_GAIN"] = info_gain or "nothing"
    if follow_camera:
        os.environ["QOED_TRAIN_FOLLOW_CAMERA"] = "1"

    if task is None and remaining and not remaining[0].startswith("-"):
        task = remaining.pop(0)

    if mode is not None:
        if mode not in GO1_MODE_TASKS:
            valid = ", ".join(sorted(GO1_MODE_TASKS))
            raise SystemExit(f"Unknown Go1 mode '{mode}'. Available modes: {valid}")
        if task is None or task in _BASE_GO1_TASKS:
            task = GO1_MODE_TASKS[mode]

    resolved_mode = mode or next((name for name, task_id in GO1_MODE_TASKS.items() if task == task_id), None)
    if resolved_mode == "finetune" and gpu_ids_value is not None:
        _record_requested_gpu_ids(gpu_ids_value)
    elif gpu_ids_args:
        remaining.extend(gpu_ids_args)

    if resolved_mode == "finetune":
        os.environ["QOED_GO1_FINETUNE"] = "1"
        checkpoint = os.environ.get("QOED_POLICY_CHECKPOINT")
        dynamics_path = os.environ.get("QOED_SYSTEM_DYNAMICS_LOAD_PATH")
        if checkpoint is None and dynamics_path is None:
            latest = _latest_go1_pretrain_checkpoint()
            if latest is None:
                raise SystemExit("No Go1 pretrain checkpoint found under logs/rsl_rl/go1_velocity/*_pretrain/")
            checkpoint = dynamics_path = str(latest)
            print(f"[INFO]: Auto-loading latest Go1 pretrain checkpoint: {latest}")
        os.environ.setdefault("QOED_POLICY_CHECKPOINT", checkpoint or dynamics_path)
        os.environ.setdefault("QOED_SYSTEM_DYNAMICS_LOAD_PATH", dynamics_path or checkpoint)

    if task is not None:
        sys.argv = [sys.argv[0], task, *remaining]


def _patch_mjlab_runner_for_current_rsl_rl() -> None:
    """Adapt MJLab's actor/critic config schema to newer rsl-rl policy config."""
    if getattr(MjlabOnPolicyRunner, "_qoed_policy_cfg_patch", False):
        return

    original_init = MjlabOnPolicyRunner.__init__
    original_load = MjlabOnPolicyRunner.load

    def patched_init(self, env, train_cfg, log_dir=None, device="cpu"):
        original_init(self, env, _normalize_train_cfg(train_cfg), log_dir, device)

    def patched_save(self, path: str, infos=None) -> None:
        env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
        infos = {**(infos or {}), "env_state": env_state}

        if hasattr(self.alg, "save"):
            saved_dict = self.alg.save()
        else:
            saved_dict = {
                "model_state_dict": self.alg.policy.state_dict(),
                "optimizer_state_dict": self.alg.optimizer.state_dict(),
            }
            if getattr(self.alg, "rnd", None):
                saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
                saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()

        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)

        if (
            self.cfg.get("upload_model", True)
            and getattr(self, "logger_type", None) in {"neptune", "wandb"}
            and not getattr(self, "disable_logs", False)
        ):
            self.writer.save_model(path, self.current_learning_iteration)

    def patched_load(self, path: str, load_cfg=None, strict: bool = True, map_location: str | None = None):
        if hasattr(self.alg, "load"):
            return original_load(self, path, load_cfg, strict, map_location)

        loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
        if "model_state_dict" not in loaded_dict:
            return original_load(self, path, load_cfg, strict, map_location)

        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"], strict=strict)
        if "optimizer_state_dict" in loaded_dict:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if getattr(self.alg, "rnd", None) and "rnd_state_dict" in loaded_dict:
            self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
        if getattr(self.alg, "rnd_optimizer", None) and "rnd_optimizer_state_dict" in loaded_dict:
            self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict.get("iter", 0)

        infos = loaded_dict.get("infos")
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos

    MjlabOnPolicyRunner.__init__ = patched_init
    MjlabOnPolicyRunner.save = patched_save
    MjlabOnPolicyRunner.load = patched_load
    MjlabOnPolicyRunner._qoed_policy_cfg_patch = True


def _normalize_train_cfg(train_cfg: dict) -> dict:
    return normalize_mjlab_rsl_rl_cfg(train_cfg)


_patch_mjlab_runner_for_current_rsl_rl()
_rewrite_legacy_go1_args()

import mjlab.scripts.train as mjlab_train
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.wandb import add_wandb_tags

from mbpo.envs.go1.mujoco_go1_train_env import DirectMujocoGo1VecEnv


_GO1_TASK_IDS = set(GO1_MODE_TASKS.values())
_DIRECT_MUJOCO_TASK_IDS = {GO1_MODE_TASKS["finetune"]}
_original_run_train = mjlab_train.run_train


def _run_train_with_direct_checkpoint(task_id, cfg, log_dir):
    if task_id not in _DIRECT_MUJOCO_TASK_IDS:
        return _original_run_train(task_id, cfg, log_dir)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    requested_cuda_visible = os.environ.get("QOED_TRAIN_REQUESTED_CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "" and requested_cuda_visible == "":
        device = "cpu"
        seed = cfg.agent.seed
        rank = 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        device = f"cuda:{local_rank}"
        seed = cfg.agent.seed + local_rank
        try:
            torch.cuda.set_device(local_rank)
            torch.empty(1, device=device)
        except Exception as exc:
            print(f"[WARN] CUDA was requested but failed to initialize; falling back to CPU. Error: {exc}")
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            device = "cpu"
            seed = cfg.agent.seed
            rank = 0

    mjlab_train.configure_torch_backends()
    cfg.agent.seed = seed
    cfg.env.seed = seed

    viewer = os.environ.get("QOED_TRAIN_VIEWER", "none").lower()
    if viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer = "native" if has_display else "none"
    if rank != 0:
        viewer = "none"
    follow_camera = os.environ.get("QOED_TRAIN_FOLLOW_CAMERA") == "1"

    if rank == 0:
        viewer_label = f"{viewer} (60 Hz throttle)" if viewer == "native" else viewer
        info_gain = os.environ.get("QOED_GO1_INFO_GAIN", "nothing")
        print(f"[INFO] Training with: direct_mujoco=True, policy_device={device}, sim_device=cpu, seed={seed}, rank={rank}")
        print(f"[INFO] Training viewer: {viewer_label}")
        print(f"[INFO] Info-gain action selection: {info_gain}")
        print(f"[INFO] Logging experiment in directory: {log_dir}")
        print("[INFO] Native MuJoCo owns simulation state on CPU; RSL-RL moves observations to the policy device.")

    if cfg.video:
        raise SystemExit("Video recording is not implemented for direct MuJoCo training.")
    if cfg.enable_nan_guard:
        print("[WARN] --enable-nan-guard is an MJLab/Warp feature and is ignored by direct MuJoCo training.")

    env = DirectMujocoGo1VecEnv(
        cfg.env,
        agent_cfg=cfg.agent,
        seed=seed,
        viewer=viewer,
        follow_camera=follow_camera,
    )
    agent_cfg = asdict(cfg.agent)
    env_cfg = asdict(cfg.env)

    if rank == 0:
        params_dir = Path(log_dir) / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        mjlab_train.dump_yaml(params_dir / "env.yaml", env_cfg)
        mjlab_train.dump_yaml(params_dir / "agent.yaml", agent_cfg)

    runner_cls = load_runner_cls(task_id)
    if runner_cls is None:
        runner_cls = MjlabOnPolicyRunner

    runner = runner_cls(env, agent_cfg, str(log_dir), device)
    add_wandb_tags(cfg.agent.wandb_tags)
    runner.add_git_repo_to_log(__file__)

    checkpoint = os.environ.get("QOED_POLICY_CHECKPOINT")
    resume_path = None
    if checkpoint is not None:
        resume_path = Path(checkpoint).expanduser()
    elif cfg.agent.resume:
        resume_path = mjlab_train.get_checkpoint_path(
            Path(log_dir).parent,
            cfg.agent.load_run,
            cfg.agent.load_checkpoint,
        )

    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    try:
        runner.learn(
            num_learning_iterations=cfg.agent.max_iterations,
            init_at_random_ep_len=False,
        )
    finally:
        env.close()


mjlab_train.run_train = _run_train_with_direct_checkpoint
main = mjlab_train.main


if __name__ == "__main__":
    main()
