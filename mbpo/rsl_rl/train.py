"""MJLab RSL-RL training entry point."""

from __future__ import annotations

import argparse
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


def _restart_without_system_cuda() -> None:
    parts = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep) if part]
    kept = [part for part in parts if not part.startswith("/usr/local/cuda")]
    cuda_visible = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if cuda_visible:
        os.environ["QOED_TRAIN_REQUESTED_CUDA_VISIBLE_DEVICES"] = cuda_visible
    if kept == parts and not cuda_visible:
        return
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
    os.execve(sys.executable, [sys.executable, *sys.argv], os.environ)


_restart_without_system_cuda()

INFO_GAIN_MODES = ("qoed", "qoed-agnostic", "boed", "nothing")
_BASE_GO1_TASKS = {"go1", "go1_velocity_flat", "Mjlab-Velocity-Flat-Unitree-Go1"}

_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
_parser.add_argument("--task")
_parser.add_argument("--mode", type=str.lower)
_parser.add_argument("--info-gain", "--info_gain", nargs="?", const="qoed", default="nothing")
_parser.add_argument("--viewer", type=str.lower, choices=("none", "auto", "native"))
_parser.add_argument("--follow-camera", "--follow_camera", action="store_true")
_parser.add_argument("--system-dynamics-load-path", "--system_dynamics_load_path")
_parser.add_argument("--checkpoint", "--checkpoint-file", "--checkpoint_file")
_parser.add_argument("--domain-randomization", "--domain_randomization", action=argparse.BooleanOptionalAction, default=True)
_parser.add_argument("--headless", action="store_true")
_args, _mjlab_args = _parser.parse_known_args()
os.environ["QOED_GO1_DOMAIN_RANDOMIZATION"] = "1" if _args.domain_randomization else "0"

import torch
from go1_tasks import GO1_MODE_TASKS, latest_go1_pretrain_checkpoint, normalize_mjlab_rsl_rl_cfg
from mjlab.rl.runner import MjlabOnPolicyRunner


def _rewrite_args_for_mjlab(args, mjlab_args: list[str]) -> None:
    info_gain = args.info_gain.strip().lower().replace("_", "-")
    if info_gain not in INFO_GAIN_MODES:
        raise SystemExit(f"Unknown --info-gain mode '{args.info_gain}'. Available modes: {', '.join(INFO_GAIN_MODES)}")
    os.environ["QOED_GO1_INFO_GAIN"] = info_gain
    if args.viewer is not None:
        os.environ["QOED_TRAIN_VIEWER"] = args.viewer
    if args.follow_camera:
        os.environ["QOED_TRAIN_FOLLOW_CAMERA"] = "1"
    if args.system_dynamics_load_path is not None:
        os.environ["QOED_SYSTEM_DYNAMICS_LOAD_PATH"] = args.system_dynamics_load_path
    if args.checkpoint is not None:
        os.environ["QOED_POLICY_CHECKPOINT"] = args.checkpoint

    task, mode = args.task, args.mode
    if task is None and mjlab_args and not mjlab_args[0].startswith("-"):
        task = mjlab_args.pop(0)
    if mode is not None:
        if mode not in GO1_MODE_TASKS:
            raise SystemExit(f"Unknown Go1 mode '{mode}'. Available modes: {', '.join(sorted(GO1_MODE_TASKS))}")
        if task is None or task in _BASE_GO1_TASKS:
            task = GO1_MODE_TASKS[mode]
    mode = mode or next((name for name, task_id in GO1_MODE_TASKS.items() if task == task_id), None)

    if mode == "finetune":
        gpu_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
        gpu_parser.add_argument("--gpu-ids", "--gpu_ids")
        gpu, mjlab_args = gpu_parser.parse_known_args(mjlab_args)
        ids = re.findall(r"\d+", gpu.gpu_ids or "")
        if ids or (gpu.gpu_ids or "").strip().lower() == "all":
            os.environ["QOED_TRAIN_REQUESTED_CUDA_VISIBLE_DEVICES"] = ",".join(ids) or "all"

        os.environ["QOED_GO1_FINETUNE"] = "1"
        checkpoint = os.environ.get("QOED_POLICY_CHECKPOINT")
        dynamics_path = os.environ.get("QOED_SYSTEM_DYNAMICS_LOAD_PATH")
        if checkpoint is None and dynamics_path is None:
            latest = latest_go1_pretrain_checkpoint()
            if latest is None:
                raise SystemExit("No Go1 pretrain checkpoint found under logs/rsl_rl/go1_velocity/*_pretrain/")
            checkpoint = dynamics_path = str(latest)
        os.environ.setdefault("QOED_POLICY_CHECKPOINT", checkpoint or dynamics_path)
        os.environ.setdefault("QOED_SYSTEM_DYNAMICS_LOAD_PATH", dynamics_path or checkpoint)

    if task is not None:
        sys.argv = [sys.argv[0], task, *mjlab_args]


def _patch_mjlab_runner_for_rsl_rl_rwm() -> None:
    """Adapt MJLab's actor/critic config schema to newer rsl-rl policy config."""
    original_init = MjlabOnPolicyRunner.__init__
    original_load = MjlabOnPolicyRunner.load

    def patched_init(self, env, train_cfg, log_dir=None, device="cpu"):
        original_init(self, env, normalize_mjlab_rsl_rl_cfg(train_cfg), log_dir, device)

    def patched_save(self, path: str, infos=None) -> None:
        infos = {**(infos or {}), "env_state": {"common_step_counter": self.env.unwrapped.common_step_counter}}
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


_patch_mjlab_runner_for_rsl_rl_rwm()
_rewrite_args_for_mjlab(_args, _mjlab_args)

import mjlab.scripts.train as mjlab_train
from mjlab.tasks.registry import load_runner_cls
from mjlab.utils.wandb import add_wandb_tags

from mbpo.envs.go1.mujoco_go1_train_env import DirectMujocoGo1VecEnv

_mjlab_run_train = mjlab_train.run_train


def _run_train(task_id, cfg, log_dir):
    if task_id != GO1_MODE_TASKS["finetune"]:
        return _mjlab_run_train(task_id, cfg, log_dir)

    if not os.environ.get("CUDA_VISIBLE_DEVICES") and not os.environ.get("QOED_TRAIN_REQUESTED_CUDA_VISIBLE_DEVICES"):
        device, seed, rank = "cpu", cfg.agent.seed, 0
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        rank = int(os.environ.get("RANK", "0"))
        device, seed = f"cuda:{local_rank}", cfg.agent.seed + local_rank
        torch.cuda.set_device(local_rank)

    mjlab_train.configure_torch_backends()
    cfg.agent.seed = cfg.env.seed = seed
    viewer = os.environ.get("QOED_TRAIN_VIEWER", "none")
    if viewer == "auto":
        viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "none"
    if rank != 0:
        viewer = "none"
    if cfg.video:
        raise SystemExit("Video recording is not implemented for direct MuJoCo training.")

    env = DirectMujocoGo1VecEnv(
        cfg.env,
        agent_cfg=cfg.agent,
        seed=seed,
        viewer=viewer,
        follow_camera=os.environ.get("QOED_TRAIN_FOLLOW_CAMERA") == "1",
    )
    agent_cfg = asdict(cfg.agent)
    if rank == 0:
        params_dir = Path(log_dir) / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        mjlab_train.dump_yaml(params_dir / "env.yaml", asdict(cfg.env))
        mjlab_train.dump_yaml(params_dir / "agent.yaml", agent_cfg)

    runner = (load_runner_cls(task_id) or MjlabOnPolicyRunner)(env, agent_cfg, str(log_dir), device)
    add_wandb_tags(cfg.agent.wandb_tags)
    runner.add_git_repo_to_log(__file__)

    checkpoint = os.environ.get("QOED_POLICY_CHECKPOINT")
    resume_path = None
    if checkpoint is not None:
        resume_path = Path(checkpoint).expanduser()
    elif cfg.agent.resume:
        resume_path = mjlab_train.get_checkpoint_path(Path(log_dir).parent, cfg.agent.load_run, cfg.agent.load_checkpoint)
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        runner.load(str(resume_path))

    runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=False)
    env.close()


mjlab_train.run_train = _run_train
main = mjlab_train.main


if __name__ == "__main__":
    main()
