"""Go1 RSL-RL playback entry point."""

from __future__ import annotations

import argparse
import ast
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from threading import Lock

import mujoco
import numpy as np

_REMOVED_CUDA_LIBRARY_PATHS: list[str] = []
_PRINTED_CUDA_LIBRARY_PATH_NOTICE = False


def _drop_system_cuda_library_paths() -> None:
    raw_path = os.environ.get("LD_LIBRARY_PATH")
    if not raw_path:
        return

    parts = [part for part in raw_path.split(os.pathsep) if part]
    kept = [part for part in parts if not part.startswith("/usr/local/cuda")]
    if len(kept) == len(parts):
        return

    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(kept)
    _REMOVED_CUDA_LIBRARY_PATHS.extend(part for part in parts if part not in kept)


_drop_system_cuda_library_paths()

import torch

import go1_tasks  # noqa: F401
from go1_tasks import GO1_MODE_TASKS
from mbpo.envs.go1.mujoco_go1_backend import PureMujocoGo1Model, draw_go1_velocity_arrows
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.modules import ActorCritic


_BASE_GO1_TASKS = {"go1", "go1_velocity_flat", "Mjlab-Velocity-Flat-Unitree-Go1"}


class X11ArrowKeyReader:
    """Best-effort held-key reader for local native MuJoCo playback on X11."""

    _KEYSYMS = {
        "forward": (0xFF52,),
        "backward": (0xFF54,),
        "turn_left": (0xFF51,),
        "turn_right": (0xFF53,),
        "stop": (0x0078, 0x0058),
    }

    def __init__(self) -> None:
        self._display = None
        self._x11 = None
        self._keycodes: dict[str, list[int]] = {}
        if not os.environ.get("DISPLAY"):
            return

        try:
            import ctypes
            import ctypes.util

            lib_path = ctypes.util.find_library("X11")
            if lib_path is None:
                return
            x11 = ctypes.CDLL(lib_path)
            x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
            x11.XOpenDisplay.restype = ctypes.c_void_p
            x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
            x11.XCloseDisplay.restype = ctypes.c_int
            x11.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            x11.XKeysymToKeycode.restype = ctypes.c_uint
            x11.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
            x11.XQueryKeymap.restype = ctypes.c_int

            display = x11.XOpenDisplay(None)
            if not display:
                return

            self._x11 = x11
            self._display = display
            for command, keysyms in self._KEYSYMS.items():
                keycodes = [
                    int(x11.XKeysymToKeycode(display, keysym))
                    for keysym in keysyms
                ]
                self._keycodes[command] = [keycode for keycode in keycodes if keycode > 0]
        except Exception:
            self._display = None
            self._x11 = None
            self._keycodes = {}

    @property
    def available(self) -> bool:
        return self._display is not None and self._x11 is not None

    def pressed_commands(self) -> set[str]:
        if not self.available:
            return set()

        import ctypes

        keymap = ctypes.create_string_buffer(32)
        if self._x11.XQueryKeymap(self._display, keymap) == 0:
            return set()

        raw = keymap.raw
        pressed = set()
        for command, keycodes in self._keycodes.items():
            for keycode in keycodes:
                if raw[keycode // 8] & (1 << (keycode % 8)):
                    pressed.add(command)
                    break
        return pressed

    def close(self) -> None:
        if self.available:
            self._x11.XCloseDisplay(self._display)
        self._display = None
        self._x11 = None


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


def _parse_bool(value: str | bool | None) -> bool:
    if value is None:
        return True
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _parse_gpu_ids(value: str) -> list[int] | str | None:
    value = value.strip()
    if value.lower() in {"", "none", "null", "cpu"}:
        return None
    if value.lower() == "all":
        return "all"

    try:
        parsed = ast.literal_eval(value)
    except (SyntaxError, ValueError):
        parsed = value

    if isinstance(parsed, int):
        return [parsed]
    if isinstance(parsed, (list, tuple)):
        try:
            return [int(gpu_id) for gpu_id in parsed]
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(f"invalid GPU ids: {value}") from exc
    if isinstance(parsed, str):
        try:
            return [int(part.strip()) for part in parsed.split(",") if part.strip()]
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid GPU ids: {value}") from exc

    raise argparse.ArgumentTypeError(f"invalid GPU ids: {value}")


def _sanitize_cuda_library_path() -> None:
    global _PRINTED_CUDA_LIBRARY_PATH_NOTICE
    _drop_system_cuda_library_paths()
    if _REMOVED_CUDA_LIBRARY_PATHS and not _PRINTED_CUDA_LIBRARY_PATH_NOTICE:
        removed = ", ".join(dict.fromkeys(_REMOVED_CUDA_LIBRARY_PATHS))
        print(f"[INFO] Ignoring system CUDA library paths for PyTorch: {removed}")
        _PRINTED_CUDA_LIBRARY_PATH_NOTICE = True


def _resolve_device(device: str | None, gpu_ids: list[int] | str | None) -> str:
    if device is not None:
        if device.startswith("cuda"):
            _sanitize_cuda_library_path()
        return device

    _sanitize_cuda_library_path()

    if gpu_ids is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"

    try:
        selected_gpus, num_gpus = select_gpus(gpu_ids)
    except (IndexError, RuntimeError) as exc:
        raise SystemExit(f"Unable to select GPU ids {gpu_ids}: {exc}") from exc

    if selected_gpus is None or num_gpus == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    return "cuda:0"


def _resolve_task(task: str | None, mode: str | None) -> str:
    if mode is not None:
        mode = mode.lower()
        if mode not in GO1_MODE_TASKS:
            valid = ", ".join(sorted(GO1_MODE_TASKS))
            raise SystemExit(f"Unknown Go1 mode '{mode}'. Available modes: {valid}")
        if task is None or task in _BASE_GO1_TASKS:
            return GO1_MODE_TASKS[mode]
    if task in _BASE_GO1_TASKS:
        return GO1_MODE_TASKS["pretrain"]
    if task is None:
        raise SystemExit("A task id is required, or use --task go1 --mode pretrain")
    return task


class MujocoArrowCommandController:
    """Held-arrow velocity controller for the direct MuJoCo player."""

    KEY_PULSE_SECONDS = 0.6
    _KEY_MAP = {
        265: "forward",  # GLFW_KEY_UP
        264: "backward",  # GLFW_KEY_DOWN
        263: "turn_left",  # GLFW_KEY_LEFT
        262: "turn_right",  # GLFW_KEY_RIGHT
        ord("X"): "stop",
        ord("x"): "stop",
        ord("R"): "reset",
        ord("r"): "reset",
    }

    def __init__(self, linear_speed: float = 0.8, yaw_speed: float = 0.6) -> None:
        self.linear_speed = float(linear_speed)
        self.yaw_speed = float(yaw_speed)
        self._active_until: dict[str, float] = {}
        self._reset_requested = False
        self._active_lock = Lock()
        self._key_reader = X11ArrowKeyReader()

    @property
    def has_held_key_polling(self) -> bool:
        return self._key_reader.available

    def handle_mujoco_key(self, key: int) -> None:
        command = self._KEY_MAP.get(key)
        if command is None:
            return
        with self._active_lock:
            if command == "reset":
                self._reset_requested = True
            elif command == "stop":
                self._active_until.clear()
            else:
                self._active_until[command] = time.monotonic() + self.KEY_PULSE_SECONDS

    def _active_commands(self) -> set[str]:
        now = time.monotonic()
        held_commands = self._key_reader.pressed_commands()
        with self._active_lock:
            if "stop" in held_commands:
                self._active_until.clear()
                held_commands.clear()
            self._active_until = {
                name: until for name, until in self._active_until.items() if until > now
            }
            return {
                name for name, until in self._active_until.items() if until > now
            } | held_commands

    def command(self) -> np.ndarray:
        active_commands = self._active_commands()
        forward = int("forward" in active_commands) - int("backward" in active_commands)
        yaw = int("turn_left" in active_commands) - int("turn_right" in active_commands)
        return np.asarray(
            [forward * self.linear_speed, 0.0, yaw * self.yaw_speed],
            dtype=np.float32,
        )

    def consume_reset_requested(self) -> bool:
        with self._active_lock:
            requested = self._reset_requested
            self._reset_requested = False
        return requested

    def close(self) -> None:
        self._key_reader.close()


class PureMujocoGo1Player:
    def __init__(self, env_cfg, agent_cfg, checkpoint_file: Path, device: str) -> None:
        self.model = PureMujocoGo1Model(env_cfg)
        self.policy_device = torch.device(device)
        if self.policy_device.type == "cuda":
            torch.cuda.set_device(self.policy_device)
        self.actor_obs_dim = 0
        self.clip_actions = agent_cfg.clip_actions
        self.policy = self._load_policy(agent_cfg, checkpoint_file)

    def _load_policy(self, agent_cfg, checkpoint_file: Path) -> ActorCritic:
        loaded = torch.load(checkpoint_file, weights_only=False, map_location="cpu")
        state_dict = loaded.get("model_state_dict", loaded)
        if not isinstance(state_dict, dict):
            raise RuntimeError(f"Checkpoint does not contain a policy state dict: {checkpoint_file}")

        self.actor_obs_dim = int(state_dict["actor.0.weight"].shape[1])
        critic_obs_dim = int(state_dict["critic.0.weight"].shape[1])
        train_cfg = go1_tasks.normalize_mjlab_rsl_rl_cfg(asdict(agent_cfg), mbpo=True)
        policy_cfg = dict(train_cfg["policy"])
        policy_cfg.pop("class_name", None)
        obs = {
            "actor": torch.zeros(1, self.actor_obs_dim, device=self.policy_device),
            "critic": torch.zeros(1, critic_obs_dim, device=self.policy_device),
        }
        policy = ActorCritic(
            obs=obs,
            obs_groups=train_cfg["obs_groups"],
            num_actions=self.model.action_dim,
            **policy_cfg,
        ).to(self.policy_device)
        state_dict = go1_tasks._migrate_policy_noise_std_state_dict(policy, state_dict)
        policy.load_state_dict(state_dict, strict=True)
        go1_tasks._patch_policy_distribution_safety(policy)
        policy.eval()
        actual_device = next(policy.parameters()).device
        print(f"[INFO] Policy network parameters are on: {actual_device}")
        return policy

    def _observation_to_policy_tensor(self, obs: np.ndarray) -> torch.Tensor:
        # MuJoCo owns the state on CPU; policy inference runs on policy_device.
        obs_cpu = torch.from_numpy(np.asarray(obs, dtype=np.float32))
        return obs_cpu.to(self.policy_device, non_blocking=True).unsqueeze(0)

    def _act(self, command: np.ndarray) -> np.ndarray:
        obs = self.model.observation(command)
        if obs.shape[0] != self.actor_obs_dim:
            raise RuntimeError(
                f"Pure MuJoCo observation has dim {obs.shape[0]}, "
                f"but checkpoint expects {self.actor_obs_dim}."
            )
        obs_tensor = self._observation_to_policy_tensor(obs)
        with torch.inference_mode():
            action = self.policy.act_inference({"actor": obs_tensor})
        action_np = action.squeeze(0).detach().cpu().numpy()
        if self.clip_actions is not None:
            limit = float(self.clip_actions)
            action_np = np.clip(action_np, -limit, limit)
        return action_np

    def _step_once(self, command: np.ndarray, auto_reset: bool) -> None:
        action = self._act(command)
        self.model.step(action)
        if auto_reset and self.model.is_fallen():
            self.model.reset()

    def run(self, args) -> None:
        if args.video:
            raise SystemExit(
                "Video recording is not implemented in the direct MuJoCo play loop yet."
            )

        controller = None
        if args.keyboard_control:
            controller = MujocoArrowCommandController(
                linear_speed=args.keyboard_linear_speed,
                yaw_speed=args.keyboard_yaw_speed,
            )
            held_status = "held-key polling active" if controller.has_held_key_polling else "native key repeats only"
            print(
                "[INFO] MuJoCo arrow control enabled: "
                f"up/down forward/back, left/right yaw, X stop, R reset ({held_status})."
            )

        viewer = args.viewer
        if viewer == "auto":
            has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
            viewer = "native" if has_display else "none"

        try:
            if viewer == "native":
                self._run_native(args, controller)
            else:
                self._run_headless(args, controller)
        finally:
            if controller is not None:
                controller.close()

    def _current_command(self, controller: MujocoArrowCommandController | None) -> np.ndarray:
        if controller is None:
            return np.zeros(3, dtype=np.float32)
        return controller.command()

    def _run_native(self, args, controller: MujocoArrowCommandController | None) -> None:
        import mujoco.viewer

        key_callback = controller.handle_mujoco_key if controller is not None else None
        max_steps = args.num_steps
        step = 0
        auto_reset = not args.no_terminations
        with mujoco.viewer.launch_passive(
            self.model.model,
            self.model.data,
            key_callback=key_callback,
        ) as handle:
            with handle.lock():
                self.model.update_camera(handle.cam)
            while handle.is_running() and (max_steps is None or step < max_steps):
                start_time = time.time()
                command = self._current_command(controller)
                active = bool(np.any(np.abs(command) > 1.0e-6))
                with handle.lock():
                    if controller is not None and controller.consume_reset_requested():
                        self.model.reset()
                    self._step_once(command, auto_reset=auto_reset)
                    self.model.set_active_color(active)
                    if args.follow_camera:
                        self.model.update_camera(handle.cam)
                    draw_go1_velocity_arrows(
                        handle.user_scn,
                        self.model.model,
                        self.model.data,
                        command,
                    )
                handle.sync()
                step += 1

                sleep_time = self.model.step_dt - (time.time() - start_time)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)

    def _run_headless(self, args, controller: MujocoArrowCommandController | None) -> None:
        max_steps = args.num_steps
        step = 0
        auto_reset = not args.no_terminations
        while max_steps is None or step < max_steps:
            start_time = time.time()
            if controller is not None and controller.consume_reset_requested():
                self.model.reset()
            command = self._current_command(controller)
            self._step_once(command, auto_reset=auto_reset)
            self.model.set_active_color(bool(np.any(np.abs(command) > 1.0e-6)))
            step += 1

            sleep_time = self.model.step_dt - (time.time() - start_time)
            if args.real_time and sleep_time > 0.0:
                time.sleep(sleep_time)


def _run_mujoco_go1_play(args, task: str, checkpoint_file: Path, device: str) -> None:
    if task not in set(GO1_MODE_TASKS.values()):
        raise SystemExit("Pure MuJoCo play currently supports the Go1 tasks only.")
    if args.num_envs not in (None, 1):
        print(f"[INFO] Pure MuJoCo play runs one environment; ignoring --num-envs {args.num_envs}.")

    print(f"[INFO] Playing with: direct_mujoco=True, policy_device={device}")
    print("[INFO] MuJoCo state is CPU-side; observations are copied to the policy device each step.")
    env_cfg = load_env_cfg(task, play=True)
    agent_cfg = load_rl_cfg(task)
    if args.no_terminations:
        env_cfg.terminations = {}

    print(f"[INFO] Loading checkpoint: {checkpoint_file}")
    player = PureMujocoGo1Player(env_cfg, agent_cfg, checkpoint_file, device=device)
    player.run(args)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Play a Go1 RSL-RL checkpoint in direct MuJoCo.")
    parser.add_argument("task_pos", nargs="?", help="Task id, or go1 alias.")
    parser.add_argument("--task", dest="task_opt", help="Task id, or go1 alias.")
    parser.add_argument("--mode", choices=sorted(GO1_MODE_TASKS), help="Go1 mode alias.")
    parser.add_argument("--checkpoint-file", "--checkpoint_file", "--checkpoint")
    parser.add_argument("--num-envs", "--num_envs", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-ids", "--gpu_ids", type=_parse_gpu_ids, default=None)
    parser.add_argument("--video", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--video-length", "--video_length", type=int, default=200)
    parser.add_argument("--video-height", "--video_height", type=int, default=None)
    parser.add_argument("--video-width", "--video_width", type=int, default=None)
    parser.add_argument("--real-time", "--real_time", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--headless", nargs="?", const=True, default=True, type=_parse_bool)
    parser.add_argument("--viewer", choices=("none", "auto", "native"), default="none")
    parser.add_argument("--follow-camera", "--follow_camera", action="store_true")
    parser.add_argument("--no-terminations", "--no_terminations", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--num-steps", "--num_steps", type=int, default=None)
    parser.add_argument("--keyboard-control", dest="keyboard_control", action="store_true", default=True)
    parser.add_argument("--no-keyboard-control", dest="keyboard_control", action="store_false")
    parser.add_argument("--keyboard-linear-speed", type=float, default=0.8)
    parser.add_argument("--keyboard-yaw-speed", type=float, default=0.6)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    device = _resolve_device(args.device, args.gpu_ids)
    configure_torch_backends()

    task = _resolve_task(args.task_opt or args.task_pos, args.mode)
    if args.checkpoint_file is None:
        checkpoint_file = _latest_go1_pretrain_checkpoint()
        if checkpoint_file is None:
            raise SystemExit("No Go1 pretrain checkpoint found under logs/rsl_rl/go1_velocity/*_pretrain/")
        print(f"[INFO] Auto-loading latest Go1 pretrain checkpoint: {checkpoint_file}")
    else:
        checkpoint_file = Path(args.checkpoint_file).expanduser()
    if not checkpoint_file.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_file}")

    _run_mujoco_go1_play(args, task, checkpoint_file, device)


if __name__ == "__main__":
    sys.exit(main())
