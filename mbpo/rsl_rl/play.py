"""RSL-RL playback entry point on direct MuJoCo."""

from __future__ import annotations

import argparse
import ast
import ctypes
import ctypes.util
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from threading import Lock

import numpy as np
import torch

from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.torch import configure_torch_backends
from rsl_rl.modules import ActorCritic

from mbpo.envs.g1.mujoco_g1_train_env import PureMujocoG1Model
from mbpo.envs.jackal.mujoco_jackal_backend import PureMujocoJackalModel
from mbpo.envs.leap_hand.mujoco_leap_train_env import PureMujocoLeapModel
from mbpo.envs.mujoco_backend import JointPositionMujocoModel, draw_velocity_arrows
from mbpo.rsl_rl.g1_tasks import G1
from mbpo.rsl_rl.go1_tasks import GO1
from mbpo.rsl_rl.jackal_tasks import JACKAL
from mbpo.rsl_rl.leap_tasks import LEAP
from mbpo.rsl_rl.mbpo_tasks import (
    MODES,
    latest_pretrain_checkpoint,
    migrate_policy_noise_std_state_dict,
    normalize_mjlab_rsl_rl_cfg,
    patch_policy_distribution_safety,
)

ROBOTS = {
    "go1": (GO1, JointPositionMujocoModel),
    "jackal": (JACKAL, PureMujocoJackalModel),
    "g1": (G1, PureMujocoG1Model),
    "leap": (LEAP, PureMujocoLeapModel),
}


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
        lib_path = ctypes.util.find_library("X11") if os.environ.get("DISPLAY") else None
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
        self._x11, self._display = x11, display
        for command, keysyms in self._KEYSYMS.items():
            keycodes = [int(x11.XKeysymToKeycode(display, keysym)) for keysym in keysyms]
            self._keycodes[command] = [keycode for keycode in keycodes if keycode > 0]

    @property
    def available(self) -> bool:
        return self._display is not None

    def pressed_commands(self) -> set[str]:
        if not self.available:
            return set()
        keymap = ctypes.create_string_buffer(32)
        if self._x11.XQueryKeymap(self._display, keymap) == 0:
            return set()
        raw = keymap.raw
        return {
            command
            for command, keycodes in self._keycodes.items()
            if any(raw[keycode // 8] & (1 << (keycode % 8)) for keycode in keycodes)
        }

    def close(self) -> None:
        if self.available:
            self._x11.XCloseDisplay(self._display)
        self._display = None
        self._x11 = None


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

    def command(self) -> np.ndarray:
        now = time.monotonic()
        held = self._key_reader.pressed_commands()
        with self._active_lock:
            if "stop" in held:
                self._active_until.clear()
                held.clear()
            self._active_until = {name: until for name, until in self._active_until.items() if until > now}
            active = set(self._active_until) | held
        forward = int("forward" in active) - int("backward" in active)
        yaw = int("turn_left" in active) - int("turn_right" in active)
        return np.asarray([forward * self.linear_speed, 0.0, yaw * self.yaw_speed], dtype=np.float32)

    def consume_reset_requested(self) -> bool:
        with self._active_lock:
            requested, self._reset_requested = self._reset_requested, False
        return requested

    def close(self) -> None:
        self._key_reader.close()


class PureMujocoPlayer:
    def __init__(self, backend_cls, env_cfg, agent_cfg, checkpoint_file: Path, device: str) -> None:
        self.model = backend_cls(env_cfg)
        self.policy_device = torch.device(device)
        if self.policy_device.type == "cuda":
            torch.cuda.set_device(self.policy_device)
        self.clip_actions = agent_cfg.clip_actions
        self.policy = self._load_policy(agent_cfg, checkpoint_file)

    def _load_policy(self, agent_cfg, checkpoint_file: Path) -> ActorCritic:
        loaded = torch.load(checkpoint_file, weights_only=False, map_location="cpu")
        state_dict = loaded.get("model_state_dict", loaded)
        self.actor_obs_dim = int(state_dict["actor.0.weight"].shape[1])
        train_cfg = normalize_mjlab_rsl_rl_cfg(asdict(agent_cfg), mbpo=True)
        policy_cfg = dict(train_cfg["policy"])
        policy_cfg.pop("class_name", None)
        obs = {
            "actor": torch.zeros(1, self.actor_obs_dim, device=self.policy_device),
            "critic": torch.zeros(1, int(state_dict["critic.0.weight"].shape[1]), device=self.policy_device),
        }
        policy = ActorCritic(obs=obs, obs_groups=train_cfg["obs_groups"], num_actions=self.model.action_dim, **policy_cfg)
        policy = policy.to(self.policy_device)
        policy.load_state_dict(migrate_policy_noise_std_state_dict(policy, state_dict), strict=True)
        patch_policy_distribution_safety(policy)
        return policy.eval()

    def _step_once(self, command: np.ndarray, auto_reset: bool) -> None:
        obs = self.model.observation(command)
        if obs.shape[0] != self.actor_obs_dim:
            raise RuntimeError(f"Pure MuJoCo observation has dim {obs.shape[0]}, but checkpoint expects {self.actor_obs_dim}.")
        obs_tensor = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(self.policy_device, non_blocking=True).unsqueeze(0)
        with torch.inference_mode():
            action = self.policy.act_inference({"actor": obs_tensor}).squeeze(0).cpu().numpy()
        if self.clip_actions is not None:
            action = np.clip(action, -float(self.clip_actions), float(self.clip_actions))
        self.model.step(action)
        if auto_reset and self.model.is_fallen():
            self.model.reset()

    def run(self, args) -> None:
        controller = None
        if args.keyboard_control:
            controller = MujocoArrowCommandController(args.keyboard_linear_speed, args.keyboard_yaw_speed)
            polling = "held-key polling active" if controller.has_held_key_polling else "native key repeats only"
            print(f"[INFO] MuJoCo arrow control: up/down forward/back, left/right yaw, X stop, R reset ({polling}).")
        viewer = args.viewer
        if viewer == "auto":
            viewer = "native" if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") else "none"
        (self._run_native if viewer == "native" else self._run_headless)(args, controller)
        if controller is not None:
            controller.close()

    def _run_native(self, args, controller: MujocoArrowCommandController | None) -> None:
        import mujoco.viewer

        key_callback = controller.handle_mujoco_key if controller is not None else None
        step = 0
        with mujoco.viewer.launch_passive(self.model.model, self.model.data, key_callback=key_callback) as handle:
            with handle.lock():
                self.model.update_camera(handle.cam)
            while handle.is_running() and (args.num_steps is None or step < args.num_steps):
                start_time = time.time()
                command = np.zeros(3, dtype=np.float32) if controller is None else controller.command()
                with handle.lock():
                    if controller is not None and controller.consume_reset_requested():
                        self.model.reset()
                    self._step_once(command, auto_reset=not args.no_terminations)
                    self.model.set_active_color(bool(np.any(np.abs(command) > 1.0e-6)))
                    if args.follow_camera:
                        self.model.update_camera(handle.cam)
                    draw_velocity_arrows(handle.user_scn, self.model.model, self.model.data, command)
                handle.sync()
                step += 1
                time.sleep(max(self.model.step_dt - (time.time() - start_time), 0.0))

    def _run_headless(self, args, controller: MujocoArrowCommandController | None) -> None:
        step = 0
        while args.num_steps is None or step < args.num_steps:
            start_time = time.time()
            if controller is not None and controller.consume_reset_requested():
                self.model.reset()
            command = np.zeros(3, dtype=np.float32) if controller is None else controller.command()
            self._step_once(command, auto_reset=not args.no_terminations)
            self.model.set_active_color(bool(np.any(np.abs(command) > 1.0e-6)))
            step += 1
            if args.real_time:
                time.sleep(max(self.model.step_dt - (time.time() - start_time), 0.0))


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value.lower() in {"1", "true", "yes", "y", "on"}:
        return True
    if value.lower() in {"0", "false", "no", "n", "off"}:
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
        parsed = [parsed]
    elif isinstance(parsed, str):
        parsed = [part for part in parsed.split(",") if part.strip()]
    try:
        return [int(gpu_id) for gpu_id in parsed]
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(f"invalid GPU ids: {value}") from exc


def _resolve_device(device: str | None, gpu_ids: list[int] | str | None) -> str:
    if device is not None:
        return device
    if gpu_ids is None:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    selected_gpus, num_gpus = select_gpus(gpu_ids)
    if selected_gpus is None or num_gpus == 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
    return "cuda:0"


def _resolve_task(task: str | None, mode: str | None):
    if task is None and mode is None:
        raise SystemExit("A task id is required, or use --task go1 --mode pretrain")
    if task is None or task in ROBOTS:
        robot, backend_cls = ROBOTS[task or "go1"]
        return robot, backend_cls, robot.mode_tasks[mode or "pretrain"]
    for robot, backend_cls in ROBOTS.values():
        if task in robot.mode_tasks.values():
            return robot, backend_cls, task
    raise SystemExit(f"Pure MuJoCo play supports the {', '.join(ROBOTS)} tasks only.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Play an RSL-RL checkpoint in direct MuJoCo.")
    parser.add_argument("task_pos", nargs="?", help="Task id, or robot alias.")
    parser.add_argument("--task", dest="task_opt", help="Task id, or robot alias.")
    parser.add_argument("--mode", choices=MODES, type=str.lower, help="Mode alias.")
    parser.add_argument("--checkpoint-file", "--checkpoint_file", "--checkpoint")
    parser.add_argument("--device", default=None)
    parser.add_argument("--gpu-ids", "--gpu_ids", type=_parse_gpu_ids, default=None)
    parser.add_argument("--real-time", "--real_time", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--viewer", choices=("none", "auto", "native"), default="none")
    parser.add_argument("--follow-camera", "--follow_camera", action="store_true")
    parser.add_argument("--no-terminations", "--no_terminations", nargs="?", const=True, default=False, type=_parse_bool)
    parser.add_argument("--num-steps", "--num_steps", type=int, default=None)
    parser.add_argument("--keyboard-control", dest="keyboard_control", action="store_true", default=True)
    parser.add_argument("--no-keyboard-control", dest="keyboard_control", action="store_false")
    parser.add_argument("--keyboard-linear-speed", type=float, default=0.8)
    parser.add_argument("--keyboard-yaw-speed", type=float, default=0.6)
    args = parser.parse_args()

    device = _resolve_device(args.device, args.gpu_ids)
    configure_torch_backends()
    robot, backend_cls, task = _resolve_task(args.task_opt or args.task_pos, args.mode)
    checkpoint_file = latest_pretrain_checkpoint(robot.experiment) if args.checkpoint_file is None else Path(args.checkpoint_file).expanduser()
    if checkpoint_file is None or not checkpoint_file.exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint_file or f'logs/rsl_rl/{robot.experiment}/*_pretrain/model_*.pt'}")
    print(f"[INFO] Loading checkpoint: {checkpoint_file}")

    env_cfg, agent_cfg = load_env_cfg(task, play=True), load_rl_cfg(task)
    if args.no_terminations:
        env_cfg.terminations = {}
    PureMujocoPlayer(backend_cls, env_cfg, agent_cfg, checkpoint_file, device=device).run(args)


if __name__ == "__main__":
    sys.exit(main())
