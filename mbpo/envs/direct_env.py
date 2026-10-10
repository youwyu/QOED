"""Direct MuJoCo VecEnv base for RSL-RL MBPO finetuning on a single robot."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import queue
import tempfile
import time
from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from .mujoco_backend import draw_velocity_arrows
from .utils import ACTIVE_RGBA, mujoco_geom_ids_with_prefix, save_mujoco_model_defaults, set_follow_camera


_VIEWER_CLOSE = "__close__"


def _put_latest(state_queue, message) -> None:
    try:
        state_queue.put_nowait(message)
    except queue.Full:
        with contextlib.suppress(queue.Empty):
            state_queue.get_nowait()
        with contextlib.suppress(queue.Full):
            state_queue.put_nowait(message)


def _mujoco_training_viewer_worker(model_path: str, state_queue, follow_camera: bool, camera_distance: float) -> None:
    import mujoco.viewer

    model = mujoco.MjModel.from_binary_path(model_path)
    data = mujoco.MjData(model)
    robot_geom_ids = mujoco_geom_ids_with_prefix(model, "robot/")
    default_rgba = model.geom_rgba[robot_geom_ids].copy()

    def latest_message():
        try:
            message = state_queue.get(timeout=1.0 / 60.0)
        except queue.Empty:
            return None
        while True:
            try:
                message = state_queue.get_nowait()
            except queue.Empty:
                return message

    with mujoco.viewer.launch_passive(model, data) as handle:
        with handle.lock():
            set_follow_camera(handle.cam, data.qpos, camera_distance)
        while handle.is_running():
            message = latest_message()
            if message == _VIEWER_CLOSE:
                break
            if message is not None:
                qpos, qvel, ctrl, command = message
                active = bool(np.any(np.abs(command) > 1.0e-6))
                with handle.lock():
                    data.qpos[:] = qpos
                    data.qvel[:] = qvel
                    data.ctrl[:] = ctrl
                    mujoco.mj_forward(model, data)
                    if robot_geom_ids:
                        model.geom_rgba[robot_geom_ids] = ACTIVE_RGBA if active else default_rgba
                    if follow_camera:
                        set_follow_camera(handle.cam, data.qpos, camera_distance)
                    draw_velocity_arrows(handle.user_scn, model, data, command)
            handle.sync()


@dataclass
class _CommandRanges:
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)


class DirectMujocoVecEnv(VecEnv):
    """Single-environment velocity task running on native MuJoCo.

    The MuJoCo model/data stay on CPU. RSL-RL moves observations/rewards to the
    policy device after each step, matching the direct MuJoCo play path.
    Subclasses provide the robot backend, rewards, observations and domain randomization.
    """

    is_vector_env = True
    backend_cls: type

    def __init__(
        self,
        env_cfg,
        agent_cfg=None,
        seed: int = 42,
        viewer: str = "none",
        follow_camera: bool = False,
    ) -> None:
        self.cfg = env_cfg
        self.agent_cfg = agent_cfg
        self.backend = self.backend_cls(env_cfg)
        self.model = self.backend.model
        self.data = self.backend.data
        self.num_envs = 1
        self.num_actions = self.backend.action_dim
        self.device = torch.device("cpu")
        self.physics_dt = float(self.model.opt.timestep)
        self.step_dt = self.backend.step_dt
        self.max_episode_length = max(
            1,
            int(round(float(getattr(env_cfg, "episode_length_s", self.step_dt)) / self.step_dt)),
        )
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.common_step_counter = 0
        self.extras: dict = {}

        self.rng = np.random.default_rng(seed)
        self.command = np.zeros(3, dtype=np.float32)
        self._command_resample_interval = self.max_episode_length
        self._command_ranges = self._read_command_ranges(env_cfg)
        self._standing_probability = self._read_standing_probability(env_cfg)

        self._episode_reward_sum = 0.0
        self.cumulative_reward = 0.0
        self._episode_term_sums: dict[str, float] = {}
        self._episode_metric_sums: dict[str, float] = {}
        self._episode_metric_count = 0

        self._setup()
        self._model_defaults = save_mujoco_model_defaults(self.model)
        self._refresh_domain_randomization_cache()
        if os.environ.get("QOED_DOMAIN_RANDOMIZATION") == "1":
            self._apply_domain_randomization()

        self._viewer_mode = viewer
        self._follow_camera = follow_camera
        self._viewer_process = None
        self._viewer_queue = None
        self._viewer_model_path = None
        self._viewer_step_period = 1.0 / 60.0 if viewer == "native" else 0.0
        self._last_viewer_step_time = 0.0
        self.reset()
        self._open_viewer()

    def _setup(self) -> None:
        pass

    def domain_randomization_vector(self) -> torch.Tensor:
        raise NotImplementedError

    def _refresh_domain_randomization_cache(self) -> None:
        raise NotImplementedError

    def _apply_domain_randomization(self) -> None:
        raise NotImplementedError

    def _obs_dict(self, system_termination: bool) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _compute_reward(self, previous_action: np.ndarray, action: np.ndarray) -> tuple[float, dict[str, float], dict[str, float]]:
        raise NotImplementedError

    def _imagination_obs_from_state(self, state: torch.Tensor, action: torch.Tensor, termination: torch.Tensor | None = None) -> TensorDict:
        raise NotImplementedError

    def _parse_imagination_termination(self, terminations: torch.Tensor | None, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def _compute_imagination_reward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        raise NotImplementedError

    def _after_reset(self) -> None:
        pass

    def _after_physics_step(self) -> None:
        pass

    def _init_imagination_buffers(self, num_envs: int, device: torch.device) -> None:
        pass

    def _reset_imagination_buffers(self, env_ids: torch.Tensor) -> None:
        pass

    def _parse_imagination_contact(self, contacts: torch.Tensor | None) -> None:
        pass

    @property
    def unwrapped(self):
        return self

    def seed(self, seed: int = -1) -> int:
        if seed == -1:
            seed = int(np.random.randint(0, 10_000))
        self.rng = np.random.default_rng(seed)
        return seed

    def reset(self) -> tuple[TensorDict, dict]:
        self._reset_robot()
        return self.get_observations(), {}

    def get_observations(self) -> TensorDict:
        obs = self._obs_dict(system_termination=False)
        return TensorDict(obs, batch_size=[self.num_envs])

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict]:
        action = actions.detach().cpu().numpy().reshape(self.num_actions).astype(np.float32)
        clip_actions = getattr(self.agent_cfg, "clip_actions", None)
        if clip_actions is not None:
            action = np.clip(action, -float(clip_actions), float(clip_actions))

        previous_action = self.backend.last_action.copy()
        self._maybe_resample_command()
        self.backend.step(action)
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._after_physics_step()

        reward, reward_terms, metrics = self._compute_reward(previous_action, action)
        fell_over = self.backend.is_fallen()
        horizon_reached = bool(self.episode_length_buf[0].item() >= self.max_episode_length)
        done = fell_over

        self._episode_reward_sum += reward
        self.cumulative_reward += reward
        for name, value in reward_terms.items():
            self._episode_term_sums[name] = self._episode_term_sums.get(name, 0.0) + value
        for name, value in metrics.items():
            self._episode_metric_sums[name] = self._episode_metric_sums.get(name, 0.0) + value
        self._episode_metric_count += 1

        extras = {"time_outs": torch.tensor([False], dtype=torch.bool, device=self.device)}
        if done:
            extras["episode"] = self._episode_log(fell_over=fell_over, time_out=False)
            self._reset_robot()
        elif horizon_reached:
            extras["episode"] = self._episode_log(fell_over=False, time_out=True)
            self._reset_episode_accounting()

        obs = self._obs_dict(system_termination=done)
        rewards = torch.tensor([reward], dtype=torch.float32, device=self.device)
        dones = torch.tensor([int(done)], dtype=torch.long, device=self.device)
        self._publish_viewer_state()
        self._throttle_viewer_step()
        return TensorDict(obs, batch_size=[self.num_envs]), rewards, dones, extras

    def close(self) -> None:
        process = self._viewer_process
        state_queue = self._viewer_queue
        model_path = self._viewer_model_path
        self._viewer_process = None
        self._viewer_queue = None
        self._viewer_model_path = None

        if state_queue is not None:
            _put_latest(state_queue, _VIEWER_CLOSE)
        if process is not None:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if model_path is not None:
            os.unlink(model_path)
        if state_queue is not None:
            state_queue.close()

    def prepare_imagination(self):
        num_envs = int(getattr(self, "num_imagination_envs", 0))
        if num_envs <= 0 or getattr(self, "system_dynamics", None) is None:
            return None

        device = torch.device(getattr(self.system_dynamics, "device", "cpu"))
        self._imagination_device = device
        self.system_dynamics.reset()
        self._imagination_model_ids = torch.randint(
            0,
            self.system_dynamics.ensemble_size,
            (1, num_envs, 1),
            device=device,
        )
        self._imagination_episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)
        self._imagination_command = torch.zeros(num_envs, 3, device=device)
        self._imagination_command_intervals = torch.ones(num_envs, device=device, dtype=torch.long)
        self._imagination_last_action = torch.zeros(num_envs, self.num_actions, device=device)
        self._init_imagination_buffers(num_envs, device)
        self._imagination_reward_sum = torch.zeros(num_envs, device=device)
        self._imagination_term_sums: dict[str, torch.Tensor] = {}
        self._imagination_metric_sums: dict[str, torch.Tensor] = {}
        self._imagination_metric_count = torch.zeros(num_envs, device=device)
        self._reset_imagination_indices(torch.arange(num_envs, device=device))
        return None

    def get_imagination_observation(self, state_history, action_history):
        self._ensure_imagination_ready()
        state = self.imagination_state_normalizer.inverse(state_history[:, -1])
        action = self.imagination_action_normalizer.inverse(action_history[:, -1])
        self._imagination_last_action = action.detach()
        return self._imagination_obs_from_state(state, action)

    def imagination_step(self, rollout_action, state_history, action_history):
        self._ensure_imagination_ready()
        rollout_action = rollout_action.to(state_history.device)
        rollout_action_normalized = self.imagination_action_normalizer(rollout_action)
        action_history = torch.cat([action_history[:, 1:], rollout_action_normalized.unsqueeze(1)], dim=1)

        (
            next_state,
            _aleatoric_uncertainty,
            epistemic_uncertainty,
            _extensions,
            contacts,
            terminations,
        ) = self.system_dynamics.forward(state_history, action_history, self._imagination_model_ids)
        next_state_denormalized = self.imagination_state_normalizer.inverse(next_state)
        self._parse_imagination_contact(contacts)
        termination = self._parse_imagination_termination(terminations, next_state_denormalized)

        rewards, reward_terms, metrics = self._compute_imagination_reward(next_state_denormalized, rollout_action)
        self._imagination_last_action = rollout_action.detach()
        self._imagination_episode_length_buf += 1
        self.common_step_counter += 1
        self._maybe_resample_imagination_commands()

        time_outs = self._imagination_episode_length_buf >= int(getattr(self, "max_imagination_episode_length", 1))
        dones_bool = termination | time_outs
        dones = dones_bool.to(dtype=torch.long)
        self._accumulate_imagination_logs(rewards, reward_terms, metrics)

        extras = {"time_outs": time_outs}
        reset_ids = dones_bool.nonzero(as_tuple=False).flatten()
        if reset_ids.numel() > 0:
            extras["log"] = self._imagination_episode_log(reset_ids, termination, time_outs)
            self._reset_imagination_indices(reset_ids)

        obs = self._imagination_obs_from_state(next_state_denormalized, rollout_action, termination)
        state_history = torch.cat([state_history[:, 1:], next_state.unsqueeze(1)], dim=1)
        return obs, rewards, dones, extras, state_history, action_history, epistemic_uncertainty

    def _ensure_imagination_ready(self) -> None:
        if not hasattr(self, "_imagination_command"):
            self.prepare_imagination()

    def _reset_imagination_indices(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        env_ids = env_ids.to(self._imagination_device, dtype=torch.long)
        if hasattr(self.system_dynamics, "reset_partial"):
            self.system_dynamics.reset_partial(env_ids)
        if hasattr(self, "_imagination_model_ids"):
            self._imagination_model_ids[:, env_ids, :] = torch.randint(
                0,
                self.system_dynamics.ensemble_size,
                (1, env_ids.numel(), 1),
                device=self._imagination_device,
            )
        self._imagination_episode_length_buf[env_ids] = 0
        self._imagination_last_action[env_ids] = 0.0
        self._reset_imagination_buffers(env_ids)
        self._imagination_reward_sum[env_ids] = 0.0
        self._imagination_metric_count[env_ids] = 0.0
        for values in self._imagination_term_sums.values():
            values[env_ids] = 0.0
        for values in self._imagination_metric_sums.values():
            values[env_ids] = 0.0
        self._sample_imagination_commands(env_ids)

    def _sample_imagination_commands(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = env_ids.numel()
        device = self._imagination_device
        command = torch.empty(n, 3, device=device)
        command[:, 0].uniform_(*self._command_ranges.lin_vel_x)
        command[:, 1].uniform_(*self._command_ranges.lin_vel_y)
        command[:, 2].uniform_(*self._command_ranges.ang_vel_z)
        standing = torch.rand(n, device=device) < self._standing_probability
        command[standing] = 0.0
        self._imagination_command[env_ids] = command

        interval_range = getattr(self, "imagination_command_resample_interval_range", None)
        if interval_range is None:
            self._imagination_command_intervals[env_ids] = max(
                1,
                int(getattr(self, "max_imagination_episode_length", 1)),
            )
        else:
            low, high = int(interval_range[0]), int(interval_range[1])
            high = max(high, low + 1)
            self._imagination_command_intervals[env_ids] = torch.randint(low, high, (n,), device=device)

    def _maybe_resample_imagination_commands(self) -> None:
        due = self._imagination_episode_length_buf % self._imagination_command_intervals == 0
        self._sample_imagination_commands(due.nonzero(as_tuple=False).flatten())

    def _accumulate_imagination_logs(
        self,
        rewards: torch.Tensor,
        reward_terms: dict[str, torch.Tensor],
        metrics: dict[str, torch.Tensor],
    ) -> None:
        self._imagination_reward_sum += rewards
        self._imagination_metric_count += 1
        for name, value in reward_terms.items():
            if name not in self._imagination_term_sums:
                self._imagination_term_sums[name] = torch.zeros_like(rewards)
            self._imagination_term_sums[name] += value
        for name, value in metrics.items():
            if name not in self._imagination_metric_sums:
                self._imagination_metric_sums[name] = torch.zeros_like(rewards)
            self._imagination_metric_sums[name] += value

    def _imagination_episode_log(
        self,
        env_ids: torch.Tensor,
        termination: torch.Tensor,
        time_outs: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        denom = torch.clamp(self._imagination_episode_length_buf[env_ids].float() * self.step_dt, min=self.step_dt)
        log = {
            "reward": self._imagination_reward_sum[env_ids].mean().view(1),
            "length": self._imagination_episode_length_buf[env_ids].float().mean().view(1),
            "Episode_Termination/fell_over": termination[env_ids].float().mean().view(1),
            "Episode_Termination/time_out": time_outs[env_ids].float().mean().view(1),
        }
        for name, value in self._imagination_term_sums.items():
            log[f"Episode_Reward/{name}"] = (value[env_ids] / denom).mean().view(1)
        metric_denom = torch.clamp(self._imagination_metric_count[env_ids], min=1.0)
        for name, value in self._imagination_metric_sums.items():
            log[name] = (value[env_ids] / metric_denom).mean().view(1)
        return log

    def _reset_robot(self) -> None:
        self.backend.reset()
        self._after_reset()
        self._reset_episode_accounting()

    def _reset_episode_accounting(self) -> None:
        self.episode_length_buf[:] = 0
        self._episode_reward_sum = 0.0
        self._episode_term_sums.clear()
        self._episode_metric_sums.clear()
        self._episode_metric_count = 0
        self._sample_command(force=True)

    def _open_viewer(self) -> None:
        if self._viewer_mode != "native":
            return
        with tempfile.NamedTemporaryFile(suffix=".mjb", delete=False) as model_file:
            self._viewer_model_path = model_file.name
        mujoco.mj_saveModel(self.model, self._viewer_model_path)
        context = mp.get_context("fork")
        self._viewer_queue = context.Queue(maxsize=2)
        self._viewer_process = context.Process(
            target=_mujoco_training_viewer_worker,
            args=(self._viewer_model_path, self._viewer_queue, self._follow_camera, self.backend.camera_distance),
            daemon=True,
        )
        self._viewer_process.start()
        self._publish_viewer_state()

    def _publish_viewer_state(self) -> None:
        if self._viewer_process is None or self._viewer_queue is None:
            return
        if not self._viewer_process.is_alive():
            self.close()
            return
        message = (self.data.qpos.copy(), self.data.qvel.copy(), self.data.ctrl.copy(), self.command.copy())
        _put_latest(self._viewer_queue, message)

    def _throttle_viewer_step(self) -> None:
        if self._viewer_step_period <= 0.0:
            return
        if self._viewer_process is None or not self._viewer_process.is_alive():
            return

        now = time.monotonic()
        if self._last_viewer_step_time > 0.0:
            sleep_time = self._last_viewer_step_time + self._viewer_step_period - now
            if sleep_time > 0.0:
                time.sleep(sleep_time)
        self._last_viewer_step_time = time.monotonic()

    def _episode_log(self, fell_over: bool, time_out: bool) -> dict[str, torch.Tensor]:
        denom = max(float(self.episode_length_buf[0].item()) * self.step_dt, self.step_dt)
        episode = {
            "reward": torch.tensor([self._episode_reward_sum], dtype=torch.float32),
            "length": self.episode_length_buf.float().clone(),
            "Episode_Termination/fell_over": torch.tensor([float(fell_over)], dtype=torch.float32),
            "Episode_Termination/time_out": torch.tensor([float(time_out)], dtype=torch.float32),
        }
        for name, value in self._episode_term_sums.items():
            episode[f"Episode_Reward/{name}"] = torch.tensor([value / denom], dtype=torch.float32)
        metric_denom = max(self._episode_metric_count, 1)
        for name, value in self._episode_metric_sums.items():
            episode[name] = torch.tensor([value / metric_denom], dtype=torch.float32)
        return episode

    def _maybe_resample_command(self) -> None:
        if int(self.episode_length_buf[0].item()) % self._command_resample_interval == 0:
            self._sample_command()

    def _sample_command(self, force: bool = False) -> None:
        command_cfg = getattr(self.cfg, "commands", {}).get("twist")
        if command_cfg is None:
            return
        if (not force) and self.rng.random() < self._standing_probability:
            self.command[:] = 0.0
        else:
            self.command[:] = (
                self.rng.uniform(*self._command_ranges.lin_vel_x),
                self.rng.uniform(*self._command_ranges.lin_vel_y),
                self.rng.uniform(*self._command_ranges.ang_vel_z),
            )
        seconds_range = command_cfg.resampling_time_range
        seconds = float(self.rng.uniform(seconds_range[0], seconds_range[1]))
        self._command_resample_interval = max(1, int(round(seconds / self.step_dt)))

    def _read_command_ranges(self, env_cfg) -> _CommandRanges:
        command_cfg = getattr(env_cfg, "commands", {}).get("twist")
        ranges = getattr(command_cfg, "ranges", None)
        if ranges is None:
            return _CommandRanges()
        return _CommandRanges(
            lin_vel_x=tuple(ranges.lin_vel_x),
            lin_vel_y=tuple(ranges.lin_vel_y),
            ang_vel_z=tuple(ranges.ang_vel_z),
        )

    def _read_standing_probability(self, env_cfg) -> float:
        command_cfg = getattr(env_cfg, "commands", {}).get("twist")
        return float(getattr(command_cfg, "rel_standing_envs", 0.1))
