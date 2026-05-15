"""Direct MuJoCo Go1 VecEnv for RSL-RL training."""

from __future__ import annotations

import math
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

from .mujoco_go1_backend import PureMujocoGo1Model, draw_go1_velocity_arrows
from .utils import (
    ACTIVE_GO1_RGBA,
    GO1_FEET_NAMES,
    apply_go1_domain_randomization,
    mujoco_body_ids_with_prefix,
    mujoco_geom_ids_with_prefix,
    mujoco_named_id,
    save_mujoco_model_defaults,
    set_go1_camera,
)


_VIEWER_CLOSE = "__close__"


def _mujoco_training_viewer_worker(
    model_path: str,
    state_queue,
    follow_camera: bool,
) -> None:
    import queue as queue_module

    import mujoco.viewer

    model = mujoco.MjModel.from_binary_path(model_path)
    data = mujoco.MjData(model)
    robot_geom_ids = mujoco_geom_ids_with_prefix(model, "robot/")
    default_rgba = model.geom_rgba[robot_geom_ids].copy()

    def latest_message():
        try:
            message = state_queue.get(timeout=1.0 / 60.0)
        except queue_module.Empty:
            return None
        while True:
            try:
                message = state_queue.get_nowait()
            except queue_module.Empty:
                return message

    with mujoco.viewer.launch_passive(model, data) as handle:
        with handle.lock():
            set_go1_camera(handle.cam, data.qpos)
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
                        model.geom_rgba[robot_geom_ids] = ACTIVE_GO1_RGBA if active else default_rgba
                    if follow_camera:
                        set_go1_camera(handle.cam, data.qpos)
                    draw_go1_velocity_arrows(handle.user_scn, model, data, command)
            handle.sync()


@dataclass
class _CommandRanges:
    lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: tuple[float, float] = (-1.0, 1.0)
    ang_vel_z: tuple[float, float] = (-0.5, 0.5)


class DirectMujocoGo1VecEnv(VecEnv):
    """Single-environment Go1 velocity task running on native MuJoCo.

    The MuJoCo model/data stay on CPU. RSL-RL moves observations/rewards to the
    policy device after each step, matching the direct MuJoCo play path.
    """

    is_vector_env = True

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
        self.mujoco_go1 = PureMujocoGo1Model(env_cfg)
        self.model = self.mujoco_go1.model
        self.data = self.mujoco_go1.data
        self.num_envs = 1
        self.num_actions = self.mujoco_go1.action_dim
        self.device = torch.device("cpu")
        self.physics_dt = float(self.model.opt.timestep)
        self.step_dt = self.mujoco_go1.step_dt
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

        self.foot_names = GO1_FEET_NAMES
        self.foot_geom_ids = np.asarray(
            [
                mujoco_named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{name}_foot_collision")
                for name in self.foot_names
            ],
            dtype=np.int32,
        )
        self.foot_site_ids = np.asarray(
            [mujoco_named_id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{name}") for name in self.foot_names],
            dtype=np.int32,
        )
        self.terrain_geom_id = mujoco_named_id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        self.prev_contact = np.zeros(4, dtype=bool)
        self.contact = np.zeros(4, dtype=bool)
        self.contact_force = np.zeros((4, 3), dtype=np.float32)
        self.foot_air_time = np.zeros(4, dtype=np.float32)
        self.peak_foot_height = np.zeros(4, dtype=np.float32)
        self.prev_foot_pos = np.zeros((4, 3), dtype=np.float32)
        self.foot_vel = np.zeros((4, 3), dtype=np.float32)

        self._ctrl_to_action = self.mujoco_go1.ctrl_action_indices
        self._episode_reward_sum = 0.0
        self.cumulative_reward = 0.0
        self._episode_term_sums: dict[str, float] = {}
        self._episode_metric_sums: dict[str, float] = {}
        self._episode_metric_count = 0

        self._joint_limits = self._read_joint_limits()
        self._robot_body_ids = mujoco_body_ids_with_prefix(self.model, "robot/")
        self._robot_dof_ids = self.mujoco_go1.obs_dof_adrs
        self._robot_qpos_ids = self.mujoco_go1.obs_qpos_adrs
        self._torso_body_id = mujoco_named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk")
        self._domain_randomization_enabled = os.environ.get("QOED_GO1_DOMAIN_RANDOMIZATION") == "1"
        self._model_defaults = save_mujoco_model_defaults(self.model)
        self._refresh_domain_randomization_cache()

        self._viewer_mode = viewer
        self._follow_camera = follow_camera
        self._viewer_process = None
        self._viewer_queue = None
        self._viewer_model_path = None
        self._viewer_step_period = 1.0 / 60.0 if viewer == "native" else 0.0
        self._last_viewer_step_time = 0.0
        self.reset()
        self._open_viewer()

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

        previous_action = self.mujoco_go1.last_action.copy()
        self._maybe_resample_command()
        self.mujoco_go1.step(action)
        self.common_step_counter += 1
        self.episode_length_buf += 1
        self._update_feet()

        reward, reward_terms, metrics = self._compute_reward(previous_action, action)
        fell_over = self.mujoco_go1.is_fallen()
        horizon_reached = bool(self.episode_length_buf[0].item() >= self.max_episode_length)
        done = fell_over

        self._episode_reward_sum += reward
        self.cumulative_reward += reward
        for name, value in reward_terms.items():
            self._episode_term_sums[name] = self._episode_term_sums.get(name, 0.0) + value
        for name, value in metrics.items():
            self._episode_metric_sums[name] = self._episode_metric_sums.get(name, 0.0) + value
        self._episode_metric_count += 1

        extras = {
            "time_outs": torch.tensor([False], dtype=torch.bool, device=self.device),
            "domain_randomization": self.domain_randomization_vector(),
        }
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
            try:
                state_queue.put_nowait(_VIEWER_CLOSE)
            except queue.Full:
                try:
                    state_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    state_queue.put_nowait(_VIEWER_CLOSE)
                except queue.Full:
                    pass
        if process is not None:
            process.join(timeout=1.0)
            if process.is_alive():
                process.terminate()
                process.join(timeout=1.0)
        if model_path is not None:
            try:
                os.unlink(model_path)
            except FileNotFoundError:
                pass

        if state_queue is None:
            return
        try:
            state_queue.close()
        except Exception as exc:
            print(f"[WARN] Failed to close MuJoCo training viewer queue cleanly: {exc}")

    def domain_randomization_vector(self) -> torch.Tensor:
        vector = np.concatenate(
            (
                np.asarray([self._floor_friction], dtype=np.float32),
                self._body_mass,
                self._dof_frictionloss,
                self._dof_armature,
            ),
            axis=0,
        )
        return torch.from_numpy(vector).view(1, -1)

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
        self._imagination_air_time = torch.zeros(num_envs, 4, device=device)
        self._imagination_contact = torch.zeros(num_envs, 4, device=device, dtype=torch.bool)
        self._imagination_reward_sum = torch.zeros(num_envs, device=device)
        self._imagination_term_sums: dict[str, torch.Tensor] = {}
        self._imagination_metric_sums: dict[str, torch.Tensor] = {}
        self._imagination_metric_count = torch.zeros(num_envs, device=device)
        self._reset_imagination_indices(torch.arange(num_envs, device=device), log=False)
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
        contact = self._parse_imagination_contact(contacts)
        termination = self._parse_imagination_termination(terminations, next_state_denormalized)

        rewards, reward_terms, metrics = self._compute_imagination_reward(
            next_state_denormalized,
            rollout_action,
            contact,
        )
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
            self._reset_imagination_indices(reset_ids, log=False)

        obs = self._imagination_obs_from_state(next_state_denormalized, rollout_action, termination)
        state_history = torch.cat([state_history[:, 1:], next_state.unsqueeze(1)], dim=1)
        return obs, rewards, dones, extras, state_history, action_history, epistemic_uncertainty

    def _ensure_imagination_ready(self) -> None:
        if not hasattr(self, "_imagination_command"):
            self.prepare_imagination()

    def _reset_imagination_indices(self, env_ids: torch.Tensor, log: bool = True) -> None:
        del log
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
        self._imagination_air_time[env_ids] = 0.0
        self._imagination_contact[env_ids] = False
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

    def _imagination_obs_from_state(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        termination: torch.Tensor | None = None,
    ) -> TensorDict:
        num_envs = state.shape[0]
        device = state.device
        actor = torch.cat(
            (
                state[:, 0:3],
                state[:, 3:6],
                state[:, 6:9],
                state[:, 9:21],
                state[:, 21:33],
                action,
                self._imagination_command.to(device=device, dtype=state.dtype),
            ),
            dim=-1,
        )
        foot_contact = self._imagination_contact.to(device=device, dtype=state.dtype)
        foot_height = torch.zeros(num_envs, 4, device=device, dtype=state.dtype)
        foot_forces = torch.zeros(num_envs, 12, device=device, dtype=state.dtype)
        critic = torch.cat(
            (
                actor,
                foot_height,
                self._imagination_air_time.to(device=device, dtype=state.dtype),
                foot_contact,
                foot_forces,
            ),
            dim=-1,
        )
        if termination is None:
            termination_value = torch.zeros(num_envs, 1, device=device, dtype=state.dtype)
        else:
            termination_value = termination.to(device=device).float().view(num_envs, 1)
        obs = {
            "actor": actor,
            "policy": actor,
            "critic": critic,
            "system_state": state,
            "system_action": action,
            "system_contact": foot_contact,
            "system_termination": termination_value,
        }
        return TensorDict(obs, batch_size=[num_envs], device=device)

    def _parse_imagination_contact(self, contacts: torch.Tensor | None) -> torch.Tensor:
        if contacts is None:
            contact = torch.zeros_like(self._imagination_contact)
        elif contacts.shape[-1] >= 8:
            contact = torch.sigmoid(contacts[:, 4:8]).round().bool()
        else:
            contact = torch.sigmoid(contacts[:, :4]).round().bool()
        self._imagination_contact = contact
        self._imagination_air_time = torch.where(
            contact,
            torch.zeros_like(self._imagination_air_time),
            self._imagination_air_time + self.step_dt,
        )
        return contact

    def _parse_imagination_termination(
        self,
        terminations: torch.Tensor | None,
        state: torch.Tensor,
    ) -> torch.Tensor:
        if terminations is None:
            termination = torch.zeros(state.shape[0], device=state.device, dtype=torch.bool)
        else:
            termination = torch.sigmoid(terminations).reshape(state.shape[0], -1).max(dim=1).values > 0.5
        projected_gravity = state[:, 6:9]
        bad_orientation = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)).abs() > math.radians(70.0)
        return termination | bad_orientation

    def _compute_imagination_reward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        contact: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        del contact
        base_lin_vel = state[:, 0:3]
        base_ang_vel = state[:, 3:6]
        projected_gravity = state[:, 6:9]
        joint_pos_rel = state[:, 9:21]
        command = self._imagination_command.to(device=state.device, dtype=state.dtype)
        command_norm = torch.linalg.norm(command[:, :2], dim=1) + command[:, 2].abs()
        active = (command_norm > 0.05).to(state.dtype)

        lin_vel_error = torch.sum(torch.square(command[:, :2] - base_lin_vel[:, :2]), dim=1) + base_lin_vel[:, 2].square()
        ang_vel_error = torch.square(command[:, 2] - base_ang_vel[:, 2]) + torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        track_linear_velocity = torch.exp(-lin_vel_error / (0.5**2))
        track_angular_velocity = torch.exp(-ang_vel_error / (math.sqrt(0.5) ** 2))
        upright = torch.exp(-torch.sum(torch.square(projected_gravity[:, :2]), dim=1) / (math.sqrt(0.2) ** 2))

        pose_std_stand = torch.tensor([0.05, 0.05, 0.1] * 4, device=state.device, dtype=state.dtype)
        pose_std_walk = torch.tensor([0.3, 0.3, 0.6] * 4, device=state.device, dtype=state.dtype)
        pose_std = torch.where((command_norm < 0.05).unsqueeze(1), pose_std_stand, pose_std_walk)
        pose = torch.exp(-torch.mean(torch.square(joint_pos_rel / pose_std), dim=1))

        default_joint_pos = torch.as_tensor(self.mujoco_go1.default_joint_pos, device=state.device, dtype=state.dtype)
        joint_pos = joint_pos_rel + default_joint_pos
        joint_limits = torch.as_tensor(self._joint_limits, device=state.device, dtype=state.dtype)
        dof_pos_limits = torch.sum(
            torch.clamp(joint_limits[:, 0] - joint_pos, min=0.0)
            + torch.clamp(joint_pos - joint_limits[:, 1], min=0.0),
            dim=1,
        )
        previous_action = self._imagination_last_action.to(device=state.device, dtype=state.dtype)
        action_rate_l2 = torch.sum(torch.square(action - previous_action), dim=1)
        air_time = torch.sum(
            ((self._imagination_air_time > 0.05) & (self._imagination_air_time < 0.5)).to(state.dtype),
            dim=1,
        ) * active
        zero = torch.zeros_like(track_linear_velocity)
        raw_terms = {
            "track_linear_velocity": track_linear_velocity,
            "track_angular_velocity": track_angular_velocity,
            "upright": upright,
            "pose": pose,
            "dof_pos_limits": dof_pos_limits,
            "action_rate_l2": action_rate_l2,
            "air_time": air_time,
            "foot_clearance": zero,
            "foot_swing_height": zero,
            "foot_slip": zero,
            "soft_landing": zero,
        }
        weights = {
            "track_linear_velocity": 2.0,
            "track_angular_velocity": 2.0,
            "upright": 1.0,
            "pose": 1.0,
            "dof_pos_limits": -1.0,
            "action_rate_l2": -0.1,
            "air_time": 0.0,
            "foot_clearance": -2.0,
            "foot_swing_height": -0.25,
            "foot_slip": -0.1,
            "soft_landing": -1.0e-5,
        }
        weighted_terms = {
            name: float(weights[name]) * value * self.step_dt
            for name, value in raw_terms.items()
        }
        reward = torch.stack(tuple(weighted_terms.values()), dim=0).sum(dim=0)
        metrics = {
            "Metrics/twist/error_vel_xy": torch.linalg.norm(command[:, :2] - base_lin_vel[:, :2], dim=1),
            "Metrics/twist/error_vel_yaw": (command[:, 2] - base_ang_vel[:, 2]).abs(),
            "Episode_Metrics/mean_action_acc": action_rate_l2,
            "Metrics/peak_height_mean": zero,
            "Metrics/slip_velocity_mean": zero,
            "Metrics/landing_force_mean": zero,
        }
        return reward, weighted_terms, metrics

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
        if self._domain_randomization_enabled:
            self._apply_domain_randomization()
        self.mujoco_go1.reset()
        self.prev_contact[:] = False
        self.contact[:] = False
        self.contact_force[:] = 0.0
        self.foot_air_time[:] = 0.0
        self.peak_foot_height[:] = 0.0
        self.prev_foot_pos[:] = self.data.site_xpos[self.foot_site_ids].astype(np.float32)
        self.foot_vel[:] = 0.0
        self._reset_episode_accounting()

    def _reset_episode_accounting(self) -> None:
        self.episode_length_buf[:] = 0
        self._episode_reward_sum = 0.0
        self._episode_term_sums.clear()
        self._episode_metric_sums.clear()
        self._episode_metric_count = 0
        self._sample_command(force=True)

    def _refresh_domain_randomization_cache(self) -> None:
        self._floor_friction = float(self.model.geom_friction[self.terrain_geom_id, 0])
        self._body_mass = self.model.body_mass[self._robot_body_ids].astype(np.float32).copy()
        self._dof_frictionloss = self.model.dof_frictionloss[self._robot_dof_ids].astype(np.float32).copy()
        self._dof_armature = self.model.dof_armature[self._robot_dof_ids].astype(np.float32).copy()

    def _apply_domain_randomization(self) -> None:
        apply_go1_domain_randomization(
            self.model,
            self.data,
            self.rng,
            self._model_defaults,
            terrain_geom_id=self.terrain_geom_id,
            robot_dof_ids=self._robot_dof_ids,
            torso_body_id=self._torso_body_id,
            body_mass_ids=self._robot_body_ids,
            robot_qpos_ids=self._robot_qpos_ids,
            dtype=np.float32,
        )
        self._refresh_domain_randomization_cache()

    def _open_viewer(self) -> None:
        if self._viewer_mode != "native":
            return
        try:
            with tempfile.NamedTemporaryFile(suffix=".mjb", delete=False) as model_file:
                self._viewer_model_path = model_file.name
            mujoco.mj_saveModel(self.model, self._viewer_model_path)
            context = mp.get_context("fork")
            self._viewer_queue = context.Queue(maxsize=2)
            self._viewer_process = context.Process(
                target=_mujoco_training_viewer_worker,
                args=(self._viewer_model_path, self._viewer_queue, self._follow_camera),
                daemon=True,
            )
            self._viewer_process.start()
            self._publish_viewer_state()
        except Exception as exc:
            self.close()
            print(f"[WARN] Failed to launch MuJoCo training viewer; continuing headless: {exc}")

    def _publish_viewer_state(self) -> None:
        if self._viewer_process is None or self._viewer_queue is None:
            return
        if not self._viewer_process.is_alive():
            self.close()
            return
        message = (
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.ctrl.copy(),
            self.command.copy(),
        )
        try:
            self._viewer_queue.put_nowait(message)
        except queue.Full:
            try:
                self._viewer_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._viewer_queue.put_nowait(message)
            except queue.Full:
                pass

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

    def _obs_dict(self, system_termination: bool) -> dict[str, torch.Tensor]:
        actor = self.mujoco_go1.observation(self.command)
        foot_height = self.data.site_xpos[self.foot_site_ids, 2].astype(np.float32)
        foot_contact = self.contact.astype(np.float32)
        foot_contact_forces = np.sign(self.contact_force) * np.log1p(np.abs(self.contact_force))
        critic = np.concatenate(
            (
                actor,
                foot_height,
                self.foot_air_time.astype(np.float32),
                foot_contact,
                foot_contact_forces.reshape(-1).astype(np.float32),
            ),
            axis=0,
        )
        system_state = np.concatenate(
            (
                actor[:9],
                actor[9:21],
                actor[21:33],
                self._actuator_force_action_order(),
            ),
            axis=0,
        ).astype(np.float32)
        obs = {
            "actor": torch.from_numpy(actor).view(1, -1),
            "policy": torch.from_numpy(actor.copy()).view(1, -1),
            "critic": torch.from_numpy(critic).view(1, -1),
            "system_state": torch.from_numpy(system_state).view(1, -1),
            "system_action": torch.from_numpy(self.mujoco_go1.last_action.copy()).view(1, -1),
            "system_contact": torch.from_numpy(foot_contact).view(1, -1),
            "system_termination": torch.tensor(
                [[float(system_termination)]], dtype=torch.float32, device=self.device
            ),
        }
        return obs

    def _compute_reward(
        self,
        previous_action: np.ndarray,
        action: np.ndarray,
    ) -> tuple[float, dict[str, float], dict[str, float]]:
        base_lin_vel = self.data.sensordata[self.mujoco_go1.lin_vel_slice].astype(np.float32)
        base_ang_vel = self.data.sensordata[self.mujoco_go1.ang_vel_slice].astype(np.float32)
        projected_gravity = self.mujoco_go1.projected_gravity()
        joint_pos = self.data.qpos[self.mujoco_go1.obs_qpos_adrs].astype(np.float32)
        joint_pos_rel = joint_pos - self.mujoco_go1.default_joint_pos
        command_norm = float(np.linalg.norm(self.command[:2]) + abs(float(self.command[2])))
        active = 1.0 if command_norm > 0.05 else 0.0

        track_linear_velocity = math.exp(
            -float(np.sum((self.command[:2] - base_lin_vel[:2]) ** 2) + base_lin_vel[2] ** 2) / (0.5**2)
        )
        track_angular_velocity = math.exp(
            -float((self.command[2] - base_ang_vel[2]) ** 2 + np.sum(base_ang_vel[:2] ** 2)) / (math.sqrt(0.5) ** 2)
        )
        upright = math.exp(-float(np.sum(projected_gravity[:2] ** 2)) / (math.sqrt(0.2) ** 2))
        pose_std = self._pose_std(command_norm)
        pose = math.exp(-float(np.mean((joint_pos_rel / pose_std) ** 2)))
        dof_pos_limits = self._joint_limit_penalty(joint_pos)
        action_rate_l2 = float(np.sum((action - previous_action) ** 2))
        air_time = float(np.sum((self.foot_air_time > 0.05) & (self.foot_air_time < 0.5))) * active
        foot_height = self.data.site_xpos[self.foot_site_ids, 2].astype(np.float32)
        foot_clearance = float(np.sum(np.abs(foot_height - 0.1) * np.linalg.norm(self.foot_vel[:, :2], axis=1))) * active
        first_contact = self.contact & ~self.prev_contact
        foot_swing_height = float(np.sum(((self.peak_foot_height / 0.1) - 1.0) ** 2 * first_contact.astype(np.float32))) * active
        foot_slip = float(np.sum((np.linalg.norm(self.foot_vel[:, :2], axis=1) ** 2) * self.contact.astype(np.float32))) * active
        soft_landing = float(np.sum(np.linalg.norm(self.contact_force, axis=1) * first_contact.astype(np.float32))) * active

        raw_terms = {
            "track_linear_velocity": track_linear_velocity,
            "track_angular_velocity": track_angular_velocity,
            "upright": upright,
            "pose": pose,
            "dof_pos_limits": dof_pos_limits,
            "action_rate_l2": action_rate_l2,
            "air_time": air_time,
            "foot_clearance": foot_clearance,
            "foot_swing_height": foot_swing_height,
            "foot_slip": foot_slip,
            "soft_landing": soft_landing,
        }
        weights = {
            "track_linear_velocity": 2.0,
            "track_angular_velocity": 2.0,
            "upright": 1.0,
            "pose": 1.0,
            "dof_pos_limits": -1.0,
            "action_rate_l2": -0.1,
            "air_time": 0.0,
            "foot_clearance": -2.0,
            "foot_swing_height": -0.25,
            "foot_slip": -0.1,
            "soft_landing": -1.0e-5,
        }
        for name, cfg in getattr(self.cfg, "rewards", {}).items():
            if name in weights:
                weights[name] = float(cfg.weight)

        weighted_terms = {name: weights[name] * value * self.step_dt for name, value in raw_terms.items()}
        reward = float(sum(weighted_terms.values()))
        metrics = {
            "Metrics/twist/error_vel_xy": float(np.linalg.norm(self.command[:2] - base_lin_vel[:2])),
            "Metrics/twist/error_vel_yaw": float(abs(self.command[2] - base_ang_vel[2])),
            "Episode_Metrics/mean_action_acc": action_rate_l2,
            "Metrics/peak_height_mean": float(np.max(self.peak_foot_height)),
            "Metrics/slip_velocity_mean": float(np.mean(np.linalg.norm(self.foot_vel[:, :2], axis=1) * self.contact.astype(np.float32))),
            "Metrics/landing_force_mean": float(np.mean(np.linalg.norm(self.contact_force, axis=1) * first_contact.astype(np.float32))),
        }
        return reward, weighted_terms, metrics

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
        if (not force) and self.rng.random() < self._standing_probability:
            self.command[:] = 0.0
        else:
            self.command[:] = (
                self.rng.uniform(*self._command_ranges.lin_vel_x),
                self.rng.uniform(*self._command_ranges.lin_vel_y),
                self.rng.uniform(*self._command_ranges.ang_vel_z),
            )
        command_cfg = getattr(self.cfg, "commands", {}).get("twist")
        seconds_range = getattr(command_cfg, "resampling_time_range", (3.0, 8.0))
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

    def _update_feet(self) -> None:
        current_pos = self.data.site_xpos[self.foot_site_ids].astype(np.float32)
        self.foot_vel[:] = (current_pos - self.prev_foot_pos) / self.step_dt
        self.prev_foot_pos[:] = current_pos
        self.prev_contact[:] = self.contact
        self.contact[:] = False
        self.contact_force[:] = 0.0
        foot_lookup = {int(geom_id): idx for idx, geom_id in enumerate(self.foot_geom_ids)}
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            foot_index = None
            if geom1 == self.terrain_geom_id and geom2 in foot_lookup:
                foot_index = foot_lookup[geom2]
            elif geom2 == self.terrain_geom_id and geom1 in foot_lookup:
                foot_index = foot_lookup[geom1]
            if foot_index is None:
                continue
            self.contact[foot_index] = True
            force6 = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(self.model, self.data, contact_id, force6)
            self.contact_force[foot_index] += force6[:3].astype(np.float32)
        self.foot_air_time[:] = np.where(self.contact, 0.0, self.foot_air_time + self.step_dt)
        foot_height = current_pos[:, 2]
        self.peak_foot_height[:] = np.where(
            self.contact,
            self.peak_foot_height,
            np.maximum(self.peak_foot_height, foot_height),
        )

    def _actuator_force_action_order(self) -> np.ndarray:
        force = np.zeros(self.num_actions, dtype=np.float32)
        force[self._ctrl_to_action] = self.data.actuator_force.astype(np.float32)
        return force

    def _pose_std(self, command_norm: float) -> np.ndarray:
        if command_norm < 0.05:
            triplet = (0.05, 0.05, 0.1)
        else:
            triplet = (0.3, 0.3, 0.6)
        return np.asarray(triplet * 4, dtype=np.float32)

    def _joint_limit_penalty(self, joint_pos: np.ndarray) -> float:
        lower = self._joint_limits[:, 0]
        upper = self._joint_limits[:, 1]
        return float(np.sum(np.maximum(lower - joint_pos, 0.0) + np.maximum(joint_pos - upper, 0.0)))

    def _read_joint_limits(self) -> np.ndarray:
        limits = []
        for name in self.mujoco_go1.obs_joint_names:
            joint_id = mujoco_named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")
            limits.append(self.model.jnt_range[joint_id])
        return np.asarray(limits, dtype=np.float32)
