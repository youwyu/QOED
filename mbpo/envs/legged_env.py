"""Direct MuJoCo legged velocity VecEnv shared by quadrupeds and humanoids."""

from __future__ import annotations

import math

import mujoco
import numpy as np
import torch
from mjlab.utils.lab_api.string import resolve_matching_names_values
from tensordict import TensorDict

from .direct_env import DirectMujocoVecEnv
from .utils import mujoco_body_ids_with_prefix, mujoco_geom_ids_with_prefix, mujoco_named_id, mujoco_sensor_slice


class LeggedVelocityVecEnv(DirectMujocoVecEnv):
    """Velocity tracking with the MJLab legged reward set; subclasses name feet and randomize dynamics.

    ``feet`` lists (site, collision geoms) per foot, ``upright_body`` is the body whose tilt and
    roll/pitch rate are penalized, and ``include_torque`` appends actuator forces to the system state.
    """

    feet: tuple[tuple[str, tuple[str, ...]], ...]
    upright_body: str
    include_torque: bool = True

    def _setup(self) -> None:
        model = self.model
        self.foot_site_ids = np.asarray([mujoco_named_id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{site}") for site, _ in self.feet], dtype=np.int32)
        self._foot_of_geom = {
            mujoco_named_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{geom}"): index for index, (_, geoms) in enumerate(self.feet) for geom in geoms
        }
        self.terrain_geom_id = mujoco_named_id(model, mujoco.mjtObj.mjOBJ_GEOM, "terrain")
        self._robot_geom_ids = set(mujoco_geom_ids_with_prefix(model, "robot/"))
        n = len(self.feet)
        self.prev_contact = np.zeros(n, dtype=bool)
        self.contact = np.zeros(n, dtype=bool)
        self.contact_force = np.zeros((n, 3), dtype=np.float32)
        self.foot_air_time = np.zeros(n, dtype=np.float32)
        self.peak_foot_height = np.zeros(n, dtype=np.float32)
        self.prev_foot_pos = np.zeros((n, 3), dtype=np.float32)
        self.foot_vel = np.zeros((n, 3), dtype=np.float32)
        self.self_collision = 0.0

        self._joint_limits = np.asarray(
            [model.jnt_range[mujoco_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}")] for name in self.backend.obs_joint_names],
            dtype=np.float32,
        )
        self._robot_body_ids = mujoco_body_ids_with_prefix(model, "robot/")
        self._robot_dof_ids = self.backend.obs_dof_adrs
        self._robot_qpos_ids = self.backend.obs_qpos_adrs
        self._upright_body_id = mujoco_named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{self.upright_body}")
        self._angmom_slice = mujoco_sensor_slice(model, "robot/root_angmom") if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "robot/root_angmom") >= 0 else None

        rewards = self.cfg.rewards
        self._reward_weights = {name: float(term.weight) for name, term in rewards.items()}
        self._reward_variance = {name: float(rewards[name].params["std"]) ** 2 for name in ("track_linear_velocity", "track_angular_velocity", "upright")}
        pose = rewards["pose"].params
        names = self.backend.obs_joint_names
        self._pose_std = np.stack([np.asarray(resolve_matching_names_values(pose[tier], names)[2], dtype=np.float32) for tier in ("std_standing", "std_walking", "std_running")])
        self._pose_thresholds = (float(pose.get("walking_threshold", 0.5)), float(pose.get("running_threshold", 1.5)))
        self._setup_robot()

    def _setup_robot(self) -> None:
        pass

    def _pose_tier(self, command_norm):
        walking, running = self._pose_thresholds
        if isinstance(command_norm, torch.Tensor):
            return (command_norm >= walking).long() + (command_norm >= running).long()
        return int(command_norm >= walking) + int(command_norm >= running)

    def _after_reset(self) -> None:
        self.prev_contact[:] = False
        self.contact[:] = False
        self.contact_force[:] = 0.0
        self.foot_air_time[:] = 0.0
        self.peak_foot_height[:] = 0.0
        self.prev_foot_pos[:] = self.data.site_xpos[self.foot_site_ids].astype(np.float32)
        self.foot_vel[:] = 0.0
        self.self_collision = 0.0

    def _after_physics_step(self) -> None:
        current_pos = self.data.site_xpos[self.foot_site_ids].astype(np.float32)
        self.foot_vel[:] = (current_pos - self.prev_foot_pos) / self.step_dt
        self.prev_foot_pos[:] = current_pos
        self.prev_contact[:] = self.contact
        self.contact[:] = False
        self.contact_force[:] = 0.0
        self.self_collision = 0.0
        force6 = np.zeros(6, dtype=np.float64)
        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            if geom1 in self._robot_geom_ids and geom2 in self._robot_geom_ids:
                mujoco.mj_contactForce(self.model, self.data, contact_id, force6)
                self.self_collision = max(self.self_collision, float(np.linalg.norm(force6[:3]) > 10.0))
                continue
            foot = self._foot_of_geom.get(geom2) if geom1 == self.terrain_geom_id else self._foot_of_geom.get(geom1) if geom2 == self.terrain_geom_id else None
            if foot is None:
                continue
            self.contact[foot] = True
            mujoco.mj_contactForce(self.model, self.data, contact_id, force6)
            self.contact_force[foot] += force6[:3].astype(np.float32)
        self.foot_air_time[:] = np.where(self.contact, 0.0, self.foot_air_time + self.step_dt)
        self.peak_foot_height[:] = np.where(self.contact, self.peak_foot_height, np.maximum(self.peak_foot_height, current_pos[:, 2]))

    def _upright_gravity(self) -> np.ndarray:
        rotation = self.data.xmat[self._upright_body_id].reshape(3, 3)
        return (rotation.T @ np.asarray([0.0, 0.0, -1.0])).astype(np.float32)

    def _obs_dict(self, system_termination: bool) -> dict[str, torch.Tensor]:
        actor = self.backend.observation(self.command)
        nj = len(self.backend.obs_joint_names)
        foot_contact = self.contact.astype(np.float32)
        critic = np.concatenate(
            (
                actor,
                self.data.site_xpos[self.foot_site_ids, 2].astype(np.float32),
                self.foot_air_time.astype(np.float32),
                foot_contact,
                (np.sign(self.contact_force) * np.log1p(np.abs(self.contact_force))).reshape(-1).astype(np.float32),
            ),
            axis=0,
        )
        parts = [self.backend.root_velocity(), actor[6 : 9 + 2 * nj]]
        if self.include_torque:
            parts.append(self.backend.actuator_force())
        return {
            "actor": torch.from_numpy(actor).view(1, -1),
            "policy": torch.from_numpy(actor.copy()).view(1, -1),
            "critic": torch.from_numpy(critic).view(1, -1),
            "system_state": torch.from_numpy(np.concatenate(parts).astype(np.float32)).view(1, -1),
            "system_action": torch.from_numpy(self.backend.last_action.copy()).view(1, -1),
            "system_contact": torch.from_numpy(foot_contact).view(1, -1),
            "system_termination": torch.tensor([[float(system_termination)]], dtype=torch.float32, device=self.device),
        }

    def _compute_reward(self, previous_action: np.ndarray, action: np.ndarray) -> tuple[float, dict[str, float], dict[str, float]]:
        lin_vel = self.data.sensordata[self.backend.lin_vel_slice].astype(np.float32)
        ang_vel = self.data.sensordata[self.backend.ang_vel_slice].astype(np.float32)
        joint_pos = self.data.qpos[self._robot_qpos_ids].astype(np.float32)
        command_norm = float(np.linalg.norm(self.command[:2]) + abs(float(self.command[2])))
        active = 1.0 if command_norm > 0.05 else 0.0
        upright_gravity = self._upright_gravity()
        body_velocity = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._upright_body_id, body_velocity, 0)
        foot_height = self.data.site_xpos[self.foot_site_ids, 2].astype(np.float32)
        slip_speed = np.linalg.norm(self.foot_vel[:, :2], axis=1)
        first_contact = (self.contact & ~self.prev_contact).astype(np.float32)
        contact = self.contact.astype(np.float32)
        std = self._pose_std[self._pose_tier(command_norm)]
        raw_terms = {
            "track_linear_velocity": math.exp(-float(np.sum((self.command[:2] - lin_vel[:2]) ** 2) + lin_vel[2] ** 2) / self._reward_variance["track_linear_velocity"]),
            "track_angular_velocity": math.exp(-float((self.command[2] - ang_vel[2]) ** 2 + np.sum(ang_vel[:2] ** 2)) / self._reward_variance["track_angular_velocity"]),
            "upright": math.exp(-float(np.sum(upright_gravity[:2] ** 2)) / self._reward_variance["upright"]),
            "pose": math.exp(-float(np.mean(((joint_pos - self.backend.default_joint_pos) / std) ** 2))),
            "body_ang_vel": float(np.sum(body_velocity[:2] ** 2)),
            "angular_momentum": 0.0 if self._angmom_slice is None else float(np.sum(self.data.sensordata[self._angmom_slice] ** 2)),
            "dof_pos_limits": float(np.sum(np.maximum(self._joint_limits[:, 0] - joint_pos, 0.0) + np.maximum(joint_pos - self._joint_limits[:, 1], 0.0))),
            "action_rate_l2": float(np.sum((action - previous_action) ** 2)),
            "air_time": float(np.sum((self.foot_air_time > 0.05) & (self.foot_air_time < 0.5))) * active,
            "foot_clearance": float(np.sum(np.abs(foot_height - 0.1) * slip_speed)) * active,
            "foot_swing_height": float(np.sum(((self.peak_foot_height / 0.1) - 1.0) ** 2 * first_contact)) * active,
            "foot_slip": float(np.sum(slip_speed**2 * contact)) * active,
            "soft_landing": float(np.sum(np.linalg.norm(self.contact_force, axis=1) * first_contact)) * active,
            "self_collisions": self.self_collision * self.backend.decimation,
        }
        weighted = {name: self._reward_weights[name] * value * self.step_dt for name, value in raw_terms.items() if name in self._reward_weights}
        metrics = {
            "Metrics/twist/error_vel_xy": float(np.linalg.norm(self.command[:2] - lin_vel[:2])),
            "Metrics/twist/error_vel_yaw": float(abs(self.command[2] - ang_vel[2])),
            "Episode_Metrics/mean_action_acc": raw_terms["action_rate_l2"],
        }
        return float(sum(weighted.values())), weighted, metrics

    def _init_imagination_buffers(self, num_envs: int, device: torch.device) -> None:
        self._imagination_air_time = torch.zeros(num_envs, len(self.feet), device=device)
        self._imagination_contact = torch.zeros(num_envs, len(self.feet), device=device, dtype=torch.bool)

    def _reset_imagination_buffers(self, env_ids: torch.Tensor) -> None:
        self._imagination_air_time[env_ids] = 0.0
        self._imagination_contact[env_ids] = False

    def _imagination_obs_from_state(self, state: torch.Tensor, action: torch.Tensor, termination: torch.Tensor | None = None) -> TensorDict:
        num_envs, n, nj = state.shape[0], len(self.feet), len(self.backend.obs_joint_names)
        actor = torch.cat((state[:, : 9 + 2 * nj], action, self._imagination_command.to(device=state.device, dtype=state.dtype)), dim=-1)
        foot_contact = self._imagination_contact.to(device=state.device, dtype=state.dtype)
        zeros = torch.zeros(num_envs, n, device=state.device, dtype=state.dtype)
        critic = torch.cat((actor, zeros, self._imagination_air_time.to(device=state.device, dtype=state.dtype), foot_contact, zeros.repeat(1, 3)), dim=-1)
        termination_value = torch.zeros(num_envs, 1, device=state.device, dtype=state.dtype) if termination is None else termination.to(device=state.device).float().view(num_envs, 1)
        obs = {
            "actor": actor,
            "policy": actor,
            "critic": critic,
            "system_state": state,
            "system_action": action,
            "system_contact": foot_contact,
            "system_termination": termination_value,
        }
        return TensorDict(obs, batch_size=[num_envs], device=state.device)

    def _parse_imagination_contact(self, contacts: torch.Tensor | None) -> None:
        n = len(self.feet)
        if contacts is None:
            contact = torch.zeros_like(self._imagination_contact)
        else:
            contact = torch.sigmoid(contacts[:, n : 2 * n] if contacts.shape[-1] >= 2 * n else contacts[:, :n]).round().bool()
        self._imagination_contact = contact
        self._imagination_air_time = torch.where(contact, torch.zeros_like(self._imagination_air_time), self._imagination_air_time + self.step_dt)

    def _parse_imagination_termination(self, terminations: torch.Tensor | None, state: torch.Tensor) -> torch.Tensor:
        if terminations is None:
            termination = torch.zeros(state.shape[0], device=state.device, dtype=torch.bool)
        else:
            termination = torch.sigmoid(terminations).reshape(state.shape[0], -1).max(dim=1).values > 0.5
        bad_orientation = torch.acos(torch.clamp(-state[:, 8], -1.0, 1.0)).abs() > math.radians(70.0)
        nj = len(self.backend.obs_joint_names)
        low, high = torch.as_tensor(self._joint_limits, device=state.device, dtype=state.dtype).unbind(1)
        joint_pos = state[:, 9 : 9 + nj] + torch.as_tensor(self.backend.default_joint_pos, device=state.device, dtype=state.dtype)
        plausible = ((joint_pos >= 2 * low - high) & (joint_pos <= 2 * high - low)).all(dim=1) & torch.isfinite(state).all(dim=1)
        return termination | bad_orientation | ~plausible

    def _compute_imagination_reward(self, state: torch.Tensor, action: torch.Tensor):
        nj = len(self.backend.obs_joint_names)
        lin_vel, ang_vel, gravity = state[:, 0:3], state[:, 3:6], state[:, 6:9]
        joint_pos = state[:, 9 : 9 + nj] + torch.as_tensor(self.backend.default_joint_pos, device=state.device, dtype=state.dtype)
        command = self._imagination_command.to(device=state.device, dtype=state.dtype)
        command_norm = torch.linalg.norm(command[:, :2], dim=1) + command[:, 2].abs()
        active = (command_norm > 0.05).to(state.dtype)
        std = torch.as_tensor(self._pose_std, device=state.device, dtype=state.dtype)[self._pose_tier(command_norm)]
        limits = torch.as_tensor(self._joint_limits, device=state.device, dtype=state.dtype)
        previous_action = self._imagination_last_action.to(device=state.device, dtype=state.dtype)
        air_time = self._imagination_air_time.to(device=state.device, dtype=state.dtype)
        zero = torch.zeros(state.shape[0], device=state.device, dtype=state.dtype)
        raw_terms = {
            "track_linear_velocity": torch.exp(-(torch.sum((command[:, :2] - lin_vel[:, :2]) ** 2, dim=1) + lin_vel[:, 2] ** 2) / self._reward_variance["track_linear_velocity"]),
            "track_angular_velocity": torch.exp(-((command[:, 2] - ang_vel[:, 2]) ** 2 + torch.sum(ang_vel[:, :2] ** 2, dim=1)) / self._reward_variance["track_angular_velocity"]),
            "upright": torch.exp(-torch.sum(gravity[:, :2] ** 2, dim=1) / self._reward_variance["upright"]),
            "pose": torch.exp(-torch.mean(((joint_pos - torch.as_tensor(self.backend.default_joint_pos, device=state.device, dtype=state.dtype)) / std) ** 2, dim=1)),
            "body_ang_vel": torch.sum(ang_vel[:, :2] ** 2, dim=1),
            "angular_momentum": zero,
            "dof_pos_limits": torch.sum(torch.clamp(limits[:, 0] - joint_pos, min=0.0) + torch.clamp(joint_pos - limits[:, 1], min=0.0), dim=1),
            "action_rate_l2": torch.sum((action - previous_action) ** 2, dim=1),
            "air_time": torch.sum(((air_time > 0.05) & (air_time < 0.5)).to(state.dtype), dim=1) * active,
            "foot_clearance": zero,
            "foot_swing_height": zero,
            "foot_slip": zero,
            "soft_landing": zero,
            "self_collisions": zero,
        }
        weighted = {name: self._reward_weights[name] * value * self.step_dt for name, value in raw_terms.items() if name in self._reward_weights}
        metrics = {
            "Metrics/twist/error_vel_xy": torch.linalg.norm(command[:, :2] - lin_vel[:, :2], dim=1),
            "Metrics/twist/error_vel_yaw": (command[:, 2] - ang_vel[:, 2]).abs(),
            "Episode_Metrics/mean_action_acc": raw_terms["action_rate_l2"],
        }
        return torch.stack(tuple(weighted.values())).sum(dim=0), weighted, metrics
