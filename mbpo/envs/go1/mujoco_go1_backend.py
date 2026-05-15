"""Shared direct MuJoCo Go1 backend for play and train."""

from __future__ import annotations

import re

import mujoco
import numpy as np
from mjlab.scene import Scene

from .utils import (
    ACTIVE_GO1_RGBA,
    mujoco_geom_ids_with_prefix,
    mujoco_named_id,
    mujoco_sensor_slice,
    quat_wxyz_to_matrix,
    set_go1_camera,
)


def _strip_robot_prefix(name: str | None) -> str:
    if not name:
        return ""
    return name.removeprefix("robot/")


def _resolve_name_values(
    values: float | int | dict[str, float],
    names: list[str],
    default: float,
) -> np.ndarray:
    if isinstance(values, (float, int)):
        return np.full(len(names), float(values), dtype=np.float32)
    if not isinstance(values, dict):
        raise TypeError(f"Unsupported action scale type: {type(values).__name__}")

    resolved = np.full(len(names), float(default), dtype=np.float32)
    matched = np.zeros(len(names), dtype=bool)
    for pattern, value in values.items():
        regex = re.compile(pattern)
        for index, name in enumerate(names):
            if regex.fullmatch(name) or regex.match(name):
                resolved[index] = float(value)
                matched[index] = True
    if not bool(matched.all()):
        missing = ", ".join(name for name, ok in zip(names, matched) if not ok)
        raise RuntimeError(f"Action scale did not match joints: {missing}")
    return resolved


def draw_go1_velocity_arrows(
    user_scn: mujoco.MjvScene | None,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    command: np.ndarray,
) -> None:
    if user_scn is None:
        return

    user_scn.ngeom = 0
    command = np.asarray(command, dtype=np.float32)
    base_lin_vel = np.asarray(
        data.sensordata[mujoco_sensor_slice(model, "robot/imu_lin_vel")],
        dtype=np.float32,
    )
    base_ang_vel = np.asarray(
        data.sensordata[mujoco_sensor_slice(model, "robot/imu_ang_vel")],
        dtype=np.float32,
    )
    rotation_body_to_world = quat_wxyz_to_matrix(data.qpos[3:7])
    base_pos = np.asarray(data.qpos[:3], dtype=np.float64)

    command_linear_world = rotation_body_to_world @ np.asarray(
        [command[0], command[1], 0.0], dtype=np.float64
    )
    current_linear_world = rotation_body_to_world @ np.asarray(
        [base_lin_vel[0], base_lin_vel[1], 0.0], dtype=np.float64
    )

    command_color = np.asarray([0.1, 0.9, 0.25, 0.85], dtype=np.float32)
    current_color = np.asarray([0.15, 0.45, 1.0, 0.85], dtype=np.float32)
    linear_scale = 0.55
    yaw_scale = 0.35
    linear_start = base_pos + np.asarray([0.0, 0.0, 0.45], dtype=np.float64)
    yaw_start = base_pos + np.asarray([0.0, 0.0, 0.85], dtype=np.float64)

    _add_arrow(
        user_scn,
        linear_start,
        linear_start + command_linear_world * linear_scale,
        command_color,
        min_length=0.04,
    )
    _add_arrow(
        user_scn,
        linear_start + np.asarray([0.0, 0.0, 0.10], dtype=np.float64),
        linear_start + np.asarray([0.0, 0.0, 0.10], dtype=np.float64)
        + current_linear_world * linear_scale,
        current_color,
        min_length=0.04,
    )
    _add_arrow(
        user_scn,
        yaw_start + rotation_body_to_world @ np.asarray([0.0, -0.32, 0.0], dtype=np.float64),
        yaw_start
        + rotation_body_to_world @ np.asarray([0.0, -0.32, 0.0], dtype=np.float64)
        + np.asarray([0.0, 0.0, float(command[2]) * yaw_scale], dtype=np.float64),
        command_color,
        min_length=0.04,
    )
    _add_arrow(
        user_scn,
        yaw_start + rotation_body_to_world @ np.asarray([0.0, 0.32, 0.0], dtype=np.float64),
        yaw_start
        + rotation_body_to_world @ np.asarray([0.0, 0.32, 0.0], dtype=np.float64)
        + np.asarray([0.0, 0.0, float(base_ang_vel[2]) * yaw_scale], dtype=np.float64),
        current_color,
        min_length=0.04,
    )


def _add_arrow(
    user_scn: mujoco.MjvScene,
    start: np.ndarray,
    end: np.ndarray,
    rgba: np.ndarray,
    min_length: float,
) -> None:
    if user_scn.ngeom >= user_scn.maxgeom:
        return
    start = np.asarray(start, dtype=np.float64)
    end = np.asarray(end, dtype=np.float64)
    length = float(np.linalg.norm(end - start))
    if length < min_length:
        return

    geom = user_scn.geoms[user_scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        rgba,
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_ARROW,
        0.035,
        start,
        end,
    )
    user_scn.ngeom += 1


class PureMujocoGo1Model:
    """Single-Go1 MuJoCo model with MJLab-compatible observations/actions."""

    _ACTIVE_RGBA = ACTIVE_GO1_RGBA

    def __init__(self, env_cfg) -> None:
        env_cfg.scene.num_envs = 1
        self.scene = Scene(env_cfg.scene, device="cpu")
        self.robot = self.scene["robot"]
        self.obs_joint_names = list(self.robot.joint_names)

        action_cfg = env_cfg.actions["joint_pos"]
        _, action_joint_names = self.robot.find_joints_by_actuator_names(action_cfg.actuator_names)
        self.action_joint_names = list(action_joint_names)
        self.action_dim = len(self.action_joint_names)

        self.model = self.scene.compile()
        env_cfg.sim.mujoco.apply(self.model)
        self.data = mujoco.MjData(self.model)
        self.decimation = int(env_cfg.decimation)
        self.step_dt = float(self.model.opt.timestep) * self.decimation

        self.obs_qpos_adrs = np.asarray(
            [self._joint_qposadr(name) for name in self.obs_joint_names],
            dtype=np.int32,
        )
        self.obs_dof_adrs = np.asarray(
            [self._joint_dofadr(name) for name in self.obs_joint_names],
            dtype=np.int32,
        )
        self.action_to_obs_indices = np.asarray(
            [self.obs_joint_names.index(name) for name in self.action_joint_names],
            dtype=np.int32,
        )

        self.default_qpos = np.asarray(self.model.key_qpos[0], dtype=np.float64).copy()
        self.default_qvel = np.asarray(self.model.key_qvel[0], dtype=np.float64).copy()
        self.default_joint_pos = self.default_qpos[self.obs_qpos_adrs].astype(np.float32)
        self.default_joint_vel = self.default_qvel[self.obs_dof_adrs].astype(np.float32)
        self.default_action_joint_pos = self.default_joint_pos[self.action_to_obs_indices]
        self.action_scale = _resolve_name_values(action_cfg.scale, self.action_joint_names, 1.0)

        action_name_to_index = {
            name: index for index, name in enumerate(self.action_joint_names)
        }
        self.ctrl_action_indices = np.empty(self.model.nu, dtype=np.int32)
        for ctrl_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[ctrl_id, 0])
            full_joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            joint_name = _strip_robot_prefix(full_joint_name)
            self.ctrl_action_indices[ctrl_id] = action_name_to_index[joint_name]
        self.default_ctrl = self.default_action_joint_pos[self.ctrl_action_indices].astype(np.float64)

        self.lin_vel_slice = self._sensor_slice("robot/imu_lin_vel")
        self.ang_vel_slice = self._sensor_slice("robot/imu_ang_vel")
        self.last_action = np.zeros(self.action_dim, dtype=np.float32)

        self.robot_geom_ids = mujoco_geom_ids_with_prefix(self.model, "robot/")
        self.default_robot_rgba = self.model.geom_rgba[self.robot_geom_ids].copy()
        self.reset()

    def _joint_qposadr(self, joint_name: str) -> int:
        joint_id = mujoco_named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{joint_name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_dofadr(self, joint_name: str) -> int:
        joint_id = mujoco_named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{joint_name}")
        return int(self.model.jnt_dofadr[joint_id])

    def _sensor_slice(self, sensor_name: str) -> slice:
        return mujoco_sensor_slice(self.model, sensor_name)

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.ctrl[:] = self.default_ctrl
        self.last_action[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def projected_gravity(self) -> np.ndarray:
        rotation_body_to_world = quat_wxyz_to_matrix(self.data.qpos[3:7])
        gravity_world = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
        return (rotation_body_to_world.T @ gravity_world).astype(np.float32)

    def observation(self, command: np.ndarray) -> np.ndarray:
        base_lin_vel = np.asarray(self.data.sensordata[self.lin_vel_slice], dtype=np.float32)
        base_ang_vel = np.asarray(self.data.sensordata[self.ang_vel_slice], dtype=np.float32)
        joint_pos = self.data.qpos[self.obs_qpos_adrs].astype(np.float32) - self.default_joint_pos
        joint_vel = self.data.qvel[self.obs_dof_adrs].astype(np.float32) - self.default_joint_vel
        obs = np.concatenate(
            [
                base_lin_vel,
                base_ang_vel,
                self.projected_gravity(),
                joint_pos,
                joint_vel,
                self.last_action,
                command.astype(np.float32),
            ],
            axis=0,
        )
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, raw_action: np.ndarray) -> None:
        action = np.nan_to_num(
            np.asarray(raw_action, dtype=np.float32).reshape(self.action_dim),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        self.last_action[:] = action
        target_action_joint_pos = self.default_action_joint_pos + action * self.action_scale
        ctrl = target_action_joint_pos[self.ctrl_action_indices].astype(np.float64)
        for _ in range(self.decimation):
            self.data.ctrl[:] = ctrl
            mujoco.mj_step(self.model, self.data)

    def is_fallen(self) -> bool:
        projected_gravity = self.projected_gravity()
        angle = np.arccos(np.clip(-float(projected_gravity[2]), -1.0, 1.0))
        return bool(angle > np.deg2rad(70.0))

    def set_active_color(self, active: bool) -> None:
        if not self.robot_geom_ids:
            return
        if active:
            self.model.geom_rgba[self.robot_geom_ids] = self._ACTIVE_RGBA
        else:
            self.model.geom_rgba[self.robot_geom_ids] = self.default_robot_rgba

    def update_camera(self, camera) -> None:
        set_go1_camera(camera, self.data.qpos)
