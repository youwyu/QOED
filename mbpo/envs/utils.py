"""Shared MuJoCo helpers for direct MBPO environments."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


ACTIVE_RGBA = np.asarray([1.0, 0.48, 0.08, 1.0], dtype=np.float32)


@dataclass(frozen=True)
class MujocoModelDefaults:
    geom_friction: np.ndarray
    dof_frictionloss: np.ndarray
    dof_armature: np.ndarray
    dof_damping: np.ndarray
    body_ipos: np.ndarray
    body_mass: np.ndarray
    qpos0: np.ndarray
    actuator_gainprm: np.ndarray
    actuator_biasprm: np.ndarray


def mujoco_named_id(model: mujoco.MjModel, obj_type, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"MuJoCo object not found: {name}")
    return int(obj_id)


def mujoco_sensor_slice(model: mujoco.MjModel, sensor_name: str) -> slice:
    sensor_id = mujoco_named_id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
    start = int(model.sensor_adr[sensor_id])
    stop = start + int(model.sensor_dim[sensor_id])
    return slice(start, stop)


def mujoco_geom_ids_with_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    return _mujoco_ids_with_prefix(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom, prefix)


def mujoco_body_ids_with_prefix(model: mujoco.MjModel, prefix: str) -> list[int]:
    return _mujoco_ids_with_prefix(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody, prefix, start=1)


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = quat / norm
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def set_follow_camera(camera, qpos: np.ndarray, distance: float = 1.5) -> None:
    camera.lookat[:] = qpos[0:3]
    camera.distance = distance
    camera.elevation = -10.0
    camera.azimuth = 90.0


def save_mujoco_model_defaults(model: mujoco.MjModel) -> MujocoModelDefaults:
    return MujocoModelDefaults(
        geom_friction=model.geom_friction.copy(),
        dof_frictionloss=model.dof_frictionloss.copy(),
        dof_armature=model.dof_armature.copy(),
        dof_damping=model.dof_damping.copy(),
        body_ipos=model.body_ipos.copy(),
        body_mass=model.body_mass.copy(),
        qpos0=model.qpos0.copy(),
        actuator_gainprm=model.actuator_gainprm.copy(),
        actuator_biasprm=model.actuator_biasprm.copy(),
    )


def restore_mujoco_model_defaults(model: mujoco.MjModel, defaults: MujocoModelDefaults) -> None:
    model.geom_friction[:] = defaults.geom_friction
    model.dof_frictionloss[:] = defaults.dof_frictionloss
    model.dof_armature[:] = defaults.dof_armature
    model.dof_damping[:] = defaults.dof_damping
    model.body_ipos[:] = defaults.body_ipos
    model.body_mass[:] = defaults.body_mass
    model.qpos0[:] = defaults.qpos0
    model.actuator_gainprm[:] = defaults.actuator_gainprm
    model.actuator_biasprm[:] = defaults.actuator_biasprm


def uniform_like(rng, low: float, high: float, values: np.ndarray, dtype) -> np.ndarray:
    return np.asarray(rng.uniform(low, high, size=np.asarray(values).shape), dtype=dtype)


def _mujoco_ids_with_prefix(model: mujoco.MjModel, obj_type, count: int, prefix: str, start: int = 0) -> list[int]:
    return [obj_id for obj_id in range(start, count) if (mujoco.mj_id2name(model, obj_type, obj_id) or "").startswith(prefix)]
