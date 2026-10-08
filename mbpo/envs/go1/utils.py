"""Unitree Go1 MuJoCo constants and domain randomization."""

from __future__ import annotations

import mujoco
import numpy as np

from mbpo.envs.utils import MujocoModelDefaults, restore_mujoco_model_defaults, uniform_like

GO1_FEET_NAMES = ("FR", "FL", "RR", "RL")
GO1_JOINT_TYPES = ("hip", "thigh", "calf")
GO1_KP_SCALE = (0.8, 1.2)
GO1_KD_SCALE = (0.5, 1.5)
GO1_PAYLOAD = (0.0, 3.0)


def mujoco_actuator_ids_by_joint_type(model: mujoco.MjModel) -> list[np.ndarray]:
    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "" for i in range(model.nu)]
    return [np.asarray([i for i, name in enumerate(names) if name.endswith(f"_{joint}_joint")]) for joint in GO1_JOINT_TYPES]


def apply_go1_domain_randomization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng,
    defaults: MujocoModelDefaults,
    *,
    terrain_geom_id: int,
    robot_dof_ids,
    torso_body_id: int,
    body_mass_ids,
    robot_qpos_ids,
    actuator_type_ids,
    dtype=np.float32,
) -> float:
    restore_mujoco_model_defaults(model, defaults)

    model.geom_friction[terrain_geom_id, 0] = float(rng.uniform(0.4, 1.0))
    model.dof_frictionloss[robot_dof_ids] *= uniform_like(
        rng, 0.9, 1.1, model.dof_frictionloss[robot_dof_ids], dtype
    )
    model.dof_armature[robot_dof_ids] *= uniform_like(
        rng, 1.0, 1.05, model.dof_armature[robot_dof_ids], dtype
    )
    model.body_ipos[torso_body_id] += np.asarray(rng.uniform(-0.05, 0.05, size=3), dtype=dtype)
    model.body_mass[body_mass_ids] *= uniform_like(
        rng, 0.9, 1.1, model.body_mass[body_mass_ids], dtype
    )
    model.body_mass[torso_body_id] += float(rng.uniform(-1.0, 1.0))
    model.qpos0[robot_qpos_ids] += uniform_like(
        rng, -0.05, 0.05, model.qpos0[robot_qpos_ids], dtype
    )
    for ids, kp, kd in zip(actuator_type_ids, rng.uniform(*GO1_KP_SCALE, size=3), rng.uniform(*GO1_KD_SCALE, size=3)):
        model.actuator_gainprm[ids, 0] *= kp
        model.actuator_biasprm[ids, 1] *= kp
        model.actuator_biasprm[ids, 2] *= kd
    payload = float(rng.uniform(*GO1_PAYLOAD))
    model.body_mass[torso_body_id] += payload

    mujoco.mj_setConst(model, data)
    return payload
