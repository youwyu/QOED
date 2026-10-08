"""Clearpath Jackal MuJoCo constants and domain randomization."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mbpo.envs.utils import MujocoModelDefaults, mujoco_named_id, restore_mujoco_model_defaults, uniform_like

JACKAL_XML = Path(__file__).resolve().parents[1] / "assets" / "robot" / "jackal" / "jackal.xml"
JACKAL_WHEELS = ("front_left_wheel", "front_right_wheel", "rear_left_wheel", "rear_right_wheel")
JACKAL_BASE = "base_link"
JACKAL_WHEEL_RADIUS = 0.095
JACKAL_TRACK_WIDTH = 0.37559
JACKAL_MAX_WHEEL_SPEED = 26.0
JACKAL_KV = 1.0
JACKAL_EFFORT_LIMIT = 20.0
JACKAL_FRICTION = (0.4, 1.2)
JACKAL_MASS_SCALE = (0.9, 1.1)
JACKAL_FRICTIONLOSS_SCALE = (0.5, 1.5)
JACKAL_ARMATURE_SCALE = (0.9, 1.1)
JACKAL_DAMPING_SCALE = (0.5, 2.0)
JACKAL_KV_SCALE = (0.5, 1.5)
JACKAL_PAYLOAD = (0.0, 10.0)


def diff_drive_mix(wheel_names) -> np.ndarray:
    turn = np.asarray([-1.0 if "left" in name else 1.0 for name in wheel_names]) * 1.5 * JACKAL_TRACK_WIDTH
    return np.stack((np.ones(len(wheel_names)), turn)) / JACKAL_WHEEL_RADIUS


def jackal_ids(model: mujoco.MjModel) -> dict[str, np.ndarray | int]:
    def ids(obj, names):
        return np.asarray([mujoco_named_id(model, obj, f"robot/{name}") for name in names])

    joints = ids(mujoco.mjtObj.mjOBJ_JOINT, JACKAL_WHEELS)
    return {
        "geom": ids(mujoco.mjtObj.mjOBJ_GEOM, JACKAL_WHEELS),
        "dof": model.jnt_dofadr[joints],
        "actuator": ids(mujoco.mjtObj.mjOBJ_ACTUATOR, JACKAL_WHEELS),
        "base": mujoco_named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{JACKAL_BASE}"),
    }


def apply_jackal_domain_randomization(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng,
    defaults: MujocoModelDefaults,
    *,
    ids: dict,
    body_ids,
    dtype=np.float32,
) -> float:
    restore_mujoco_model_defaults(model, defaults)
    geom, dof, actuator = ids["geom"], ids["dof"], ids["actuator"]
    model.geom_friction[geom, 0] = uniform_like(rng, *JACKAL_FRICTION, geom, dtype)
    model.body_mass[body_ids] *= uniform_like(rng, *JACKAL_MASS_SCALE, model.body_mass[body_ids], dtype)
    model.dof_frictionloss[dof] *= uniform_like(rng, *JACKAL_FRICTIONLOSS_SCALE, dof, dtype)
    model.dof_armature[dof] *= uniform_like(rng, *JACKAL_ARMATURE_SCALE, dof, dtype)
    model.dof_damping[dof] *= uniform_like(rng, *JACKAL_DAMPING_SCALE, dof, dtype)
    kv_scale = uniform_like(rng, *JACKAL_KV_SCALE, actuator, dtype)
    model.actuator_gainprm[actuator, 0] *= kv_scale
    model.actuator_biasprm[actuator, 2] *= kv_scale
    payload = float(rng.uniform(*JACKAL_PAYLOAD))
    model.body_mass[ids["base"]] += payload
    mujoco.mj_setConst(model, data)
    return payload
