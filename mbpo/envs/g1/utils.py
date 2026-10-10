"""Unitree G1 MuJoCo constants and domain randomization."""

from __future__ import annotations

import re

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_ARTICULATION

from mbpo.envs.utils import MujocoModelDefaults, mujoco_named_id, restore_mujoco_model_defaults

G1_FEET = tuple((f"{side}_foot", tuple(f"{side}_foot{i}_collision" for i in range(1, 8))) for side in ("left", "right"))
G1_TORSO = "torso_link"
G1_FOOT_FRICTION = (0.3, 1.2)
G1_TORSO_MASS_SCALE = (0.9, 1.1)
G1_KP_SCALE = (0.8, 1.2)
G1_KD_SCALE = (0.5, 1.5)
G1_ARMATURE_SCALE = (0.9, 1.1)
G1_PAYLOAD = (0.0, 5.0)


def g1_ids(model: mujoco.MjModel) -> dict:
    names = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").removeprefix("robot/") for i in range(model.nu)]
    groups = [np.asarray([i for i, name in enumerate(names) if any(re.fullmatch(p, name) for p in actuator.target_names_expr)]) for actuator in G1_ARTICULATION.actuators]
    return {
        "foot": np.asarray([mujoco_named_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{geom}") for _, geoms in G1_FEET for geom in geoms]),
        "torso": mujoco_named_id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot/{G1_TORSO}"),
        "actuator": groups,
        "dof": [model.jnt_dofadr[model.actuator_trnid[ids, 0]] for ids in groups],
    }


def apply_g1_domain_randomization(model: mujoco.MjModel, data: mujoco.MjData, rng, defaults: MujocoModelDefaults, *, ids: dict) -> float:
    restore_mujoco_model_defaults(model, defaults)
    model.geom_friction[ids["foot"], 0] = rng.uniform(*G1_FOOT_FRICTION)
    model.body_mass[ids["torso"]] *= rng.uniform(*G1_TORSO_MASS_SCALE)
    for actuators, dofs, kp, kd, armature in zip(
        ids["actuator"], ids["dof"], *(rng.uniform(*scale, size=len(ids["actuator"])) for scale in (G1_KP_SCALE, G1_KD_SCALE, G1_ARMATURE_SCALE))
    ):
        model.actuator_gainprm[actuators, 0] *= kp
        model.actuator_biasprm[actuators, 1] *= kp
        model.actuator_biasprm[actuators, 2] *= kd
        model.dof_armature[dofs] *= armature
    payload = float(rng.uniform(*G1_PAYLOAD))
    model.body_mass[ids["torso"]] += payload
    mujoco.mj_setConst(model, data)
    return payload
