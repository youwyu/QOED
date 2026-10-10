"""LEAP hand MuJoCo constants and domain randomization."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from mbpo.envs.utils import MujocoModelDefaults, mujoco_named_id, restore_mujoco_model_defaults, uniform_like

LEAP_ASSETS = Path(__file__).resolve().parents[1] / "assets" / "robot" / "leap_hand"
LEAP_XML = LEAP_ASSETS / "leap_hand.xml"
CUBE_XML = LEAP_ASSETS / "cube.xml"
LEAP_FINGERS = ("if", "mf", "rf", "th")
LEAP_TIPS = ("th_tip", "if_tip", "mf_tip", "rf_tip")
LEAP_PALM_SITE = "grasp_site"
LEAP_HOME = {r"(if|mf|rf)_(mcp|pip|dip)": 0.8, r"(if|mf|rf)_rot": 0.0, r"th_(cmc|axl|mcp)": 0.8, r"th_ipl": 0.0}
CUBE_INIT_POS = (0.1, 0.0, 0.05)
CUBE_DROP_HEIGHT = -0.05
LEAP_TIP_FRICTION = (0.5, 1.0)
LEAP_CUBE_MASS_SCALE = (0.8, 1.2)
LEAP_CUBE_COM = (-5.0e-3, 5.0e-3)
LEAP_KP_SCALE = (0.8, 1.2)
LEAP_DAMPING_SCALE = (0.8, 1.2)
LEAP_FRICTIONLOSS_SCALE = (0.5, 2.0)
LEAP_ARMATURE_SCALE = (1.0, 1.05)


def leap_ids(model: mujoco.MjModel) -> dict:
    actuator_names = [(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) or "").removeprefix("robot/") for i in range(model.nu)]
    actuators = [np.asarray([i for i, name in enumerate(actuator_names) if name.startswith(f"{finger}_")]) for finger in LEAP_FINGERS]
    return {
        "tip": np.asarray([mujoco_named_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"robot/{tip}") for tip in LEAP_TIPS]),
        "cube": mujoco_named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube/cube"),
        "actuator": actuators,
        "dof": [model.jnt_dofadr[model.actuator_trnid[ids, 0]] for ids in actuators],
    }


def apply_leap_domain_randomization(model: mujoco.MjModel, data: mujoco.MjData, rng, defaults: MujocoModelDefaults, *, ids: dict, dtype=np.float32) -> None:
    restore_mujoco_model_defaults(model, defaults)
    model.geom_friction[ids["tip"], 0] = rng.uniform(*LEAP_TIP_FRICTION)
    cube = ids["cube"]
    mass_scale = rng.uniform(*LEAP_CUBE_MASS_SCALE)
    model.body_mass[cube] *= mass_scale
    model.body_inertia[cube] = defaults.body_inertia[cube] * mass_scale
    model.body_ipos[cube] += uniform_like(rng, *LEAP_CUBE_COM, model.body_ipos[cube], dtype)
    scales = (rng.uniform(*scale, size=len(LEAP_FINGERS)) for scale in (LEAP_KP_SCALE, LEAP_DAMPING_SCALE, LEAP_FRICTIONLOSS_SCALE, LEAP_ARMATURE_SCALE))
    for actuators, dofs, kp, damping, frictionloss, armature in zip(ids["actuator"], ids["dof"], *scales):
        model.actuator_gainprm[actuators, 0] *= kp
        model.actuator_biasprm[actuators, 1] *= kp
        model.dof_damping[dofs] *= damping
        model.dof_frictionloss[dofs] *= frictionloss
        model.dof_armature[dofs] *= armature
    mujoco.mj_setConst(model, data)
