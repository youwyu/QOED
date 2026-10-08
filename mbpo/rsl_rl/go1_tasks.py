"""Unitree Go1 MBPO task for QOED."""

from __future__ import annotations

import torch
from mjlab.envs.mdp import observations as obs_mdp
from mjlab.managers import EventTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg

from mbpo.envs.go1.utils import GO1_KD_SCALE, GO1_KP_SCALE, GO1_PAYLOAD, mujoco_actuator_ids_by_joint_type
from mbpo.rsl_rl.mbpo_tasks import (
    MbpoRobot,
    MbpoRunner,
    add_mbpo_observations,
    add_privilege_observation,
    configure_agent_cfg,
    domain_randomization_enabled,
    joint_effort,
    named_global_id,
    register_tasks,
    resolve_env_ids,
    uniform_like,
    uniform_mean_var,
)

_BODY_MASS_DEFAULTS = (5.204, *([0.68, 1.009, 0.195862] * 4))
_DOF_FRICTIONLOSS_DEFAULTS = (0.0,) * 12
_DOF_ARMATURE_DEFAULTS = (0.004026312, 0.004026312, 0.009059202) * 4

_STATE_IDX = {
    r"$v_b$\n$[m/s]$": [0, 1, 2],
    r"$\omega_b$\n$[rad/s]$": [3, 4, 5],
    r"$g_b$\n$[1]$": [6, 7, 8],
    r"$q - q_0$\n$[rad]$": list(range(9, 21)),
    r"$\dot{q}$\n$[rad/s]$": list(range(21, 33)),
    r"$\tau$\n$[Nm]$": list(range(33, 45)),
}
_STATE_MEAN = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, *([0.0] * 36))
_STATE_STD = (
    0.5, 0.5, 0.1,
    0.3, 0.3, 0.5,
    0.02, 0.02, 0.04,
    *[0.3, 0.3, 0.6] * 4,
    *[1.0, 1.5, 2.5] * 4,
    *[23.7, 23.7, 35.55] * 4,
)
_STATE_TERMS = {
    "base_lin_vel": obs_mdp.base_lin_vel,
    "base_ang_vel": obs_mdp.base_ang_vel,
    "projected_gravity": obs_mdp.projected_gravity,
    "joint_pos": obs_mdp.joint_pos_rel,
    "joint_vel": obs_mdp.joint_vel_rel,
    "joint_torque": joint_effort,
}


def _go1_actuator_type_ids(env) -> list[torch.Tensor]:
    if getattr(env, "qoed_actuator_type_ids", None) is None:
        env.qoed_actuator_type_ids = [torch.as_tensor(ids, device=env.device) for ids in mujoco_actuator_ids_by_joint_type(env.sim.mj_model)]
    return env.qoed_actuator_type_ids


def _go1_payload(env) -> torch.Tensor:
    if getattr(env, "qoed_payload", None) is None:
        env.qoed_payload = torch.zeros(env.num_envs, device=env.device)
    return env.qoed_payload


def _go1_domain_randomization_vector(env) -> torch.Tensor:
    robot, terrain, model = env.scene["robot"], env.scene["terrain"], env.sim.model
    floor_geom_id = named_global_id(terrain.geom_names, terrain.indexing.geom_ids, "terrain")
    torso_body_id = named_global_id(robot.body_names, robot.indexing.body_ids, "trunk")
    joint_v_ids = robot.indexing.joint_v_adr
    payload = _go1_payload(env)
    body_mass = model.body_mass[:, robot.indexing.body_ids] - payload[:, None] * (robot.indexing.body_ids == torso_body_id)
    gain, bias = env.sim.get_default_field("actuator_gainprm"), env.sim.get_default_field("actuator_biasprm")
    types = _go1_actuator_type_ids(env)
    return torch.cat(
        (
            model.geom_friction[:, floor_geom_id, 0:1],
            body_mass,
            model.dof_frictionloss[:, joint_v_ids],
            model.dof_armature[:, joint_v_ids],
            torch.stack([(model.actuator_gainprm[:, ids, 0] / gain[ids, 0]).mean(1) for ids in types], dim=-1),
            torch.stack([(model.actuator_biasprm[:, ids, 2] / bias[ids, 2]).mean(1) for ids in types], dim=-1),
            payload[:, None],
        ),
        dim=-1,
    )


@requires_model_fields(
    "geom_friction",
    "body_ipos",
    "dof_frictionloss",
    "dof_armature",
    "body_mass",
    "qpos0",
    "actuator_gainprm",
    "actuator_biasprm",
    recompute=RecomputeLevel.set_const,
)
def _go1_pretrain_domain_randomize(env, env_ids: torch.Tensor | slice | None) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    num_envs = len(env_ids)

    robot, terrain, model = env.scene["robot"], env.scene["terrain"], env.sim.model
    floor_geom_id = named_global_id(terrain.geom_names, terrain.indexing.geom_ids, "terrain")
    torso_body_id = named_global_id(robot.body_names, robot.indexing.body_ids, "trunk")
    body_ids = robot.indexing.body_ids
    joint_q_ids = robot.indexing.joint_q_adr
    joint_v_ids = robot.indexing.joint_v_adr
    default = env.sim.get_default_field

    # Match mujoco_playground Go1 randomize.py.
    model.geom_friction[env_ids, floor_geom_id] = default("geom_friction")[floor_geom_id]
    model.geom_friction[env_ids, floor_geom_id, 0] = uniform_like(model.geom_friction, (num_envs,), 0.4, 1.0)

    # Joint friction loss: default * U(0.9, 1.1).
    env_grid, dof_grid = torch.meshgrid(env_ids, joint_v_ids, indexing="ij")
    frictionloss_scale = uniform_like(model.dof_frictionloss, (num_envs, len(joint_v_ids)), 0.9, 1.1)
    model.dof_frictionloss[env_grid, dof_grid] = default("dof_frictionloss")[joint_v_ids] * frictionloss_scale

    # Armature: default * U(1.0, 1.05).
    armature_scale = uniform_like(model.dof_armature, (num_envs, len(joint_v_ids)), 1.0, 1.05)
    model.dof_armature[env_grid, dof_grid] = default("dof_armature")[joint_v_ids] * armature_scale

    # Torso COM position: default + U(-0.05, 0.05).
    model.body_ipos[env_ids, torso_body_id] = default("body_ipos")[torso_body_id] + uniform_like(
        model.body_ipos, (num_envs, 3), -0.05, 0.05
    )

    # Link masses: default * U(0.9, 1.1), plus torso mass U(-1.0, 1.0).
    env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
    mass_scale = uniform_like(model.body_mass, (num_envs, len(body_ids)), 0.9, 1.1)
    model.body_mass[env_grid, body_grid] = default("body_mass")[body_ids] * mass_scale
    model.body_mass[env_ids, torso_body_id] += uniform_like(model.body_mass, (num_envs,), -1.0, 1.0)

    # qpos0 joint defaults: default + U(-0.05, 0.05).
    env_grid, qpos_grid = torch.meshgrid(env_ids, joint_q_ids, indexing="ij")
    model.qpos0[env_grid, qpos_grid] = default("qpos0")[joint_q_ids] + uniform_like(
        model.qpos0, (num_envs, len(joint_q_ids)), -0.05, 0.05
    )

    kp_scale = uniform_like(model.actuator_gainprm, (num_envs, 3), *GO1_KP_SCALE)
    kd_scale = uniform_like(model.actuator_biasprm, (num_envs, 3), *GO1_KD_SCALE)
    gain, bias = default("actuator_gainprm"), default("actuator_biasprm")
    for k, ids in enumerate(_go1_actuator_type_ids(env)):
        rows = env_ids[:, None].long()
        model.actuator_gainprm[rows, ids, 0] = gain[ids, 0] * kp_scale[:, k : k + 1]
        model.actuator_biasprm[rows, ids, 1] = bias[ids, 1] * kp_scale[:, k : k + 1]
        model.actuator_biasprm[rows, ids, 2] = bias[ids, 2] * kd_scale[:, k : k + 1]
    payload = uniform_like(model.body_mass, (num_envs,), *GO1_PAYLOAD)
    model.body_mass[env_ids, torso_body_id] += payload
    _go1_payload(env)[env_ids.long()] = payload


def _go1_env_cfg(mode: str, play: bool = False):
    cfg = unitree_go1_flat_env_cfg(play=play or mode == "visualize")

    if not play and mode in {"init", "pretrain", "finetune"}:
        cfg.scene.num_envs = 4096

    if mode in {"finetune", "visualize"}:
        if play or mode == "visualize":
            cfg.scene.num_envs = 10
        cfg.scene.env_spacing = 2.5
        cfg.observations["actor"].enable_corruption = False

    if mode == "visualize":
        cfg.events.pop("push_robot", None)
        cfg.commands["twist"].resampling_time_range = (2.0, 2.0)

    if mode in {"pretrain", "finetune", "visualize"}:
        add_mbpo_observations(cfg, _STATE_TERMS)

    if play:
        cfg.scene.num_envs = 1

    if not play and mode == "pretrain" and domain_randomization_enabled():
        cfg.events["qoed_pretrain_domain_randomization"] = EventTermCfg(mode="reset", func=_go1_pretrain_domain_randomize)
        add_privilege_observation(cfg, _go1_domain_randomization_vector)
    elif mode in {"finetune", "visualize"}:
        add_privilege_observation(cfg, _go1_domain_randomization_vector)

    return cfg


def _go1_privilege_ranges() -> list[tuple[float, float]]:
    return [
        (0.4, 1.0),
        *((mass * 0.9, mass * 1.1) for mass in _BODY_MASS_DEFAULTS),
        *((value * 0.9, value * 1.1) for value in _DOF_FRICTIONLOSS_DEFAULTS),
        *((value, value * 1.05) for value in _DOF_ARMATURE_DEFAULTS),
        *(GO1_KP_SCALE,) * 3,
        *(GO1_KD_SCALE,) * 3,
        GO1_PAYLOAD,
    ]


def _go1_robot() -> MbpoRobot:
    ranges = _go1_privilege_ranges()
    low, high = map(list, zip(*ranges))
    low[1] -= 1.0
    high[1] += 1.0
    means, variances = map(list, zip(*(uniform_mean_var(*bounds) for bounds in ranges)))
    variances[1] += uniform_mean_var(-1.0, 1.0)[1]
    return MbpoRobot(
        name="Unitree-Go1",
        experiment="go1_velocity",
        state_mean=_STATE_MEAN,
        state_std=_STATE_STD,
        state_idx=_STATE_IDX,
        action_dim=12,
        privilege=_go1_domain_randomization_vector,
        privilege_low=tuple(low),
        privilege_high=tuple(high),
        prior_mean=tuple(means),
        prior_var=tuple(variances),
    )


GO1 = _go1_robot()


class Go1MBPOOnPolicyRunner(MbpoRunner):
    robot = GO1


def _go1_agent_cfg(mode: str):
    return configure_agent_cfg(unitree_go1_ppo_runner_cfg(), mode, GO1)


register_tasks(GO1, _go1_env_cfg, _go1_agent_cfg, Go1MBPOOnPolicyRunner)
