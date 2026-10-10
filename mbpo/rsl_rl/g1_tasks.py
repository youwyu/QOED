"""Unitree G1 MBPO task for QOED."""

from __future__ import annotations

import torch
from mjlab.envs.mdp import observations as obs_mdp
from mjlab.managers import EventTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_flat_env_cfg
from mjlab.tasks.velocity.config.g1.rl_cfg import unitree_g1_ppo_runner_cfg

from mbpo.envs.g1.utils import (
    G1_ARMATURE_SCALE,
    G1_FOOT_FRICTION,
    G1_KD_SCALE,
    G1_KP_SCALE,
    G1_PAYLOAD,
    G1_TORSO_MASS_SCALE,
    g1_ids,
)
from mbpo.rsl_rl.mbpo_tasks import (
    MbpoRobot,
    MbpoRunner,
    add_mbpo_observations,
    add_privilege_observation,
    configure_agent_cfg,
    domain_randomization_enabled,
    register_tasks,
    resolve_env_ids,
    uniform_like,
    uniform_mean_var,
)

_TORSO_MASS_DEFAULT = 7.818
_NUM_GROUPS = 6
_NUM_JOINTS = 29

_STATE_IDX = {
    r"$v_b$\n$[m/s]$": [0, 1, 2],
    r"$\omega_b$\n$[rad/s]$": [3, 4, 5],
    r"$g_b$\n$[1]$": [6, 7, 8],
    r"$q - q_0$\n$[rad]$": list(range(9, 9 + _NUM_JOINTS)),
    r"$\dot{q}$\n$[rad/s]$": list(range(9 + _NUM_JOINTS, 9 + 2 * _NUM_JOINTS)),
}
_STATE_MEAN = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, *([0.0] * (2 * _NUM_JOINTS)))
_STATE_STD = (0.5, 0.5, 0.1, 0.3, 0.3, 0.5, 0.02, 0.02, 0.04, *([0.2] * _NUM_JOINTS), *([1.5] * _NUM_JOINTS))
_STATE_TERMS = {
    "base_lin_vel": obs_mdp.base_lin_vel,
    "base_ang_vel": obs_mdp.base_ang_vel,
    "projected_gravity": obs_mdp.projected_gravity,
    "joint_pos": obs_mdp.joint_pos_rel,
    "joint_vel": obs_mdp.joint_vel_rel,
}


def _g1_cache(env) -> dict:
    if getattr(env, "qoed_g1_ids", None) is None:
        ids = g1_ids(env.sim.mj_model)
        env.qoed_g1_ids = {
            "foot": torch.as_tensor(ids["foot"], device=env.device),
            "torso": ids["torso"],
            "actuator": [torch.as_tensor(a, device=env.device) for a in ids["actuator"]],
            "dof": [torch.as_tensor(d, device=env.device) for d in ids["dof"]],
            "payload": torch.zeros(env.num_envs, device=env.device),
        }
    return env.qoed_g1_ids


def _g1_domain_randomization_vector(env) -> torch.Tensor:
    ids, model, default = _g1_cache(env), env.sim.model, env.sim.get_default_field
    gain, bias, armature = default("actuator_gainprm"), default("actuator_biasprm"), default("dof_armature")
    payload = ids["payload"]
    return torch.cat(
        (
            model.geom_friction[:, ids["foot"][0], 0:1],
            (model.body_mass[:, ids["torso"]] - payload)[:, None],
            torch.stack([(model.actuator_gainprm[:, a, 0] / gain[a, 0]).mean(1) for a in ids["actuator"]], dim=-1),
            torch.stack([(model.actuator_biasprm[:, a, 2] / bias[a, 2]).mean(1) for a in ids["actuator"]], dim=-1),
            torch.stack([(model.dof_armature[:, d] / armature[d]).mean(1) for d in ids["dof"]], dim=-1),
            payload[:, None],
        ),
        dim=-1,
    )


@requires_model_fields("geom_friction", "body_mass", "actuator_gainprm", "actuator_biasprm", "dof_armature", recompute=RecomputeLevel.set_const)
def _g1_pretrain_domain_randomize(env, env_ids: torch.Tensor | slice | None) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    rows, n = env_ids[:, None].long(), len(env_ids)
    ids, model, default = _g1_cache(env), env.sim.model, env.sim.get_default_field
    model.geom_friction[rows, ids["foot"], 0] = uniform_like(model.geom_friction, (n, 1), *G1_FOOT_FRICTION)
    torso = ids["torso"]
    model.body_mass[env_ids, torso] = default("body_mass")[torso] * uniform_like(model.body_mass, (n,), *G1_TORSO_MASS_SCALE)
    gain, bias, armature = default("actuator_gainprm"), default("actuator_biasprm"), default("dof_armature")
    kp, kd, arm = (uniform_like(model.actuator_gainprm, (n, _NUM_GROUPS), *scale) for scale in (G1_KP_SCALE, G1_KD_SCALE, G1_ARMATURE_SCALE))
    for k, (actuators, dofs) in enumerate(zip(ids["actuator"], ids["dof"])):
        model.actuator_gainprm[rows, actuators, 0] = gain[actuators, 0] * kp[:, k : k + 1]
        model.actuator_biasprm[rows, actuators, 1] = bias[actuators, 1] * kp[:, k : k + 1]
        model.actuator_biasprm[rows, actuators, 2] = bias[actuators, 2] * kd[:, k : k + 1]
        model.dof_armature[rows, dofs] = armature[dofs] * arm[:, k : k + 1]
    payload = uniform_like(model.body_mass, (n,), *G1_PAYLOAD)
    model.body_mass[env_ids, torso] += payload
    ids["payload"][env_ids.long()] = payload


def _g1_env_cfg(mode: str, play: bool = False):
    cfg = unitree_g1_flat_env_cfg(play=play or mode == "visualize")
    for name in ("foot_friction", "encoder_bias", "base_com"):
        cfg.events.pop(name, None)

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
        cfg.events["qoed_pretrain_domain_randomization"] = EventTermCfg(mode="reset", func=_g1_pretrain_domain_randomize)
        add_privilege_observation(cfg, _g1_domain_randomization_vector)
    elif mode in {"finetune", "visualize"}:
        add_privilege_observation(cfg, _g1_domain_randomization_vector)

    return cfg


def _g1_robot() -> MbpoRobot:
    ranges = [
        G1_FOOT_FRICTION,
        tuple(_TORSO_MASS_DEFAULT * s for s in G1_TORSO_MASS_SCALE),
        *(G1_KP_SCALE,) * _NUM_GROUPS,
        *(G1_KD_SCALE,) * _NUM_GROUPS,
        *(G1_ARMATURE_SCALE,) * _NUM_GROUPS,
        G1_PAYLOAD,
    ]
    low, high = zip(*ranges)
    means, variances = zip(*(uniform_mean_var(*bounds) for bounds in ranges))
    return MbpoRobot(
        name="Unitree-G1",
        experiment="g1_velocity",
        state_mean=_STATE_MEAN,
        state_std=_STATE_STD,
        state_idx=_STATE_IDX,
        action_dim=_NUM_JOINTS,
        privilege=_g1_domain_randomization_vector,
        privilege_low=low,
        privilege_high=high,
        prior_mean=means,
        prior_var=variances,
        pretrain_iterations=4000,
    )


G1 = _g1_robot()


class G1MBPOOnPolicyRunner(MbpoRunner):
    robot = G1


def _g1_agent_cfg(mode: str):
    return configure_agent_cfg(unitree_g1_ppo_runner_cfg(), mode, G1)


register_tasks(G1, _g1_env_cfg, _g1_agent_cfg, G1MBPOOnPolicyRunner)
