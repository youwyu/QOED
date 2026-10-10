"""LEAP hand cube-rotation MBPO task for QOED."""

from __future__ import annotations

import math

import mujoco
import torch
from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mbpo.envs.leap_hand.utils import (
    CUBE_DROP_HEIGHT,
    CUBE_INIT_POS,
    CUBE_XML,
    LEAP_ARMATURE_SCALE,
    LEAP_CUBE_COM,
    LEAP_CUBE_MASS_SCALE,
    LEAP_DAMPING_SCALE,
    LEAP_FINGERS,
    LEAP_FRICTIONLOSS_SCALE,
    LEAP_HOME,
    LEAP_KP_SCALE,
    LEAP_PALM_SITE,
    LEAP_TIP_FRICTION,
    LEAP_TIPS,
    LEAP_XML,
    leap_ids,
)
from mbpo.rsl_rl.mbpo_tasks import (
    MbpoRobot,
    MbpoRunner,
    add_mbpo_observations,
    add_privilege_observation,
    configure_agent_cfg,
    domain_randomization_enabled,
    joint_effort,
    register_tasks,
    resolve_env_ids,
    uniform_like,
    uniform_mean_var,
)

_CUBE_MASS_DEFAULT = 0.108
_NUM_JOINTS = 16
_ROBOT = SceneEntityCfg("robot")

_STATE_IDX = {
    r"$q - q_0$\n$[rad]$": list(range(0, 16)),
    r"$\dot{q}$\n$[rad/s]$": list(range(16, 32)),
    r"$p_{palm} - p_{cube}$\n$[m]$": [32, 33, 34],
    r"$q_{cube}$\n$[1]$": [35, 36, 37, 38],
    r"$v_{cube}$\n$[m/s]$": [39, 40, 41],
    r"$\omega_{cube}$\n$[rad/s]$": [42, 43, 44],
}
_STATE_MEAN = (*([0.0] * 32), 0.0, 0.0, -0.02, *([0.0] * 10))
_STATE_STD = (*([0.3] * 16), *([3.0] * 16), 0.02, 0.02, 0.02, *([0.5] * 4), *([0.2] * 3), *([2.0] * 3))


def _palm_pos(env) -> torch.Tensor:
    robot = env.scene["robot"]
    return robot.data.site_pos_w[:, robot.site_names.index(LEAP_PALM_SITE)]


def cube_pos_error(env) -> torch.Tensor:
    return _palm_pos(env) - env.scene["cube"].data.root_link_pos_w


def cube_quat(env) -> torch.Tensor:
    return env.scene["cube"].data.root_link_quat_w


def cube_lin_vel(env) -> torch.Tensor:
    return env.scene["cube"].data.root_link_lin_vel_w


def cube_ang_vel(env) -> torch.Tensor:
    return env.scene["cube"].data.root_link_ang_vel_w


def fingertip_pos(env) -> torch.Tensor:
    robot = env.scene["robot"]
    ids = [robot.site_names.index(tip) for tip in LEAP_TIPS]
    return (robot.data.site_pos_w[:, ids] - _palm_pos(env)[:, None]).flatten(1)


def cube_dropped(env) -> torch.Tensor:
    return env.scene["cube"].data.root_link_pos_w[:, 2] < CUBE_DROP_HEIGHT


def cube_dropped_obs(env) -> torch.Tensor:
    return cube_dropped(env).float().unsqueeze(-1)


def cube_spin(env) -> torch.Tensor:
    return cube_ang_vel(env)[:, 2]


_STATE_TERMS = {
    "joint_pos": mdp.joint_pos_rel,
    "joint_vel": mdp.joint_vel_rel,
    "cube_pos_error": cube_pos_error,
    "cube_quat": cube_quat,
    "cube_lin_vel": cube_lin_vel,
    "cube_ang_vel": cube_ang_vel,
}


def _leap_cache(env) -> dict:
    if getattr(env, "qoed_leap_ids", None) is None:
        ids = leap_ids(env.sim.mj_model)
        env.qoed_leap_ids = {
            "tip": torch.as_tensor(ids["tip"], device=env.device),
            "cube": ids["cube"],
            "actuator": [torch.as_tensor(a, device=env.device) for a in ids["actuator"]],
            "dof": [torch.as_tensor(d, device=env.device) for d in ids["dof"]],
        }
    return env.qoed_leap_ids


def _leap_domain_randomization_vector(env) -> torch.Tensor:
    ids, model, default = _leap_cache(env), env.sim.model, env.sim.get_default_field
    gain = default("actuator_gainprm")
    cube = ids["cube"]

    def dof_scale(field):
        base = default(field)
        return torch.stack([(getattr(model, field)[:, d] / base[d]).mean(1) for d in ids["dof"]], dim=-1)

    return torch.cat(
        (
            model.geom_friction[:, ids["tip"][0], 0:1],
            model.body_mass[:, cube : cube + 1],
            model.body_ipos[:, cube],
            torch.stack([(model.actuator_gainprm[:, a, 0] / gain[a, 0]).mean(1) for a in ids["actuator"]], dim=-1),
            dof_scale("dof_damping"),
            dof_scale("dof_frictionloss"),
            dof_scale("dof_armature"),
        ),
        dim=-1,
    )


@requires_model_fields(
    "geom_friction",
    "body_mass",
    "body_inertia",
    "body_ipos",
    "actuator_gainprm",
    "actuator_biasprm",
    "dof_damping",
    "dof_frictionloss",
    "dof_armature",
    recompute=RecomputeLevel.set_const,
)
def _leap_pretrain_domain_randomize(env, env_ids: torch.Tensor | slice | None) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    rows, n = env_ids[:, None].long(), len(env_ids)
    ids, model, default = _leap_cache(env), env.sim.model, env.sim.get_default_field
    model.geom_friction[rows, ids["tip"], 0] = uniform_like(model.geom_friction, (n, 1), *LEAP_TIP_FRICTION)
    cube = ids["cube"]
    mass_scale = uniform_like(model.body_mass, (n,), *LEAP_CUBE_MASS_SCALE)
    model.body_mass[env_ids, cube] = default("body_mass")[cube] * mass_scale
    model.body_inertia[env_ids, cube] = default("body_inertia")[cube] * mass_scale[:, None]
    model.body_ipos[env_ids, cube] = default("body_ipos")[cube] + uniform_like(model.body_ipos, (n, 3), *LEAP_CUBE_COM)
    gain, bias = default("actuator_gainprm"), default("actuator_biasprm")
    kp, damping, frictionloss, armature = (
        uniform_like(model.dof_damping, (n, len(LEAP_FINGERS)), *scale) for scale in (LEAP_KP_SCALE, LEAP_DAMPING_SCALE, LEAP_FRICTIONLOSS_SCALE, LEAP_ARMATURE_SCALE)
    )
    for k, (actuators, dofs) in enumerate(zip(ids["actuator"], ids["dof"])):
        model.actuator_gainprm[rows, actuators, 0] = gain[actuators, 0] * kp[:, k : k + 1]
        model.actuator_biasprm[rows, actuators, 1] = bias[actuators, 1] * kp[:, k : k + 1]
        for field, scale in (("dof_damping", damping), ("dof_frictionloss", frictionloss), ("dof_armature", armature)):
            getattr(model, field)[rows, dofs] = default(field)[dofs] * scale[:, k : k + 1]


def _leap_base_env_cfg() -> ManagerBasedRlEnvCfg:
    actor_terms = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.05, n_max=0.05)),
        "actions": ObservationTermCfg(func=mdp.last_action),
    }
    critic_terms = {
        "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
        "joint_torque": ObservationTermCfg(func=joint_effort),
        "fingertip_pos": ObservationTermCfg(func=fingertip_pos),
        "cube_pos_error": ObservationTermCfg(func=cube_pos_error),
        "cube_quat": ObservationTermCfg(func=cube_quat),
        "cube_ang_vel": ObservationTermCfg(func=cube_ang_vel),
        "cube_lin_vel": ObservationTermCfg(func=cube_lin_vel),
    }
    robot = EntityCfg(
        init_state=EntityCfg.InitialStateCfg(joint_pos=LEAP_HOME),
        spec_fn=lambda: mujoco.MjSpec.from_file(str(LEAP_XML)),
        articulation=EntityArticulationInfoCfg(actuators=(XmlActuatorCfg(target_names_expr=(".*",)),)),
    )
    cube = EntityCfg(init_state=EntityCfg.InitialStateCfg(pos=CUBE_INIT_POS), spec_fn=lambda: mujoco.MjSpec.from_file(str(CUBE_XML)))
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(entities={"robot": robot, "cube": cube}, num_envs=1, env_spacing=0.5),
        observations={
            "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=True),
            "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
        },
        actions={"joint_pos": JointPositionActionCfg(entity_name="robot", actuator_names=(".*",), scale=0.6, use_default_offset=True)},
        events={
            "reset_hand": EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={"position_range": (-0.1, 0.1), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            ),
            "reset_cube": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={
                    "pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01), "z": (-0.01, 0.01), "roll": (-math.pi, math.pi), "pitch": (-math.pi, math.pi), "yaw": (-math.pi, math.pi)},
                    "velocity_range": {},
                    "asset_cfg": SceneEntityCfg("cube"),
                },
            ),
        },
        rewards={
            "cube_spin": RewardTermCfg(func=cube_spin, weight=1.0),
            "termination": RewardTermCfg(func=mdp.is_terminated, weight=-100.0),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "cube_dropped": TerminationTermCfg(func=cube_dropped),
        },
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="cube", body_name="cube", distance=0.6, elevation=-30.0, azimuth=120.0),
        sim=SimulationCfg(njmax=200, mujoco=MujocoCfg(timestep=0.01, integrator="euler", iterations=50, ls_iterations=50)),
        decimation=5,
        episode_length_s=25.0,
    )


def _leap_env_cfg(mode: str, play: bool = False):
    cfg = _leap_base_env_cfg()

    if not play and mode in {"init", "pretrain", "finetune"}:
        cfg.scene.num_envs = 4096

    if mode in {"finetune", "visualize"}:
        if play or mode == "visualize":
            cfg.scene.num_envs = 10
        cfg.observations["actor"].enable_corruption = False

    if mode in {"pretrain", "finetune", "visualize"}:
        add_mbpo_observations(cfg, _STATE_TERMS, termination=ObservationTermCfg(func=cube_dropped_obs))

    if play:
        cfg.scene.num_envs = 1
        cfg.observations["actor"].enable_corruption = False

    if not play and mode == "pretrain" and domain_randomization_enabled():
        cfg.events["qoed_pretrain_domain_randomization"] = EventTermCfg(mode="reset", func=_leap_pretrain_domain_randomize)
        add_privilege_observation(cfg, _leap_domain_randomization_vector)
    elif mode in {"finetune", "visualize"}:
        add_privilege_observation(cfg, _leap_domain_randomization_vector)

    return cfg


def _leap_robot() -> MbpoRobot:
    ranges = [
        LEAP_TIP_FRICTION,
        tuple(_CUBE_MASS_DEFAULT * s for s in LEAP_CUBE_MASS_SCALE),
        *(LEAP_CUBE_COM,) * 3,
        *(LEAP_KP_SCALE,) * len(LEAP_FINGERS),
        *(LEAP_DAMPING_SCALE,) * len(LEAP_FINGERS),
        *(LEAP_FRICTIONLOSS_SCALE,) * len(LEAP_FINGERS),
        *(LEAP_ARMATURE_SCALE,) * len(LEAP_FINGERS),
    ]
    low, high = zip(*ranges)
    means, variances = zip(*(uniform_mean_var(*bounds) for bounds in ranges))
    return MbpoRobot(
        name="Leap-Hand",
        task="Cube-RotateZ",
        experiment="leap_rotate_z",
        state_mean=_STATE_MEAN,
        state_std=_STATE_STD,
        state_idx=_STATE_IDX,
        action_dim=_NUM_JOINTS,
        privilege=_leap_domain_randomization_vector,
        privilege_low=low,
        privilege_high=high,
        prior_mean=means,
        prior_var=variances,
    )


LEAP = _leap_robot()


class LeapMBPOOnPolicyRunner(MbpoRunner):
    robot = LEAP


def _leap_agent_cfg(mode: str):
    cfg = unitree_go1_ppo_runner_cfg()
    cfg.clip_actions = 1.0
    cfg.algorithm.entropy_coef = 0.001
    cfg.algorithm.gamma = 0.97
    return configure_agent_cfg(cfg, mode, LEAP)


register_tasks(LEAP, _leap_env_cfg, _leap_agent_cfg, LeapMBPOOnPolicyRunner)
