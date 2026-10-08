"""Clearpath Jackal MBPO task for QOED."""

from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import torch
from mjlab.actuator import BuiltinVelocityActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointVelocityAction, JointVelocityActionCfg
from mjlab.managers import EventTermCfg, ObservationGroupCfg, ObservationTermCfg, RewardTermCfg, SceneEntityCfg, TerminationTermCfg
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from mbpo.envs.jackal.utils import (
    JACKAL_ARMATURE_SCALE,
    JACKAL_DAMPING_SCALE,
    JACKAL_EFFORT_LIMIT,
    JACKAL_FRICTION,
    JACKAL_FRICTIONLOSS_SCALE,
    JACKAL_KV,
    JACKAL_KV_SCALE,
    JACKAL_MASS_SCALE,
    JACKAL_MAX_WHEEL_SPEED,
    JACKAL_PAYLOAD,
    JACKAL_XML,
    diff_drive_mix,
    jackal_ids,
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

_BODY_MASS_DEFAULTS = (4.548, 16.523, 0.477, 0.477, 0.477, 0.477)
_DOF_FRICTIONLOSS_DEFAULT = 0.1
_DOF_ARMATURE_DEFAULT = 0.05
_DOF_DAMPING_DEFAULT = 1.0

_STATE_IDX = {
    r"$v_b$\n$[m/s]$": [0, 1, 2],
    r"$\omega_b$\n$[rad/s]$": [3, 4, 5],
    r"$g_b$\n$[1]$": [6, 7, 8],
    r"$\dot{q}$\n$[rad/s]$": list(range(9, 13)),
}
_STATE_MEAN = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, *([0.0] * 4))
_STATE_STD = (0.7, 0.1, 0.05, 0.2, 0.2, 0.7, 0.02, 0.02, 0.01, *([10.0] * 4))
_STATE_TERMS = {
    "base_lin_vel": mdp.base_lin_vel,
    "base_ang_vel": mdp.base_ang_vel,
    "projected_gravity": mdp.projected_gravity,
    "joint_vel": mdp.joint_vel_rel,
}


@dataclass(kw_only=True)
class DiffDriveActionCfg(JointVelocityActionCfg):
    use_default_offset: bool = False

    def build(self, env):
        return DiffDriveAction(self, env)


class DiffDriveAction(JointVelocityAction):
    """Maps (forward speed, yaw rate) to skid-steer wheel velocity targets."""

    def __init__(self, cfg: DiffDriveActionCfg, env):
        super().__init__(cfg, env)
        self._mix = torch.as_tensor(diff_drive_mix(self._target_names), device=self.device, dtype=torch.float32)
        self._action_dim = 2
        self._raw_actions = torch.zeros(self.num_envs, 2, device=self.device)

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions = (actions @ self._mix).clamp(-JACKAL_MAX_WHEEL_SPEED, JACKAL_MAX_WHEEL_SPEED)


def _jackal_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=EntityCfg.InitialStateCfg(pos=(0.0, 0.0, 0.07)),
        spec_fn=lambda: mujoco.MjSpec.from_file(str(JACKAL_XML)),
        articulation=EntityArticulationInfoCfg(
            actuators=(BuiltinVelocityActuatorCfg(target_names_expr=(".*_wheel",), damping=JACKAL_KV, effort_limit=JACKAL_EFFORT_LIMIT),)
        ),
    )


def _jackal_cache(env) -> dict[str, torch.Tensor]:
    if getattr(env, "qoed_jackal_ids", None) is None:
        ids = {key: torch.as_tensor(value, device=env.device) for key, value in jackal_ids(env.sim.mj_model).items()}
        ids["body"] = env.scene["robot"].indexing.body_ids
        ids["payload"] = torch.zeros(env.num_envs, device=env.device)
        env.qoed_jackal_ids = ids
    return env.qoed_jackal_ids


def _jackal_domain_randomization_vector(env) -> torch.Tensor:
    ids, model = _jackal_cache(env), env.sim.model
    payload, dof = ids["payload"], ids["dof"]
    body_mass = model.body_mass[:, ids["body"]] - payload[:, None] * (ids["body"] == ids["base"])
    return torch.cat(
        (
            model.geom_friction[:, ids["geom"], 0],
            body_mass,
            model.dof_frictionloss[:, dof],
            model.dof_armature[:, dof],
            model.dof_damping[:, dof],
            model.actuator_gainprm[:, ids["actuator"], 0],
            payload[:, None],
        ),
        dim=-1,
    )


@requires_model_fields(
    "geom_friction",
    "body_mass",
    "dof_frictionloss",
    "dof_armature",
    "dof_damping",
    "actuator_gainprm",
    "actuator_biasprm",
    recompute=RecomputeLevel.set_const,
)
def _jackal_pretrain_domain_randomize(env, env_ids: torch.Tensor | slice | None) -> None:
    env_ids = resolve_env_ids(env, env_ids)
    rows, n = env_ids[:, None].long(), len(env_ids)
    ids, model, default = _jackal_cache(env), env.sim.model, env.sim.get_default_field
    geom, dof, actuator, body = ids["geom"], ids["dof"], ids["actuator"], ids["body"]
    model.geom_friction[rows, geom, 0] = uniform_like(model.geom_friction, (n, len(geom)), *JACKAL_FRICTION)
    model.body_mass[rows, body] = default("body_mass")[body] * uniform_like(model.body_mass, (n, len(body)), *JACKAL_MASS_SCALE)
    for field, scale in (("dof_frictionloss", JACKAL_FRICTIONLOSS_SCALE), ("dof_armature", JACKAL_ARMATURE_SCALE), ("dof_damping", JACKAL_DAMPING_SCALE)):
        values = getattr(model, field)
        values[rows, dof] = default(field)[dof] * uniform_like(values, (n, len(dof)), *scale)
    kv_scale = uniform_like(model.actuator_gainprm, (n, len(actuator)), *JACKAL_KV_SCALE)
    model.actuator_gainprm[rows, actuator, 0] = default("actuator_gainprm")[actuator, 0] * kv_scale
    model.actuator_biasprm[rows, actuator, 2] = default("actuator_biasprm")[actuator, 2] * kv_scale
    payload = uniform_like(model.body_mass, (n,), *JACKAL_PAYLOAD)
    model.body_mass[env_ids, ids["base"]] += payload
    ids["payload"][env_ids.long()] = payload


def _jackal_base_env_cfg() -> ManagerBasedRlEnvCfg:
    actor_terms = {
        "base_lin_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_lin_vel"}, noise=Unoise(n_min=-0.1, n_max=0.1)),
        "base_ang_vel": ObservationTermCfg(func=mdp.builtin_sensor, params={"sensor_name": "robot/imu_ang_vel"}, noise=Unoise(n_min=-0.1, n_max=0.1)),
        "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05)),
        "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5)),
        "actions": ObservationTermCfg(func=mdp.last_action),
        "command": ObservationTermCfg(func=mdp.generated_commands, params={"command_name": "twist"}),
    }
    critic_terms = {name: ObservationTermCfg(func=term.func, params=term.params) for name, term in actor_terms.items()}
    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), entities={"robot": _jackal_robot_cfg()}, num_envs=1, extent=2.0),
        observations={
            "actor": ObservationGroupCfg(terms=actor_terms, concatenate_terms=True, enable_corruption=True),
            "critic": ObservationGroupCfg(terms=critic_terms, concatenate_terms=True, enable_corruption=False),
        },
        actions={"wheel_vel": DiffDriveActionCfg(entity_name="robot", actuator_names=(".*_wheel",))},
        commands={
            "twist": UniformVelocityCommandCfg(
                entity_name="robot",
                resampling_time_range=(3.0, 8.0),
                rel_standing_envs=0.1,
                rel_heading_envs=0.0,
                heading_command=False,
                debug_vis=True,
                ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(-1.0, 1.5), lin_vel_y=(0.0, 0.0), ang_vel_z=(-1.0, 1.0)),
            )
        },
        events={
            "reset_base": EventTermCfg(
                func=mdp.reset_root_state_uniform,
                mode="reset",
                params={"pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)}, "velocity_range": {}},
            ),
            "reset_robot_joints": EventTermCfg(
                func=mdp.reset_joints_by_offset,
                mode="reset",
                params={"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0), "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
            ),
        },
        rewards={
            "track_linear_velocity": RewardTermCfg(func=mdp.track_linear_velocity, weight=2.0, params={"command_name": "twist", "std": math.sqrt(0.25)}),
            "track_angular_velocity": RewardTermCfg(func=mdp.track_angular_velocity, weight=2.0, params={"command_name": "twist", "std": math.sqrt(0.5)}),
            "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1),
        },
        terminations={
            "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
            "fell_over": TerminationTermCfg(func=mdp.bad_orientation, params={"limit_angle": math.radians(70.0)}),
        },
        metrics={"mean_action_acc": MetricsTermCfg(func=mdp.mean_action_acc)},
        viewer=ViewerConfig(origin_type=ViewerConfig.OriginType.ASSET_BODY, entity_name="robot", body_name="base_link", distance=2.5, elevation=-10.0, azimuth=90.0),
        sim=SimulationCfg(njmax=300, mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20)),
        decimation=4,
        episode_length_s=20.0,
    )


def _jackal_env_cfg(mode: str, play: bool = False):
    cfg = _jackal_base_env_cfg()

    if not play and mode in {"init", "pretrain", "finetune"}:
        cfg.scene.num_envs = 4096

    if mode in {"finetune", "visualize"}:
        if play or mode == "visualize":
            cfg.scene.num_envs = 10
        cfg.scene.env_spacing = 2.5
        cfg.observations["actor"].enable_corruption = False

    if mode == "visualize":
        cfg.commands["twist"].resampling_time_range = (2.0, 2.0)

    if mode in {"pretrain", "finetune", "visualize"}:
        add_mbpo_observations(cfg, _STATE_TERMS)

    if play:
        cfg.scene.num_envs = 1
        cfg.observations["actor"].enable_corruption = False

    if not play and mode == "pretrain" and domain_randomization_enabled():
        cfg.events["qoed_pretrain_domain_randomization"] = EventTermCfg(mode="reset", func=_jackal_pretrain_domain_randomize)
        add_privilege_observation(cfg, _jackal_domain_randomization_vector)
    elif mode in {"finetune", "visualize"}:
        add_privilege_observation(cfg, _jackal_domain_randomization_vector)

    return cfg


def _jackal_privilege_ranges() -> list[tuple[float, float]]:
    def scaled(value, scale):
        return (value * scale[0], value * scale[1])

    return [
        *(JACKAL_FRICTION,) * 4,
        *(scaled(mass, JACKAL_MASS_SCALE) for mass in _BODY_MASS_DEFAULTS),
        *(scaled(_DOF_FRICTIONLOSS_DEFAULT, JACKAL_FRICTIONLOSS_SCALE),) * 4,
        *(scaled(_DOF_ARMATURE_DEFAULT, JACKAL_ARMATURE_SCALE),) * 4,
        *(scaled(_DOF_DAMPING_DEFAULT, JACKAL_DAMPING_SCALE),) * 4,
        *(scaled(JACKAL_KV, JACKAL_KV_SCALE),) * 4,
        JACKAL_PAYLOAD,
    ]


def _jackal_robot() -> MbpoRobot:
    ranges = _jackal_privilege_ranges()
    low, high = zip(*ranges)
    means, variances = zip(*(uniform_mean_var(*bounds) for bounds in ranges))
    return MbpoRobot(
        name="Clearpath-Jackal",
        experiment="jackal_velocity",
        state_mean=_STATE_MEAN,
        state_std=_STATE_STD,
        state_idx=_STATE_IDX,
        action_dim=2,
        privilege=_jackal_domain_randomization_vector,
        privilege_low=low,
        privilege_high=high,
        prior_mean=means,
        prior_var=variances,
    )


JACKAL = _jackal_robot()


class JackalMBPOOnPolicyRunner(MbpoRunner):
    robot = JACKAL


def _jackal_agent_cfg(mode: str):
    return configure_agent_cfg(unitree_go1_ppo_runner_cfg(), mode, JACKAL)


register_tasks(JACKAL, _jackal_env_cfg, _jackal_agent_cfg, JackalMBPOOnPolicyRunner)
