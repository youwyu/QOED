"""Local MJLab task aliases for QOED Go1 workflows."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path

import torch
from mjlab.envs.mdp import observations as obs_mdp
from mjlab.envs.mdp import terminations as term_mdp
from mjlab.managers import (
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    SceneEntityCfg,
    TerminationTermCfg,
)
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_flat_env_cfg
from mjlab.tasks.velocity.config.go1.rl_cfg import unitree_go1_ppo_runner_cfg
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from rsl_rl.runners import MBPOOnPolicyRunner

from mbpo.envs.go1.utils import GO1_KD_SCALE, GO1_KP_SCALE, GO1_PAYLOAD, mujoco_actuator_ids_by_joint_type

GO1_MODE_TASKS = {
    "init": "QOED-Mjlab-Velocity-Flat-Unitree-Go1-Init-v0",
    "pretrain": "QOED-Mjlab-Velocity-Flat-Unitree-Go1-Pretrain-v0",
    "finetune": "QOED-Mjlab-Velocity-Flat-Unitree-Go1-Finetune-v0",
    "visualize": "QOED-Mjlab-Velocity-Flat-Unitree-Go1-Visualize-v0",
}
GO1_SYSTEM_PRIVILEGE_OBS_KEY = "system_privilege"
GO1_FINETUNE_NUM_STEPS_PER_ENV = 100
GO1_POLICY_NOISE_STD_TYPE = "log"
GO1_INFO_GAIN_DEFAULT_NUM_ACTIONS = 1024

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
_STATE_MEAN = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, *([0.0] * 36)]
_STATE_STD = [
    0.5, 0.5, 0.1,
    0.3, 0.3, 0.5,
    0.02, 0.02, 0.04,
    *[0.3, 0.3, 0.6] * 4,
    *[1.0, 1.5, 2.5] * 4,
    *[23.7, 23.7, 35.55] * 4,
]
_SYSTEM_DYNAMICS = {
    "ensemble_size": 1,
    "history_horizon": 32,
    "architecture_config": {
        "type": "shortcut",
        "latent_dim": 128,
        "timestep_embed_dim": 64,
        "depth": 6,
        "num_heads": 4,
        "mlp_ratio": 4.0,
        "num_registers": 8,
        "num_steps": 1,
        "use_shortcut": True,
        "shortcut_self_consistency": 0.25,
        "shortcut_min_dt": 0.0078125,
        "privilege_source_noise": True,
        "privilege_dropout_prob": 0.1,
        "privilege_availability_feature": True,
    },
    "freeze_auxiliary": False,
}


def latest_go1_pretrain_checkpoint() -> Path | None:
    roots = {Path.cwd(), Path(__file__).resolve().parents[2]}
    checkpoints = [path for root in roots for path in (root / "logs/rsl_rl/go1_velocity").glob("*_pretrain/model_*.pt")]
    if not checkpoints:
        return None

    def sort_key(path: Path):
        match = re.search(r"model_(\d+)\.pt$", path.name)
        return path.parent.name, int(match.group(1)) if match else -1, path.stat().st_mtime

    nonzero = [path for path in checkpoints if re.search(r"model_([1-9]\d*)\.pt$", path.name)]
    return max(nonzero or checkpoints, key=sort_key)


def _uniform_mean_var(low: float, high: float) -> tuple[float, float]:
    return 0.5 * (low + high), ((high - low) ** 2) / 12.0


def _joint_effort(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    return asset.data.actuator_force[:, asset_cfg.actuator_ids]


def _bad_orientation(env, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    return term_mdp.bad_orientation(env, limit_angle=limit_angle, asset_cfg=asset_cfg).float().unsqueeze(-1)


def _named_global_id(names: tuple[str, ...], ids: torch.Tensor, name: str) -> torch.Tensor:
    return ids[names.index(name)] if name in names else ids[0]


def _uniform_like(reference: torch.Tensor, shape: tuple[int, ...], low: float, high: float) -> torch.Tensor:
    return torch.empty(shape, device=reference.device, dtype=reference.dtype).uniform_(low, high)


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
    floor_geom_id = _named_global_id(terrain.geom_names, terrain.indexing.geom_ids, "terrain")
    torso_body_id = _named_global_id(robot.body_names, robot.indexing.body_ids, "trunk")
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


def _go1_fisher_parameter_bounds() -> tuple[list[float], list[float]]:
    low, high = map(list, zip(*_go1_privilege_ranges()))
    low[1] -= 1.0
    high[1] += 1.0
    return low, high


def _go1_fisher_parameter_prior() -> tuple[list[float], list[float]]:
    means, variances = map(list, zip(*(_uniform_mean_var(low, high) for low, high in _go1_privilege_ranges())))
    variances[1] += _uniform_mean_var(-1.0, 1.0)[1]
    return means, variances


def _add_go1_domain_randomization_privilege_observation(cfg) -> None:
    cfg.observations[GO1_SYSTEM_PRIVILEGE_OBS_KEY] = ObservationGroupCfg(
        terms={"domain_randomization": ObservationTermCfg(func=_go1_domain_randomization_vector)},
        enable_corruption=False,
        concatenate_terms=True,
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
    all_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    if env_ids is None:
        env_ids = all_ids
    elif isinstance(env_ids, slice):
        env_ids = all_ids[env_ids]
    else:
        env_ids = env_ids.to(env.device, dtype=torch.int)
    num_envs = len(env_ids)

    robot, terrain, model = env.scene["robot"], env.scene["terrain"], env.sim.model
    floor_geom_id = _named_global_id(terrain.geom_names, terrain.indexing.geom_ids, "terrain")
    torso_body_id = _named_global_id(robot.body_names, robot.indexing.body_ids, "trunk")
    body_ids = robot.indexing.body_ids
    joint_q_ids = robot.indexing.joint_q_adr
    joint_v_ids = robot.indexing.joint_v_adr
    default = env.sim.get_default_field

    # Match mujoco_playground Go1 randomize.py.
    model.geom_friction[env_ids, floor_geom_id] = default("geom_friction")[floor_geom_id]
    model.geom_friction[env_ids, floor_geom_id, 0] = _uniform_like(model.geom_friction, (num_envs,), 0.4, 1.0)

    # Joint friction loss: default * U(0.9, 1.1).
    env_grid, dof_grid = torch.meshgrid(env_ids, joint_v_ids, indexing="ij")
    frictionloss_scale = _uniform_like(model.dof_frictionloss, (num_envs, len(joint_v_ids)), 0.9, 1.1)
    model.dof_frictionloss[env_grid, dof_grid] = default("dof_frictionloss")[joint_v_ids] * frictionloss_scale

    # Armature: default * U(1.0, 1.05).
    armature_scale = _uniform_like(model.dof_armature, (num_envs, len(joint_v_ids)), 1.0, 1.05)
    model.dof_armature[env_grid, dof_grid] = default("dof_armature")[joint_v_ids] * armature_scale

    # Torso COM position: default + U(-0.05, 0.05).
    model.body_ipos[env_ids, torso_body_id] = default("body_ipos")[torso_body_id] + _uniform_like(
        model.body_ipos, (num_envs, 3), -0.05, 0.05
    )

    # Link masses: default * U(0.9, 1.1), plus torso mass U(-1.0, 1.0).
    env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
    mass_scale = _uniform_like(model.body_mass, (num_envs, len(body_ids)), 0.9, 1.1)
    model.body_mass[env_grid, body_grid] = default("body_mass")[body_ids] * mass_scale
    model.body_mass[env_ids, torso_body_id] += _uniform_like(model.body_mass, (num_envs,), -1.0, 1.0)

    # qpos0 joint defaults: default + U(-0.05, 0.05).
    env_grid, qpos_grid = torch.meshgrid(env_ids, joint_q_ids, indexing="ij")
    model.qpos0[env_grid, qpos_grid] = default("qpos0")[joint_q_ids] + _uniform_like(
        model.qpos0, (num_envs, len(joint_q_ids)), -0.05, 0.05
    )

    kp_scale = _uniform_like(model.actuator_gainprm, (num_envs, 3), *GO1_KP_SCALE)
    kd_scale = _uniform_like(model.actuator_biasprm, (num_envs, 3), *GO1_KD_SCALE)
    gain, bias = default("actuator_gainprm"), default("actuator_biasprm")
    for k, ids in enumerate(_go1_actuator_type_ids(env)):
        rows = env_ids[:, None].long()
        model.actuator_gainprm[rows, ids, 0] = gain[ids, 0] * kp_scale[:, k : k + 1]
        model.actuator_biasprm[rows, ids, 1] = bias[ids, 1] * kp_scale[:, k : k + 1]
        model.actuator_biasprm[rows, ids, 2] = bias[ids, 2] * kd_scale[:, k : k + 1]
    payload = _uniform_like(model.body_mass, (num_envs,), *GO1_PAYLOAD)
    model.body_mass[env_ids, torso_body_id] += payload
    _go1_payload(env)[env_ids.long()] = payload


def _add_mbpo_observations(cfg) -> None:
    def group(**terms):
        return ObservationGroupCfg(terms=terms, enable_corruption=False, concatenate_terms=True)

    cfg.observations.setdefault("policy", deepcopy(cfg.observations["actor"]))
    cfg.observations["system_state"] = group(
        base_lin_vel=ObservationTermCfg(func=obs_mdp.base_lin_vel),
        base_ang_vel=ObservationTermCfg(func=obs_mdp.base_ang_vel),
        projected_gravity=ObservationTermCfg(func=obs_mdp.projected_gravity),
        joint_pos=ObservationTermCfg(func=obs_mdp.joint_pos_rel),
        joint_vel=ObservationTermCfg(func=obs_mdp.joint_vel_rel),
        joint_torque=ObservationTermCfg(func=_joint_effort),
    )
    cfg.observations["system_action"] = group(actions=ObservationTermCfg(func=obs_mdp.last_action))
    if "foot_contact" in cfg.observations["critic"].terms:
        cfg.observations["system_contact"] = group(foot_contact=deepcopy(cfg.observations["critic"].terms["foot_contact"]))
    fell_over_cfg = cfg.terminations.get("fell_over")
    fell_over_params = deepcopy(fell_over_cfg.params) if fell_over_cfg is not None else {"limit_angle": 1.2217304763960306}
    cfg.observations["system_termination"] = group(
        fell_over=ObservationTermCfg(func=_bad_orientation, params=fell_over_params)
    )


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
        _add_mbpo_observations(cfg)

    if play:
        cfg.scene.num_envs = 1

    if not play and mode == "pretrain" and os.environ.get("QOED_GO1_DOMAIN_RANDOMIZATION") == "1":
        cfg.events["qoed_pretrain_domain_randomization"] = EventTermCfg(mode="reset", func=_go1_pretrain_domain_randomize)
        _add_go1_domain_randomization_privilege_observation(cfg)
    elif mode in {"finetune", "visualize"}:
        _add_go1_domain_randomization_privilege_observation(cfg)

    return cfg


def normalize_mjlab_rsl_rl_cfg(train_cfg: dict, mbpo: bool = False) -> dict:
    train_cfg = deepcopy(train_cfg)

    obs_groups = train_cfg.get("obs_groups")
    if isinstance(obs_groups, dict) and "policy" not in obs_groups and "actor" in obs_groups:
        obs_groups["policy"] = obs_groups["actor"]

    if "policy" not in train_cfg and "actor" in train_cfg and "critic" in train_cfg:
        actor, critic = train_cfg["actor"], train_cfg["critic"]
        distribution = actor.get("distribution_cfg") or {}
        recurrent = actor.get("rnn_type") is not None or actor.get("class_name") == "RNNModel"
        noise_std_type = distribution.get("std_type", "scalar")
        policy = {
            "class_name": "ActorCriticRecurrent" if recurrent else "ActorCritic",
            "actor_hidden_dims": actor.get("hidden_dims", [512, 256, 128]),
            "critic_hidden_dims": critic.get("hidden_dims", [512, 256, 128]),
            "activation": actor.get("activation", "elu"),
            "actor_obs_normalization": actor.get("obs_normalization", False),
            "critic_obs_normalization": critic.get("obs_normalization", False),
            "init_noise_std": distribution.get("init_std", 1.0),
            "noise_std_type": GO1_POLICY_NOISE_STD_TYPE if noise_std_type == "scalar" else noise_std_type,
        }
        if recurrent:
            policy["rnn_type"] = actor.get("rnn_type", "lstm")
            policy["rnn_hidden_dim"] = actor.get("rnn_hidden_dim", 256)
            policy["rnn_num_layers"] = actor.get("rnn_num_layers", 1)
        train_cfg["policy"] = policy

    algorithm = train_cfg.get("algorithm")
    if isinstance(algorithm, dict):
        algorithm.pop("optimizer", None)
        algorithm.pop("share_cnn_encoders", None)
        if mbpo:
            algorithm["class_name"] = "MBPOPPO"
            algorithm["policy_learning_rate"] = algorithm.pop("learning_rate", algorithm.get("policy_learning_rate", 1.0e-3))
            defaults = {
                "system_dynamics_learning_rate": 1.0e-3,
                "system_dynamics_weight_decay": 0.0,
                "system_dynamics_forecast_horizon": 8,
                "system_dynamics_loss_weights": {
                    "state": 1.0,
                    "sequence": 1.0,
                    "bound": 1.0,
                    "kl": 0.1,
                    "extension": 1.0,
                    "contact": 1.0,
                    "termination": 1.0,
                },
                "system_dynamics_num_mini_batches": 20,
                "system_dynamics_mini_batch_size": 1024,
                "system_dynamics_replay_buffer_size": 1000,
                "system_dynamics_num_eval_trajectories": 1,
                "system_dynamics_len_eval_trajectory": 400,
                "system_dynamics_eval_traj_noise_scale": [0.1, 0.2, 0.4, 0.5, 0.8],
            }
            for key, value in defaults.items():
                algorithm.setdefault(key, value)
    return train_cfg


def _migrate_policy_noise_std_state_dict(policy, state_dict: dict) -> dict:
    state_dict = dict(state_dict)
    policy_state = policy.state_dict()
    if "log_std" in policy_state and "log_std" not in state_dict and "std" in state_dict:
        state_dict["log_std"] = state_dict.pop("std").clamp_min(1.0e-6).log()
    elif "std" in policy_state and "std" not in state_dict and "log_std" in state_dict:
        state_dict["std"] = state_dict.pop("log_std").exp()
    return state_dict


def _patch_policy_distribution_safety(policy) -> None:
    original_update_distribution = policy.update_distribution

    def safe_update_distribution(obs):
        with torch.no_grad():
            if hasattr(policy, "log_std"):
                policy.log_std.nan_to_num_(nan=0.0, posinf=2.0, neginf=-5.0).clamp_(-5.0, 2.0)
            if hasattr(policy, "std"):
                policy.std.nan_to_num_(nan=1.0, posinf=7.389, neginf=1.0e-3).clamp_(1.0e-3, 7.389)
        original_update_distribution(obs)
        scale = policy.distribution.scale
        if (not torch.isfinite(scale).all()) or (scale < 0.0).any():
            safe_scale = scale.nan_to_num(nan=1.0, posinf=7.389, neginf=1.0e-3).clamp_min(1.0e-3)
            policy.distribution = torch.distributions.Normal(policy.distribution.loc, safe_scale)

    policy.update_distribution = safe_update_distribution


def _add_go1_mbpo_cfg(train_cfg: dict, mode: str, supports_imagination: bool) -> dict:
    train_cfg["system_dynamics"] = deepcopy(_SYSTEM_DYNAMICS)
    fisher_param_min, fisher_param_max = _go1_fisher_parameter_bounds()
    fisher_prior_mean, fisher_prior_cov_diag = _go1_fisher_parameter_prior()
    algorithm = train_cfg.setdefault("algorithm", {})
    algorithm.update(
        fisher_param_min=fisher_param_min,
        fisher_param_max=fisher_param_max,
        fisher_prior_mean=fisher_prior_mean,
        fisher_prior_cov_diag=fisher_prior_cov_diag,
        fisher_var_threshold_for_update=1.0e-12,
        fisher_fd_delta_floor=0.1,
        fisher_obs_noise_std=0.1,
        info_gain_mode=os.environ.get("QOED_GO1_INFO_GAIN", "nothing") if mode == "finetune" else "nothing",
        info_gain_num_actions=GO1_INFO_GAIN_DEFAULT_NUM_ACTIONS,
    )
    if train_cfg.get("clip_actions") is not None:
        algorithm["info_gain_clip_actions"] = train_cfg["clip_actions"]

    use_imagination = mode == "finetune" and supports_imagination
    train_cfg["imagination"] = {
        "num_envs": 8192 if use_imagination else 0,
        "num_steps_per_env": 24 if use_imagination else 0,
        "max_episode_length": 256 if use_imagination else 0,
        "command_resample_interval_range": [100, 120] if use_imagination else None,
        "uncertainty_penalty_weight": -0.0,
        "state_normalizer": {"mean": list(_STATE_MEAN), "std": list(_STATE_STD)},
        "action_normalizer": {"mean": [0.0] * 12, "std": [1.0] * 12},
    }

    train_cfg.setdefault("load_system_dynamics", False)
    train_cfg["system_dynamics_load_path"] = os.environ.get("QOED_SYSTEM_DYNAMICS_LOAD_PATH")
    train_cfg["system_dynamics_warmup_iterations"] = 0
    train_cfg["system_dynamics_num_visualizations"] = 1
    train_cfg["system_dynamics_state_idx_dict"] = deepcopy(_STATE_IDX)
    train_cfg["pca_obs_buf_size"] = 10000
    if mode == "finetune":
        algorithm["system_dynamics_len_eval_trajectory"] = GO1_FINETUNE_NUM_STEPS_PER_ENV
    if train_cfg["system_dynamics_load_path"] is not None:
        train_cfg["load_system_dynamics"] = True
    return train_cfg


def _checkpoint_privilege_dim(checkpoint_path: str | None, architecture_config: dict) -> int | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        path = Path.cwd() / path
    if not path.exists():
        return None
    state_dict = torch.load(path, weights_only=False, map_location="cpu").get("system_dynamics_state_dict")
    if not isinstance(state_dict, dict):
        return None
    for key, tensor in state_dict.items():
        if key.endswith("model.flow.privilege_encoder.0.weight") and tensor.ndim == 2:
            condition_dim = int(tensor.shape[1]) - int(architecture_config.get("privilege_availability_feature", True))
            return max(condition_dim, 0)
    return None


class Go1MBPOOnPolicyRunner(MBPOOnPolicyRunner):
    """QOED Go1 adapter for rsl-rl-rwm's MBPO runner."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        import rsl_rl.runners.mbpo_on_policy_runner as mbpo_runner

        from mbpo.model_based.system_dynamics import QOED_MBPOPPO, build_system_dynamics

        mbpo_runner.SystemDynamicsEnsemble = build_system_dynamics
        mbpo_runner.MBPOPPO = QOED_MBPOPPO

        self.qoed_go1_mode = train_cfg.get("run_name", "pretrain")
        supports_imagination = all(
            hasattr(env.unwrapped, name)
            for name in ("prepare_imagination", "get_imagination_observation", "imagination_step")
        )
        train_cfg = normalize_mjlab_rsl_rl_cfg(train_cfg, mbpo=True)
        train_cfg = _add_go1_mbpo_cfg(train_cfg, self.qoed_go1_mode, supports_imagination)
        architecture_config = train_cfg["system_dynamics"]["architecture_config"]
        privilege_dim = _checkpoint_privilege_dim(train_cfg.get("system_dynamics_load_path"), architecture_config)
        if privilege_dim is None and hasattr(env, "get_observations"):
            privilege_obs = env.get_observations().get(GO1_SYSTEM_PRIVILEGE_OBS_KEY)
            if privilege_obs is not None:
                privilege_dim = int(privilege_obs.shape[-1])
        if privilege_dim is not None:
            architecture_config["privilege_dim"] = privilege_dim
        super().__init__(env, train_cfg, log_dir, device)
        _patch_policy_distribution_safety(self.alg.policy)
        self.imagination_infos = []

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if os.environ.get("QOED_GO1_FINETUNE") == "1":
            init_at_random_ep_len = False
        super().learn(num_learning_iterations, init_at_random_ep_len)
        if self.qoed_go1_mode == "finetune":
            self._print_final_summary()

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width, pad)
        if "mean_system_state_loss" in locs:

            def scalar(name: str) -> float:
                value = locs.get(name, 0.0)
                return float(value.detach().mean().item() if isinstance(value, torch.Tensor) else value)

            print(
                f"[Dynamics] iter={locs['it']} "
                f"state={scalar('mean_system_state_loss'):.4g} "
                f"seq={scalar('mean_system_sequence_loss'):.4g} "
                f"bound={scalar('mean_system_bound_loss'):.4g} "
                f"kl={scalar('mean_system_kl_loss'):.4g} "
                f"ext={scalar('mean_system_extension_loss'):.4g} "
                f"contact={scalar('mean_system_contact_loss'):.4g} "
                f"term={scalar('mean_system_termination_loss'):.4g}"
            )
        if self.qoed_go1_mode != "pretrain":
            stats = self._fisher_parameter_error()
            if stats is not None:
                print(f"[Fisher] iter={locs['it']} history={int(stats['history'])} rmse={stats['rmse']:.4g}")

    def _fisher_parameter_error(self) -> dict[str, float] | None:
        estimator = self.alg.parameter_estimator
        if estimator is None:
            return None
        estimated = self.alg.effective_system_privilege_mean().detach().float().cpu()
        env = self.env.unwrapped
        truth = env.domain_randomization_vector() if hasattr(env, "domain_randomization_vector") else _go1_domain_randomization_vector(env)
        truth = truth.detach().float().cpu().mean(dim=0)
        dim = min(estimated.numel(), truth.numel())
        if dim == 0:
            return None
        error = estimated[:dim] - truth[:dim]
        return {"rmse": float(error.square().mean().sqrt().item()), "history": float(len(estimator.history))}

    def _print_final_summary(self) -> None:
        param_stats = self._fisher_parameter_error()
        replay_buffer = self.alg.system_replay_buffer
        if replay_buffer.replay_buf is None or replay_buffer.num_transitions <= 0:
            dynamics_error = self.alg.latest_system_dynamics_autoregressive_error
        else:
            dynamics_error = self.alg.evaluate_system_dynamics(privilege_override="current_mean", num_trajectories=256)[-2]
        if self.alg.num_info_gain_selections > 0:
            fisher_gain = self.alg.cumulative_info_gain
        elif self.alg.parameter_estimator is not None and self.alg.parameter_estimator.history:
            fisher_gain = torch.trace(self.alg.parameter_estimator.compute_fisher())
        else:
            fisher_gain = None
        values = {
            "parameter_rmse": None if param_stats is None else param_stats["rmse"],
            "dynamics_prediction_error": dynamics_error,
            "fisher_information_gain": fisher_gain,
            "cumulative_reward": getattr(self.env.unwrapped, "cumulative_reward", None),
        }
        print("[Final Summary] " + " ".join(f"{k}={'n/a' if v is None else f'{float(v):.6g}'}" for k, v in values.items()))

    def load(self, path: str, load_cfg: dict | bool | None = None, strict: bool = True, map_location: str | None = None, load_optimizer: bool = True):
        map_location = map_location or self.device
        if isinstance(load_cfg, bool):
            load_optimizer, load_cfg = load_cfg, None
        if os.environ.get("QOED_POLICY_CHECKPOINT") is not None:
            load_optimizer = False
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        model_state_dict = _migrate_policy_noise_std_state_dict(self.alg.policy, loaded_dict["model_state_dict"])

        if load_cfg is None:
            resumed_training = self.alg.policy.load_state_dict(model_state_dict, strict=strict)
            if self.cfg["load_system_dynamics"]:
                dynamics_path = self.cfg["system_dynamics_load_path"]
                dynamics_dict = loaded_dict if dynamics_path is None else torch.load(dynamics_path, weights_only=False, map_location=map_location)
                self.alg.system_dynamics.load_state_dict(dynamics_dict["system_dynamics_state_dict"])
            if getattr(self.alg, "rnd", None):
                self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            if load_optimizer and resumed_training:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                self.alg.system_dynamics_optimizer.load_state_dict(loaded_dict["system_dynamics_optimizer_state_dict"])
                if getattr(self.alg, "rnd", None):
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            if resumed_training and os.environ.get("QOED_GO1_FINETUNE") != "1":
                self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict.get("infos")

        if load_cfg.get("actor", True):
            self.alg.policy.load_state_dict(model_state_dict, strict=strict)
        if load_cfg.get("system_dynamics", False) and "system_dynamics_state_dict" in loaded_dict:
            self.alg.system_dynamics.load_state_dict(loaded_dict["system_dynamics_state_dict"], strict=strict)
        self.current_learning_iteration = loaded_dict.get("iter", self.current_learning_iteration)
        infos = loaded_dict.get("infos")
        if infos and "env_state" in infos:
            self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
        return infos


def _go1_agent_cfg(mode: str):
    cfg = unitree_go1_ppo_runner_cfg()
    cfg.run_name = mode
    cfg.actor.distribution_cfg["std_type"] = GO1_POLICY_NOISE_STD_TYPE
    cfg.actor.obs_normalization = cfg.critic.obs_normalization = False
    if mode == "pretrain":
        cfg.max_iterations = 1000
    elif mode == "finetune":
        cfg.max_iterations = 20
        cfg.num_steps_per_env = GO1_FINETUNE_NUM_STEPS_PER_ENV
    return cfg


def _go1_finetune_env_cfg():
    cfg = _go1_env_cfg("pretrain", play=True)
    cfg.episode_length_s = GO1_FINETUNE_NUM_STEPS_PER_ENV * cfg.decimation * cfg.sim.mujoco.timestep
    cfg.auto_reset = True
    if "time_out" in cfg.terminations:
        cfg.terminations["time_out"].time_out = True
    else:
        cfg.terminations["time_out"] = TerminationTermCfg(func=term_mdp.time_out, time_out=True)
    return cfg


def register_go1_tasks() -> None:
    registered = set(list_tasks())
    for mode, task_id in GO1_MODE_TASKS.items():
        if task_id in registered:
            continue
        register_mjlab_task(
            task_id=task_id,
            env_cfg=_go1_finetune_env_cfg() if mode == "finetune" else _go1_env_cfg(mode),
            play_env_cfg=_go1_env_cfg(mode, play=True),
            rl_cfg=_go1_agent_cfg(mode),
            runner_cls=Go1MBPOOnPolicyRunner if mode in {"pretrain", "finetune", "visualize"} else VelocityOnPolicyRunner,
        )


register_go1_tasks()
