"""Local MJLab task aliases for QOED Go1 workflows."""

from __future__ import annotations

import os
import sys
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


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from mbpo.model_based.configs.go1_velocity_flat_cfg import Go1VelocityFlatConfig, go1_rsl_rl_policy_cfg


GO1_MODE_TASKS = {
    "init": "QOED-Mjlab-Velocity-Flat-Unitree-Go1-Init-v0",
    "pretrain": "QOED-Mjlab-Velocity-Flat-Unitree-Go1-Pretrain-v0",
    "finetune": "QOED-Mjlab-Velocity-Flat-Unitree-Go1-Finetune-v0",
    "visualize": "QOED-Mjlab-Velocity-Flat-Unitree-Go1-Visualize-v0",
}
GO1_DOMAIN_RANDOMIZATION_INFO_KEY = "domain_randomization"
GO1_SYSTEM_PRIVILEGE_OBS_KEY = "system_privilege"
GO1_FINETUNE_NUM_STEPS_PER_ENV = 100
GO1_POLICY_NOISE_STD_TYPE = "log"
GO1_INFO_GAIN_DEFAULT_NUM_ACTIONS = 1024
_GO1_PRIVILEGE_BODY_MASS_DEFAULTS = (5.204, *([0.68, 1.009, 0.195862] * 4))
_GO1_PRIVILEGE_DOF_FRICTIONLOSS_DEFAULTS = (0.0,) * 12
_GO1_PRIVILEGE_DOF_ARMATURE_DEFAULTS = (0.004026312, 0.004026312, 0.009059202) * 4


def _uniform_mean_var(low: float, high: float) -> tuple[float, float]:
    return 0.5 * (low + high), ((high - low) ** 2) / 12.0


def _joint_effort(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    asset = env.scene[asset_cfg.name]
    return asset.data.actuator_force[:, asset_cfg.actuator_ids]


def _bad_orientation(env, limit_angle: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    fell_over = term_mdp.bad_orientation(env, limit_angle=limit_angle, asset_cfg=asset_cfg)
    return fell_over.float().unsqueeze(-1)


def _event_env_ids(env, env_ids: torch.Tensor | slice | None) -> torch.Tensor:
    if env_ids is None:
        return torch.arange(env.num_envs, device=env.device, dtype=torch.int)
    if isinstance(env_ids, slice):
        return torch.arange(env.num_envs, device=env.device, dtype=torch.int)[env_ids]
    return env_ids.to(env.device, dtype=torch.int)


def _uniform_like(
    reference: torch.Tensor,
    shape: tuple[int, ...],
    low: float,
    high: float,
) -> torch.Tensor:
    return torch.empty(shape, device=reference.device, dtype=reference.dtype).uniform_(low, high)


def _named_global_id(names: tuple[str, ...], ids: torch.Tensor, name: str) -> torch.Tensor:
    if name in names:
        return ids[names.index(name)]
    return ids[0]


def _go1_domain_randomization_vector(env) -> torch.Tensor:
    robot = env.scene["robot"]
    terrain = env.scene["terrain"]
    model = env.sim.model

    floor_geom_id = _named_global_id(terrain.geom_names, terrain.indexing.geom_ids, "terrain")
    body_ids = robot.indexing.body_ids
    joint_v_ids = robot.indexing.joint_v_adr

    floor_friction = model.geom_friction[:, floor_geom_id, 0:1]
    body_mass = model.body_mass[:, body_ids]
    frictionloss = model.dof_frictionloss[:, joint_v_ids]
    armature = model.dof_armature[:, joint_v_ids]
    return torch.cat((floor_friction, body_mass, frictionloss, armature), dim=-1)


def _go1_fisher_parameter_bounds() -> tuple[list[float], list[float]]:
    ranges = _go1_privilege_randomization_ranges()
    return [low for low, _ in ranges], [high for _, high in ranges]


def _go1_privilege_randomization_ranges() -> list[tuple[float, float]]:
    mass_ranges = [
        (mass * 0.9, mass * 1.1)
        for mass in _GO1_PRIVILEGE_BODY_MASS_DEFAULTS
    ]
    torso_low, torso_high = mass_ranges[0]
    mass_ranges[0] = (torso_low - 1.0, torso_high + 1.0)

    frictionloss_ranges = [
        (value * 0.9, value * 1.1)
        for value in _GO1_PRIVILEGE_DOF_FRICTIONLOSS_DEFAULTS
    ]
    armature_ranges = [
        (value, value * 1.05)
        for value in _GO1_PRIVILEGE_DOF_ARMATURE_DEFAULTS
    ]
    return [(0.4, 1.0), *mass_ranges, *frictionloss_ranges, *armature_ranges]


def _go1_fisher_parameter_prior() -> tuple[list[float], list[float]]:
    means = []
    variances = []

    mean, variance = _uniform_mean_var(0.4, 1.0)
    means.append(mean)
    variances.append(variance)

    for body_id, mass in enumerate(_GO1_PRIVILEGE_BODY_MASS_DEFAULTS):
        low, high = mass * 0.9, mass * 1.1
        mean, variance = _uniform_mean_var(low, high)
        if body_id == 0:
            variance += _uniform_mean_var(-1.0, 1.0)[1]
        means.append(mean)
        variances.append(variance)

    for value in _GO1_PRIVILEGE_DOF_FRICTIONLOSS_DEFAULTS:
        mean, variance = _uniform_mean_var(value * 0.9, value * 1.1)
        means.append(mean)
        variances.append(variance)

    for value in _GO1_PRIVILEGE_DOF_ARMATURE_DEFAULTS:
        mean, variance = _uniform_mean_var(value, value * 1.05)
        means.append(mean)
        variances.append(variance)
    return means, variances


def _go1_domain_randomization_privilege(env) -> torch.Tensor:
    return _go1_domain_randomization_vector(env)


def _add_go1_domain_randomization_privilege_observation(cfg) -> None:
    cfg.observations[GO1_SYSTEM_PRIVILEGE_OBS_KEY] = ObservationGroupCfg(
        terms={
            GO1_DOMAIN_RANDOMIZATION_INFO_KEY: ObservationTermCfg(
                func=_go1_domain_randomization_privilege
            )
        },
        enable_corruption=False,
        concatenate_terms=True,
    )


def _publish_go1_domain_randomization_info(env, env_ids: torch.Tensor | slice | None = None) -> None:
    del env_ids
    env.extras[GO1_DOMAIN_RANDOMIZATION_INFO_KEY] = _go1_domain_randomization_vector(env)


@requires_model_fields(
    "geom_friction",
    "body_ipos",
    "dof_frictionloss",
    "dof_armature",
    "body_mass",
    "qpos0",
    recompute=RecomputeLevel.set_const,
)
def _go1_pretrain_domain_randomize(env, env_ids: torch.Tensor | slice | None) -> None:
    env_ids = _event_env_ids(env, env_ids)
    num_envs = len(env_ids)

    robot = env.scene["robot"]
    terrain = env.scene["terrain"]

    floor_geom_id = _named_global_id(terrain.geom_names, terrain.indexing.geom_ids, "terrain")
    torso_body_id = _named_global_id(robot.body_names, robot.indexing.body_ids, "trunk")
    body_ids = robot.indexing.body_ids
    joint_q_ids = robot.indexing.joint_q_adr
    joint_v_ids = robot.indexing.joint_v_adr

    model = env.sim.model

    # Match mujoco_playground Go1 randomize.py.
    default_geom_friction = env.sim.get_default_field("geom_friction")
    model.geom_friction[env_ids, floor_geom_id] = default_geom_friction[floor_geom_id]
    model.geom_friction[env_ids, floor_geom_id, 0] = _uniform_like(
        model.geom_friction, (num_envs,), 0.4, 1.0
    )

    # Joint friction loss: default * U(0.9, 1.1).
    env_grid, dof_grid = torch.meshgrid(env_ids, joint_v_ids, indexing="ij")
    default_frictionloss = env.sim.get_default_field("dof_frictionloss")
    frictionloss_scale = _uniform_like(
        model.dof_frictionloss, (num_envs, len(joint_v_ids)), 0.9, 1.1
    )
    model.dof_frictionloss[env_grid, dof_grid] = (
        default_frictionloss[joint_v_ids] * frictionloss_scale
    )

    # Armature: default * U(1.0, 1.05).
    default_armature = env.sim.get_default_field("dof_armature")
    armature_scale = _uniform_like(
        model.dof_armature, (num_envs, len(joint_v_ids)), 1.0, 1.05
    )
    model.dof_armature[env_grid, dof_grid] = default_armature[joint_v_ids] * armature_scale

    # Torso COM position: default + U(-0.05, 0.05).
    default_body_ipos = env.sim.get_default_field("body_ipos")
    model.body_ipos[env_ids, torso_body_id] = (
        default_body_ipos[torso_body_id]
        + _uniform_like(model.body_ipos, (num_envs, 3), -0.05, 0.05)
    )

    # Link masses: default * U(0.9, 1.1), plus torso mass U(-1.0, 1.0).
    env_grid, body_grid = torch.meshgrid(env_ids, body_ids, indexing="ij")
    default_body_mass = env.sim.get_default_field("body_mass")
    mass_scale = _uniform_like(model.body_mass, (num_envs, len(body_ids)), 0.9, 1.1)
    model.body_mass[env_grid, body_grid] = default_body_mass[body_ids] * mass_scale
    model.body_mass[env_ids, torso_body_id] += _uniform_like(
        model.body_mass, (num_envs,), -1.0, 1.0
    )

    # qpos0 joint defaults: default + U(-0.05, 0.05).
    env_grid, qpos_grid = torch.meshgrid(env_ids, joint_q_ids, indexing="ij")
    default_qpos0 = env.sim.get_default_field("qpos0")
    model.qpos0[env_grid, qpos_grid] = (
        default_qpos0[joint_q_ids]
        + _uniform_like(model.qpos0, (num_envs, len(joint_q_ids)), -0.05, 0.05)
    )


def _add_go1_pretrain_domain_randomization(cfg) -> None:
    cfg.events["qoed_pretrain_domain_randomization"] = EventTermCfg(
        mode="reset",
        func=_go1_pretrain_domain_randomize,
    )
    cfg.events["qoed_domain_randomization_info_reset"] = EventTermCfg(
        mode="reset",
        func=_publish_go1_domain_randomization_info,
    )
    cfg.events["qoed_domain_randomization_info_step"] = EventTermCfg(
        mode="step",
        func=_publish_go1_domain_randomization_info,
    )
    _add_go1_domain_randomization_privilege_observation(cfg)


def _add_mbpo_observations(cfg) -> None:
    cfg.observations.setdefault("policy", deepcopy(cfg.observations["actor"]))
    cfg.observations["system_state"] = ObservationGroupCfg(
        terms={
            "base_lin_vel": ObservationTermCfg(func=obs_mdp.base_lin_vel),
            "base_ang_vel": ObservationTermCfg(func=obs_mdp.base_ang_vel),
            "projected_gravity": ObservationTermCfg(func=obs_mdp.projected_gravity),
            "joint_pos": ObservationTermCfg(func=obs_mdp.joint_pos_rel),
            "joint_vel": ObservationTermCfg(func=obs_mdp.joint_vel_rel),
            "joint_torque": ObservationTermCfg(func=_joint_effort),
        },
        enable_corruption=False,
        concatenate_terms=True,
    )
    cfg.observations["system_action"] = ObservationGroupCfg(
        terms={"actions": ObservationTermCfg(func=obs_mdp.last_action)},
        enable_corruption=False,
        concatenate_terms=True,
    )
    if "foot_contact" in cfg.observations["critic"].terms:
        cfg.observations["system_contact"] = ObservationGroupCfg(
            terms={"foot_contact": deepcopy(cfg.observations["critic"].terms["foot_contact"])},
            enable_corruption=False,
            concatenate_terms=True,
        )
    fell_over_cfg = cfg.terminations.get("fell_over")
    if fell_over_cfg is not None:
        fell_over_params = deepcopy(fell_over_cfg.params)
    else:
        fell_over_params = {"limit_angle": 1.2217304763960306}
    cfg.observations["system_termination"] = ObservationGroupCfg(
        terms={"fell_over": ObservationTermCfg(func=_bad_orientation, params=fell_over_params)},
        enable_corruption=False,
        concatenate_terms=True,
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

    if (
        not play
        and mode == "pretrain"
        and os.environ.get("QOED_GO1_DOMAIN_RANDOMIZATION") == "1"
    ):
        _add_go1_pretrain_domain_randomization(cfg)
    elif mode in {"finetune", "visualize"}:
        _add_go1_domain_randomization_privilege_observation(cfg)

    return cfg


def normalize_mjlab_rsl_rl_cfg(train_cfg: dict, mbpo: bool = False) -> dict:
    train_cfg = deepcopy(train_cfg)

    obs_groups = train_cfg.get("obs_groups")
    if isinstance(obs_groups, dict) and "policy" not in obs_groups and "actor" in obs_groups:
        obs_groups["policy"] = obs_groups["actor"]

    if "policy" not in train_cfg and "actor" in train_cfg and "critic" in train_cfg:
        actor_cfg = train_cfg["actor"]
        critic_cfg = train_cfg["critic"]
        distribution_cfg = actor_cfg.get("distribution_cfg") or {}
        recurrent = actor_cfg.get("rnn_type") is not None or actor_cfg.get("class_name") == "RNNModel"
        go1_policy_cfg = go1_rsl_rl_policy_cfg(recurrent=recurrent)

        noise_std_type = distribution_cfg.get("std_type", go1_policy_cfg["noise_std_type"])
        if noise_std_type == "scalar":
            noise_std_type = GO1_POLICY_NOISE_STD_TYPE

        policy_cfg = {
            "class_name": go1_policy_cfg["class_name"],
            "actor_hidden_dims": actor_cfg.get("hidden_dims", go1_policy_cfg["actor_hidden_dims"]),
            "critic_hidden_dims": critic_cfg.get("hidden_dims", go1_policy_cfg["critic_hidden_dims"]),
            "activation": actor_cfg.get("activation", go1_policy_cfg["activation"]),
            "actor_obs_normalization": actor_cfg.get(
                "obs_normalization", go1_policy_cfg["actor_obs_normalization"]
            ),
            "critic_obs_normalization": critic_cfg.get(
                "obs_normalization", go1_policy_cfg["critic_obs_normalization"]
            ),
            "init_noise_std": distribution_cfg.get("init_std", go1_policy_cfg["init_noise_std"]),
            "noise_std_type": noise_std_type,
        }
        if recurrent:
            policy_cfg.update(
                {
                    "rnn_type": actor_cfg.get("rnn_type", go1_policy_cfg["rnn_type"]),
                    "rnn_hidden_dim": actor_cfg.get("rnn_hidden_dim", go1_policy_cfg["rnn_hidden_dim"]),
                    "rnn_num_layers": actor_cfg.get("rnn_num_layers", go1_policy_cfg["rnn_num_layers"]),
                }
            )
        train_cfg["policy"] = policy_cfg

    algorithm_cfg = train_cfg.get("algorithm")
    if isinstance(algorithm_cfg, dict):
        for key in ("optimizer", "share_cnn_encoders"):
            algorithm_cfg.pop(key, None)
        if mbpo:
            algorithm_cfg["class_name"] = "MBPOPPO"
            algorithm_cfg["policy_learning_rate"] = algorithm_cfg.pop(
                "learning_rate", algorithm_cfg.get("policy_learning_rate", 1.0e-3)
            )
            algorithm_cfg.setdefault("system_dynamics_learning_rate", 1.0e-3)
            algorithm_cfg.setdefault("system_dynamics_weight_decay", 0.0)
            algorithm_cfg.setdefault("system_dynamics_forecast_horizon", 8)
            algorithm_cfg.setdefault(
                "system_dynamics_loss_weights",
                {
                    "state": 1.0,
                    "sequence": 1.0,
                    "bound": 1.0,
                    "kl": 0.1,
                    "extension": 1.0,
                    "contact": 1.0,
                    "termination": 1.0,
                },
            )
            algorithm_cfg.setdefault("system_dynamics_num_mini_batches", 20)
            algorithm_cfg.setdefault("system_dynamics_mini_batch_size", 1024)
            algorithm_cfg.setdefault("system_dynamics_replay_buffer_size", 1000)
            algorithm_cfg.setdefault("system_dynamics_num_eval_trajectories", 1)
            algorithm_cfg.setdefault("system_dynamics_len_eval_trajectory", 400)
            algorithm_cfg.setdefault(
                "system_dynamics_eval_traj_noise_scale",
                [0.1, 0.2, 0.4, 0.5, 0.8],
            )

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
    if getattr(policy, "_qoed_distribution_safety_patch", False):
        return

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
    policy._qoed_distribution_safety_patch = True


def _add_go1_mbpo_cfg(train_cfg: dict, mode: str, supports_imagination: bool) -> dict:
    cfg = Go1VelocityFlatConfig()
    data_cfg = cfg.data_config
    model_cfg = cfg.model_architecture_config
    env_cfg = cfg.environment_config

    train_cfg["system_dynamics"] = {
        "ensemble_size": model_cfg.ensemble_size,
        "history_horizon": model_cfg.history_horizon,
        "architecture_config": dict(model_cfg.architecture_config),
        "freeze_auxiliary": False,
    }
    fisher_param_min, fisher_param_max = _go1_fisher_parameter_bounds()
    fisher_prior_mean, fisher_prior_cov_diag = _go1_fisher_parameter_prior()
    train_cfg.setdefault("algorithm", {})
    train_cfg["algorithm"]["fisher_param_min"] = fisher_param_min
    train_cfg["algorithm"]["fisher_param_max"] = fisher_param_max
    train_cfg["algorithm"]["fisher_prior_mean"] = fisher_prior_mean
    train_cfg["algorithm"]["fisher_prior_cov_diag"] = fisher_prior_cov_diag
    train_cfg["algorithm"]["fisher_var_threshold_for_update"] = 1.0e-12
    train_cfg["algorithm"]["fisher_fd_delta_floor"] = 0.1
    train_cfg["algorithm"]["fisher_obs_noise_std"] = 0.1
    info_gain_mode = os.environ.get("QOED_GO1_INFO_GAIN", "nothing")
    if mode != "finetune":
        info_gain_mode = "nothing"
    train_cfg["algorithm"]["info_gain_mode"] = info_gain_mode
    train_cfg["algorithm"]["info_gain_num_actions"] = GO1_INFO_GAIN_DEFAULT_NUM_ACTIONS
    if train_cfg.get("clip_actions") is not None:
        train_cfg["algorithm"]["info_gain_clip_actions"] = train_cfg["clip_actions"]

    use_imagination = mode == "finetune" and supports_imagination
    train_cfg["imagination"] = {
        "num_envs": 8192 if use_imagination else 0,
        "num_steps_per_env": 24 if use_imagination else 0,
        "max_episode_length": 256 if use_imagination else 0,
        "command_resample_interval_range": env_cfg.command_resample_interval_range if use_imagination else None,
        "uncertainty_penalty_weight": env_cfg.uncertainty_penalty_weight if use_imagination else -0.0,
        "state_normalizer": {"mean": data_cfg.state_data_mean, "std": data_cfg.state_data_std},
        "action_normalizer": {"mean": data_cfg.action_data_mean, "std": data_cfg.action_data_std},
    }

    train_cfg.setdefault("load_system_dynamics", False)
    train_cfg["system_dynamics_load_path"] = os.environ.get("QOED_SYSTEM_DYNAMICS_LOAD_PATH")
    train_cfg["system_dynamics_warmup_iterations"] = 0
    train_cfg["system_dynamics_num_visualizations"] = 1
    train_cfg["system_dynamics_state_idx_dict"] = data_cfg.state_idx_dict
    train_cfg["pca_obs_buf_size"] = 10000
    if mode == "finetune":
        train_cfg["algorithm"]["system_dynamics_len_eval_trajectory"] = GO1_FINETUNE_NUM_STEPS_PER_ENV
    if train_cfg["system_dynamics_load_path"] is not None:
        train_cfg["load_system_dynamics"] = True
    return train_cfg


def _infer_system_dynamics_privilege_dim_from_checkpoint(
    checkpoint_path: str | None,
    architecture_config: dict,
) -> int | None:
    if checkpoint_path is None:
        return None
    path = Path(checkpoint_path).expanduser()
    if not path.exists():
        path = Path.cwd() / path
    if not path.exists():
        return None

    loaded_dict = torch.load(path, weights_only=False, map_location="cpu")
    state_dict = loaded_dict.get("system_dynamics_state_dict")
    if not isinstance(state_dict, dict):
        return None

    for key, tensor in state_dict.items():
        if key.endswith("model.flow.privilege_encoder.0.weight") and tensor.ndim == 2:
            condition_dim = int(tensor.shape[1])
            if architecture_config.get("privilege_availability_feature", True):
                condition_dim -= 1
            return max(condition_dim, 0)
    return None


class Go1MBPOOnPolicyRunner(MBPOOnPolicyRunner):
    """QOED Go1 adapter for rsl-rl-rwm's MBPO runner."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        from mbpo.model_based.system_dynamics import QOED_MBPOPPO, build_system_dynamics
        import rsl_rl.runners.mbpo_on_policy_runner as mbpo_runner

        mbpo_runner.SystemDynamicsEnsemble = build_system_dynamics
        mbpo_runner.MBPOPPO = QOED_MBPOPPO

        mode = train_cfg.get("run_name", "pretrain")
        self.qoed_go1_mode = mode
        imagination_methods = ("prepare_imagination", "get_imagination_observation", "imagination_step")
        supports_imagination = all(
            hasattr(env.unwrapped, name)
            for name in imagination_methods
        )
        train_cfg = normalize_mjlab_rsl_rl_cfg(train_cfg, mbpo=True)
        train_cfg = _add_go1_mbpo_cfg(train_cfg, mode, supports_imagination)
        architecture_config = train_cfg["system_dynamics"]["architecture_config"]
        privilege_dim = _infer_system_dynamics_privilege_dim_from_checkpoint(
            train_cfg.get("system_dynamics_load_path"),
            architecture_config,
        )
        if privilege_dim is None and hasattr(env, "get_observations"):
            observations = env.get_observations()
            privilege_obs = observations.get(GO1_SYSTEM_PRIVILEGE_OBS_KEY)
            if privilege_obs is not None:
                privilege_dim = int(privilege_obs.shape[-1])
        if privilege_dim is not None:
            architecture_config["privilege_dim"] = privilege_dim
        super().__init__(env, train_cfg, log_dir, device)
        _patch_policy_distribution_safety(self.alg.policy)
        self.imagination_infos = []
        self._printed_final_summary = False

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False):
        if os.environ.get("QOED_GO1_FINETUNE") == "1":
            init_at_random_ep_len = False
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self._print_final_summary()

    def log(self, locs: dict, width: int = 80, pad: int = 35):
        super().log(locs, width, pad)
        self._log_system_dynamics_losses(locs)
        if self.qoed_go1_mode != "pretrain":
            self._log_fisher_parameter_estimate(locs["it"])

    def _log_system_dynamics_losses(self, locs: dict) -> None:
        if "mean_system_state_loss" not in locs:
            return

        def scalar(name: str) -> float:
            value = locs.get(name, 0.0)
            if isinstance(value, torch.Tensor):
                value = value.detach().mean().item()
            return float(value)

        print(
            "[Dynamics] "
            f"iter={locs['it']} "
            f"state={scalar('mean_system_state_loss'):.4g} "
            f"seq={scalar('mean_system_sequence_loss'):.4g} "
            f"bound={scalar('mean_system_bound_loss'):.4g} "
            f"kl={scalar('mean_system_kl_loss'):.4g} "
            f"ext={scalar('mean_system_extension_loss'):.4g} "
            f"contact={scalar('mean_system_contact_loss'):.4g} "
            f"term={scalar('mean_system_termination_loss'):.4g}"
        )

    def _log_fisher_parameter_estimate(self, step: int) -> None:
        stats = self._fisher_parameter_error()
        if stats is None:
            return

        print(
            "[Fisher] "
            f"iter={step} "
            f"history={int(stats['history'])} "
            f"rmse={stats['rmse']:.4g}"
        )

    def _fisher_parameter_error(self) -> dict[str, float] | None:
        estimator = getattr(self.alg, "parameter_estimator", None)
        if estimator is None:
            return None

        effective_mean = getattr(self.alg, "effective_system_privilege_mean", None)
        if effective_mean is not None:
            estimated_tensor = effective_mean()
        else:
            estimated_tensor = estimator.dist.mean
        estimated = estimated_tensor.detach().float().cpu()
        if hasattr(self.env.unwrapped, "domain_randomization_vector"):
            ground_truth = self.env.unwrapped.domain_randomization_vector().detach().float().cpu().mean(dim=0)
        else:
            ground_truth = _go1_domain_randomization_vector(self.env.unwrapped).detach().float().cpu().mean(dim=0)
        dim = min(estimated.numel(), ground_truth.numel())
        if dim == 0:
            return None

        error = estimated[:dim] - ground_truth[:dim]
        return {
            "rmse": float(error.square().mean().sqrt().item()),
            "history": float(len(estimator.history)),
        }

    def _final_dynamics_prediction_error(self) -> float | None:
        replay_buffer = getattr(self.alg, "system_replay_buffer", None)
        if replay_buffer is None or replay_buffer.replay_buf is None or replay_buffer.num_transitions <= 0:
            error = getattr(self.alg, "latest_system_dynamics_autoregressive_error", None)
            return None if error is None else float(error)
        try:
            result = self.alg.evaluate_system_dynamics(privilege_override="current_mean")
        except Exception as exc:
            print(f"[WARN] Could not compute final dynamics prediction error: {exc}")
            error = getattr(self.alg, "latest_system_dynamics_autoregressive_error", None)
            return None if error is None else float(error)
        return float(result[-2])

    def _final_fisher_information_gain(self) -> float | None:
        if getattr(self.alg, "num_info_gain_selections", 0) > 0:
            return float(getattr(self.alg, "cumulative_info_gain", 0.0))

        estimator = getattr(self.alg, "parameter_estimator", None)
        if estimator is None or not estimator.history:
            return None
        try:
            return float(torch.trace(estimator.compute_fisher()).detach().cpu())
        except Exception as exc:
            print(f"[WARN] Could not compute final Fisher information: {exc}")
            return None

    def _format_final_value(self, value: float | None) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.6g}"

    def _print_final_summary(self) -> None:
        if self._printed_final_summary or self.qoed_go1_mode != "finetune":
            return
        self._printed_final_summary = True

        param_stats = self._fisher_parameter_error()
        dynamics_error = self._final_dynamics_prediction_error()
        fisher_gain = self._final_fisher_information_gain()
        cumulative_reward = getattr(self.env.unwrapped, "cumulative_reward", None)

        param_rmse = None if param_stats is None else param_stats["rmse"]
        print(
            "[Final Summary] "
            f"parameter_rmse={self._format_final_value(param_rmse)} "
            f"dynamics_prediction_error={self._format_final_value(dynamics_error)} "
            f"fisher_information_gain={self._format_final_value(fisher_gain)} "
            f"cumulative_reward={self._format_final_value(cumulative_reward)}"
        )

    def load(
        self,
        path: str,
        load_cfg: dict | bool | None = None,
        strict: bool = True,
        map_location: str | None = None,
        load_optimizer: bool = True,
    ):
        map_location = map_location or self.device
        if isinstance(load_cfg, bool):
            load_optimizer = load_cfg
            load_cfg = None
        if os.environ.get("QOED_POLICY_CHECKPOINT") is not None:
            load_optimizer = False

        import torch

        if load_cfg is None:
            loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
            finetune_load = os.environ.get("QOED_GO1_FINETUNE") == "1"
            model_state_dict = _migrate_policy_noise_std_state_dict(
                self.alg.policy,
                loaded_dict["model_state_dict"],
            )
            resumed_training = self.alg.policy.load_state_dict(model_state_dict, strict=strict)
            if self.cfg["load_system_dynamics"]:
                if self.cfg["system_dynamics_load_path"] is not None:
                    system_dynamics_loaded_dict = torch.load(
                        self.cfg["system_dynamics_load_path"], weights_only=False, map_location=map_location
                    )
                else:
                    system_dynamics_loaded_dict = loaded_dict
                self.alg.system_dynamics.load_state_dict(system_dynamics_loaded_dict["system_dynamics_state_dict"])
            if getattr(self.alg, "rnd", None):
                self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"])
            if load_optimizer and resumed_training:
                self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
                self.alg.system_dynamics_optimizer.load_state_dict(loaded_dict["system_dynamics_optimizer_state_dict"])
                if getattr(self.alg, "rnd", None):
                    self.alg.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
            if resumed_training and not finetune_load:
                self.current_learning_iteration = loaded_dict["iter"]
            return loaded_dict.get("infos")

        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        if load_cfg.get("actor", True):
            model_state_dict = _migrate_policy_noise_std_state_dict(
                self.alg.policy,
                loaded_dict["model_state_dict"],
            )
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
        env_cfg = _go1_finetune_env_cfg() if mode == "finetune" else _go1_env_cfg(mode)
        if mode in {"pretrain", "finetune", "visualize"}:
            runner_cls = Go1MBPOOnPolicyRunner
        else:
            runner_cls = VelocityOnPolicyRunner
        register_mjlab_task(
            task_id=task_id,
            env_cfg=env_cfg,
            play_env_cfg=_go1_env_cfg(mode, play=True),
            rl_cfg=_go1_agent_cfg(mode),
            runner_cls=runner_cls,
        )


register_go1_tasks()
