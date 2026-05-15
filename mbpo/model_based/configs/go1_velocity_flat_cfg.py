from dataclasses import dataclass, field
from typing import Dict, List

from .base_cfg import BaseConfig


MJLAB_GO1_FLAT_TASK_ID = "Mjlab-Velocity-Flat-Unitree-Go1"
_GO1_JOINT_POS_REL_STD = [0.3, 0.3, 0.6] * 4
_GO1_JOINT_VEL_REL_STD = [1.0, 1.5, 2.5] * 4
_GO1_ACTUATOR_FORCE_STD = [23.7, 23.7, 35.55] * 4


@dataclass
class Go1VelocityFlatConfig(BaseConfig):
    experiment_name: str = "go1_velocity"

    @dataclass
    class ExperimentConfig(BaseConfig.ExperimentConfig):
        environment: str = "go1_velocity_flat"

    @dataclass
    class EnvironmentConfig(BaseConfig.EnvironmentConfig):
        reward_term_weights: Dict[str, float] = field(
            default_factory=lambda: {
                "track_linear_velocity": 2.0,
                "track_angular_velocity": 2.0,
                "upright": 1.0,
                "pose": 1.0,
                "dof_pos_limits": -1.0,
                "action_rate_l2": -0.1,
                "air_time": 0.0,
            }
        )
        uncertainty_penalty_weight: float = -0.0
        command_resample_interval_range: List[int] | None = field(default_factory=lambda: [100, 120])
        event_interval_range: List[int] = field(default_factory=lambda: [50, 151])

    @dataclass
    class DataConfig(BaseConfig.DataConfig):
        dataset_root: str = "assets"
        dataset_folder: str = "data"
        batch_data_size: int = 10000
        state_idx_dict: Dict[str, List[int]] = field(
            default_factory=lambda: {
                r"$v_b$\n$[m/s]$": [0, 1, 2],
                r"$\omega_b$\n$[rad/s]$": [3, 4, 5],
                r"$g_b$\n$[1]$": [6, 7, 8],
                r"$q - q_0$\n$[rad]$": [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                r"$\dot{q}$\n$[rad/s]$": [21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32],
                r"$\tau$\n$[Nm]$": [33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44],
            }
        )
        state_data_mean: List[float] = field(
            default_factory=lambda: [
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, -1.0,
                *([0.0] * 12),
                *([0.0] * 12),
                *([0.0] * 12),
            ]
        )
        state_data_std: List[float] = field(
            default_factory=lambda: [
                0.5, 0.5, 0.1,
                0.3, 0.3, 0.5,
                0.02, 0.02, 0.04,
                *_GO1_JOINT_POS_REL_STD,
                *_GO1_JOINT_VEL_REL_STD,
                *_GO1_ACTUATOR_FORCE_STD,
            ]
        )
        action_data_mean: List[float] = field(default_factory=lambda: [0.0] * 12)
        action_data_std: List[float] = field(default_factory=lambda: [1.0] * 12)

    @dataclass
    class ModelArchitectureConfig(BaseConfig.ModelArchitectureConfig):
        history_horizon: int = 32
        forecast_horizon: int = 8
        ensemble_size: int = 1
        contact_dim: int = 4
        termination_dim: int = 1
        architecture_config: Dict[str, object] = field(
            default_factory=lambda: {
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
            }
        )
        resume_path: str | None = None

    @dataclass
    class PolicyArchitectureConfig(BaseConfig.PolicyArchitectureConfig):
        class_name: str = "ActorCritic"
        recurrent_class_name: str = "ActorCriticRecurrent"
        observation_dim: int = 48
        action_dim: int = 12
        actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
        critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
        activation: str = "elu"
        actor_obs_normalization: bool = False
        critic_obs_normalization: bool = False
        init_noise_std: float = 1.0
        noise_std_type: str = "scalar"
        rnn_type: str = "lstm"
        rnn_hidden_dim: int = 256
        rnn_num_layers: int = 1
        resume_path: str | None = None

    @dataclass
    class PolicyAlgorithmConfig(BaseConfig.PolicyAlgorithmConfig):
        learning_rate: float = 1.0e-3
        entropy_coef: float = 0.005

    @dataclass
    class PolicyTrainingConfig(BaseConfig.PolicyTrainingConfig):
        save_interval: int = 50
        max_iterations: int = 10000

    experiment_config: ExperimentConfig = field(default_factory=ExperimentConfig)
    environment_config: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_architecture_config: ModelArchitectureConfig = field(default_factory=ModelArchitectureConfig)
    policy_architecture_config: PolicyArchitectureConfig = field(default_factory=PolicyArchitectureConfig)
    policy_algorithm_config: PolicyAlgorithmConfig = field(default_factory=PolicyAlgorithmConfig)
    policy_training_config: PolicyTrainingConfig = field(default_factory=PolicyTrainingConfig)


def go1_rsl_rl_policy_cfg(recurrent: bool = False) -> Dict[str, object]:
    cfg = Go1VelocityFlatConfig().policy_architecture_config
    policy_cfg = {
        "class_name": cfg.recurrent_class_name if recurrent else cfg.class_name,
        "actor_hidden_dims": cfg.actor_hidden_dims,
        "critic_hidden_dims": cfg.critic_hidden_dims,
        "activation": cfg.activation,
        "actor_obs_normalization": cfg.actor_obs_normalization,
        "critic_obs_normalization": cfg.critic_obs_normalization,
        "init_noise_std": cfg.init_noise_std,
        "noise_std_type": cfg.noise_std_type,
    }
    if recurrent:
        policy_cfg.update(
            {
                "rnn_type": cfg.rnn_type,
                "rnn_hidden_dim": cfg.rnn_hidden_dim,
                "rnn_num_layers": cfg.rnn_num_layers,
            }
        )
    return policy_cfg
