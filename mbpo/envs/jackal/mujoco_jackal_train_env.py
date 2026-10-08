"""Direct MuJoCo Jackal VecEnv for RSL-RL training."""

from __future__ import annotations

import math

import numpy as np
import torch
from tensordict import TensorDict

from mbpo.envs.direct_env import DirectMujocoVecEnv
from mbpo.envs.utils import mujoco_body_ids_with_prefix

from .mujoco_jackal_backend import PureMujocoJackalModel
from .utils import JACKAL_MAX_WHEEL_SPEED, apply_jackal_domain_randomization, jackal_ids


class DirectMujocoJackalVecEnv(DirectMujocoVecEnv):
    """Single-environment Jackal velocity task running on native MuJoCo."""

    backend_cls = PureMujocoJackalModel

    def _setup(self) -> None:
        self._ids = jackal_ids(self.model)
        self._robot_body_ids = mujoco_body_ids_with_prefix(self.model, "robot/")
        self._reward_weights = {name: float(term.weight) for name, term in self.cfg.rewards.items()}
        self._reward_variance = {name: float(term.params["std"]) ** 2 for name, term in self.cfg.rewards.items() if "std" in term.params}
        self._payload = 0.0

    def domain_randomization_vector(self) -> torch.Tensor:
        vector = np.concatenate(
            (
                self._friction,
                self._body_mass,
                self._dof_frictionloss,
                self._dof_armature,
                self._dof_damping,
                self._kv,
                np.asarray([self._payload], dtype=np.float32),
            ),
            axis=0,
        )
        return torch.from_numpy(vector).view(1, -1)

    def _refresh_domain_randomization_cache(self) -> None:
        ids, model = self._ids, self.model
        self._friction = model.geom_friction[ids["geom"], 0].astype(np.float32)
        self._body_mass = model.body_mass[self._robot_body_ids].astype(np.float32)
        self._body_mass[self._robot_body_ids.index(ids["base"])] -= self._payload
        self._dof_frictionloss = model.dof_frictionloss[ids["dof"]].astype(np.float32)
        self._dof_armature = model.dof_armature[ids["dof"]].astype(np.float32)
        self._dof_damping = model.dof_damping[ids["dof"]].astype(np.float32)
        self._kv = model.actuator_gainprm[ids["actuator"], 0].astype(np.float32)

    def _apply_domain_randomization(self) -> None:
        self._payload = apply_jackal_domain_randomization(
            self.model, self.data, self.rng, self._model_defaults, ids=self._ids, body_ids=self._robot_body_ids
        )
        self._refresh_domain_randomization_cache()

    def _obs_dict(self, system_termination: bool) -> dict[str, torch.Tensor]:
        actor = self.backend.observation(self.command)
        system_state = actor[:13].astype(np.float32)
        return {
            "actor": torch.from_numpy(actor).view(1, -1),
            "policy": torch.from_numpy(actor.copy()).view(1, -1),
            "critic": torch.from_numpy(actor.copy()).view(1, -1),
            "system_state": torch.from_numpy(system_state).view(1, -1),
            "system_action": torch.from_numpy(self.backend.last_action.copy()).view(1, -1),
            "system_termination": torch.tensor([[float(system_termination)]], dtype=torch.float32, device=self.device),
        }

    def _reward_terms(self, command, lin_vel, ang_vel, action, previous_action, xp):
        lin_vel_error = xp.sum((command[..., :2] - lin_vel[..., :2]) ** 2, -1) + lin_vel[..., 2] ** 2
        ang_vel_error = (command[..., 2] - ang_vel[..., 2]) ** 2 + xp.sum(ang_vel[..., :2] ** 2, -1)
        raw_terms = {
            "track_linear_velocity": xp.exp(-lin_vel_error / self._reward_variance["track_linear_velocity"]),
            "track_angular_velocity": xp.exp(-ang_vel_error / self._reward_variance["track_angular_velocity"]),
            "action_rate_l2": xp.sum((action - previous_action) ** 2, -1),
        }
        terms = {name: self._reward_weights[name] * value * self.step_dt for name, value in raw_terms.items()}
        metrics = {
            "Metrics/twist/error_vel_xy": xp.sqrt(xp.sum((command[..., :2] - lin_vel[..., :2]) ** 2, -1)),
            "Metrics/twist/error_vel_yaw": xp.abs(command[..., 2] - ang_vel[..., 2]),
            "Episode_Metrics/mean_action_acc": raw_terms["action_rate_l2"],
        }
        return sum(terms.values()), terms, metrics

    def _compute_reward(self, previous_action: np.ndarray, action: np.ndarray) -> tuple[float, dict[str, float], dict[str, float]]:
        lin_vel = self.data.sensordata[self.backend.lin_vel_slice].astype(np.float32)
        ang_vel = self.data.sensordata[self.backend.ang_vel_slice].astype(np.float32)
        reward, terms, metrics = self._reward_terms(self.command, lin_vel, ang_vel, action, previous_action, np)
        return float(reward), {k: float(v) for k, v in terms.items()}, {k: float(v) for k, v in metrics.items()}

    def _compute_imagination_reward(self, state: torch.Tensor, action: torch.Tensor):
        command = self._imagination_command.to(device=state.device, dtype=state.dtype)
        previous_action = self._imagination_last_action.to(device=state.device, dtype=state.dtype)
        return self._reward_terms(command, state[:, 0:3], state[:, 3:6], action, previous_action, torch)

    def _imagination_obs_from_state(self, state: torch.Tensor, action: torch.Tensor, termination: torch.Tensor | None = None) -> TensorDict:
        num_envs = state.shape[0]
        actor = torch.cat((state[:, 0:13], action, self._imagination_command.to(device=state.device, dtype=state.dtype)), dim=-1)
        if termination is None:
            termination_value = torch.zeros(num_envs, 1, device=state.device, dtype=state.dtype)
        else:
            termination_value = termination.to(device=state.device).float().view(num_envs, 1)
        obs = {
            "actor": actor,
            "policy": actor,
            "critic": actor,
            "system_state": state,
            "system_action": action,
            "system_termination": termination_value,
        }
        return TensorDict(obs, batch_size=[num_envs], device=state.device)

    def _parse_imagination_termination(self, terminations: torch.Tensor | None, state: torch.Tensor) -> torch.Tensor:
        if terminations is None:
            termination = torch.zeros(state.shape[0], device=state.device, dtype=torch.bool)
        else:
            termination = torch.sigmoid(terminations).reshape(state.shape[0], -1).max(dim=1).values > 0.5
        bad_orientation = torch.acos(torch.clamp(-state[:, 8], -1.0, 1.0)).abs() > math.radians(70.0)
        plausible = (state[:, 9:13].abs() <= 2 * JACKAL_MAX_WHEEL_SPEED).all(dim=1) & torch.isfinite(state).all(dim=1)
        return termination | bad_orientation | ~plausible
