"""Direct MuJoCo LEAP hand cube-rotation VecEnv for RSL-RL training."""

from __future__ import annotations

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from mbpo.envs.direct_env import DirectMujocoVecEnv
from mbpo.envs.mujoco_backend import JointPositionMujocoModel
from mbpo.envs.utils import mujoco_named_id, quat_wxyz_to_matrix

from .utils import CUBE_DROP_HEIGHT, CUBE_INIT_POS, LEAP_PALM_SITE, LEAP_TIPS, apply_leap_domain_randomization, leap_ids


def _quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    return np.asarray([cy * cr * cp + sy * sr * sp, cy * sr * cp - sy * cr * sp, cy * cr * sp + sy * sr * cp, sy * cr * cp - cy * sr * sp])


class PureMujocoLeapModel(JointPositionMujocoModel):
    """LEAP hand with a free cube; observations are joint offsets and the last action."""

    camera_distance = 0.6

    def __init__(self, env_cfg) -> None:
        super().__init__(env_cfg)
        model = self.model
        cube_joint = mujoco_named_id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube/cube_freejoint")
        self.cube_body = mujoco_named_id(model, mujoco.mjtObj.mjOBJ_BODY, "cube/cube")
        self.cube_qpos = int(model.jnt_qposadr[cube_joint])
        self.cube_dof = int(model.jnt_dofadr[cube_joint])
        self.palm_site = mujoco_named_id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{LEAP_PALM_SITE}")
        self.tip_sites = [mujoco_named_id(model, mujoco.mjtObj.mjOBJ_SITE, f"robot/{tip}") for tip in LEAP_TIPS]

    def observation(self, command: np.ndarray) -> np.ndarray:
        return np.nan_to_num(np.concatenate([self.joint_pos_rel(), self.last_action]), nan=0.0, posinf=0.0, neginf=0.0)

    def cube_state(self) -> np.ndarray:
        quat = self.data.qpos[self.cube_qpos + 3 : self.cube_qpos + 7]
        lin_vel = self.data.qvel[self.cube_dof : self.cube_dof + 3]
        ang_vel = quat_wxyz_to_matrix(quat) @ self.data.qvel[self.cube_dof + 3 : self.cube_dof + 6]
        error = self.data.site_xpos[self.palm_site] - self.data.qpos[self.cube_qpos : self.cube_qpos + 3]
        return np.concatenate((error, quat, lin_vel, ang_vel)).astype(np.float32)

    def fingertips(self) -> np.ndarray:
        return (self.data.site_xpos[self.tip_sites] - self.data.site_xpos[self.palm_site]).reshape(-1).astype(np.float32)

    def is_fallen(self) -> bool:
        return bool(self.data.qpos[self.cube_qpos + 2] < CUBE_DROP_HEIGHT)


class DirectMujocoLeapVecEnv(DirectMujocoVecEnv):
    """Single-environment LEAP cube rotation running on native MuJoCo."""

    backend_cls = PureMujocoLeapModel

    def _setup(self) -> None:
        self._ids = leap_ids(self.model)
        self._reward_weights = {name: float(term.weight) for name, term in self.cfg.rewards.items()}
        self._joint_limits = self.model.jnt_range[[mujoco_named_id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"robot/{name}") for name in self.backend.obs_joint_names]]
        self._palm_z = float(self.data.site_xpos[self.backend.palm_site, 2])

    def _after_reset(self) -> None:
        backend, data = self.backend, self.data
        data.qpos[backend.obs_qpos_adrs] = np.clip(backend.default_joint_pos + self.rng.uniform(-0.1, 0.1, len(backend.obs_qpos_adrs)), *self._joint_limits.T)
        data.qpos[backend.cube_qpos : backend.cube_qpos + 3] = np.asarray(CUBE_INIT_POS) + self.rng.uniform(-0.01, 0.01, 3)
        data.qpos[backend.cube_qpos + 3 : backend.cube_qpos + 7] = _quat_from_euler_xyz(*self.rng.uniform(-np.pi, np.pi, 3))
        data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, data)

    def domain_randomization_vector(self) -> torch.Tensor:
        vector = np.concatenate(([self._tip_friction, self._cube_mass], self._cube_com, self._kp_scale, self._damping_scale, self._frictionloss_scale, self._armature_scale)).astype(np.float32)
        return torch.from_numpy(vector).view(1, -1)

    def _refresh_domain_randomization_cache(self) -> None:
        ids, model, defaults = self._ids, self.model, self._model_defaults
        self._tip_friction = float(model.geom_friction[ids["tip"][0], 0])
        self._cube_mass = float(model.body_mass[ids["cube"]])
        self._cube_com = model.body_ipos[ids["cube"]].astype(np.float32).copy()
        self._kp_scale = np.asarray([(model.actuator_gainprm[a, 0] / defaults.actuator_gainprm[a, 0]).mean() for a in ids["actuator"]], dtype=np.float32)

        def dof_scale(field):
            return np.asarray([(getattr(model, field)[d] / getattr(defaults, field)[d]).mean() for d in ids["dof"]], dtype=np.float32)

        self._damping_scale, self._frictionloss_scale, self._armature_scale = dof_scale("dof_damping"), dof_scale("dof_frictionloss"), dof_scale("dof_armature")

    def _apply_domain_randomization(self) -> None:
        apply_leap_domain_randomization(self.model, self.data, self.rng, self._model_defaults, ids=self._ids)
        self._refresh_domain_randomization_cache()

    def _obs_dict(self, system_termination: bool) -> dict[str, torch.Tensor]:
        backend = self.backend
        actor = backend.observation(self.command)
        cube = backend.cube_state()
        critic = np.concatenate((actor, backend.joint_vel_rel(), backend.actuator_force(), backend.fingertips(), cube[:7], cube[10:13], cube[7:10])).astype(np.float32)
        system_state = np.concatenate((backend.joint_pos_rel(), backend.joint_vel_rel(), cube)).astype(np.float32)
        return {
            "actor": torch.from_numpy(actor.astype(np.float32)).view(1, -1),
            "policy": torch.from_numpy(actor.astype(np.float32)).view(1, -1),
            "critic": torch.from_numpy(critic).view(1, -1),
            "system_state": torch.from_numpy(system_state).view(1, -1),
            "system_action": torch.from_numpy(backend.last_action.copy()).view(1, -1),
            "system_termination": torch.tensor([[float(system_termination)]], dtype=torch.float32, device=self.device),
        }

    def _compute_reward(self, previous_action: np.ndarray, action: np.ndarray) -> tuple[float, dict[str, float], dict[str, float]]:
        cube = self.backend.cube_state()
        terms = {"cube_spin": float(cube[12]), "termination": float(self.backend.is_fallen())}
        weighted = {name: self._reward_weights[name] * value * self.step_dt for name, value in terms.items()}
        return float(sum(weighted.values())), weighted, {"Metrics/cube_spin": terms["cube_spin"]}

    def _cube_dropped(self, state: torch.Tensor) -> torch.Tensor:
        return self._palm_z - state[:, 34] < CUBE_DROP_HEIGHT

    def _imagination_obs_from_state(self, state: torch.Tensor, action: torch.Tensor, termination: torch.Tensor | None = None) -> TensorDict:
        num_envs = state.shape[0]
        zeros = torch.zeros(num_envs, 28, device=state.device, dtype=state.dtype)
        actor = torch.cat((state[:, :16], action), dim=-1)
        critic = torch.cat((actor, state[:, 16:32], zeros, state[:, 32:39], state[:, 42:45], state[:, 39:42]), dim=-1)
        termination_value = torch.zeros(num_envs, 1, device=state.device, dtype=state.dtype) if termination is None else termination.to(device=state.device).float().view(num_envs, 1)
        obs = {"actor": actor, "policy": actor, "critic": critic, "system_state": state, "system_action": action, "system_termination": termination_value}
        return TensorDict(obs, batch_size=[num_envs], device=state.device)

    def _parse_imagination_termination(self, terminations: torch.Tensor | None, state: torch.Tensor) -> torch.Tensor:
        termination = torch.zeros(state.shape[0], device=state.device, dtype=torch.bool) if terminations is None else torch.sigmoid(terminations).reshape(state.shape[0], -1).max(dim=1).values > 0.5
        low, high = torch.as_tensor(self._joint_limits, device=state.device, dtype=state.dtype).unbind(1)
        joint_pos = state[:, :16] + torch.as_tensor(self.backend.default_joint_pos, device=state.device, dtype=state.dtype)
        plausible = ((joint_pos >= 2 * low - high) & (joint_pos <= 2 * high - low)).all(dim=1) & torch.isfinite(state).all(dim=1)
        return termination | self._cube_dropped(state) | ~plausible

    def _compute_imagination_reward(self, state: torch.Tensor, action: torch.Tensor):
        terms = {"cube_spin": state[:, 44], "termination": self._cube_dropped(state).to(state.dtype)}
        weighted = {name: self._reward_weights[name] * value * self.step_dt for name, value in terms.items()}
        return weighted["cube_spin"] + weighted["termination"], weighted, {"Metrics/cube_spin": terms["cube_spin"]}
