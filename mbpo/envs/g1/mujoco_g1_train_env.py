"""Direct MuJoCo G1 VecEnv for RSL-RL training."""

from __future__ import annotations

import numpy as np
import torch

from mbpo.envs.legged_env import LeggedVelocityVecEnv
from mbpo.envs.mujoco_backend import JointPositionMujocoModel

from .utils import G1_FEET, G1_TORSO, apply_g1_domain_randomization, g1_ids


class PureMujocoG1Model(JointPositionMujocoModel):
    camera_distance = 3.0


class DirectMujocoG1VecEnv(LeggedVelocityVecEnv):
    """Single-environment G1 velocity task running on native MuJoCo."""

    backend_cls = PureMujocoG1Model
    feet = G1_FEET
    upright_body = G1_TORSO
    include_torque = False

    def _setup_robot(self) -> None:
        self._ids = g1_ids(self.model)
        self._payload = 0.0

    def domain_randomization_vector(self) -> torch.Tensor:
        vector = np.concatenate(
            (
                np.asarray([self._foot_friction, self._torso_mass], dtype=np.float32),
                self._kp_scale,
                self._kd_scale,
                self._armature_scale,
                np.asarray([self._payload], dtype=np.float32),
            ),
            axis=0,
        )
        return torch.from_numpy(vector).view(1, -1)

    def _refresh_domain_randomization_cache(self) -> None:
        ids, model, defaults = self._ids, self.model, self._model_defaults
        self._foot_friction = float(model.geom_friction[ids["foot"][0], 0])
        self._torso_mass = float(model.body_mass[ids["torso"]]) - self._payload
        self._kp_scale = np.asarray([(model.actuator_gainprm[a, 0] / defaults.actuator_gainprm[a, 0]).mean() for a in ids["actuator"]], dtype=np.float32)
        self._kd_scale = np.asarray([(model.actuator_biasprm[a, 2] / defaults.actuator_biasprm[a, 2]).mean() for a in ids["actuator"]], dtype=np.float32)
        self._armature_scale = np.asarray([(model.dof_armature[d] / defaults.dof_armature[d]).mean() for d in ids["dof"]], dtype=np.float32)

    def _apply_domain_randomization(self) -> None:
        self._payload = apply_g1_domain_randomization(self.model, self.data, self.rng, self._model_defaults, ids=self._ids)
        self._refresh_domain_randomization_cache()
