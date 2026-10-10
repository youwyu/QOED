"""Direct MuJoCo Go1 VecEnv for RSL-RL training."""

from __future__ import annotations

import mujoco
import numpy as np
import torch

from mbpo.envs.legged_env import LeggedVelocityVecEnv
from mbpo.envs.mujoco_backend import JointPositionMujocoModel
from mbpo.envs.utils import mujoco_named_id

from .utils import GO1_FEET_NAMES, apply_go1_domain_randomization, mujoco_actuator_ids_by_joint_type


class DirectMujocoGo1VecEnv(LeggedVelocityVecEnv):
    """Single-environment Go1 velocity task running on native MuJoCo."""

    backend_cls = JointPositionMujocoModel
    feet = tuple((name, (f"{name}_foot_collision",)) for name in GO1_FEET_NAMES)
    upright_body = "trunk"

    def _setup_robot(self) -> None:
        self._torso_body_id = mujoco_named_id(self.model, mujoco.mjtObj.mjOBJ_BODY, "robot/trunk")
        self._actuator_type_ids = mujoco_actuator_ids_by_joint_type(self.model)
        self._payload = 0.0

    def domain_randomization_vector(self) -> torch.Tensor:
        vector = np.concatenate(
            (
                np.asarray([self._floor_friction], dtype=np.float32),
                self._body_mass,
                self._dof_frictionloss,
                self._dof_armature,
                self._kp_scale,
                self._kd_scale,
                np.asarray([self._payload], dtype=np.float32),
            ),
            axis=0,
        )
        return torch.from_numpy(vector).view(1, -1)

    def _refresh_domain_randomization_cache(self) -> None:
        self._floor_friction = float(self.model.geom_friction[self.terrain_geom_id, 0])
        self._body_mass = self.model.body_mass[self._robot_body_ids].astype(np.float32).copy()
        self._body_mass[self._robot_body_ids.index(self._torso_body_id)] -= self._payload
        self._dof_frictionloss = self.model.dof_frictionloss[self._robot_dof_ids].astype(np.float32).copy()
        self._dof_armature = self.model.dof_armature[self._robot_dof_ids].astype(np.float32).copy()
        defaults = self._model_defaults
        self._kp_scale = np.asarray([(self.model.actuator_gainprm[ids, 0] / defaults.actuator_gainprm[ids, 0]).mean() for ids in self._actuator_type_ids], dtype=np.float32)
        self._kd_scale = np.asarray([(self.model.actuator_biasprm[ids, 2] / defaults.actuator_biasprm[ids, 2]).mean() for ids in self._actuator_type_ids], dtype=np.float32)

    def _apply_domain_randomization(self) -> None:
        self._payload = apply_go1_domain_randomization(
            self.model,
            self.data,
            self.rng,
            self._model_defaults,
            terrain_geom_id=self.terrain_geom_id,
            robot_dof_ids=self._robot_dof_ids,
            torso_body_id=self._torso_body_id,
            body_mass_ids=self._robot_body_ids,
            robot_qpos_ids=self._robot_qpos_ids,
            actuator_type_ids=self._actuator_type_ids,
            dtype=np.float32,
        )
        self._refresh_domain_randomization_cache()
