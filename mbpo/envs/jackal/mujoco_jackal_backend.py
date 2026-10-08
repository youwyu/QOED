"""Direct MuJoCo Jackal backend for play and train."""

from __future__ import annotations

import numpy as np

from mbpo.envs.mujoco_backend import PureMujocoModel

from .utils import JACKAL_MAX_WHEEL_SPEED, diff_drive_mix


class PureMujocoJackalModel(PureMujocoModel):
    """Single-Jackal MuJoCo model with MJLab-compatible differential-drive actions."""

    action_name = "wheel_vel"
    camera_distance = 2.5

    def _setup_actions(self, action_cfg) -> int:
        self.ctrl_mix = diff_drive_mix(self.action_joint_names)[:, self.ctrl_action_indices].astype(np.float32)
        self.default_ctrl = np.zeros(self.model.nu, dtype=np.float64)
        return 2

    def ctrl(self, action: np.ndarray) -> np.ndarray:
        return np.clip(action @ self.ctrl_mix, -JACKAL_MAX_WHEEL_SPEED, JACKAL_MAX_WHEEL_SPEED).astype(np.float64)

    def joint_observation(self) -> list[np.ndarray]:
        return [self.joint_vel_rel()]
