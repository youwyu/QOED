"""Direct MuJoCo Go1 backend for play and train."""

from __future__ import annotations

import re

import numpy as np

from mbpo.envs.mujoco_backend import PureMujocoModel


def _resolve_name_values(
    values: float | int | dict[str, float],
    names: list[str],
    default: float,
) -> np.ndarray:
    if isinstance(values, (float, int)):
        return np.full(len(names), float(values), dtype=np.float32)
    resolved = np.full(len(names), float(default), dtype=np.float32)
    matched = np.zeros(len(names), dtype=bool)
    for pattern, value in values.items():
        regex = re.compile(pattern)
        for index, name in enumerate(names):
            if regex.match(name):
                resolved[index] = float(value)
                matched[index] = True
    if not bool(matched.all()):
        missing = ", ".join(name for name, ok in zip(names, matched) if not ok)
        raise RuntimeError(f"Action scale did not match joints: {missing}")
    return resolved


class PureMujocoGo1Model(PureMujocoModel):
    """Single-Go1 MuJoCo model with MJLab-compatible joint position actions."""

    action_name = "joint_pos"

    def _setup_actions(self, action_cfg) -> int:
        action_to_obs_indices = np.asarray([self.obs_joint_names.index(name) for name in self.action_joint_names], dtype=np.int32)
        self.default_action_joint_pos = self.default_joint_pos[action_to_obs_indices]
        self.action_scale = _resolve_name_values(action_cfg.scale, self.action_joint_names, 1.0)
        self.default_ctrl = self.default_action_joint_pos[self.ctrl_action_indices].astype(np.float64)
        return len(self.action_joint_names)

    def ctrl(self, action: np.ndarray) -> np.ndarray:
        target_action_joint_pos = self.default_action_joint_pos + action * self.action_scale
        return target_action_joint_pos[self.ctrl_action_indices].astype(np.float64)

    def joint_observation(self) -> list[np.ndarray]:
        return [self.joint_pos_rel(), self.joint_vel_rel()]
