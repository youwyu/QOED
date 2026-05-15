from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ..common import axis_angle_to_quat, quat_mul
from .utils import (
    GO1_FEET_NAMES,
    apply_go1_domain_randomization,
    save_mujoco_model_defaults,
)

GO1_FEET_SITES = GO1_FEET_NAMES
GO1_SENSOR_NAMES = (
    "upvector",
    "global_linvel",
    "global_angvel",
    "local_linvel",
    "accelerometer",
    "gyro",
)


class Go1JoystickEnv(gym.Env):
    """
    Classic MuJoCo port of the upstream Go1 joystick task.

    The policy observation in `obs["state"]` matches the upstream layout:
    [local_linvel, gyro, gravity, joint_pos_rel, joint_vel, last_act, command]
    """

    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        xml_path: str,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        path = Path(xml_path)
        if not path.exists():
            raise FileNotFoundError(f"XML not found: {path}")

        self.render_mode = render_mode
        self.dtype = np.float32
        self.model = mujoco.MjModel.from_xml_path(str(path))
        self.data = mujoco.MjData(self.model)

        # ------------------------------------------------------------------
        # Config aligned with the upstream Go1 joystick reference task.
        # ------------------------------------------------------------------
        self.ctrl_dt = 0.02
        self.sim_dt = 0.004
        self.episode_length = 1000
        self.Kp = 35.0
        self.Kd = 0.5
        self.action_repeat = 1
        self.soft_joint_pos_limit_factor = 0.95

        self.noise_level = 1.0
        self.noise_scales = {
            "joint_pos": 0.03,
            "joint_vel": 1.5,
            "gyro": 0.2,
            "gravity": 0.05,
            "linvel": 0.1,
        }

        self.reward_scales = {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -0.5,
            "ang_vel_xy": -0.05,
            "orientation": -5.0,
            "dof_pos_limits": -1.0,
            "pose": 0.5,
            "termination": -1.0,
            "stand_still": -1.0,
            "torques": -0.0002,
            "action_rate": -0.01,
            "energy": -0.001,
            "feet_clearance": -2.0,
            "feet_height": -0.2,
            "feet_slip": -0.1,
            "feet_air_time": 0.1,
        }
        self.tracking_sigma = 0.25
        self.max_foot_height = 0.1

        self.pert_enable = False
        self.pert_velocity_kick = [0.0, 3.0]
        self.pert_kick_durations = [0.05, 0.2]
        self.pert_kick_wait_times = [1.0, 3.0]

        self._cmd_a = np.array([1.5, 0.8, 1.2], dtype=self.dtype)
        self._cmd_b = np.array([0.9, 0.25, 0.5], dtype=self.dtype)

        self._rng = np.random.RandomState(0)

        self.model.opt.timestep = self.sim_dt
        self.model.dof_damping[6:] = self.Kd
        self.model.actuator_gainprm[:, 0] = self.Kp
        self.model.actuator_biasprm[:, 1] = -self.Kp

        mujoco.mj_forward(self.model, self.data)

        self._dt = self.ctrl_dt
        self._skip = int(round(self._dt / self.model.opt.timestep))
        self.max_episode_length = int(self.episode_length)

        self.n_body = self.model.nbody
        self.nu = int(self.model.nu)
        self._joint_qpos_slice = slice(7, 7 + self.nu)
        self._joint_qvel_slice = slice(6, 6 + self.nu)
        self._domain_param_dim = 1 + 3 + self.n_body + 12 + 12 + 12

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.nu,),
            dtype=self.dtype,
        )
        self.single_action_space = self.action_space

        self.state_dim = 48
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.state_dim,),
                    dtype=self.dtype,
                ),
            }
        )
        self.privilege_space = spaces.Dict(
            {
                "privilege": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self._domain_param_dim,),
                    dtype=self.dtype,
                ),
            }
        )

        self.renderer = None
        self.model.vis.map.znear = 0.001
        self.model.vis.map.zfar = 50
        self.cam = -1
        for camera_name in ("camera", "track", "side", "back", "top"):
            camera_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
            if camera_id >= 0:
                self.cam = camera_id
                break

        self._init_q = self.model.keyframe("home").qpos.copy()
        self._default_pose = self._init_q[7:].copy()

        lowers, uppers = self.model.jnt_range[1:].T
        self._physical_joint_lowers = lowers.astype(self.dtype)
        self._physical_joint_uppers = uppers.astype(self.dtype)
        self._soft_lowers = (lowers * self.soft_joint_pos_limit_factor).astype(self.dtype)
        self._soft_uppers = (uppers * self.soft_joint_pos_limit_factor).astype(self.dtype)

        self._torso_body_id = self.model.body("trunk").id
        self._torso_mass = self.model.body_subtreemass[self._torso_body_id]

        self._feet_sites = list(GO1_FEET_SITES)
        self._feet_site_ids = np.array(
            [self.model.site(name).id for name in self._feet_sites],
            dtype=int,
        )
        self._floor_geom_id = self.model.geom("floor").id

        foot_linvel_sensor_adr = []
        for site in self._feet_sites:
            sensor_id = self.model.sensor(f"{site}_global_linvel").id
            adr = self.model.sensor_adr[sensor_id]
            dim = self.model.sensor_dim[sensor_id]
            foot_linvel_sensor_adr.append(list(range(adr, adr + dim)))
        self._foot_linvel_sensor_adr = np.array(foot_linvel_sensor_adr, dtype=int)

        self._imu_site_id = self.model.site("imu").id

        self._feet_floor_found_sensor_ids = [
            self.model.sensor(f"{site}_floor_found").id for site in self._feet_sites
        ]

        self._sensor_ids: Dict[str, int] = {}
        for name in GO1_SENSOR_NAMES:
            self._sensor_ids[name] = self.model.sensor(name).id

        self._step_count = 0
        self.last_act = np.zeros(self.nu, dtype=self.dtype)
        self.last_last_act = np.zeros(self.nu, dtype=self.dtype)
        self.command = np.zeros(3, dtype=self.dtype)
        self.feet_air_time = np.zeros(4, dtype=self.dtype)
        self.last_contact = np.zeros(4, dtype=bool)
        self.swing_peak = np.zeros(4, dtype=self.dtype)
        self.steps_until_next_cmd = 0
        self.steps_since_last_pert = 0
        self.steps_until_next_pert = 0
        self.pert_duration_steps = 0
        self.pert_duration_seconds = 0.0
        self.pert_dir = np.zeros(3, dtype=self.dtype)
        self.pert_mag = 0.0
        self.pert_steps = 0
        self.crash_condition = False
        self.truncation = False
        self._last_state = np.zeros(self.state_dim, dtype=self.dtype)

        self._model_defaults = save_mujoco_model_defaults(self.model)

    # ------------------------------------------------------------------
    # Data utilities
    # ------------------------------------------------------------------

    def set_data(self, new_data):
        self.data = new_data
        mujoco.mj_forward(self.model, self.data)

    def get_data(self):
        return copy.deepcopy(self.data)

    def _get_domain_rand_vector(self) -> np.ndarray:
        floor_fric = np.array(
            [self.model.geom_friction[self._floor_geom_id, 0]],
            dtype=self.dtype,
        )
        torso_ipos = self.model.body_ipos[self._torso_body_id].astype(self.dtype)
        body_mass = self.model.body_mass.astype(self.dtype)
        qpos0 = self.model.qpos0[self._joint_qpos_slice].astype(self.dtype)
        frictionloss = self.model.dof_frictionloss[self._joint_qvel_slice].astype(self.dtype)
        armature = self.model.dof_armature[self._joint_qvel_slice].astype(self.dtype)

        vec = np.concatenate(
            [
                floor_fric,
                torso_ipos,
                body_mass,
                qpos0,
                frictionloss,
                armature,
            ]
        ).astype(self.dtype)
        assert vec.shape[0] == self._domain_param_dim
        return vec

    # ------------------------------------------------------------------
    # Sensor helpers
    # ------------------------------------------------------------------

    def _get_sensor(self, name: str) -> np.ndarray:
        sid = self._sensor_ids[name]
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return self.data.sensordata[adr : adr + dim].astype(self.dtype)

    def get_upvector(self) -> np.ndarray:
        return self._get_sensor("upvector")

    def get_global_linvel(self) -> np.ndarray:
        return self._get_sensor("global_linvel")

    def get_global_angvel(self) -> np.ndarray:
        return self._get_sensor("global_angvel")

    def get_local_linvel(self) -> np.ndarray:
        return self._get_sensor("local_linvel")

    def get_accelerometer(self) -> np.ndarray:
        return self._get_sensor("accelerometer")

    def get_gyro(self) -> np.ndarray:
        return self._get_sensor("gyro")

    def get_gravity(self) -> np.ndarray:
        r_flat = self.data.site_xmat[self._imu_site_id].copy()
        r_mat = r_flat.reshape(3, 3)
        return (r_mat.T @ np.array([0.0, 0.0, -1.0], dtype=self.dtype)).astype(self.dtype)

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(self, seed: Optional[int] = None):
        if seed is not None:
            self._rng = np.random.RandomState(seed)

        self.domain_randomize()
        mujoco.mj_resetData(self.model, self.data)

        qpos = self._init_q.copy()
        qvel = np.zeros(self.model.nv, dtype=self.dtype)

        dxy = self._rng.uniform(-0.5, 0.5, size=2).astype(self.dtype)
        qpos[0:2] += dxy

        yaw = float(self._rng.uniform(-3.14, 3.14))
        quat_yaw = axis_angle_to_quat([0.0, 0.0, 1.0], yaw)
        qpos[3:7] = quat_mul(qpos[3:7], quat_yaw)

        qvel[0:6] = self._rng.uniform(-0.5, 0.5, size=6).astype(self.dtype)

        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[:] = qpos[self._joint_qpos_slice]
        mujoco.mj_forward(self.model, self.data)

        t_next = self._rng.uniform(
            self.pert_kick_wait_times[0],
            self.pert_kick_wait_times[1],
        )
        self.steps_until_next_pert = int(round(t_next / self._dt))

        dur_sec = self._rng.uniform(
            self.pert_kick_durations[0],
            self.pert_kick_durations[1],
        )
        self.pert_duration_seconds = float(dur_sec)
        self.pert_duration_steps = int(round(dur_sec / self._dt))
        self.pert_mag = float(
            self._rng.uniform(self.pert_velocity_kick[0], self.pert_velocity_kick[1])
        )

        self.steps_until_next_cmd = self._sample_command_wait_steps()
        self.command = self._rng.uniform(-self._cmd_a, self._cmd_a).astype(self.dtype)

        self._step_count = 0
        self.last_act.fill(0.0)
        self.last_last_act.fill(0.0)
        self.feet_air_time.fill(0.0)
        self.last_contact.fill(False)
        self.swing_peak.fill(0.0)
        self.steps_since_last_pert = 0
        self.pert_steps = 0
        self.pert_dir.fill(0.0)
        self.crash_condition = False
        self.truncation = False

        obs = self.compute_obs()
        info = self.compute_info(obs["state"])
        return obs, info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=self.dtype)
        # action = np.clip(action, -1.0, 1.0)

        if self.pert_enable:
            self._maybe_apply_perturbation()

        motor_targets = self._policy_to_motor_targets(action)
        self.data.ctrl[: self.nu] = motor_targets

        for _ in range(self._skip):
            mujoco.mj_step(self.model, self.data)

        self._step_count += 1

        contact = np.array(
            [
                self.data.sensordata[self.model.sensor_adr[sid]] > 0.0
                for sid in self._feet_floor_found_sensor_ids
            ],
            dtype=bool,
        )
        contact_filt = np.logical_or(contact, self.last_contact)
        first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self._dt

        feet_pos = self.data.site_xpos[self._feet_site_ids]
        self.swing_peak = np.maximum(self.swing_peak, feet_pos[:, 2])

        obs = self.compute_obs()
        reward, terminated = self.compute_reward_and_done(action, contact, first_contact)
        info = self.compute_info(obs["state"])

        self.last_last_act = self.last_act.copy()
        self.last_act = action.copy()

        self.steps_until_next_cmd -= 1
        if self.steps_until_next_cmd <= 0:
            self.command = self.sample_command(self.command)
        if terminated or self.steps_until_next_cmd <= 0:
            self.steps_until_next_cmd = self._sample_command_wait_steps()

        self.feet_air_time *= (~contact).astype(self.dtype)
        self.last_contact = contact
        self.swing_peak *= (~contact).astype(self.dtype)

        self.crash_condition = bool(terminated)
        self.truncation = self._step_count >= self.max_episode_length

        return obs, reward, self.crash_condition, self.truncation, info

    # ------------------------------------------------------------------
    # Observation / info
    # ------------------------------------------------------------------

    def _policy_to_motor_targets(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=self.dtype)
        # action_normalized = (action + 1.0) * 0.5
        # motor_targets = self._soft_lowers + action_normalized * (
        #     self._soft_uppers - self._soft_lowers
        # )
        motor_targets = self._default_pose + action
        return np.clip(
            motor_targets,
            self._physical_joint_lowers,
            self._physical_joint_uppers,
        ).astype(self.dtype)

    def _sample_command_wait_steps(self) -> int:
        time_until_next_cmd = self._rng.exponential() * 5.0
        return int(round(time_until_next_cmd / self._dt))

    def sample_command(self, current_command: np.ndarray) -> np.ndarray:
        y_k = self._rng.uniform(-self._cmd_a, self._cmd_a).astype(self.dtype)
        z_k = self._rng.binomial(1, self._cmd_b, size=3).astype(self.dtype)
        w_k = self._rng.binomial(1, 0.5, size=3).astype(self.dtype)
        return (current_command - w_k * (current_command - y_k * z_k)).astype(self.dtype)

    def compute_obs(self) -> Dict[str, np.ndarray]:
        gyro = self.get_gyro()
        gravity = self.get_gravity()
        joint_angles = self.data.qpos[self._joint_qpos_slice].astype(self.dtype)
        joint_vel = self.data.qvel[self._joint_qvel_slice].astype(self.dtype)
        linvel = self.get_local_linvel()

        def add_noise(x: np.ndarray, key: str) -> np.ndarray:
            if self.noise_level <= 0.0:
                return x
            noise = self._rng.uniform(-1.0, 1.0, size=x.shape).astype(self.dtype)
            return x + noise * self.noise_level * self.noise_scales[key]

        noisy_gyro = add_noise(gyro, "gyro")
        noisy_gravity = add_noise(gravity, "gravity")
        noisy_joint_angles = add_noise(joint_angles, "joint_pos")
        noisy_joint_vel = add_noise(joint_vel, "joint_vel")
        noisy_linvel = add_noise(linvel, "linvel")

        state = np.concatenate(
            [
                noisy_linvel,
                noisy_gyro,
                noisy_gravity,
                noisy_joint_angles - self._default_pose,
                noisy_joint_vel,
                self.last_act,
                self.command,
            ]
        ).astype(self.dtype)

        self._last_state = state
        return {"state": state}

    def _build_privileged_state(self, state: np.ndarray) -> np.ndarray:
        gyro = self.get_gyro()
        accelerometer = self.get_accelerometer()
        gravity = self.get_gravity()
        linvel = self.get_local_linvel()
        angvel = self.get_global_angvel()
        joint_angles = self.data.qpos[self._joint_qpos_slice].astype(self.dtype)
        joint_vel = self.data.qvel[self._joint_qvel_slice].astype(self.dtype)
        feet_vel = self.data.sensordata[self._foot_linvel_sensor_adr].ravel().astype(self.dtype)
        pert_active = np.array(
            [float(self.steps_since_last_pert >= self.steps_until_next_pert)],
            dtype=self.dtype,
        )

        return np.concatenate(
            [
                state,
                gyro.astype(self.dtype),
                accelerometer.astype(self.dtype),
                gravity.astype(self.dtype),
                linvel.astype(self.dtype),
                angvel.astype(self.dtype),
                joint_angles - self._default_pose,
                joint_vel,
                self.data.actuator_force[: self.nu].astype(self.dtype),
                self.last_contact.astype(self.dtype),
                feet_vel,
                self.feet_air_time.astype(self.dtype),
                self.data.xfrc_applied[self._torso_body_id, :3].astype(self.dtype),
                pert_active,
            ]
        ).astype(self.dtype)

    def compute_info(self, state: Optional[np.ndarray] = None) -> Dict[str, Any]:
        state = self._last_state if state is None else state
        return {
            "privilege": self._get_domain_rand_vector(),
            "privileged_state": self._build_privileged_state(state),
            "command": self.command.copy(),
            "feet_air_time": self.feet_air_time.copy(),
            "last_contact": self.last_contact.copy(),
            "swing_peak": self.swing_peak.copy(),
            "steps_since_last_pert": int(self.steps_since_last_pert),
            "steps_until_next_pert": int(self.steps_until_next_pert),
        }

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def _get_termination_flag(self) -> bool:
        return bool(self.get_upvector()[-1] < 0.0)

    def compute_reward_and_done(
        self,
        action: np.ndarray,
        contact: np.ndarray,
        first_contact: np.ndarray,
    ) -> tuple[float, bool]:
        rewards = {
            "tracking_lin_vel": self._reward_tracking_lin_vel(self.command, self.get_local_linvel()),
            "tracking_ang_vel": self._reward_tracking_ang_vel(self.command, self.get_gyro()),
            "lin_vel_z": self._cost_lin_vel_z(self.get_global_linvel()),
            "ang_vel_xy": self._cost_ang_vel_xy(self.get_global_angvel()),
            "orientation": self._cost_orientation(self.get_upvector()),
            "stand_still": self._cost_stand_still(
                self.command,
                self.data.qpos[self._joint_qpos_slice],
            ),
            "termination": self._cost_termination(self._get_termination_flag()),
            "pose": self._reward_pose(self.data.qpos[self._joint_qpos_slice]),
            "torques": self._cost_torques(self.data.actuator_force[: self.nu]),
            "action_rate": self._cost_action_rate(action, self.last_act, self.last_last_act),
            "energy": self._cost_energy(
                self.data.qvel[self._joint_qvel_slice],
                self.data.actuator_force[: self.nu],
            ),
            "feet_slip": self._cost_feet_slip(contact),
            "feet_clearance": self._cost_feet_clearance(),
            "feet_height": self._cost_feet_height(first_contact),
            "feet_air_time": self._reward_feet_air_time(first_contact),
            "dof_pos_limits": self._cost_joint_pos_limits(
                self.data.qpos[self._joint_qpos_slice]
            ),
        }

        reward = sum(self.reward_scales[k] * float(v) for k, v in rewards.items())
        reward = reward * self._dt #float(np.clip(reward * self._dt, 0.0, 10000.0))
        terminated = self._get_termination_flag()
        return reward, terminated

    def _reward_tracking_lin_vel(self, commands: np.ndarray, local_vel: np.ndarray) -> float:
        lin_vel_error = np.sum((commands[:2] - local_vel[:2]) ** 2)
        return float(np.exp(-lin_vel_error / self.tracking_sigma))

    def _reward_tracking_ang_vel(self, commands: np.ndarray, ang_vel: np.ndarray) -> float:
        ang_vel_error = (commands[2] - ang_vel[2]) ** 2
        return float(np.exp(-ang_vel_error / self.tracking_sigma))

    def _cost_lin_vel_z(self, global_linvel: np.ndarray) -> float:
        return float(global_linvel[2] ** 2)

    def _cost_ang_vel_xy(self, global_angvel: np.ndarray) -> float:
        return float(np.sum(global_angvel[:2] ** 2))

    def _cost_orientation(self, torso_zaxis: np.ndarray) -> float:
        return float(np.sum(torso_zaxis[:2] ** 2))

    def _cost_torques(self, torques: np.ndarray) -> float:
        return float(np.sqrt(np.sum(torques**2)) + np.sum(np.abs(torques)))

    def _cost_energy(self, qvel: np.ndarray, qfrc_actuator: np.ndarray) -> float:
        return float(np.sum(np.abs(qvel) * np.abs(qfrc_actuator)))

    def _cost_action_rate(
        self,
        act: np.ndarray,
        last_act: np.ndarray,
        last_last_act: np.ndarray,
    ) -> float:
        del last_last_act
        return float(np.sum((act - last_act) ** 2))

    def _reward_pose(self, qpos: np.ndarray) -> float:
        weight = np.array([1.0, 1.0, 0.1] * 4, dtype=self.dtype)
        return float(np.exp(-np.sum(((qpos - self._default_pose) ** 2) * weight)))

    def _cost_stand_still(self, commands: np.ndarray, qpos: np.ndarray) -> float:
        cmd_norm = np.linalg.norm(commands)
        return float(np.sum(np.abs(qpos - self._default_pose)) * (cmd_norm < 0.01))

    def _cost_termination(self, done_flag: bool) -> float:
        return float(done_flag)

    def _cost_joint_pos_limits(self, qpos: np.ndarray) -> float:
        out_of_limits = -np.clip(qpos - self._soft_lowers, None, 0.0)
        out_of_limits += np.clip(qpos - self._soft_uppers, 0.0, None)
        return float(np.sum(out_of_limits))

    def _cost_feet_slip(self, contact: np.ndarray) -> float:
        cmd_norm = np.linalg.norm(self.command)
        feet_vel = self.data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[:, :2]
        vel_xy_norm_sq = np.sum(vel_xy**2, axis=-1)
        return float(np.sum(vel_xy_norm_sq * contact.astype(self.dtype)) * (cmd_norm > 0.01))

    def _cost_feet_clearance(self) -> float:
        feet_vel = self.data.sensordata[self._foot_linvel_sensor_adr]
        vel_xy = feet_vel[:, :2]
        vel_norm = np.sqrt(np.linalg.norm(vel_xy, axis=-1) + 1e-8)
        foot_pos = self.data.site_xpos[self._feet_site_ids]
        foot_z = foot_pos[:, 2]
        delta = np.abs(foot_z - self.max_foot_height)
        return float(np.sum(delta * vel_norm))

    def _cost_feet_height(self, first_contact: np.ndarray) -> float:
        cmd_norm = np.linalg.norm(self.command)
        error = self.swing_peak / self.max_foot_height - 1.0
        return float(np.sum((error**2) * first_contact.astype(self.dtype)) * (cmd_norm > 0.01))

    def _reward_feet_air_time(self, first_contact: np.ndarray) -> float:
        cmd_norm = np.linalg.norm(self.command)
        rew_air_time = np.sum((self.feet_air_time - 0.1) * first_contact.astype(self.dtype))
        rew_air_time *= cmd_norm > 0.01
        return float(rew_air_time)

    # ------------------------------------------------------------------
    # Perturbation
    # ------------------------------------------------------------------

    def domain_randomize(self, seed: int | None = None) -> None:
        rng = self._rng if seed is None else np.random.RandomState(seed)
        apply_go1_domain_randomization(
            self.model,
            self.data,
            rng,
            self._model_defaults,
            terrain_geom_id=self._floor_geom_id,
            robot_dof_ids=self._joint_qvel_slice,
            torso_body_id=self._torso_body_id,
            body_mass_ids=slice(None),
            robot_qpos_ids=self._joint_qpos_slice,
            dtype=self.dtype,
        )

    def _maybe_apply_perturbation(self):
        if not self.pert_enable:
            self.data.xfrc_applied[:, :] = 0.0
            return

        if self.steps_since_last_pert >= self.steps_until_next_pert:
            t = self.pert_steps * self._dt
            u_t = 0.5 * np.sin(np.pi * t / max(self.pert_duration_seconds, 1e-6))
            force = (
                u_t
                * self._torso_mass
                * self.pert_mag
                / max(self.pert_duration_seconds, 1e-6)
            )
            self.data.xfrc_applied[:, :] = 0.0
            self.data.xfrc_applied[self._torso_body_id, :3] = force * self.pert_dir
            if self.pert_steps >= self.pert_duration_steps:
                self.steps_since_last_pert = 0
            self.pert_steps += 1
        else:
            self.steps_since_last_pert += 1
            self.data.xfrc_applied[:, :] = 0.0
            if self.steps_since_last_pert >= self.steps_until_next_pert:
                angle = self._rng.uniform(0.0, 2.0 * np.pi)
                self.pert_dir = np.array([np.cos(angle), np.sin(angle), 0.0], dtype=self.dtype)
                self.pert_steps = 0

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self, mode: str = "rgb_array"):
        if mode != "rgb_array":
            raise NotImplementedError("Only 'rgb_array' render mode is supported.")
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, width=1280, height=720)
        self.renderer.update_scene(self.data, camera=self.cam)
        return self.renderer.render()


__all__ = ["Go1JoystickEnv"]
