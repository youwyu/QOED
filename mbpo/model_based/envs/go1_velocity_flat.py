import math

import torch
from tensordict import TensorDict

from envs.base import BaseEnv


class Go1VelocityFlatEnv(BaseEnv):
    _COMMAND_RANGES = {
        "lin_vel_x": (-1.0, 1.0),
        "lin_vel_y": (-1.0, 1.0),
        "ang_vel_z": (-0.5, 0.5),
    }
    _STANDING_ENV_PROBABILITY = 0.1
    _WALKING_THRESHOLD = 0.05
    _RUNNING_THRESHOLD = 1.5
    _BAD_ORIENTATION_LIMIT = math.radians(70.0)

    def _init_additional_imagination_attributes(self):
        self.current_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self._pose_std_standing = torch.tensor(
            [0.05, 0.05, 0.1] * 4,
            device=self.device,
        )
        self._pose_std_walking = torch.tensor(
            [0.3, 0.3, 0.6] * 4,
            device=self.device,
        )
        self._pose_std_running = torch.tensor(
            [0.3, 0.3, 0.6] * 4,
            device=self.device,
        )

    def _reset_additional_imagination_attributes(self, env_ids):
        self.current_air_time[env_ids] = 0.0

    def _init_imagination_command(self):
        self.base_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._reset_imagination_command(torch.arange(self.num_envs, device=self.device))

    def _reset_imagination_command(self, env_ids):
        if len(env_ids) == 0:
            return

        self.base_velocity[env_ids, 0] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self._COMMAND_RANGES["lin_vel_x"]
        )
        self.base_velocity[env_ids, 1] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self._COMMAND_RANGES["lin_vel_y"]
        )
        self.base_velocity[env_ids, 2] = torch.empty(len(env_ids), device=self.device).uniform_(
            *self._COMMAND_RANGES["ang_vel_z"]
        )

        standing = torch.empty(len(env_ids), device=self.device).uniform_(0.0, 1.0) <= self._STANDING_ENV_PROBABILITY
        self.is_standing_env[env_ids] = standing
        self.base_velocity[env_ids[standing]] = 0.0

        if hasattr(self, "last_obs"):
            self.last_obs["policy"][env_ids, 45:48] = self.base_velocity[env_ids]

    def get_imagination_observation(self, state_history, action_history):
        state_history_denormalized, action_history_denormalized = self.dataset.denormalize(
            state_history[:, -1],
            action_history[:, -1],
        )
        obs_base_lin_vel = state_history_denormalized[:, 0:3]
        obs_base_ang_vel = state_history_denormalized[:, 3:6]
        obs_projected_gravity = state_history_denormalized[:, 6:9]
        obs_joint_pos = state_history_denormalized[:, 9:21]
        obs_joint_vel = state_history_denormalized[:, 21:33]
        self.obs_last_action = action_history_denormalized

        if self.observation_noise:
            obs_base_lin_vel += 2 * (torch.rand_like(obs_base_lin_vel) - 0.5) * 0.5
            obs_base_ang_vel += 2 * (torch.rand_like(obs_base_ang_vel) - 0.5) * 0.2
            obs_projected_gravity += 2 * (torch.rand_like(obs_projected_gravity) - 0.5) * 0.05
            obs_joint_pos += 2 * (torch.rand_like(obs_joint_pos) - 0.5) * 0.01
            obs_joint_vel += 2 * (torch.rand_like(obs_joint_vel) - 0.5) * 1.5

        obs = torch.cat(
            [
                obs_base_lin_vel,
                obs_base_ang_vel,
                obs_projected_gravity,
                obs_joint_pos,
                obs_joint_vel,
                self.obs_last_action,
                self.base_velocity,
            ],
            dim=1,
        )
        obs = TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
        self.last_obs = obs
        return obs

    def _parse_imagination_states(self, imagination_states_denormalized):
        base_lin_vel = imagination_states_denormalized[:, 0:3]
        base_ang_vel = imagination_states_denormalized[:, 3:6]
        projected_gravity = imagination_states_denormalized[:, 6:9]
        joint_pos = imagination_states_denormalized[:, 9:21]
        joint_vel = imagination_states_denormalized[:, 21:33]
        joint_torque = imagination_states_denormalized[:, 33:45]

        return {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
        }

    def _parse_extensions(self, extensions):
        if extensions is None:
            return None
        return {}

    def _parse_contacts(self, contacts):
        if contacts is None:
            foot_contact = None
        elif contacts.shape[-1] >= 8:
            foot_contact = torch.sigmoid(contacts[:, 4:8]).round()
        else:
            foot_contact = torch.sigmoid(contacts[:, :4]).round()

        return {"foot_contact": foot_contact}

    def _parse_terminations(self, terminations):
        if terminations is None:
            return None
        return torch.sigmoid(terminations).squeeze(-1).round().bool()

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        del parsed_extensions

        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        projected_gravity = parsed_imagination_states["projected_gravity"]
        joint_pos = parsed_imagination_states["joint_pos"]
        foot_contact = parsed_contacts["foot_contact"]

        self.termination_flags = self._combine_bad_orientation_termination(projected_gravity)

        lin_vel_error = torch.sum(torch.square(self.base_velocity[:, :2] - base_lin_vel[:, :2]), dim=1)
        lin_vel_error += torch.square(base_lin_vel[:, 2])
        ang_vel_error = torch.square(self.base_velocity[:, 2] - base_ang_vel[:, 2])
        ang_vel_error += torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)

        track_linear_velocity = torch.exp(-lin_vel_error / 0.25)
        track_angular_velocity = torch.exp(-ang_vel_error / 0.5)
        upright = torch.exp(-torch.sum(torch.square(projected_gravity[:, :2]), dim=1) / 0.2)
        pose = self._compute_pose_reward(joint_pos)
        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)
        air_time = self._compute_air_time_reward(foot_contact)
        dof_pos_limits = torch.zeros(self.num_envs, device=self.device)

        self.imagination_reward_per_step = {
            "track_linear_velocity": track_linear_velocity,
            "track_angular_velocity": track_angular_velocity,
            "upright": upright,
            "pose": pose,
            "dof_pos_limits": dof_pos_limits,
            "action_rate_l2": action_rate_l2,
            "air_time": air_time,
        }

        last_obs = torch.cat(
            [
                base_lin_vel,
                base_ang_vel,
                projected_gravity,
                joint_pos,
                parsed_imagination_states["joint_vel"],
                rollout_action,
                self.base_velocity,
            ],
            dim=1,
        )
        self.last_obs = TensorDict({"policy": last_obs}, batch_size=[self.num_envs], device=self.device)

    def _combine_bad_orientation_termination(self, projected_gravity):
        bad_orientation = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)).abs() > self._BAD_ORIENTATION_LIMIT
        if self.termination_flags is None:
            return bad_orientation
        return self.termination_flags | bad_orientation

    def _compute_pose_reward(self, joint_pos):
        linear_speed = torch.norm(self.base_velocity[:, :2], dim=1)
        angular_speed = torch.abs(self.base_velocity[:, 2])
        total_speed = linear_speed + angular_speed

        standing_mask = (total_speed < self._WALKING_THRESHOLD).float()
        walking_mask = ((total_speed >= self._WALKING_THRESHOLD) & (total_speed < self._RUNNING_THRESHOLD)).float()
        running_mask = (total_speed >= self._RUNNING_THRESHOLD).float()
        std = (
            self._pose_std_standing * standing_mask.unsqueeze(1)
            + self._pose_std_walking * walking_mask.unsqueeze(1)
            + self._pose_std_running * running_mask.unsqueeze(1)
        )
        return torch.exp(-torch.mean(torch.square(joint_pos) / torch.square(std), dim=1))

    def _compute_air_time_reward(self, foot_contact):
        if foot_contact is None:
            return torch.zeros(self.num_envs, device=self.device)

        in_contact = foot_contact.bool()
        self.current_air_time = torch.where(
            in_contact,
            torch.zeros_like(self.current_air_time),
            self.current_air_time + self._step_dt,
        )
        in_range = (self.current_air_time > 0.05) & (self.current_air_time < 0.5)
        command_active = (torch.norm(self.base_velocity[:, :2], dim=1) + torch.abs(self.base_velocity[:, 2])) > 0.5
        return torch.sum(in_range.float(), dim=1) * command_active.float()

    def _apply_interval_events(self, imagination_states_denormalized, parsed_imagination_states, event_ids):
        if len(event_ids) == 0:
            imagination_states, _ = self.dataset.normalize(imagination_states_denormalized, None)
        else:
            base_lin_vel = parsed_imagination_states["base_lin_vel"]
            r = torch.empty(len(event_ids), device=self.device)
            base_lin_vel[event_ids, 0] += r.uniform_(-0.5, 0.5)
            base_lin_vel[event_ids, 1] += r.uniform_(-0.5, 0.5)
            imagination_states_denormalized[event_ids, 0:3] = base_lin_vel[event_ids, 0:3]
            imagination_states, _ = self.dataset.normalize(imagination_states_denormalized, None)
        return imagination_states

    @property
    def state_dim(self):
        return 45

    @property
    def observation_dim(self):
        return 48

    @property
    def action_dim(self):
        return 12
