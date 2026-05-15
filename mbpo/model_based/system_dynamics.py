from __future__ import annotations

import os
import time

import torch
import torch.nn as nn
from boed.fisher import FisherEstimator, ParameterDistribution
from boed.objectives import BOED, QOED, QOED_AGNOSTIC, VALID_MODES, score_mask
from mbpo.neural import FlowDynamics
from rsl_rl.algorithms import MBPOPPO
from rsl_rl.modules import SystemDynamicsEnsemble
from rsl_rl.storage.replay_buffer import ReplayBuffer as _RslReplayBuffer


INFO_GAIN_NOTHING = "nothing"


def _normalize_info_gain_mode(mode) -> str:
    key = str(mode).strip().lower().replace("_", "-")
    if key in {"", INFO_GAIN_NOTHING, "none", "off", "false", "0"}:
        return INFO_GAIN_NOTHING
    if key in {QOED, "q-oed"}:
        return QOED
    if key in {QOED_AGNOSTIC, "q-oed-agnostic", "agnostic"}:
        return QOED_AGNOSTIC
    if key == BOED:
        return BOED
    return key


class ReplayBuffer(_RslReplayBuffer):
    """Replay buffer sampler that avoids CPU RNG and large reset-mask unfolds."""

    def _sequence_replay_buf(self, sequence_length):
        if self.num_transitions < sequence_length:
            padding_size = sequence_length - self.num_transitions
            if isinstance(self.replay_buf, list):
                return [
                    torch.cat(
                        [
                            torch.zeros(buf.shape[0], padding_size, buf.shape[-1], device=self.device),
                            buf[:, : self.num_transitions],
                        ],
                        dim=1,
                    )
                    if buf is not None
                    else None
                    for buf in self.replay_buf
                ]
            return torch.cat(
                [
                    torch.zeros(
                        self.replay_buf.shape[0],
                        padding_size,
                        self.replay_buf.shape[-1],
                        device=self.device,
                    ),
                    self.replay_buf[:, : self.num_transitions],
                ],
                dim=1,
            )
        return self.replay_buf

    def has_valid_sequences(self, sequence_length: int) -> bool:
        if self.replay_buf is None or self.num_transitions <= 0:
            return False
        replay_buf = self._sequence_replay_buf(sequence_length)
        reset_data = replay_buf[-1] if isinstance(replay_buf, list) else None
        if reset_data is None:
            return True
        env_indices, _ = self._generate_valid_indices(reset_data, sequence_length)
        return env_indices.numel() > 0

    def sample_batch(self, sequence_length: int, mini_batch_size: int, require_valid: bool = True):
        assert self.replay_buf is not None, "Replay buffer is not initialized."
        replay_buf = self._sequence_replay_buf(sequence_length)
        valid_indices = None
        if require_valid and isinstance(replay_buf, list):
            reset_data = replay_buf[-1]
            if reset_data is not None:
                valid_indices = self._generate_valid_indices(reset_data, sequence_length)
        return self._generate_batch(replay_buf, valid_indices, sequence_length, mini_batch_size)

    def _generate_valid_indices(self, reset_data, sequence_length):
        reset_flags = reset_data[:, : max(self.num_transitions, sequence_length)].to(torch.bool)
        if reset_flags.ndim == 3:
            reset_flags = reset_flags.squeeze(-1)

        num_starts = reset_flags.shape[1] - sequence_length + 1
        if num_starts <= 0:
            return (
                torch.zeros(0, dtype=torch.long, device=self.device),
                torch.zeros(0, dtype=torch.long, device=self.device),
            )

        window = sequence_length - 1
        if window <= 0:
            valid_mask = torch.ones(
                reset_flags.shape[0],
                num_starts,
                dtype=torch.bool,
                device=reset_flags.device,
            )
        else:
            reset_cumsum = torch.nn.functional.pad(reset_flags.to(torch.int32), (1, 0)).cumsum(dim=1)
            reset_count = reset_cumsum[:, window : window + num_starts] - reset_cumsum[:, :num_starts]
            valid_mask = reset_count == 0
        return torch.where(valid_mask)

    def _generate_batch(self, replay_buf, valid_indices, sequence_length, mini_batch_size):
        if valid_indices is None:
            max_start_idx = max(self.num_transitions - sequence_length, 0) + 1
            sampled_envs = torch.randint(self.num_envs, (mini_batch_size,), device=self.device)
            sampled_starts = torch.randint(max_start_idx, (mini_batch_size,), device=self.device)
        else:
            env_indices, start_indices = valid_indices
            if len(env_indices) == 0:
                raise RuntimeError("No valid replay-buffer sequences available for sampling.")
            sampled_idxs = torch.randint(len(env_indices), (mini_batch_size,), device=self.device)
            sampled_envs = env_indices[sampled_idxs]
            sampled_starts = start_indices[sampled_idxs]

        offsets = torch.arange(sequence_length, device=self.device)
        if isinstance(replay_buf, list):
            return [
                buf[sampled_envs[:, None], sampled_starts[:, None] + offsets]
                if buf is not None
                else None
                for buf in replay_buf
            ]
        return replay_buf[sampled_envs[:, None], sampled_starts[:, None] + offsets]


def build_system_dynamics(
    state_dim: int,
    action_dim: int,
    extension_dim: int,
    contact_dim: int,
    termination_dim: int,
    device: str,
    ensemble_size: int = 1,
    history_horizon: int = 1,
    architecture_config: dict | None = None,
    freeze_auxiliary: bool = False,
):
    architecture_config = architecture_config or {}
    if architecture_config.get("type") == "shortcut":
        dynamics_cls = ShortcutSystemDynamicsEnsemble
    else:
        dynamics_cls = SystemDynamicsEnsemble
    return dynamics_cls(
        state_dim,
        action_dim,
        extension_dim,
        contact_dim,
        termination_dim,
        device,
        ensemble_size=ensemble_size,
        history_horizon=history_horizon,
        architecture_config=architecture_config,
        freeze_auxiliary=freeze_auxiliary,
    )


class QOED_MBPOPPO(MBPOPPO):
    """MBPOPPO variant that keeps RSL-RL's ReplayBuffer on the runner device."""

    class _FisherDynamicsAdapter:
        def __init__(self, system_dynamics):
            self.system_dynamics = system_dynamics

        def step(self, state, action, theta):
            pred, *_ = self.system_dynamics.forward(
                state.unsqueeze(1),
                action.unsqueeze(1),
                privilege_batch=theta,
            )
            return pred

    def __init__(self, *args, **kwargs):
        import rsl_rl.algorithms.mbpo_ppo as mbpo_ppo

        fisher_param_min = kwargs.pop("fisher_param_min", -100.0)
        fisher_param_max = kwargs.pop("fisher_param_max", 100.0)
        fisher_prior_mean = kwargs.pop("fisher_prior_mean", None)
        fisher_prior_cov = kwargs.pop("fisher_prior_cov", None)
        fisher_prior_cov_diag = kwargs.pop("fisher_prior_cov_diag", None)
        fisher_var_threshold_for_update = float(kwargs.pop("fisher_var_threshold_for_update", 0.0025))
        fisher_fd_delta_floor = float(kwargs.pop("fisher_fd_delta_floor", 1.0))
        fisher_obs_noise_std = float(kwargs.pop("fisher_obs_noise_std", 0.025))
        self.info_gain_mode = _normalize_info_gain_mode(kwargs.pop("info_gain_mode", INFO_GAIN_NOTHING))
        self.info_gain_num_actions = int(kwargs.pop("info_gain_num_actions", 1024))
        clip_actions = kwargs.pop("info_gain_clip_actions", None)
        self.info_gain_clip_actions = None if clip_actions is None else float(clip_actions)
        self._fisher_param_min_cfg = fisher_param_min
        self._fisher_param_max_cfg = fisher_param_max
        self._fisher_param_min = fisher_param_min
        self._fisher_param_max = fisher_param_max
        self._fisher_prior_mean_cfg = fisher_prior_mean
        self._fisher_prior_cov_cfg = fisher_prior_cov if fisher_prior_cov is not None else fisher_prior_cov_diag
        self._fisher_prior_dist = None
        self.latest_info_gain_score = None
        self.cumulative_info_gain = 0.0
        self.num_info_gain_selections = 0
        self.latest_system_dynamics_autoregressive_error = None
        if self.info_gain_mode not in (*VALID_MODES, INFO_GAIN_NOTHING):
            valid = ", ".join((*VALID_MODES, INFO_GAIN_NOTHING))
            raise ValueError(f"Invalid info_gain_mode '{self.info_gain_mode}'. Expected one of: {valid}")

        original_replay_buffer = mbpo_ppo.ReplayBuffer

        def device_replay_buffer(dim, buffer_size, device):
            return ReplayBuffer(dim, buffer_size, device)

        mbpo_ppo.ReplayBuffer = device_replay_buffer
        try:
            super().__init__(*args, **kwargs)
        finally:
            mbpo_ppo.ReplayBuffer = original_replay_buffer
        self._latest_system_privilege = None
        self._fisher_prev_state = None
        self._fisher_prev_valid = None
        self._fisher_window = 20
        self.parameter_estimator = None
        self.system_privilege_dim = int(getattr(self.system_dynamics, "shortcut_privilege_dim", 0))
        if self.system_privilege_dim > 0:
            buffer_size = self.system_replay_buffer.buffer_size
            buffer_device = self.system_replay_buffer.device
            self.system_replay_buffer = ReplayBuffer(
                [
                    self.system_dynamics.state_dim,
                    self.system_dynamics.action_dim,
                    self.system_dynamics.extension_dim,
                    self.system_dynamics.contact_dim,
                    self.system_privilege_dim,
                    self.system_dynamics.termination_dim,
                ],
                buffer_size,
                buffer_device,
            )
            mean, cov = self._fisher_prior_tensors(self.system_privilege_dim, self.device)
            self._fisher_prior_dist = ParameterDistribution(mean, cov)
            self._fisher_param_min = self._fisher_bound_tensor(
                self._fisher_param_min_cfg,
                self.system_privilege_dim,
                self.device,
                mean.dtype,
                "fisher_param_min",
            )
            self._fisher_param_max = self._fisher_bound_tensor(
                self._fisher_param_max_cfg,
                self.system_privilege_dim,
                self.device,
                mean.dtype,
                "fisher_param_max",
            )
            self.parameter_estimator = FisherEstimator(
                self._FisherDynamicsAdapter(self.system_dynamics),
                ParameterDistribution(mean, cov),
                obs_noise_std=fisher_obs_noise_std,
                max_history=None,
                cem_samples=2048,
                cem_iters=5,
                param_min=self._fisher_param_min,
                param_max=self._fisher_param_max,
                qoed=self.info_gain_mode != BOED,
                var_threshold_for_update=fisher_var_threshold_for_update,
                fd_delta_floor=fisher_fd_delta_floor,
            )
            self._latest_system_privilege = mean.unsqueeze(0)
            self._sync_system_dynamics_privilege_source()

    def act(self, obs):
        if (
            self.info_gain_mode == INFO_GAIN_NOTHING
            or self.parameter_estimator is None
            or "system_state" not in obs
            or self._is_imagination_observation(obs)
        ):
            return super().act(obs)

        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        base_action = self.policy.act(obs).detach()
        candidates = self._sample_info_gain_action_candidates(base_action)
        selected_action = self._select_info_gain_action(obs, candidates, base_action).detach()

        self.transition.actions = selected_action
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def _sample_info_gain_action_candidates(self, base_action: torch.Tensor) -> torch.Tensor:
        num_candidates = max(1, self.info_gain_num_actions)
        candidates = self.policy.distribution.sample((num_candidates,)).detach()
        candidates = torch.nan_to_num(candidates, nan=0.0, posinf=0.0, neginf=0.0)
        candidates[0] = base_action
        return candidates

    def _select_info_gain_action(
        self,
        obs,
        candidates: torch.Tensor,
        fallback_action: torch.Tensor,
    ) -> torch.Tensor:
        if "system_state" not in obs:
            return fallback_action

        selected = []
        system_state = self.state_normalizer(obs["system_state"].to(self.device))
        num_envs = candidates.shape[1]
        for env_idx in range(num_envs):
            action_batch = candidates[:, env_idx].to(self.device)
            if self.info_gain_clip_actions is not None:
                limit = self.info_gain_clip_actions
                action_batch = action_batch.clamp(-limit, limit)
            scores = self._score_info_gain_actions(system_state[env_idx : env_idx + 1], action_batch)
            if scores is None:
                selected.append(fallback_action[env_idx].to(self.device))
                continue
            safe_scores = torch.nan_to_num(scores, nan=-torch.inf, posinf=1.0e30, neginf=-torch.inf)
            if not torch.isfinite(safe_scores).any():
                selected.append(fallback_action[env_idx].to(self.device))
                continue
            best_idx = torch.argmax(safe_scores)
            self.parameter_estimator.min_idx = best_idx.detach()
            if getattr(self.parameter_estimator, "F_cur", None) is not None:
                self.parameter_estimator.F_history = self.parameter_estimator.F_cur[best_idx].detach()
            best_score = safe_scores[best_idx].detach()
            self.latest_info_gain_score = best_score
            self.cumulative_info_gain = self.cumulative_info_gain + best_score
            self.num_info_gain_selections += 1
            selected.append(action_batch[best_idx])
        return torch.stack(selected, dim=0).to(device=fallback_action.device, dtype=fallback_action.dtype)

    def _score_info_gain_actions(self, state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor | None:
        if self.parameter_estimator is None:
            return None
        normalized_actions = self.action_normalizer(actions)
        controls = normalized_actions.unsqueeze(1)
        was_training = self.system_dynamics.training
        self.system_dynamics.eval()
        try:
            return self.parameter_estimator.local_fisher_trace_path(
                state,
                controls,
                baseline=self.info_gain_mode,
            )
        finally:
            self.system_dynamics.train(was_training)

    def _is_imagination_observation(self, obs) -> bool:
        imagination_storage = getattr(self, "imagination_storage", None)
        storage = getattr(self, "storage", None)
        if imagination_storage is None or storage is None or "policy" not in obs:
            return False
        batch_size = int(obs["policy"].shape[0])
        return batch_size == imagination_storage.num_envs and batch_size != storage.num_envs

    def _fisher_prior_tensors(self, dim: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        if self._fisher_prior_mean_cfg is None:
            mean = torch.zeros(dim, device=device)
        else:
            mean = torch.as_tensor(self._fisher_prior_mean_cfg, device=device, dtype=torch.float32).reshape(-1)
            if mean.numel() != dim:
                raise ValueError(f"Expected fisher_prior_mean dim {dim}, got {mean.numel()}")

        if self._fisher_prior_cov_cfg is None:
            cov = torch.eye(dim, device=device, dtype=mean.dtype) * 25.0
        else:
            cov_cfg = torch.as_tensor(self._fisher_prior_cov_cfg, device=device, dtype=mean.dtype)
            if cov_cfg.ndim == 1:
                if cov_cfg.numel() != dim:
                    raise ValueError(f"Expected fisher_prior_cov_diag dim {dim}, got {cov_cfg.numel()}")
                cov = torch.diag(torch.clamp(cov_cfg, min=0.0))
            elif cov_cfg.ndim == 2:
                if tuple(cov_cfg.shape) != (dim, dim):
                    raise ValueError(f"Expected fisher_prior_cov shape {(dim, dim)}, got {tuple(cov_cfg.shape)}")
                cov = 0.5 * (cov_cfg + cov_cfg.T)
            else:
                raise ValueError("fisher prior covariance must be a vector or matrix")
        return mean, cov

    def _fisher_bound_tensor(self, value, dim: int, device, dtype, name: str) -> torch.Tensor:
        bound = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
        if bound.numel() == 1:
            return bound.expand(dim).detach().clone()
        if bound.numel() != dim:
            raise ValueError(f"Expected {name} dim {dim}, got {bound.numel()}")
        return bound.detach().clone()

    def _clamp_fisher_params(self, params: torch.Tensor) -> torch.Tensor:
        param_min = torch.as_tensor(self._fisher_param_min, device=params.device, dtype=params.dtype).reshape(1, -1)
        param_max = torch.as_tensor(self._fisher_param_max, device=params.device, dtype=params.dtype).reshape(1, -1)
        return torch.minimum(torch.maximum(params, param_min), param_max)

    def _active_privilege_distribution(self) -> ParameterDistribution | None:
        if self.parameter_estimator is not None and self.info_gain_mode != INFO_GAIN_NOTHING:
            return self.parameter_estimator.dist
        return self._fisher_prior_dist

    def _qoed_identifiable_mask(self, cov: torch.Tensor) -> torch.Tensor | None:
        if self.info_gain_mode != QOED or self.parameter_estimator is None:
            return None
        try:
            mask = score_mask(
                self.parameter_estimator.compute_fisher(),
                cov,
                self.info_gain_mode,
                bool(self.parameter_estimator.history),
                self.parameter_estimator.eig_ratio_thresh,
                self.parameter_estimator.dist_threshold,
                self.parameter_estimator.param_contrib_ratio,
                self.parameter_estimator.var_threshold_for_update,
                self.parameter_estimator.smallest_eigval_threshold,
            )
        except Exception as exc:
            print(f"[WARN] Could not compute QOED identifiable mask for dynamics DR: {exc}")
            return None
        mask = mask.to(device=cov.device, dtype=torch.bool).reshape(-1)
        self.parameter_estimator.score_mask = mask
        return mask

    def _effective_privilege_mean_cov(self):
        dist = self._active_privilege_distribution()
        if dist is None:
            return None, None
        mean = dist.mean
        cov = dist.cov
        mask = self._qoed_identifiable_mask(cov)
        if mask is not None:
            if self._fisher_prior_dist is not None:
                prior_mean = self._fisher_prior_dist.mean.to(device=mean.device, dtype=mean.dtype)
                mean = torch.where(mask.to(device=mean.device), mean, prior_mean)
            mask_value = mask.to(dtype=cov.dtype)
            cov = cov * mask_value[:, None] * mask_value[None, :]
        return mean, cov

    def effective_system_privilege_mean(self) -> torch.Tensor | None:
        mean, _ = self._effective_privilege_mean_cov()
        return mean

    def _estimated_system_privilege(self, batch_size: int, like: torch.Tensor):
        mean, _ = self._effective_privilege_mean_cov()
        if mean is None:
            return None
        return mean.to(device=like.device, dtype=like.dtype).expand(batch_size, -1)

    def _sample_system_privilege(self, batch_size: int, like: torch.Tensor):
        mean, cov = self._effective_privilege_mean_cov()
        if mean is None or cov is None:
            return None
        mean = mean.to(device=like.device, dtype=like.dtype).reshape(-1)
        cov = cov.to(device=like.device, dtype=like.dtype).reshape(mean.numel(), mean.numel())
        cov = 0.5 * (cov + cov.T)
        noise = torch.randn(batch_size, mean.numel(), device=like.device, dtype=like.dtype)
        eye = torch.eye(mean.numel(), device=like.device, dtype=like.dtype)
        try:
            scale = torch.linalg.cholesky(cov + 1.0e-9 * eye)
            sample = mean.unsqueeze(0) + noise @ scale.T
        except RuntimeError:
            std = torch.sqrt(torch.clamp(torch.diagonal(cov), min=0.0))
            sample = mean.unsqueeze(0) + noise * std.unsqueeze(0)
        return self._clamp_fisher_params(sample)

    def _sync_system_dynamics_privilege_source(self) -> None:
        setter = getattr(self.system_dynamics, "set_source_privilege_distribution", None)
        if setter is None:
            return
        mean, cov = self._effective_privilege_mean_cov()
        if mean is None or cov is None:
            return
        setter(mean, cov, self._fisher_param_min, self._fisher_param_max)

    def _update_parameter_estimator(self, system_state, system_action, system_termination=None):
        if self.parameter_estimator is None or self.info_gain_mode == INFO_GAIN_NOTHING:
            self._latest_system_privilege = self._estimated_system_privilege(system_state.shape[0], system_state)
            return
        done = torch.zeros(system_state.shape[0], dtype=torch.bool, device=system_state.device)
        if system_termination is not None:
            done = system_termination.reshape(system_state.shape[0], -1).any(dim=1).to(torch.bool)

        if self._fisher_prev_state is not None:
            valid = self._fisher_prev_valid & ~done
            for idx in valid.nonzero(as_tuple=False).flatten().tolist():
                self.parameter_estimator.add_sample(
                    self._fisher_prev_state[idx],
                    system_action[idx],
                    system_state[idx],
                )
            if len(self.parameter_estimator.est_history) >= self._fisher_window:
                self.parameter_estimator.history = self.parameter_estimator.history[-self._fisher_window :]
                self.parameter_estimator.update_posterior()
                self.parameter_estimator.history = self.parameter_estimator.history[-self._fisher_window :]
                self.parameter_estimator.est_history.clear()

        self._fisher_prev_state = system_state.detach().clone()
        self._fisher_prev_valid = ~done
        self._latest_system_privilege = self._estimated_system_privilege(system_state.shape[0], system_state)
        self._sync_system_dynamics_privilege_source()

    def process_env_step(self, obs, rewards, dones, infos, imagination=False):
        return super().process_env_step(obs, rewards, dones, infos, imagination=imagination)

    def fill_history_buffer(self, obs):
        system_state = self.state_normalizer(obs["system_state"])
        system_action = self.action_normalizer(obs["system_action"])
        system_extension = obs.get("system_extension")
        system_contact = obs.get("system_contact")
        system_privilege = obs.get("system_privilege")
        if system_privilege is None:
            self._update_parameter_estimator(system_state, system_action, obs.get("system_termination"))
            system_privilege = self._latest_system_privilege
            if system_privilege is None and self.system_privilege_dim > 0:
                system_privilege = self._estimated_system_privilege(system_state.shape[0], system_state)
        system_termination = obs.get("system_termination")
        if system_privilege is not None:
            system_privilege = system_privilege.to(device=system_state.device, dtype=system_state.dtype)
        if system_privilege is None and self.system_privilege_dim > 0:
            system_privilege = torch.full(
                (
                    system_state.shape[0],
                    self.system_privilege_dim,
                ),
                float("nan"),
                dtype=system_state.dtype,
                device=system_state.device,
            )

        def to_storage(tensor):
            if tensor is None:
                return None
            if self.system_replay_buffer.device == "cpu":
                return tensor.detach().cpu()
            return tensor

        replay_items = [
            to_storage(system_state).unsqueeze(1),
            to_storage(system_action).unsqueeze(1),
            to_storage(system_extension).unsqueeze(1) if system_extension is not None else None,
            to_storage(system_contact).unsqueeze(1) if system_contact is not None else None,
        ]
        if self.system_privilege_dim > 0:
            replay_items.append(to_storage(system_privilege).unsqueeze(1))
        replay_items.append(
            to_storage(system_termination).unsqueeze(1) if system_termination is not None else None
        )
        self.system_replay_buffer.insert(replay_items)

    def prepare_imagination(self):
        self._sync_system_dynamics_privilege_source()
        return super().prepare_imagination()

    def update_system_dynamics(self):
        mean_system_state_loss = 0
        mean_system_sequence_loss = 0
        mean_system_bound_loss = 0
        mean_system_kl_loss = 0
        mean_system_extension_loss = 0
        mean_system_contact_loss = 0
        mean_system_termination_loss = 0
        system_generator = self.system_replay_buffer.mini_batch_generator(
            self.system_dynamics.history_horizon + self.system_dynamics_forecast_horizon,
            self.system_dynamics_num_mini_batches,
            self.system_dynamics_mini_batch_size,
        )
        for system_batch in system_generator:
            if len(system_batch) == 6:
                (
                    system_state_batch,
                    system_action_batch,
                    system_extension_batch,
                    system_contact_batch,
                    system_privilege_batch,
                    system_termination_batch,
                ) = system_batch
            else:
                (
                    system_state_batch,
                    system_action_batch,
                    system_extension_batch,
                    system_contact_batch,
                    system_termination_batch,
                ) = system_batch
                system_privilege_batch = None
            self.system_dynamics.reset()
            if hasattr(self.system_dynamics, "shortcut_privilege_dim"):
                (
                    state_loss,
                    sequence_loss,
                    bound_loss,
                    kl_loss,
                    extension_loss,
                    contact_loss,
                    termination_loss,
                ) = self.system_dynamics.compute_loss(
                    system_state_batch,
                    system_action_batch,
                    system_extension_batch,
                    system_contact_batch,
                    system_termination_batch,
                    privilege_batch=system_privilege_batch,
                    bootstrap=True,
                )
            else:
                (
                    state_loss,
                    sequence_loss,
                    bound_loss,
                    kl_loss,
                    extension_loss,
                    contact_loss,
                    termination_loss,
                ) = self.system_dynamics.compute_loss(
                    system_state_batch,
                    system_action_batch,
                    system_extension_batch,
                    system_contact_batch,
                    system_termination_batch,
                    bootstrap=True,
                )
            loss = (
                self.system_dynamics_loss_weights["state"] * state_loss
                + self.system_dynamics_loss_weights["sequence"] * sequence_loss
                + self.system_dynamics_loss_weights["bound"] * bound_loss
                + self.system_dynamics_loss_weights["kl"] * kl_loss
                + self.system_dynamics_loss_weights["extension"] * extension_loss
                + self.system_dynamics_loss_weights["contact"] * contact_loss
                + self.system_dynamics_loss_weights["termination"] * termination_loss
            )
            self.system_dynamics_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.system_dynamics.parameters(), self.max_grad_norm)
            self.system_dynamics_optimizer.step()
            mean_system_state_loss += state_loss.item()
            mean_system_sequence_loss += sequence_loss.item()
            mean_system_bound_loss += bound_loss.item()
            mean_system_kl_loss += kl_loss.item()
            mean_system_extension_loss += extension_loss.item()
            mean_system_contact_loss += contact_loss.item()
            mean_system_termination_loss += termination_loss.item()

        system_dynamics_num_updates = self.system_dynamics_num_mini_batches
        mean_system_state_loss /= system_dynamics_num_updates
        mean_system_sequence_loss /= system_dynamics_num_updates
        mean_system_bound_loss /= system_dynamics_num_updates
        mean_system_kl_loss /= system_dynamics_num_updates
        mean_system_extension_loss /= system_dynamics_num_updates
        mean_system_contact_loss /= system_dynamics_num_updates
        mean_system_termination_loss /= system_dynamics_num_updates
        return (
            mean_system_state_loss,
            mean_system_sequence_loss,
            mean_system_bound_loss,
            mean_system_kl_loss,
            mean_system_extension_loss,
            mean_system_contact_loss,
            mean_system_termination_loss,
        )

    def _system_dynamics_autoregressive_prediction_with_privilege(
        self,
        state_traj,
        action_traj,
        extension_traj,
        contact_traj,
        termination_traj,
        privilege_traj,
    ):
        state_traj_pred = torch.zeros_like(state_traj, device=self.device)
        aleatoric_uncertainty_traj_pred = torch.zeros(
            state_traj.shape[0], state_traj.shape[1], device=self.device
        )
        epistemic_uncertainty_traj_pred = torch.zeros(
            state_traj.shape[0], state_traj.shape[1], device=self.device
        )
        action_traj_pred = action_traj.clone()
        extension_traj_pred = torch.zeros_like(extension_traj, device=self.device) if extension_traj is not None else None
        contact_traj_pred = torch.zeros_like(contact_traj, device=self.device) if contact_traj is not None else None
        termination_traj_pred = torch.zeros_like(termination_traj, device=self.device) if termination_traj is not None else None

        state_traj_pred[:, : self.system_dynamics.history_horizon] = state_traj[
            :, : self.system_dynamics.history_horizon
        ]
        if extension_traj_pred is not None:
            extension_traj_pred[:, : self.system_dynamics.history_horizon] = extension_traj[
                :, : self.system_dynamics.history_horizon
            ]
        if contact_traj_pred is not None:
            contact_traj_pred[:, : self.system_dynamics.history_horizon] = contact_traj[
                :, : self.system_dynamics.history_horizon
            ]
        if termination_traj_pred is not None:
            termination_traj_pred[:, : self.system_dynamics.history_horizon] = termination_traj[
                :, : self.system_dynamics.history_horizon
            ]

        self.system_dynamics.reset()
        with torch.inference_mode():
            for i in range(
                self.system_dynamics.history_horizon,
                self.system_dynamics_len_eval_trajectory,
            ):
                if (
                    self.system_dynamics.architecture_config["type"] in ["rnn", "rssm"]
                    and i > self.system_dynamics.history_horizon
                ):
                    state_input = state_traj_pred[:, i - 1 : i]
                    action_input = action_traj_pred[:, i - 1 : i]
                    privilege_input = privilege_traj[:, i - 1 : i] if privilege_traj is not None else None
                else:
                    state_input = state_traj_pred[
                        :, i - self.system_dynamics.history_horizon : i
                    ]
                    action_input = action_traj_pred[
                        :, i - self.system_dynamics.history_horizon : i
                    ]
                    privilege_input = (
                        privilege_traj[
                            :, i - self.system_dynamics.history_horizon : i
                        ]
                        if privilege_traj is not None
                        else None
                    )
                (
                    state_pred,
                    aleatoric_uncertainty,
                    epistemic_uncertainty,
                    extension_pred,
                    contact_pred,
                    termination_pred,
                ) = self.system_dynamics.forward(
                    state_input,
                    action_input,
                    privilege_batch=privilege_input,
                )
                state_traj_pred[:, i] = state_pred
                aleatoric_uncertainty_traj_pred[:, i] = aleatoric_uncertainty
                epistemic_uncertainty_traj_pred[:, i] = epistemic_uncertainty
                if extension_traj_pred is not None and extension_pred is not None:
                    extension_traj_pred[:, i] = extension_pred
                if contact_traj_pred is not None and contact_pred is not None:
                    contact_traj_pred[:, i] = torch.sigmoid(contact_pred).round().int()
                if termination_traj_pred is not None and termination_pred is not None:
                    termination_traj_pred[:, i] = torch.sigmoid(termination_pred).round().int()
        return (
            state_traj_pred,
            aleatoric_uncertainty_traj_pred,
            epistemic_uncertainty_traj_pred,
            action_traj_pred,
            extension_traj_pred,
            contact_traj_pred,
            termination_traj_pred,
        )

    def evaluate_system_dynamics(self, privilege_override: str | None = None):
        requested_eval_len = self.system_dynamics_len_eval_trajectory
        eval_len = self._resolve_system_dynamics_eval_length(requested_eval_len)
        require_valid_eval_sequence = eval_len is not None
        if eval_len is None:
            eval_len = min(
                requested_eval_len,
                max(self.system_dynamics.history_horizon + 1, self.system_replay_buffer.num_transitions),
            )
            if not getattr(self, "_warned_relaxed_system_dynamics_eval", False):
                print(
                    "[WARN] No reset-free replay-buffer sequence is long enough for system-dynamics "
                    "evaluation yet; using a best-effort logging sample."
                )
                self._warned_relaxed_system_dynamics_eval = True
        elif eval_len < requested_eval_len and not getattr(self, "_warned_short_system_dynamics_eval", False):
            print(
                "[WARN] System-dynamics evaluation requested "
                f"{requested_eval_len} steps, but only {eval_len} reset-free steps are available; "
                "using the shorter trajectory for this log."
            )
            self._warned_short_system_dynamics_eval = True

        system_batch = self.system_replay_buffer.sample_batch(
            eval_len,
            self.system_dynamics_num_eval_trajectories,
            require_valid=require_valid_eval_sequence,
        )
        if len(system_batch) == 6:
            (
                state_traj,
                action_traj,
                extension_traj,
                contact_traj,
                privilege_traj,
                termination_traj,
            ) = system_batch
        else:
            state_traj, action_traj, extension_traj, contact_traj, termination_traj = system_batch
            privilege_traj = None
        if privilege_override == "current_mean" and self.system_privilege_dim > 0:
            current_privilege = self._estimated_system_privilege(state_traj.shape[0], state_traj)
            if current_privilege is not None:
                privilege_traj = current_privilege.unsqueeze(1).expand(-1, eval_len, -1)
        training_device = self.device
        eval_device = self._resolve_system_dynamics_eval_device(training_device)
        self._synchronize_if_cuda(eval_device)
        eval_start_time = time.perf_counter()
        state_traj = state_traj.to(eval_device)
        action_traj = action_traj.to(eval_device)
        extension_traj = extension_traj.to(eval_device) if extension_traj is not None else None
        contact_traj = contact_traj.to(eval_device) if contact_traj is not None else None
        privilege_traj = privilege_traj.to(eval_device) if privilege_traj is not None else None
        termination_traj = termination_traj.to(eval_device) if termination_traj is not None else None

        was_training = self.system_dynamics.training
        dynamics_devices = {
            module: module.device
            for module in self.system_dynamics.modules()
            if hasattr(module, "device")
        }
        self.system_dynamics.to(eval_device)
        for module in dynamics_devices:
            module.device = eval_device
        self.device = eval_device
        self.system_dynamics.eval()
        self.system_dynamics_len_eval_trajectory = eval_len
        try:
            if privilege_traj is None:
                (
                    state_traj_pred,
                    _,
                    _,
                    action_traj_pred,
                    extension_traj_pred,
                    contact_traj_pred,
                    termination_traj_pred,
                ) = super().system_dynamics_autoregressive_prediction(
                    state_traj,
                    action_traj,
                    extension_traj,
                    contact_traj,
                    termination_traj,
                )
            else:
                (
                    state_traj_pred,
                    _,
                    _,
                    action_traj_pred,
                    extension_traj_pred,
                    contact_traj_pred,
                    termination_traj_pred,
                ) = self._system_dynamics_autoregressive_prediction_with_privilege(
                    state_traj,
                    action_traj,
                    extension_traj,
                    contact_traj,
                    termination_traj,
                    privilege_traj,
                )
            denominator = state_traj[:, self.system_dynamics.history_horizon :].abs().sum(dim=-1).clamp_min(1.0e-8)
            traj_autoregressive_error = (
                (state_traj_pred[:, self.system_dynamics.history_horizon :] - state_traj[:, self.system_dynamics.history_horizon :])
                .abs()
                .sum(dim=-1)
                / denominator
            ).mean().item()
            traj_autoregressive_error_noised_dict = {}
            for noise_scale in self.system_dynamics_eval_traj_noise_scale:
                state_traj_noised = state_traj + torch.randn_like(state_traj) * noise_scale
                action_traj_noised = action_traj + torch.randn_like(action_traj) * noise_scale
                if privilege_traj is None:
                    state_traj_pred_noised, _, _, _, _, _, _ = super().system_dynamics_autoregressive_prediction(
                        state_traj_noised,
                        action_traj_noised,
                        extension_traj,
                        contact_traj,
                        termination_traj,
                    )
                else:
                    state_traj_pred_noised, _, _, _, _, _, _ = (
                        self._system_dynamics_autoregressive_prediction_with_privilege(
                            state_traj_noised,
                            action_traj_noised,
                            extension_traj,
                            contact_traj,
                            termination_traj,
                            privilege_traj,
                        )
                    )
                noised_denominator = state_traj_noised[:, self.system_dynamics.history_horizon :].abs().sum(dim=-1).clamp_min(1.0e-8)
                traj_autoregressive_error_noised = (
                    (
                        state_traj_pred_noised[:, self.system_dynamics.history_horizon :]
                        - state_traj_noised[:, self.system_dynamics.history_horizon :]
                    )
                    .abs()
                    .sum(dim=-1)
                    / noised_denominator
                ).mean().item()
                traj_autoregressive_error_noised_dict[noise_scale] = traj_autoregressive_error_noised
        finally:
            self.system_dynamics_len_eval_trajectory = requested_eval_len
            self.system_dynamics.train(was_training)
            self.system_dynamics.to(training_device)
            for module, device in dynamics_devices.items():
                module.device = device
            self.device = training_device

        self._synchronize_if_cuda(eval_device)
        print(
            "[MBPO Eval] "
            f"device={eval_device} "
            f"len={eval_len} "
            f"num_traj={self.system_dynamics_num_eval_trajectories} "
            f"autoregressive_error={traj_autoregressive_error:.4g} "
            f"time={time.perf_counter() - eval_start_time:.3f}s"
        )
        self.latest_system_dynamics_autoregressive_error = float(traj_autoregressive_error)

        def to_training_device(tensor):
            if tensor is None:
                return None
            return tensor.to(training_device)

        return (
            to_training_device(state_traj),
            to_training_device(action_traj),
            to_training_device(extension_traj),
            to_training_device(contact_traj),
            to_training_device(termination_traj),
            to_training_device(state_traj_pred),
            to_training_device(action_traj_pred),
            to_training_device(extension_traj_pred),
            to_training_device(contact_traj_pred),
            to_training_device(termination_traj_pred),
            traj_autoregressive_error,
            traj_autoregressive_error_noised_dict,
        )

    @staticmethod
    def _resolve_system_dynamics_eval_device(training_device):
        override = os.environ.get("QOED_SYSTEM_DYNAMICS_EVAL_DEVICE")
        if override is None or override.strip().lower() in {"", "training", "train", "same"}:
            return training_device
        if override.strip().lower() == "gpu":
            return training_device if str(training_device).startswith("cuda") else "cuda:0"
        return override.strip()

    @staticmethod
    def _synchronize_if_cuda(device) -> None:
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize(torch.device(device))

    def _resolve_system_dynamics_eval_length(self, requested_eval_len: int) -> int | None:
        if self.system_replay_buffer.replay_buf is None:
            return None
        min_eval_len = self.system_dynamics.history_horizon + 1
        max_eval_len = min(requested_eval_len, self.system_replay_buffer.num_transitions)
        if max_eval_len < min_eval_len:
            return None
        if not self.system_replay_buffer.has_valid_sequences(min_eval_len):
            return None

        low, high = min_eval_len, max_eval_len
        while low < high:
            mid = (low + high + 1) // 2
            if self.system_replay_buffer.has_valid_sequences(mid):
                low = mid
            else:
                high = mid - 1
        return low


class ShortcutSystemDynamicsEnsemble(nn.Module):
    """shortcut dynamics ensemble with the RSL-RL system dynamics interface."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        extension_dim: int,
        contact_dim: int,
        termination_dim: int,
        device: str,
        ensemble_size: int = 1,
        history_horizon: int = 1,
        architecture_config: dict | None = None,
        freeze_auxiliary: bool = False,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device
        self.ensemble_size = ensemble_size
        self.history_horizon = history_horizon
        self.architecture_config = architecture_config or {}
        self.freeze_auxiliary = freeze_auxiliary
        self.prediction_type = "single"
        self.num_steps = int(self.architecture_config.get("num_steps", 1))
        self.shortcut_privilege_dim = int(self.architecture_config.get("privilege_dim", 0))
        self.shortcut_privilege_source_noise = bool(
            self.architecture_config.get("privilege_source_noise", True)
        )
        self.shortcut_privilege_dropout_prob = float(
            self.architecture_config.get("privilege_dropout_prob", 0.1)
        )
        if not 0.0 <= self.shortcut_privilege_dropout_prob <= 1.0:
            raise ValueError("privilege_dropout_prob must be in [0, 1]")
        self.shortcut_privilege_availability = bool(
            self.architecture_config.get("privilege_availability_feature", True)
        )
        self._source_privilege_mean = None
        self._source_privilege_cov = None
        self._source_privilege_min = None
        self._source_privilege_max = None
        shortcut_condition_dim = self.shortcut_privilege_dim
        if self.shortcut_privilege_dim > 0 and self.shortcut_privilege_availability:
            shortcut_condition_dim += 1
        self.models = nn.ModuleList(
            [
                FlowDynamics(
                    state_dim,
                    action_dim,
                    shortcut_condition_dim,
                    latent_dim=int(self.architecture_config.get("latent_dim", state_dim)),
                    timestep_embed_dim=self.architecture_config.get("timestep_embed_dim"),
                    use_shortcut=bool(self.architecture_config.get("use_shortcut", True)),
                    shortcut_self_consistency=float(
                        self.architecture_config.get("shortcut_self_consistency", 0.25)
                    ),
                    shortcut_min_dt=float(self.architecture_config.get("shortcut_min_dt", 0.0078125)),
                    depth=int(self.architecture_config.get("depth", 6)),
                    num_heads=int(self.architecture_config.get("num_heads", 4)),
                    mlp_ratio=float(self.architecture_config.get("mlp_ratio", 4.0)),
                    num_registers=int(self.architecture_config.get("num_registers", 8)),
                )
                for _ in range(ensemble_size)
            ]
        )

    def set_source_privilege_distribution(
        self,
        mean: torch.Tensor,
        cov: torch.Tensor,
        param_min: float | torch.Tensor | None = None,
        param_max: float | torch.Tensor | None = None,
    ) -> None:
        if self.shortcut_privilege_dim <= 0:
            return
        mean = torch.as_tensor(mean, device=self.device, dtype=torch.float32).reshape(-1)
        cov = torch.as_tensor(cov, device=self.device, dtype=mean.dtype).reshape(mean.numel(), mean.numel())
        if mean.numel() != self.shortcut_privilege_dim:
            raise ValueError(
                f"Expected source privilege dim {self.shortcut_privilege_dim}, got {mean.numel()}"
            )
        self._source_privilege_mean = mean.detach().clone()
        self._source_privilege_cov = (0.5 * (cov + cov.T)).detach().clone()
        self._source_privilege_min = self._source_privilege_bound(param_min, mean, "param_min")
        self._source_privilege_max = self._source_privilege_bound(param_max, mean, "param_max")

    def _source_privilege_bound(self, value, mean: torch.Tensor, name: str) -> torch.Tensor | None:
        if value is None:
            return None
        bound = torch.as_tensor(value, device=mean.device, dtype=mean.dtype).reshape(-1)
        if bound.numel() == 1:
            return bound.expand(mean.numel()).detach().clone()
        if bound.numel() != mean.numel():
            raise ValueError(f"Expected source privilege {name} dim {mean.numel()}, got {bound.numel()}")
        return bound.detach().clone()

    def _privilege(
        self,
        batch_size: int,
        like: torch.Tensor,
        privilege_batch: torch.Tensor | None = None,
        ids: torch.Tensor | None = None,
        step_index: int = -1,
        allow_dropout: bool = False,
    ):
        if self.shortcut_privilege_dim <= 0:
            return None
        if privilege_batch is None:
            privilege = self._sample_source_privilege(batch_size, like)
            source_available = 1.0 if self._source_privilege_mean is not None else 0.0
            available = torch.full(
                (batch_size, 1),
                source_available,
                dtype=like.dtype,
                device=like.device,
            )
            return self._append_privilege_availability(privilege, available)
        privilege_batch = privilege_batch.to(device=like.device, dtype=like.dtype)
        if ids is not None:
            privilege_batch = privilege_batch[ids]
        if privilege_batch.ndim == 3:
            privilege = privilege_batch[:, step_index]
        elif privilege_batch.ndim == 2:
            privilege = privilege_batch
        else:
            raise ValueError(
                f"Expected privilege batch with rank 2 or 3, got shape {tuple(privilege_batch.shape)}"
            )
        if privilege.shape[-1] != self.shortcut_privilege_dim:
            raise ValueError(
                f"Expected privilege dim {self.shortcut_privilege_dim}, got {privilege.shape[-1]}"
            )
        available = torch.isfinite(privilege).all(dim=-1, keepdim=True).to(dtype=like.dtype)
        privilege = torch.nan_to_num(privilege, nan=0.0, posinf=0.0, neginf=0.0)

        if allow_dropout and self.training and self.shortcut_privilege_dropout_prob > 0.0:
            drop = torch.rand(batch_size, 1, device=like.device) < self.shortcut_privilege_dropout_prob
            available = torch.where(drop, torch.zeros_like(available), available)

        missing = available <= 0.0
        if missing.any():
            source_privilege = self._sample_source_privilege(batch_size, like)
            privilege = torch.where(missing, source_privilege, privilege)
        return self._append_privilege_availability(privilege, available)

    def _sample_source_privilege(self, batch_size: int, like: torch.Tensor) -> torch.Tensor:
        if self._source_privilege_mean is not None and self._source_privilege_cov is not None:
            mean = self._source_privilege_mean.to(device=like.device, dtype=like.dtype)
            cov = self._source_privilege_cov.to(device=like.device, dtype=like.dtype)
            noise = torch.randn(batch_size, mean.numel(), dtype=like.dtype, device=like.device)
            eye = torch.eye(mean.numel(), dtype=like.dtype, device=like.device)
            try:
                scale = torch.linalg.cholesky(cov + 1.0e-9 * eye)
                privilege = mean.unsqueeze(0) + noise @ scale.T
            except RuntimeError:
                std = torch.sqrt(torch.clamp(torch.diagonal(cov), min=0.0))
                privilege = mean.unsqueeze(0) + noise * std.unsqueeze(0)
            fixed = torch.diagonal(cov) <= 0.0
            if fixed.any():
                privilege[:, fixed] = mean[fixed]
            if self._source_privilege_min is not None or self._source_privilege_max is not None:
                if self._source_privilege_min is not None:
                    min_value = self._source_privilege_min.to(device=like.device, dtype=like.dtype).reshape(1, -1)
                    privilege = torch.maximum(privilege, min_value)
                if self._source_privilege_max is not None:
                    max_value = self._source_privilege_max.to(device=like.device, dtype=like.dtype).reshape(1, -1)
                    privilege = torch.minimum(privilege, max_value)
            return privilege
        if self.shortcut_privilege_source_noise:
            return torch.randn(
                batch_size,
                self.shortcut_privilege_dim,
                dtype=like.dtype,
                device=like.device,
            )
        return torch.zeros(batch_size, self.shortcut_privilege_dim, dtype=like.dtype, device=like.device)

    def _append_privilege_availability(
        self,
        privilege: torch.Tensor,
        available: torch.Tensor,
    ) -> torch.Tensor:
        if not self.shortcut_privilege_availability:
            return privilege
        return torch.cat((privilege, available), dim=-1)

    def forward(self, x_state_batch, x_action_batch, model_ids=None, privilege_batch=None):
        x_state_batch = x_state_batch.to(self.device)
        x_action_batch = x_action_batch.to(self.device)
        state = x_state_batch[:, -1]
        action = x_action_batch[:, -1]
        privilege = self._privilege(state.shape[0], state, privilege_batch=privilege_batch)
        state_means = []
        for model in self.models:
            state_means.append(model(state, action, privilege, n_step=self.num_steps).unsqueeze(0))
        state_means = torch.cat(state_means, dim=0)

        if model_ids is None:
            output_state_means = state_means.mean(dim=0)
        else:
            output_state_means = torch.gather(
                state_means,
                0,
                model_ids.repeat(1, 1, self.state_dim),
            ).squeeze(0)

        aleatoric_uncertainty = torch.zeros(output_state_means.shape[0], device=output_state_means.device)
        if self.ensemble_size > 1:
            epistemic_uncertainty = state_means.std(dim=0).sum(dim=1)
        else:
            epistemic_uncertainty = torch.zeros(output_state_means.shape[0], device=output_state_means.device)
        return output_state_means, aleatoric_uncertainty, epistemic_uncertainty, None, None, None

    def compute_loss(
        self,
        state_batch,
        action_batch,
        extension_batch,
        contact_batch,
        termination_batch,
        privilege_batch=None,
        bootstrap=False,
    ):
        del extension_batch, contact_batch, termination_batch
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        privilege_batch = privilege_batch.to(self.device) if privilege_batch is not None else None
        forecast_horizon = state_batch.shape[1] - self.history_horizon
        loss_rho = float(self.architecture_config.get("rho", 0.5))
        state_losses = []
        for model in self.models:
            if bootstrap:
                ids = torch.randint(0, state_batch.shape[0], (state_batch.shape[0],), device=state_batch.device)
            else:
                ids = torch.arange(0, state_batch.shape[0], device=state_batch.device)
            x_state = state_batch[ids, self.history_horizon - 1]
            model_losses = []
            for step in range(forecast_horizon):
                target_index = self.history_horizon + step
                x_action = action_batch[ids, target_index]
                state_target = state_batch[ids, target_index]
                privilege = self._privilege(
                    x_state.shape[0],
                    x_state,
                    privilege_batch=privilege_batch,
                    ids=ids,
                    step_index=target_index - 1,
                    allow_dropout=True,
                )
                x_next = model(x_state, x_action, privilege, n_step=self.num_steps)
                model_losses.append((loss_rho**step) * model.loss(state_target, x_state, x_action, privilege))
                x_state = x_next
            state_losses.append(torch.stack(model_losses).sum() / max(forecast_horizon, 1))

        state_loss = torch.stack(state_losses).mean()
        zero = torch.zeros((), dtype=state_batch.dtype, device=state_batch.device)
        return state_loss, zero, zero, zero, zero, zero, zero

    def reset(self):
        return None

    def reset_partial(self, batch_indices):
        del batch_indices
        return None
