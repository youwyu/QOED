from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import MBPOPPO
from rsl_rl.modules import SystemDynamicsEnsemble
from rsl_rl.storage.replay_buffer import ReplayBuffer as _RslReplayBuffer

from boed.fisher import FisherEstimator, ParameterDistribution
from boed.objectives import VALID_MODES
from mbpo.neural import FlowDynamics

INFO_GAIN_NOTHING = "nothing"
LOSS_NAMES = ("state", "sequence", "bound", "kl", "extension", "contact", "termination")


class ReplayBuffer(_RslReplayBuffer):
    """Replay buffer sampler that avoids CPU RNG and large reset-mask unfolds."""

    def _sequence_replay_buf(self, sequence_length):
        if self.num_transitions >= sequence_length:
            return self.replay_buf

        def pad(buf):
            zeros = torch.zeros(buf.shape[0], sequence_length - self.num_transitions, buf.shape[-1], device=self.device)
            return torch.cat([zeros, buf[:, : self.num_transitions]], dim=1)

        if isinstance(self.replay_buf, list):
            return [None if buf is None else pad(buf) for buf in self.replay_buf]
        return pad(self.replay_buf)

    def has_valid_sequences(self, sequence_length: int) -> bool:
        if self.replay_buf is None or self.num_transitions <= 0:
            return False
        replay_buf = self._sequence_replay_buf(sequence_length)
        reset_data = replay_buf[-1] if isinstance(replay_buf, list) else None
        return reset_data is None or self._generate_valid_indices(reset_data, sequence_length)[0].numel() > 0

    def sample_batch(self, sequence_length: int, mini_batch_size: int, require_valid: bool = True):
        replay_buf = self._sequence_replay_buf(sequence_length)
        valid_indices = None
        if require_valid and isinstance(replay_buf, list) and replay_buf[-1] is not None:
            valid_indices = self._generate_valid_indices(replay_buf[-1], sequence_length)
        return self._generate_batch(replay_buf, valid_indices, sequence_length, mini_batch_size)

    def _generate_valid_indices(self, reset_data, sequence_length):
        reset_flags = reset_data[:, : max(self.num_transitions, sequence_length)].to(torch.bool)
        if reset_flags.ndim == 3:
            reset_flags = reset_flags.squeeze(-1)
        num_starts = reset_flags.shape[1] - sequence_length + 1
        if num_starts <= 0:
            empty = torch.zeros(0, dtype=torch.long, device=self.device)
            return empty, empty.clone()
        window = sequence_length - 1
        if window <= 0:
            valid_mask = torch.ones(reset_flags.shape[0], num_starts, dtype=torch.bool, device=reset_flags.device)
        else:
            reset_cumsum = torch.nn.functional.pad(reset_flags.to(torch.int32), (1, 0)).cumsum(dim=1)
            valid_mask = reset_cumsum[:, window : window + num_starts] == reset_cumsum[:, :num_starts]
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

        index = (sampled_envs[:, None], sampled_starts[:, None] + torch.arange(sequence_length, device=self.device))
        if isinstance(replay_buf, list):
            return [None if buf is None else buf[index] for buf in replay_buf]
        return replay_buf[index]


def build_system_dynamics(*args, architecture_config: dict | None = None, **kwargs):
    architecture_config = architecture_config or {}
    cls = ShortcutSystemDynamicsEnsemble if architecture_config.get("type") == "shortcut" else SystemDynamicsEnsemble
    return cls(*args, architecture_config=architecture_config, **kwargs)


class QOED_MBPOPPO(MBPOPPO):
    """MBPOPPO variant that keeps RSL-RL's ReplayBuffer on the runner device."""

    class _FisherDynamicsAdapter:
        def __init__(self, system_dynamics, privilege, index, scale):
            self.system_dynamics = system_dynamics
            self.privilege, self.index, self.scale = privilege, index, scale

        def step(self, state, action, theta):
            inputs, index = torch.unique(torch.cat([state, action], dim=-1), dim=0, return_inverse=True)
            noise = torch.randn(inputs.shape[0], state.shape[-1], device=state.device, dtype=state.dtype)[index]
            privilege = self.privilege.repeat(theta.shape[0], 1).index_copy_(1, self.index, theta * self.scale)
            return self.system_dynamics.forward(state.unsqueeze(1), action.unsqueeze(1), privilege_batch=privilege, noise=noise)[0]

    def __init__(
        self,
        *args,
        fisher_param_min=-100.0,
        fisher_param_max=100.0,
        fisher_prior_mean=None,
        fisher_prior_cov=None,
        fisher_prior_cov_diag=None,
        fisher_var_threshold_for_update: float = 0.0025,
        fisher_fd_delta_floor: float = 1.0,
        fisher_obs_noise_std: float = 0.025,
        info_gain_mode: str = INFO_GAIN_NOTHING,
        info_gain_num_actions: int = 1024,
        info_gain_clip_actions: float | None = None,
        **kwargs,
    ):
        if info_gain_mode not in (*VALID_MODES, INFO_GAIN_NOTHING):
            raise ValueError(f"Invalid info_gain_mode '{info_gain_mode}', expected one of {(*VALID_MODES, INFO_GAIN_NOTHING)}")
        self.info_gain_mode = info_gain_mode
        self.info_gain_num_actions = max(1, int(info_gain_num_actions))
        self.info_gain_clip_actions = None if info_gain_clip_actions is None else float(info_gain_clip_actions)
        self.cumulative_info_gain = 0.0
        self.num_info_gain_selections = 0
        self.latest_system_dynamics_autoregressive_error = None
        super().__init__(*args, **kwargs)

        self._fisher_prev_state = None
        self._fisher_prev_valid = None
        self._fisher_residual_var = None
        self._fisher_window = 20
        self._fisher_param_min = fisher_param_min
        self._fisher_param_max = fisher_param_max
        self._fisher_prior_dist = None
        self.parameter_estimator = None
        self.system_privilege_dim = int(getattr(self.system_dynamics, "shortcut_privilege_dim", 0))

        buffer = self.system_replay_buffer
        dims = list(buffer.dim)
        if self.system_privilege_dim > 0:
            dims.insert(4, self.system_privilege_dim)
        self.system_replay_buffer = ReplayBuffer(dims, buffer.buffer_size, buffer.device)
        if self.system_privilege_dim <= 0:
            return

        p = self.system_privilege_dim
        if fisher_prior_mean is None:
            mean = torch.zeros(p, device=self.device)
        else:
            mean = torch.as_tensor(fisher_prior_mean, device=self.device, dtype=torch.float32).reshape(p)
        cov_cfg = fisher_prior_cov if fisher_prior_cov is not None else fisher_prior_cov_diag
        if cov_cfg is None:
            cov = torch.eye(p, device=self.device, dtype=mean.dtype) * 25.0
        else:
            cov = torch.as_tensor(cov_cfg, device=self.device, dtype=mean.dtype)
            cov = torch.diag(torch.clamp(cov.reshape(p), min=0.0)) if cov.ndim == 1 else 0.5 * (cov + cov.T)

        def bound(value):
            return torch.as_tensor(value, device=self.device, dtype=mean.dtype).reshape(-1).expand(p).clone()

        self._fisher_prior_dist = ParameterDistribution(mean, cov)
        self._fisher_param_min, self._fisher_param_max = bound(fisher_param_min), bound(fisher_param_max)
        scale = cov.diagonal().clamp_min(0.0).sqrt()
        idx = self._fisher_index = (scale > 0).nonzero().flatten()
        s = self._fisher_scale = scale[idx]
        self.parameter_estimator = FisherEstimator(
            self._FisherDynamicsAdapter(self.system_dynamics, mean, idx, s),
            ParameterDistribution(mean[idx] / s, cov[idx][:, idx] / (s[:, None] * s[None])),
            obs_noise_std=fisher_obs_noise_std,
            max_history=None,
            cem_samples=2048,
            cem_iters=5,
            param_min=self._fisher_param_min[idx] / s,
            param_max=self._fisher_param_max[idx] / s,
            var_threshold_for_update=fisher_var_threshold_for_update,
            fd_delta_floor=fisher_fd_delta_floor,
        )
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
        candidates = self.policy.distribution.sample((self.info_gain_num_actions,)).detach()
        candidates = torch.nan_to_num(candidates, nan=0.0, posinf=0.0, neginf=0.0)
        candidates[0] = base_action

        self.transition.actions = self._select_info_gain_action(obs, candidates, base_action).detach()
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def _select_info_gain_action(self, obs, candidates: torch.Tensor, fallback_action: torch.Tensor) -> torch.Tensor:
        system_state = self.state_normalizer(obs["system_state"].to(self.device))
        selected = []
        for env_idx in range(candidates.shape[1]):
            action_batch = candidates[:, env_idx].to(self.device)
            if self.info_gain_clip_actions is not None:
                action_batch = action_batch.clamp(-self.info_gain_clip_actions, self.info_gain_clip_actions)
            was_training = self.system_dynamics.training
            self.system_dynamics.eval()
            scores = self.parameter_estimator.local_fisher_trace_path(
                system_state[env_idx : env_idx + 1],
                self.action_normalizer(action_batch).unsqueeze(1),
                baseline=self.info_gain_mode,
            )
            self.system_dynamics.train(was_training)
            scores = torch.nan_to_num(scores, nan=-torch.inf, posinf=1.0e30, neginf=-torch.inf)
            if not torch.isfinite(scores).any():
                selected.append(fallback_action[env_idx].to(self.device))
                continue
            best_idx = torch.argmax(scores)
            self.cumulative_info_gain = self.cumulative_info_gain + scores[best_idx].detach()
            self.num_info_gain_selections += 1
            selected.append(action_batch[best_idx])
        return torch.stack(selected, dim=0).to(device=fallback_action.device, dtype=fallback_action.dtype)

    def _is_imagination_observation(self, obs) -> bool:
        imagination_storage = getattr(self, "imagination_storage", None)
        storage = getattr(self, "storage", None)
        if imagination_storage is None or storage is None or "policy" not in obs:
            return False
        batch_size = int(obs["policy"].shape[0])
        return batch_size == imagination_storage.num_envs and batch_size != storage.num_envs

    def _effective_privilege_mean_cov(self):
        prior = self._fisher_prior_dist
        if prior is None:
            return None, None
        if self.parameter_estimator is None or self.info_gain_mode == INFO_GAIN_NOTHING:
            return prior.mean, prior.cov
        dist, idx, s = self.parameter_estimator.dist, self._fisher_index, self._fisher_scale
        cov = torch.zeros_like(prior.cov).index_put((idx[:, None], idx), dist.cov * s[:, None] * s[None])
        return prior.mean.index_copy(0, idx, dist.mean * s), cov

    def effective_system_privilege_mean(self) -> torch.Tensor | None:
        return self._effective_privilege_mean_cov()[0]

    def _estimated_system_privilege(self, batch_size: int, like: torch.Tensor):
        mean = self.effective_system_privilege_mean()
        return None if mean is None else mean.to(device=like.device, dtype=like.dtype).expand(batch_size, -1)

    def _sync_system_dynamics_privilege_source(self) -> None:
        setter = getattr(self.system_dynamics, "set_source_privilege_distribution", None)
        if setter is None:
            return
        mean, cov = self._effective_privilege_mean_cov()
        if mean is not None:
            setter(mean, cov, self._fisher_param_min, self._fisher_param_max)

    def _update_parameter_estimator(self, system_state, system_action, system_termination=None):
        n = system_state.shape[0]
        estimator = self.parameter_estimator
        if estimator is None or self.info_gain_mode == INFO_GAIN_NOTHING:
            return self._estimated_system_privilege(n, system_state)
        done = torch.zeros(n, dtype=torch.bool, device=system_state.device)
        if system_termination is not None:
            done = system_termination.reshape(n, -1).any(dim=1).to(torch.bool)

        if self._fisher_prev_state is not None:
            for idx in (self._fisher_prev_valid & ~done).nonzero(as_tuple=False).flatten().tolist():
                estimator.add_sample(self._fisher_prev_state[idx], system_action[idx], system_state[idx])
            if len(estimator.est_history) >= self._fisher_window:
                estimator.history = estimator.history[-self._fisher_window :]
                self._calibrate_fisher_noise()
                estimator.update_posterior()
                estimator.est_history.clear()

        self._fisher_prev_state = system_state.detach().clone()
        self._fisher_prev_valid = ~done
        privilege = self._estimated_system_privilege(n, system_state)
        self._sync_system_dynamics_privilege_source()
        return privilege

    def _calibrate_fisher_noise(self):
        estimator = self.parameter_estimator
        states, actions, observed, _ = estimator.stack(estimator.est_history)
        with torch.no_grad():
            predicted = estimator.dyn.step(states, actions, estimator.dist.mean.reshape(1, -1).expand(states.shape[0], -1))
        variance = (observed - predicted).square().mean(0)
        previous = self._fisher_residual_var
        self._fisher_residual_var = variance if previous is None else 0.9 * previous + 0.1 * variance
        estimator.R_inv = torch.diag(1.0 / self._fisher_residual_var.clamp_min(1.0e-4))
        estimator._fisher_cache = None

    def fill_history_buffer(self, obs):
        system_state = self.state_normalizer(obs["system_state"])
        system_action = self.action_normalizer(obs["system_action"])
        system_termination = obs.get("system_termination")
        system_privilege = obs.get("system_privilege")
        if system_privilege is None:
            system_privilege = self._update_parameter_estimator(system_state, system_action, system_termination)
        if system_privilege is not None:
            system_privilege = system_privilege.to(device=system_state.device, dtype=system_state.dtype)

        items = [system_state, system_action, obs.get("system_extension"), obs.get("system_contact")]
        if self.system_privilege_dim > 0:
            items.append(system_privilege)
        items.append(system_termination)
        on_cpu = self.system_replay_buffer.device == "cpu"
        self.system_replay_buffer.insert(
            [None if x is None else (x.detach().cpu() if on_cpu else x).unsqueeze(1) for x in items]
        )

    def prepare_imagination(self):
        self._sync_system_dynamics_privilege_source()
        return super().prepare_imagination()

    def update_system_dynamics(self):
        totals = [0.0] * len(LOSS_NAMES)
        generator = self.system_replay_buffer.mini_batch_generator(
            self.system_dynamics.history_horizon + self.system_dynamics_forecast_horizon,
            self.system_dynamics_num_mini_batches,
            self.system_dynamics_mini_batch_size,
        )
        for state, action, extension, contact, *privilege, termination in generator:
            self.system_dynamics.reset()
            kwargs = {"privilege_batch": privilege[0]} if privilege else {}
            losses = self.system_dynamics.compute_loss(state, action, extension, contact, termination, bootstrap=True, **kwargs)
            loss = sum(self.system_dynamics_loss_weights[name] * value for name, value in zip(LOSS_NAMES, losses))
            self.system_dynamics_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.system_dynamics.parameters(), self.max_grad_norm)
            self.system_dynamics_optimizer.step()
            for i, value in enumerate(losses):
                totals[i] += value.item()
        return tuple(total / self.system_dynamics_num_mini_batches for total in totals)

    def system_dynamics_autoregressive_prediction(
        self, state_traj, action_traj, extension_traj, contact_traj, termination_traj, privilege_traj=None
    ):
        h = self.system_dynamics.history_horizon
        recurrent = self.system_dynamics.architecture_config["type"] in ["rnn", "rssm"]
        state_pred = torch.zeros_like(state_traj, device=self.device)
        aleatoric = torch.zeros(state_traj.shape[0], state_traj.shape[1], device=self.device)
        epistemic = torch.zeros(state_traj.shape[0], state_traj.shape[1], device=self.device)
        action_pred = action_traj.clone()

        def init(traj):
            if traj is None:
                return None
            pred = torch.zeros_like(traj, device=self.device)
            pred[:, :h] = traj[:, :h]
            return pred

        extension_pred, contact_pred, termination_pred = init(extension_traj), init(contact_traj), init(termination_traj)
        state_pred[:, :h] = state_traj[:, :h]

        self.system_dynamics.reset()
        with torch.inference_mode():
            for i in range(h, self.system_dynamics_len_eval_trajectory):
                window = slice(i - 1, i) if recurrent and i > h else slice(i - h, i)
                kwargs = {} if privilege_traj is None else {"privilege_batch": privilege_traj[:, window]}
                state_i, aleatoric_i, epistemic_i, extension_i, contact_i, termination_i = self.system_dynamics.forward(
                    state_pred[:, window], action_pred[:, window.start + 1 : window.stop + 1], **kwargs
                )
                state_pred[:, i] = state_i
                aleatoric[:, i] = aleatoric_i
                epistemic[:, i] = epistemic_i
                if extension_pred is not None and extension_i is not None:
                    extension_pred[:, i] = extension_i
                if contact_pred is not None and contact_i is not None:
                    contact_pred[:, i] = torch.sigmoid(contact_i).round().int()
                if termination_pred is not None and termination_i is not None:
                    termination_pred[:, i] = torch.sigmoid(termination_i).round().int()
        return state_pred, aleatoric, epistemic, action_pred, extension_pred, contact_pred, termination_pred

    def _relative_error(self, pred, truth) -> float:
        h = self.system_dynamics.history_horizon
        denominator = truth[:, h:].abs().sum(dim=-1).clamp_min(1.0e-8)
        return ((pred[:, h:] - truth[:, h:]).abs().sum(dim=-1) / denominator).mean().item()

    def evaluate_system_dynamics(self, privilege_override: str | None = None, num_trajectories: int | None = None):
        requested_len = self.system_dynamics_len_eval_trajectory
        eval_len = self._resolve_system_dynamics_eval_length(requested_len)
        require_valid = eval_len is not None
        if eval_len is None:
            eval_len = min(requested_len, max(self.system_dynamics.history_horizon + 1, self.system_replay_buffer.num_transitions))

        batch = self.system_replay_buffer.sample_batch(eval_len, num_trajectories or self.system_dynamics_num_eval_trajectories, require_valid=require_valid)
        state_traj, action_traj, extension_traj, contact_traj, *privilege, termination_traj = batch
        privilege_traj = privilege[0] if privilege else None
        if privilege_override == "current_mean" and self.system_privilege_dim > 0:
            current = self._estimated_system_privilege(state_traj.shape[0], state_traj)
            if current is not None:
                privilege_traj = current.unsqueeze(1).expand(-1, eval_len, -1)

        was_training = self.system_dynamics.training
        self.system_dynamics.eval()
        self.system_dynamics_len_eval_trajectory = eval_len
        state_pred, _, _, action_pred, extension_pred, contact_pred, termination_pred = self.system_dynamics_autoregressive_prediction(
            state_traj, action_traj, extension_traj, contact_traj, termination_traj, privilege_traj
        )
        error = self._relative_error(state_pred, state_traj)
        noised_errors = {}
        for noise_scale in self.system_dynamics_eval_traj_noise_scale:
            state_noised = state_traj + torch.randn_like(state_traj) * noise_scale
            action_noised = action_traj + torch.randn_like(action_traj) * noise_scale
            state_pred_noised = self.system_dynamics_autoregressive_prediction(
                state_noised, action_noised, extension_traj, contact_traj, termination_traj, privilege_traj
            )[0]
            noised_errors[noise_scale] = self._relative_error(state_pred_noised, state_noised)
        self.system_dynamics_len_eval_trajectory = requested_len
        self.system_dynamics.train(was_training)
        self.latest_system_dynamics_autoregressive_error = float(error)
        return (
            state_traj,
            action_traj,
            extension_traj,
            contact_traj,
            termination_traj,
            state_pred,
            action_pred,
            extension_pred,
            contact_pred,
            termination_pred,
            error,
            noised_errors,
        )

    def _resolve_system_dynamics_eval_length(self, requested_eval_len: int) -> int | None:
        buffer = self.system_replay_buffer
        low, high = self.system_dynamics.history_horizon + 1, min(requested_eval_len, buffer.num_transitions)
        if buffer.replay_buf is None or high < low or not buffer.has_valid_sequences(low):
            return None
        while low < high:
            mid = (low + high + 1) // 2
            if buffer.has_valid_sequences(mid):
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
        cfg = architecture_config or {}
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.extension_dim = extension_dim
        self.contact_dim = contact_dim
        self.termination_dim = termination_dim
        self.device = device
        self.ensemble_size = ensemble_size
        self.history_horizon = history_horizon
        self.architecture_config = cfg
        self.freeze_auxiliary = freeze_auxiliary
        self.num_steps = int(cfg.get("num_steps", 1))
        self.shortcut_privilege_dim = int(cfg.get("privilege_dim", 0))
        self.shortcut_privilege_source_noise = bool(cfg.get("privilege_source_noise", True))
        self.shortcut_privilege_dropout_prob = float(cfg.get("privilege_dropout_prob", 0.1))
        self.shortcut_privilege_availability = bool(cfg.get("privilege_availability_feature", True))
        self._source_privilege_mean = None
        self._source_privilege_cov = None
        self._source_privilege_min = None
        self._source_privilege_max = None
        condition_dim = self.shortcut_privilege_dim
        if self.shortcut_privilege_dim > 0 and self.shortcut_privilege_availability:
            condition_dim += 1
        self.models = nn.ModuleList(
            [
                FlowDynamics(
                    state_dim,
                    action_dim,
                    condition_dim,
                    latent_dim=int(cfg.get("latent_dim", state_dim)),
                    timestep_embed_dim=cfg.get("timestep_embed_dim"),
                    use_shortcut=bool(cfg.get("use_shortcut", True)),
                    shortcut_self_consistency=float(cfg.get("shortcut_self_consistency", 0.25)),
                    shortcut_min_dt=float(cfg.get("shortcut_min_dt", 0.0078125)),
                    depth=int(cfg.get("depth", 6)),
                    num_heads=int(cfg.get("num_heads", 4)),
                    mlp_ratio=float(cfg.get("mlp_ratio", 4.0)),
                    num_registers=int(cfg.get("num_registers", 8)),
                )
                for _ in range(ensemble_size)
            ]
        )

    def set_source_privilege_distribution(self, mean, cov, param_min=None, param_max=None) -> None:
        if self.shortcut_privilege_dim <= 0:
            return
        mean = torch.as_tensor(mean, device=self.device, dtype=torch.float32).reshape(self.shortcut_privilege_dim)
        cov = torch.as_tensor(cov, device=self.device, dtype=mean.dtype).reshape(mean.numel(), mean.numel())

        def bound(value):
            if value is None:
                return None
            return torch.as_tensor(value, device=mean.device, dtype=mean.dtype).reshape(-1).expand(mean.numel()).clone()

        self._source_privilege_mean = mean.detach().clone()
        self._source_privilege_cov = (0.5 * (cov + cov.T)).detach().clone()
        self._source_privilege_min = bound(param_min)
        self._source_privilege_max = bound(param_max)

    def _privilege(self, batch_size, like, privilege_batch=None, ids=None, step_index=-1, allow_dropout=False):
        if self.shortcut_privilege_dim <= 0:
            return None
        if privilege_batch is None:
            privilege = self._sample_source_privilege(batch_size, like)
            available = torch.full(
                (batch_size, 1), float(self._source_privilege_mean is not None), dtype=like.dtype, device=like.device
            )
            return self._append_privilege_availability(privilege, available)
        privilege_batch = privilege_batch.to(device=like.device, dtype=like.dtype)
        if ids is not None:
            privilege_batch = privilege_batch[ids]
        privilege = privilege_batch[:, step_index] if privilege_batch.ndim == 3 else privilege_batch
        available = torch.isfinite(privilege).all(dim=-1, keepdim=True).to(dtype=like.dtype)
        privilege = torch.nan_to_num(privilege, nan=0.0, posinf=0.0, neginf=0.0)

        if allow_dropout and self.training and self.shortcut_privilege_dropout_prob > 0.0:
            drop = torch.rand(batch_size, 1, device=like.device) < self.shortcut_privilege_dropout_prob
            available = torch.where(drop, torch.zeros_like(available), available)

        missing = available <= 0.0
        if missing.any():
            privilege = torch.where(missing, self._sample_source_privilege(batch_size, like), privilege)
        return self._append_privilege_availability(privilege, available)

    def _sample_source_privilege(self, batch_size: int, like: torch.Tensor) -> torch.Tensor:
        if self._source_privilege_mean is None or self._source_privilege_cov is None:
            if self.shortcut_privilege_source_noise:
                return torch.randn(batch_size, self.shortcut_privilege_dim, dtype=like.dtype, device=like.device)
            return torch.zeros(batch_size, self.shortcut_privilege_dim, dtype=like.dtype, device=like.device)

        mean = self._source_privilege_mean.to(device=like.device, dtype=like.dtype)
        cov = self._source_privilege_cov.to(device=like.device, dtype=like.dtype)
        noise = torch.randn(batch_size, mean.numel(), dtype=like.dtype, device=like.device)
        eye = torch.eye(mean.numel(), dtype=like.dtype, device=like.device)
        try:
            privilege = mean.unsqueeze(0) + noise @ torch.linalg.cholesky(cov + 1.0e-9 * eye).T
        except RuntimeError:
            privilege = mean.unsqueeze(0) + noise * torch.sqrt(torch.clamp(torch.diagonal(cov), min=0.0)).unsqueeze(0)
        fixed = torch.diagonal(cov) <= 0.0
        if fixed.any():
            privilege[:, fixed] = mean[fixed]
        if self._source_privilege_min is not None:
            privilege = torch.maximum(privilege, self._source_privilege_min.to(device=like.device, dtype=like.dtype).reshape(1, -1))
        if self._source_privilege_max is not None:
            privilege = torch.minimum(privilege, self._source_privilege_max.to(device=like.device, dtype=like.dtype).reshape(1, -1))
        return privilege

    def _append_privilege_availability(self, privilege: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
        return torch.cat((privilege, available), dim=-1) if self.shortcut_privilege_availability else privilege

    def forward(self, x_state_batch, x_action_batch, model_ids=None, privilege_batch=None, noise=None):
        state = x_state_batch.to(self.device)[:, -1]
        action = x_action_batch.to(self.device)[:, -1]
        privilege = self._privilege(state.shape[0], state, privilege_batch=privilege_batch)
        state_means = torch.cat([model(state, action, privilege, n_step=self.num_steps, noise=noise).unsqueeze(0) for model in self.models], dim=0)
        if model_ids is None:
            output_state_means = state_means.mean(dim=0)
        else:
            output_state_means = torch.gather(state_means, 0, model_ids.repeat(1, 1, self.state_dim)).squeeze(0)

        zeros = torch.zeros(output_state_means.shape[0], device=output_state_means.device)
        epistemic_uncertainty = state_means.std(dim=0).sum(dim=1) if self.ensemble_size > 1 else zeros.clone()
        return output_state_means, zeros, epistemic_uncertainty, None, None, None

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
                privilege = self._privilege(
                    x_state.shape[0],
                    x_state,
                    privilege_batch=privilege_batch,
                    ids=ids,
                    step_index=target_index - 1,
                    allow_dropout=True,
                )
                x_next = model(x_state, x_action, privilege, n_step=self.num_steps)
                model_losses.append((loss_rho**step) * model.loss(state_batch[ids, target_index], x_state, x_action, privilege))
                x_state = x_next
            state_losses.append(torch.stack(model_losses).sum() / max(forecast_horizon, 1))

        zero = torch.zeros((), dtype=state_batch.dtype, device=state_batch.device)
        return torch.stack(state_losses).mean(), zero, zero, zero, zero, zero, zero

    def reset(self):
        pass

    def reset_partial(self, batch_indices):
        pass
