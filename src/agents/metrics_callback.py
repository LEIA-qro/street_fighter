# metrics_callback.py
#
# The TensorBoard callback the whole sf2-sota-rl-upgrade branch was missing:
# the env has emitted reward_parts, spacing aggregates and episode diagnostics
# since Task 2/3, and MacroActionWrapper emits macro_action -- but nothing
# consumed them, so none of the branch's success metrics were measurable.
# This callback reads ONLY info dicts (no env methods, no extra IPC) and logs:
#
#   spacing/*     per-episode rel_dist mean / median / fraction >= 80, plus
#                 ep_air_frac (fraction of non-sentinel steps airborne).
#                 THE metrics: baseline is median 83 with 52.2% of steps far;
#                 the movement fix is judged by this falling toward 70. And
#                 ep_air_frac ~0.33 is what uniform random looks like (3 of 9
#                 directions are jumps) -- walking drives it well under 0.15.
#   reward/*      per-step mean of every reward component (damage, taken,
#                 combo, shaping, time, terminal). Confirms shaping is now the
#                 same order as damage instead of 0.2% of it.
#   macros/*      fraction of agent actions that are macros, plus per-macro
#                 usage. If ~0 after 1M steps, raise ent_coef before
#                 concluding macros do not help.
#   episodes/*    win rate, double-KO rate, timeout rate, mean length.
#   env/*         hp_sentinel frame rate, socket deaths.
#   throughput/*  aggregate agent steps/s overall and during collection only,
#                 plus seconds spent in the gradient phase. The number the
#                 whole optimization roadmap is scored against.
#
# Works under PPO (one _on_rollout_end per 2048-vec-step rollout) and QR-DQN
# (one per 4-step collect cycle): records are flushed only when at least
# `min_steps_per_log` transitions have accumulated.

import time
from collections import defaultdict

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from envs.action_macros import MACRO_NAMES, N_PRIMITIVES


class MetricsCallback(BaseCallback):
    def __init__(self, min_steps_per_log: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.min_steps_per_log = min_steps_per_log
        self._reset_window()
        # Throughput bookkeeping
        self._t_train_start = None
        self._ts_last_flush = 0
        self._t_last_flush = None
        self._t_rollout_start = None
        self._ts_rollout_start = 0
        self._collect_seconds = 0.0
        self._grad_seconds = 0.0
        self._t_last_rollout_end = None

    def _reset_window(self):
        self._steps_seen = 0
        self._sentinel_steps = 0
        self._socket_deaths = 0
        self._reward_sums = defaultdict(float)
        self._reward_steps = 0
        self._actions_seen = 0
        self._macro_counts = defaultdict(int)
        self._ep_means = []
        self._ep_medians = []
        self._ep_frac_far = []
        self._ep_air_fracs = []
        self._ep_lengths = []
        self._ep_wins = []
        self._ep_double_kos = []
        self._ep_timeouts = []

    # ------------------------------------------------------------------ ingest

    def _ingest_infos(self, infos) -> None:
        """Pure accumulation over one vec-step's info dicts. Unit-tested."""
        for info in infos:
            self._steps_seen += 1
            if info.get("hp_sentinel"):
                self._sentinel_steps += 1
            if info.get("socket_death"):
                self._socket_deaths += 1

            parts = info.get("reward_parts")
            if parts:
                self._reward_steps += 1
                for key, val in parts.items():
                    self._reward_sums[key] += float(val)

            macro_action = info.get("macro_action")
            if macro_action is not None:
                self._actions_seen += 1
                if macro_action >= N_PRIMITIVES:
                    self._macro_counts[macro_action - N_PRIMITIVES] += 1

            if "win" in info:
                self._ep_wins.append(info["win"])
                self._ep_double_kos.append(1 if info.get("double_ko") else 0)
                self._ep_timeouts.append(1 if info.get("timeout") else 0)
                if "episode_steps" in info:
                    self._ep_lengths.append(info["episode_steps"])
                if "ep_rel_dist_mean" in info:
                    self._ep_means.append(info["ep_rel_dist_mean"])
                    self._ep_medians.append(info["ep_rel_dist_median"])
                    self._ep_frac_far.append(info["ep_rel_dist_frac_far"])
                if "ep_air_frac" in info:
                    self._ep_air_fracs.append(info["ep_air_frac"])

    # ----------------------------------------------------------------- compute

    def _compute_records(self) -> dict:
        """Turns the accumulated window into a {tag: value} dict. Unit-tested."""
        records = {}
        if self._reward_steps:
            for key, total in self._reward_sums.items():
                records[f"reward/{key}_per_step"] = total / self._reward_steps
        if self._ep_means:
            records["spacing/ep_rel_dist_mean"] = float(np.mean(self._ep_means))
            records["spacing/ep_rel_dist_median"] = float(np.mean(self._ep_medians))
            records["spacing/frac_steps_far"] = float(np.mean(self._ep_frac_far))
        if self._ep_air_fracs:
            records["spacing/ep_air_frac"] = float(np.mean(self._ep_air_fracs))
        if self._actions_seen:
            n_macros = sum(self._macro_counts.values())
            records["macros/frac_macro_actions"] = n_macros / self._actions_seen
            for idx, name in enumerate(MACRO_NAMES):
                records[f"macros/use_{name}"] = self._macro_counts[idx] / self._actions_seen
        if self._ep_wins:
            records["episodes/win_rate"] = float(np.mean(self._ep_wins))
            records["episodes/double_ko_rate"] = float(np.mean(self._ep_double_kos))
            records["episodes/timeout_rate"] = float(np.mean(self._ep_timeouts))
        if self._ep_lengths:
            records["episodes/len_mean"] = float(np.mean(self._ep_lengths))
        if self._steps_seen:
            records["env/hp_sentinel_frac"] = self._sentinel_steps / self._steps_seen
            records["env/socket_deaths"] = self._socket_deaths
        return records

    # -------------------------------------------------------------- SB3 hooks

    def _on_training_start(self) -> None:
        now = time.perf_counter()
        self._t_train_start = now
        self._t_last_flush = now
        self._ts_last_flush = self.num_timesteps

    def _on_rollout_start(self) -> None:
        now = time.perf_counter()
        self._t_rollout_start = now
        self._ts_rollout_start = self.num_timesteps
        if self._t_last_rollout_end is not None:
            self._grad_seconds += now - self._t_last_rollout_end

    def _on_step(self) -> bool:
        self._ingest_infos(self.locals.get("infos", []))
        return True

    def _on_rollout_end(self) -> None:
        now = time.perf_counter()
        if self._t_rollout_start is not None:
            self._collect_seconds += now - self._t_rollout_start
        self._t_last_rollout_end = now

        if self.num_timesteps - self._ts_last_flush < self.min_steps_per_log:
            return

        for tag, value in self._compute_records().items():
            self.logger.record(tag, value)

        wall = now - self._t_last_flush
        steps = self.num_timesteps - self._ts_last_flush
        if wall > 0:
            self.logger.record("throughput/agent_steps_per_s", steps / wall)
        if self._collect_seconds > 0:
            self.logger.record(
                "throughput/collect_steps_per_s", steps / self._collect_seconds
            )
        self.logger.record("throughput/grad_seconds", self._grad_seconds)

        self._t_last_flush = now
        self._ts_last_flush = self.num_timesteps
        self._collect_seconds = 0.0
        self._grad_seconds = 0.0
        self._reset_window()
