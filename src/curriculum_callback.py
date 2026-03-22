# curriculum_callback.py — Full updated file

import os
import json
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

import config


class CurriculumCallback(BaseCallback):
    MAX_CHECKPOINTS_TO_KEEP = 3

    def __init__(self, save_path: str, verbose: int = 1, start_phase: int = 0):
        super().__init__(verbose)
        self.save_path = save_path
        self.current_phase = start_phase

        self.win_buffer    = deque(maxlen=config.WIN_RATE_WINDOW)
        self.reward_buffer = deque(maxlen=300)
        self.last_save_step   = 0

        # FIX: Per-phase bests instead of global bests.
        # Structure: { phase_idx: {"reward": float, "win_rate": float} }
        # Each phase starts at -inf so a new phase never inherits the old one's threshold.
        self._phase_bests: dict[int, dict] = {}
        self._checkpoint_registry: list[tuple[str, str]] = []

    # ------------------------------------------------------------------
    # Per-phase best helpers
    # ------------------------------------------------------------------
    def _get_phase_best(self, key: str) -> float:
        return self._phase_bests.get(self.current_phase, {}).get(key, -np.inf)

    def _set_phase_best(self, key: str, value: float):
        if self.current_phase not in self._phase_bests:
            self._phase_bests[self.current_phase] = {}
        self._phase_bests[self.current_phase][key] = value

    # ------------------------------------------------------------------
    # Metric helpers
    # ------------------------------------------------------------------
    def _win_rate(self) -> float:
        if len(self.win_buffer) < config.WIN_RATE_WINDOW // 2:
            return 0.0
        return float(np.mean(self.win_buffer))

    def _mean_reward(self) -> float:
        if len(self.reward_buffer) == 0:
            return -np.inf
        return float(np.mean(self.reward_buffer))

    # ------------------------------------------------------------------
    # Phase state persistence
    # ------------------------------------------------------------------
    def _save_phase_state(self):
        """Write curriculum progress to disk so resume scripts can restore it."""
        state = {
            "current_phase":  self.current_phase,
            "num_timesteps":  self.num_timesteps,
            # Serialize per-phase bests — keys must be strings for JSON
            "phase_bests": {
                str(k): v for k, v in self._phase_bests.items()
            },
        }
        path = os.path.join(self.save_path, "curriculum_state.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        if self.verbose:
            print(f"[Curriculum] State saved → Phase {self.current_phase + 1} "
                  f"at {self.num_timesteps:,} steps")

    @staticmethod
    def load_phase_state(save_path: str) -> dict:
        """
        Load persisted curriculum state.
        Returns safe defaults if no file exists (fresh training run).
        """
        path = os.path.join(save_path, "curriculum_state.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                raw = json.load(f)
            # Re-key phase_bests back to int
            raw["phase_bests"] = {
                int(k): v for k, v in raw.get("phase_bests", {}).items()
            }
            print(f"[Curriculum] Restored: Phase {raw['current_phase'] + 1} "
                  f"| {raw['num_timesteps']:,} steps")
            return raw

        print("[Curriculum] No saved state found — starting from Phase 1.")
        return {
            "current_phase":  0,
            "num_timesteps":  0,
            "phase_bests":    {},
        }

    # ------------------------------------------------------------------
    # Hyperparameter injection
    # ------------------------------------------------------------------
    def _set_phase_hyperparams(self, phase_idx: int):
        params = config.PHASE_HYPERPARAMS[phase_idx]
        lr   = params["lr"]
        ent  = params["ent_coef"]
        clip = params["clip"]

        self.model.learning_rate = lambda _: lr
        self.model.clip_range    = lambda _: clip
        self.model.ent_coef      = ent

        for pg in self.model.policy.optimizer.param_groups:
            pg["lr"] = lr

        if self.verbose:
            print(f"\n[Curriculum] Phase {phase_idx + 1} hyperparams → "
                  f"LR={lr:.2e} | ent={ent:.4f} | clip={clip:.3f}")

    # ------------------------------------------------------------------
    # Curriculum advancement
    # ------------------------------------------------------------------
    def _advance_phase(self):
        next_phase = self.current_phase + 1
        if next_phase >= len(config.CURRICULUM_PHASES):
            if self.verbose:
                print("\n[Curriculum] Final phase — continuing at max difficulty.")
            return

        self.current_phase = next_phase
        new_states = config.CURRICULUM_PHASES[self.current_phase]

        try:
            self.training_env.env_method("set_training_states", new_states)
        except AttributeError:
            config.TRAINING_STATES = new_states

        self._set_phase_hyperparams(self.current_phase)

        # FIX: Reset scalar bests for the new phase.
        # The new phase's _phase_bests entry doesn't exist yet → _get_phase_best returns -inf.
        # No explicit reset needed — the per-phase dict handles it automatically.

        # Permanent phase-transition checkpoint
        tag = f"phase{self.current_phase + 1}_entry"
        self.model.save(
            os.path.join(self.save_path, f"{config.MODEL_NAME}_{tag}"))
        self.training_env.save(
            os.path.join(self.save_path, f"{config.MODEL_NAME}_vecnorm_{tag}.pkl"))

        # Persist state immediately after transition
        self._save_phase_state()

        # Decay normalizer count
        try:
            norm = self.training_env
            if hasattr(norm, "count"):
                norm.count = min(norm.count, 5_000.0)
        except Exception:
            pass

        # Clear measurement windows
        self.win_buffer.clear()
        self.reward_buffer.clear()

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[Curriculum] *** ADVANCING TO PHASE {self.current_phase + 1} ***")
            print(f"  States: {new_states}")
            print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------
    def _save_best_reward(self):
        path     = os.path.join(self.save_path, f"{config.MODEL_NAME}_BEST_REWARD")
        vec_path = os.path.join(self.save_path, f"{config.MODEL_NAME}_vecnorm_BEST_REWARD.pkl")
        self.model.save(path)
        self.training_env.save(vec_path)
        if self.verbose:
            print(f"[Best-Reward ✓] {self.num_timesteps:,} steps | "
                  f"Phase {self.current_phase + 1} | "
                  f"New best: {self._get_phase_best('reward'):.2f}")

    def _save_best_winrate(self):
        path     = os.path.join(self.save_path, f"{config.MODEL_NAME}_BEST_WINRATE")
        vec_path = os.path.join(self.save_path, f"{config.MODEL_NAME}_vecnorm_BEST_WINRATE.pkl")
        self.model.save(path)
        self.training_env.save(vec_path)
        if self.verbose:
            print(f"[Best-WinRate ✓] {self.num_timesteps:,} steps | "
                  f"Phase {self.current_phase + 1} | "
                  f"New best: {self._win_rate():.1%}")

    def _save_periodic_checkpoint(self):
        path = os.path.join(
            self.save_path,
            f"{config.MODEL_NAME}_ckpt_{self.num_timesteps}_steps"
        )
        vec_path = os.path.join(
            self.save_path,
            f"{config.MODEL_NAME}_vecnorm_ckpt_{self.num_timesteps}_steps.pkl"
        )
        self.model.save(path)
        self.training_env.save(vec_path)
        self._checkpoint_registry.append((path + ".zip", vec_path))

        while len(self._checkpoint_registry) > self.MAX_CHECKPOINTS_TO_KEEP:
            old_model, old_vec = self._checkpoint_registry.pop(0)
            for filepath in (old_model, old_vec):
                try:
                    os.remove(filepath)
                    if self.verbose:
                        print(f"[Prune] {os.path.basename(filepath)}")
                except FileNotFoundError:
                    pass

        self.last_save_step = self.num_timesteps

        if self.verbose:
            print(f"\n[Checkpoint] {self.num_timesteps:,} steps | "
                  f"Phase {self.current_phase + 1} | "
                  f"Win Rate: {self._win_rate():.1%} | "
                  f"Mean Reward: {self._mean_reward():.2f}")

    # ------------------------------------------------------------------
    # SB3 hook
    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "win" in info:
                self.win_buffer.append(info["win"])
            if "episode" in info:
                self.reward_buffer.append(info["episode"]["r"])

        if self.num_timesteps % 500 == 0:
            mr = self._mean_reward()
            wr = self._win_rate()

            if mr > self._get_phase_best("reward") and len(self.reward_buffer) >= 20:
                self._set_phase_best("reward", mr)
                self._save_best_reward()

            if wr > self._get_phase_best("win_rate") and len(self.win_buffer) >= config.WIN_RATE_WINDOW // 2:
                self._set_phase_best("win_rate", wr)
                self._save_best_winrate()

        if (
            self.num_timesteps % 1_000 == 0
            and len(self.win_buffer) >= config.WIN_RATE_WINDOW // 2
        ):
            if self._win_rate() >= config.WIN_RATE_THRESHOLD:
                self._advance_phase()

        if self.num_timesteps - self.last_save_step >= config.SAVE_FREQ_STEPS:
            self._save_periodic_checkpoint()

        return True


# ------------------------------------------------------------------
# Phase-stopping subclass — used exclusively by phase_supervisor.py
# ------------------------------------------------------------------
class PhaseStoppingCallback(CurriculumCallback):
    """
    Identical to CurriculumCallback but signals model.learn() to exit
    cleanly immediately after a phase transition. The supervisor uses
    this to insert an Optuna run between phases without any deadlock.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_just_advanced = False

    def _advance_phase(self):
        super()._advance_phase()
        self.phase_just_advanced = True  # Arm the stop flag

    def _on_step(self) -> bool:
        super()._on_step()
        if self.phase_just_advanced:
            print("\n[Supervisor] Phase transition detected — stopping for Optuna.")
            return False  # Stops model.learn() cleanly
        return True