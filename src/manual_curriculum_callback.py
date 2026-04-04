# manual_curriculum_callback.py
#
# Drops the automatic phase-advancement gate from CurriculumCallback.
# YOU decide when the model is ready to move to the next phase by calling:
#
#       callback.set_phase(new_phase_idx)
#
# Everything else is preserved:
#   - Best-reward model saving
#   - Best-winrate model saving
#   - Periodic checkpoints with rolling pruning
#   - Per-phase best tracking (bests reset cleanly when you advance)
#   - Full state persistence to JSON for crash recovery / resume
#   - Live hyperparameter injection on phase change
#   - Normalizer count decay on phase change
#
# Usage in train_production_v2.py:
#
#   callback = ManualCurriculumCallback(
#       save_path=directories["production"],
#       start_phase=0,          # Which phase config to start on
#       eval_interval=500,      # How often (steps) to check for new best models
#       save_interval=1_000_000 # How often (steps) to save a periodic checkpoint
#   )
#
# To advance phase mid-training (from another thread, script, or after
# re-launching with a resume script):
#
#   callback.set_phase(1)  # moves to Phase 2 states + hyperparams
#
import os
import json
import numpy as np
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback

import config


class ManualCurriculumCallback(BaseCallback):
    MAX_CHECKPOINTS_TO_KEEP = 3

    def __init__(self, save_path: str, verbose: int = 1, start_phase: int = 0, 
                 eval_interval: int = 500, save_interval: int = None):
        
        super().__init__(verbose)
        self.save_path = save_path
        self.current_phase = start_phase
        self.eval_interval = eval_interval
        self.save_interval = save_interval if save_interval is not None else config.SAVE_FREQ_STEPS

        self.win_buffer    = deque(maxlen=config.WIN_RATE_WINDOW)
        self.reward_buffer = deque(maxlen=300)

        # FIX: Per-phase bests instead of global bests.
        # Structure: { phase_idx: {"reward": float, "win_rate": float} }
        # Each phase starts at -inf so a new phase never inherits the old one's threshold.
        self._phase_bests: dict[int, dict] = {}
        self._checkpoint_registry: list[tuple[str, str]] = []

        self.last_eval_step = 0
        self.last_save_step   = 0

        # Threshold milestone tracker
        self._threshold_save_fired: set[int] = set()

    def set_phase(self, new_phase_idx: int):
        """
        Manually advance (or rewind) to any phase index.
 
        Call this whenever YOU decide the model is ready:
 
            callback.set_phase(1)   # advance to Phase 2
 
        What happens automatically:
          1. New save-states broadcast to all parallel envs
          2. LR / ent_coef / clip_range injected into live optimizer
          3. Normalizer count decayed to force faster re-adaptation
          4. Win/reward buffers cleared (fresh measurement window)
          5. Per-phase bests reset (new phase starts from -inf)
          6. A permanent phase-entry checkpoint is written to disk
          7. curriculum_state.json updated
        """
        if new_phase_idx < 0 or new_phase_idx >= len(config.CURRICULUM_PHASES):
            print(f"[ManualCurriculum] Invalid phase index {new_phase_idx}. "
                  f"Valid range: 0 - {len(config.CURRICULUM_PHASES) - 1}")
            return
 
        self.current_phase = new_phase_idx
        new_states = config.CURRICULUM_PHASES[self.current_phase]
 
        # 1. Broadcast new states to every parallel env
        try:
            self.training_env.env_method("set_training_states", new_states)
        except AttributeError:
            # Fallback for non-vectorized envs (e.g. during testing)
            config.TRAINING_STATES = new_states
 
        # 2. Inject new hyperparameters into the live optimizer
        self._apply_phase_hyperparams(self.current_phase)
 
        # 3. Decay the normalizer's running count so it adapts faster
        #    to the new distribution without fully resetting learned stats.
        try:
            norm = self.training_env
            if hasattr(norm, "count"):
                norm.count = min(norm.count, 5_000.0)
        except Exception:
            pass
 
        # 4. Clear measurement windows — fresh slate for the new phase
        self.win_buffer.clear()
        self.reward_buffer.clear()
 
        # 5. Per-phase bests reset automatically because the new phase_idx
        #    has no entry in _phase_bests yet → _get_phase_best returns -inf.
 
        # 6. Write a permanent phase-entry checkpoint
        tag = f"phase{self.current_phase + 1}_entry"
        self.model.save(
            os.path.join(self.save_path, f"{config.MODEL_NAME}_{tag}"))
        self.training_env.save(
            os.path.join(self.save_path, f"{config.MODEL_NAME}_vecnorm_{tag}.pkl"))
 
        # 7. Persist updated state to disk
        self._save_phase_state() 
 
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[ManualCurriculum] *** MOVED TO PHASE {self.current_phase + 1} ***")
            print(f"  States : {new_states}")
            print(f"  Steps  : {self.num_timesteps:,}")
            print(f"{'='*60}\n")



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

    # ==================================================================
    # Hyperparameter injection
    # ==================================================================
 
    def _apply_phase_hyperparams(self, phase_idx: int):
        """Push LR / ent_coef / clip_range from config into the live model."""
        params = config.PHASE_HYPERPARAMS[phase_idx]
        lr    = params["lr"]
        ent   = params["ent_coef"]
        clip  = params["clip"]
 
        self.model.learning_rate = lambda _: lr
        self.model.clip_range    = lambda _: clip
        self.model.ent_coef      = ent
 
        for pg in self.model.policy.optimizer.param_groups:
            pg["lr"] = lr
 
        if self.verbose:
            print(f"[ManualCurriculum] Hyperparams applied → "
                  f"LR={lr:.2e} | ent_coef={ent:.4f} | clip={clip:.3f}")

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
            "threshold_save_fired":  list(self._threshold_save_fired),  # ADD THIS
        }
        path = os.path.join(self.save_path, "curriculum_state.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2)
        if self.verbose:
            print(f"[Curriculum] State saved → Phase {self.current_phase + 1} "
                  f"at {self.num_timesteps:,} steps")

    @staticmethod
    def load_state(save_path: str) -> dict:
        """
        Load persisted state from disk.
        Returns safe defaults if no file exists (first-ever run).
 
        Usage in your resume script:
            state = ManualCurriculumCallback.load_state(directories["production"])
            callback = ManualCurriculumCallback(
                save_path=directories["production"],
                start_phase=state["current_phase"],
            )
        """
        path = os.path.join(save_path, "curriculum_state.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                raw = json.load(f)
            raw["phase_bests"] = {
                int(k): v for k, v in raw.get("phase_bests", {}).items()
            }
            raw["threshold_save_fired"] = set(raw.get("threshold_save_fired", []))
            
            print(f"[ManualCurriculum] Restored → Phase {raw['current_phase'] + 1} "
                  f"| {raw['num_timesteps']:,} steps")
            return raw
 
        print("[ManualCurriculum] No saved state found — starting fresh from Phase 1.")
        return {
            "current_phase": 0,
            "num_timesteps": 0,
            "phase_bests":   {},
        }

    # ------------------------------------------------------------------
    # Save helpers
    # ------------------------------------------------------------------
    def _save_best_reward(self):
        path     = os.path.join(self.save_path, f"{config.MODEL_NAME}_BEST_REWARD")
        vec_path = os.path.join(self.save_path, f"{config.MODEL_NAME}_vecnorm_BEST_REWARD.pkl")
        self.model.save(path)
        self.training_env.save(vec_path)
        self._save_phase_state()  # ADD THIS
        if self.verbose:
            print(f"[Best-Reward ✓] {self.num_timesteps:,} steps | "
                  f"Phase {self.current_phase + 1} | "
                  f"New best: {self._get_phase_best('reward'):.2f}")

    def _save_best_winrate(self):
        path     = os.path.join(self.save_path, f"{config.MODEL_NAME}_BEST_WINRATE")
        vec_path = os.path.join(self.save_path, f"{config.MODEL_NAME}_vecnorm_BEST_WINRATE.pkl")
        self.model.save(path)
        self.training_env.save(vec_path)
        self._save_phase_state()  # ADD THIS
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

    def _save_threshold_milestone(self, win_rate: float):
        """
        Fires ONCE per phase when win rate crosses WIN_RATE_THRESHOLD.
        Does NOT advance the phase — that remains your manual decision.
        Creates a uniquely named artifact so it's never overwritten.
        """
        tag = f"phase{self.current_phase + 1}_WR{int(win_rate * 100)}pct_{self.num_timesteps}steps"
        model_path = os.path.join(self.save_path, f"{config.MODEL_NAME}_{tag}")
        vec_path   = os.path.join(self.save_path, f"{config.MODEL_NAME}_vecnorm_{tag}.pkl")
        
        self.model.save(model_path)
        self.training_env.save(vec_path)
        self._save_phase_state()

        # Mark this phase as fired so we never duplicate this save
        self._threshold_save_fired.add(self.current_phase)

        if self.verbose:
            print(f"\n{'*'*60}")
            print(f"[THRESHOLD MILESTONE] Phase {self.current_phase + 1} cleared!")
            print(f"  Win Rate : {win_rate:.1%} >= {config.WIN_RATE_THRESHOLD:.1%}")
            print(f"  Steps    : {self.num_timesteps:,}")
            print(f"  Saved    : {model_path}.zip")
            print(f"  Manually call callback.set_phase({self.current_phase + 1}) to advance.")
            print(f"{'*'*60}\n")
    # ==================================================================
    # Status print  — call anytime to see current metrics
    # ==================================================================
 
    def print_status(self):
        """
        Print a snapshot of current training metrics on demand.
        Useful for monitoring from a REPL or interactive session.
        """
        print(f"\n{'─'*55}")
        print(f"  Phase        : {self.current_phase + 1} / {len(config.CURRICULUM_PHASES)}")
        print(f"  Timesteps    : {self.num_timesteps:,}")
        print(f"  Win Rate     : {self._win_rate():.1%}  "
              f"(buffer: {len(self.win_buffer)}/{config.WIN_RATE_WINDOW})")
        print(f"  Mean Reward  : {self._mean_reward():.2f}  "
              f"(buffer: {len(self.reward_buffer)}/300)")
        print(f"  Best Reward  : {self._get_phase_best('reward'):.2f}")
        print(f"  Best WinRate : {self._get_phase_best('win_rate'):.1%}")
        print(f"{'─'*55}\n")
 
    # ------------------------------------------------------------------
    # SB3 hook
    # ------------------------------------------------------------------
    
    def _on_step(self) -> bool:
 
        # ---- Collect episode outcomes from Monitor's info dict ----
        for info in self.locals.get("infos", []):
            if "win" in info:
                self.win_buffer.append(info["win"])
            if "episode" in info:
                self.reward_buffer.append(info["episode"]["r"])
 
        # ---- Check for new best models every eval_interval steps ----
        if self.num_timesteps - self.last_eval_step >= self.eval_interval:
            self.last_eval_step = self.num_timesteps
            mr = self._mean_reward()
            wr = self._win_rate()
 
            # Best reward: require at least 20 episodes to avoid saving on noise
            if mr > self._get_phase_best("reward") and len(self.reward_buffer) >= 20:
                self._set_phase_best("reward", mr)
                self._save_best_reward()
 
            # Best win rate: require buffer to be at least half full
            if wr > self._get_phase_best("win_rate") and len(self.win_buffer) >= config.WIN_RATE_WINDOW // 2:
                self._set_phase_best("win_rate", wr)
                self._save_best_winrate()

            # Threshold milestone: if we cross the win rate threshold for the first time in this phase, save a milestone checkpoint
            if (
                self.current_phase not in self._threshold_save_fired 
                and len(self.win_buffer) >= config.WIN_RATE_WINDOW
                and wr >= config.WIN_RATE_THRESHOLD
            ):
                self._save_threshold_milestone(wr)
 
        # ---- Periodic checkpoint every save_interval steps ----
        if self.num_timesteps - self.last_save_step >= self.save_interval:
            self._save_periodic_checkpoint()
 
        return True  # Always True — YOU control when training stops