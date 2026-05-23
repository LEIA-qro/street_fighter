# auto_curriculum_callback.py
#
# Dedicated automated curriculum callback that implements:
#   - Rehearsal Weighted Lottery (dynamic state weighting via replication)
#   - Active-Only Gated Advancement (monitors Active + New states winrate >= 75%)
#   - 3-cycle consecutive evaluation stability checks to prevent promotion on noise
#   - Micro-Steps (introducing 2 states from the next level at a time)
#   - Continuous 8-level hyperparameter injection with grace fallbacks
#   - Full JSON state persistence for crash recovery / resume safety
#

import os
import json
import random
import numpy as np
from collections import deque
from typing import Dict, List, Any, Set, Tuple
from stable_baselines3.common.callbacks import BaseCallback

import core.config as config


class AutoCurriculumCallback(BaseCallback):
    MAX_CHECKPOINTS_TO_KEEP = 3

    def __init__(self, save_path: str, phase_hyperparams: dict, verbose: int = 1,
                 start_level: int = 1, eval_interval: int = 500, save_interval: int = None,
                 algo: str = "ppo", env_version: str = "v2", model_name: str = "league",
                 state_name: str = None, win_rate_threshold: float = 0.75,
                 stability_threshold: int = 3, min_episodes_for_eval: int = 100):
        
        super().__init__(verbose)
        self.save_path = save_path
        self.phase_hyperparams = phase_hyperparams
        self.current_level = start_level  # Level range: 1 - 8
        self.eval_interval = eval_interval
        self.save_interval = save_interval if save_interval is not None else config.SAVE_FREQ_STEPS

        # Metadata for dynamic naming
        self.algo = algo.lower()
        self.env_version = env_version.lower()
        self.model_name = model_name
        self.state_name = state_name

        # Curriculum Gating Thresholds
        self.win_rate_threshold = win_rate_threshold
        self.stability_threshold = stability_threshold
        self.min_episodes_for_eval = min_episodes_for_eval

        # Stability counter tracking consecutive successful evaluations
        self.stability_counter = 0

        # Micro-step tracker: list of states from Level `current_level + 1` introduced so far
        self.introduced_states: List[str] = []

        # Decentralized per-state win buffers (maxlen=100 each to gather local rolling win rates)
        self.state_win_buffers: Dict[str, deque] = {}
        for lvl, states in config.DIFFICULTY_LEVELS.items():
            for state in states:
                self.state_win_buffers[state] = deque(maxlen=100)

        # Rehearsal Lottery Pool representation
        self.active_pool: List[str] = []

        # Keep track of periodic checkpoints and milestone saving
        self._checkpoint_registry: List[Tuple[str, str]] = []
        self.last_eval_step = 0
        self.last_save_step = 0
        self._threshold_save_fired: Set[int] = set()

        # Active Best Model Pruning references to prevent disk bloat
        self._last_best_reward_path: str = None
        self._last_best_winrate_path: str = None

        # Gating JSON I/O and responsive telemetry saves
        self.last_json_save_step = 0

        # Self-restore previous training state if resuming/restarting
        self._load_and_restore_state()

    def _get_base_filename(self, metric_tag: str) -> str:
        """Construct the dynamic base filename: {algo}_{env}_{customName}_{state}_{metricTag}_{steps}"""
        state_tag = self.state_name if self.state_name is not None else f"lvl{self.current_level}"
        if len(self.introduced_states) > 0 and self.state_name is None:
            state_tag = f"lvl{self.current_level}_plus{len(self.introduced_states)}"
        return f"{self.algo}_{self.env_version}_{self.model_name}_{state_tag}_{metric_tag}_{self.num_timesteps}steps"

    def generate_weighted_lottery_pool(self) -> List[str]:
        """
        Generate the broadcasted training state list with balanced category weighting:
          - Past Rehearsal states (levels < current_level): Capped at 12 states to prevent dilution. Total weight = 12.
          - Mastered active states (current_level with WR >= 75%): Total weight = 24.
          - Standard Active states (current_level with < 10 eps): Total weight = 36.
          - Contested / Weakness states (current_level with WR < 75%): Total weight = 48.
          - Newly introduced states (introduced from current_level + 1): Total weight = 60.
        
        This normalized category probability distribution guarantees that:
          1. Weakness states are heavily prioritized (57.1% selection probability at Level 2).
          2. Selection probability of weaknesses remains stable and does not dilute as levels advance.
          3. Catastrophic forgetting is avoided by rotating a sub-sampled representative past cohort.
        """
        state_categories = {
            "past": [],
            "mastered": [],
            "active": [],
            "weakness": [],
            "new": []
        }

        # 1. Classify Past Rehearsal states (levels < current_level)
        past_states = []
        for lvl in range(1, self.current_level):
            if lvl in config.DIFFICULTY_LEVELS:
                for state in config.DIFFICULTY_LEVELS[lvl]:
                    past_states.append(state)
        
        # Sub-sample past states to a maximum of 12 to prevent lottery dilution
        if len(past_states) > 12:
            state_categories["past"] = random.sample(past_states, 12)
        else:
            state_categories["past"] = past_states

        # 2. Classify Active states (current_level)
        if self.current_level in config.DIFFICULTY_LEVELS:
            for state in config.DIFFICULTY_LEVELS[self.current_level]:
                buf = self.state_win_buffers.get(state, [])
                if len(buf) >= 10:
                    wr = sum(buf) / len(buf)
                    if wr < self.win_rate_threshold:
                        state_categories["weakness"].append(state)
                    else:
                        state_categories["mastered"].append(state)
                else:
                    state_categories["active"].append(state)

        # 3. Classify Newly Introduced states
        for state in self.introduced_states:
            state_categories["new"].append(state)

        # 4. Generate balanced lottery pool matching targeted category weight ratios
        # Total Category Weights (proportional to past=1, mastered=2, active=3, weakness=4, new=5)
        cat_weights = {
            "past": 12,
            "mastered": 24,
            "active": 36,
            "weakness": 48,
            "new": 60
        }

        pool: List[str] = []
        for cat, states in state_categories.items():
            if not states:
                continue
            n_states = len(states)
            target_sum = cat_weights[cat]
            if n_states >= target_sum:
                # If there are more states than target sum, each gets at least multiplicity 1
                for s in states:
                    pool.append(s)
            else:
                # Deterministically distribute target sum among the states
                base_mult = target_sum // n_states
                remainder = target_sum % n_states
                for idx, s in enumerate(states):
                    m = base_mult + (1 if idx < remainder else 0)
                    pool.extend([s] * max(1, m))

        self.active_pool = pool
        return pool

    def update_environment_states(self):
        """Broadcast the updated weighted lottery pool to all environment workers."""
        pool = self.generate_weighted_lottery_pool()
        if self.verbose:
            print(f"[AutoCurriculum] Broadcasting Lottery Pool: "
                  f"Level {self.current_level} (Weight 3) | "
                  f"{len(self.introduced_states)} Introduced from Lvl {self.current_level+1} (Weight 5) | "
                  f"Total replicated pool size: {len(pool)}")
        try:
            self.training_env.env_method("set_training_states", pool)
        except AttributeError:
            config.TRAINING_STATES = pool

    def set_level(self, new_level: int):
        """Force advance or adjust the active difficulty level directly."""
        if new_level < 1 or new_level > 8:
            print(f"[AutoCurriculum] Invalid level {new_level}. Range is 1-8.")
            return

        self.current_level = new_level
        self.introduced_states.clear()
        self.stability_counter = 0

        # Broadcast and update hyperparams
        self.update_environment_states()
        self._apply_level_hyperparams(self.current_level)

        # Decay normalizer
        try:
            norm = self.training_env
            if hasattr(norm, "count"):
                norm.count = min(norm.count, 5_000.0)
        except Exception:
            pass

        # Clear ALL gameplay win buffers to ensure a clean slate of assessments for the new level
        for state in list(self.state_win_buffers.keys()):
            if not state.startswith("best_"):
                self.state_win_buffers[state].clear()

        # Reset best-model pruning references to prevent deleting the final milestone of the previous level
        self._last_best_reward_path = None
        self._last_best_winrate_path = None

        # Entry save
        tag = f"lvl{self.current_level}_entry"
        base_name = self._get_base_filename(tag)
        self.model.save(os.path.join(self.save_path, base_name))
        self.training_env.save(os.path.join(self.save_path, f"{base_name}_vecnorm.pkl"))

        self._save_curriculum_state(force=True)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"[AutoCurriculum] *** MOVED TO ACTIVE LEVEL {self.current_level} ***")
            print(f"  Steps  : {self.num_timesteps:,}")
            print(f"{'='*60}\n")

    def _apply_level_hyperparams(self, level: int):
        """Push LR / ent_coef / clip_range / tau from configuration into the live model."""
        # Check if algorithm has LEVEL_HYPERPARAMS defined, else map dynamically using phase = (level - 1) // 2
        # This guarantees absolute backward compatibility with existing 4-phase DQN/SAC setups.
        phase_idx = (level - 1) // 2
        
        # Pull hyperparams dict safely
        params = {}
        try:
            # Check if there is an explicit level-based parameter set or level config
            params = self.phase_hyperparams[level - 1]  # Check if they reorganized it fully
        except KeyError:
            # Graceful Fallback: map to phase bounds
            params = self.phase_hyperparams.get(phase_idx, {})

        if not params:
            return

        # 1. Learning Rate
        if "lr" in params:
            lr = params["lr"]
            self.model.learning_rate = lambda _: lr
            for pg in self.model.policy.optimizer.param_groups:
                pg["lr"] = lr

        # 2. Entropy Coefficient (PPO/SAC)
        if "ent_coef" in params and hasattr(self.model, "ent_coef"):
            self.model.ent_coef = params["ent_coef"]

        # 3. Clip Range (PPO)
        if "clip" in params and hasattr(self.model, "clip_range"):
            clip = params["clip"]
            self.model.clip_range = lambda _: clip

        # 4. Tau (SAC)
        if "tau" in params and hasattr(self.model, "tau"):
            self.model.tau = params["tau"]

        if self.verbose:
            log_parts = [f"{k}={v:.2e}" if isinstance(v, (float, np.float32, np.float64)) else f"{k}={v}" 
                        for k, v in params.items()]
            print(f"[AutoCurriculum] Applied Hyperparams (Lvl {level} -> Phase Fallback {phase_idx}) -> {' | '.join(log_parts)}")

    def _save_curriculum_state(self, force: bool = False):
        """Serialize current curriculum state variables to disk safely, gated to avoid massive overwrites."""
        if not force and self.num_timesteps > 0 and (self.num_timesteps - self.last_json_save_step) < 5000:
            return

        state = {
            "current_level": self.current_level,
            "introduced_states": self.introduced_states,
            "stability_counter": self.stability_counter,
            "num_timesteps": self.num_timesteps,
            "last_save_step": self.last_save_step,
            "last_eval_step": self.last_eval_step,
            "state_name": self.state_name,
            "threshold_save_fired": list(self._threshold_save_fired),
            # Serialize deques as clean lists of 1s and 0s
            "state_win_buffers": {
                state: list(buf) for state, buf in self.state_win_buffers.items() if len(buf) > 0
            }
        }
        path = os.path.join(self.save_path, "auto_curriculum_state.json")
        temp_path = path + ".tmp"
        try:
            with open(temp_path, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(temp_path, path)
            self.last_json_save_step = self.num_timesteps
            if self.verbose:
                print(f"[AutoCurriculum] Progress saved -> Lvl {self.current_level} "
                      f"(+ {len(self.introduced_states)} introduced) at {self.num_timesteps:,} steps")
        except Exception as e:
            print(f"[AutoCurriculum][WARN] Failed to write curriculum state safely: {e}")

    def _load_and_restore_state(self):
        """Load curriculum state from disk if it exists, restoring all metrics and buffers."""
        path = os.path.join(self.save_path, "auto_curriculum_state.json")
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    state = json.load(f)
                
                self.current_level = state.get("current_level", self.current_level)
                self.introduced_states = state.get("introduced_states", [])
                self.stability_counter = state.get("stability_counter", 0)
                self.last_save_step = state.get("last_save_step", 0)
                self.last_eval_step = state.get("last_eval_step", 0)
                self.last_json_save_step = state.get("num_timesteps", 0)
                self._threshold_save_fired = set(state.get("threshold_save_fired", []))
                
                # Restore deques
                saved_buffers = state.get("state_win_buffers", {})
                for state_key, buf_list in saved_buffers.items():
                    if state_key in self.state_win_buffers:
                        self.state_win_buffers[state_key] = deque(buf_list, maxlen=100)
                
                if self.verbose:
                    print(f"[AutoCurriculum] Callback self-restored state from disk -> Level {self.current_level} "
                          f"| Introduced: {len(self.introduced_states)} | Stability: {self.stability_counter} | Steps: {self.last_json_save_step:,}")
            except Exception as e:
                print(f"[AutoCurriculum][WARN] Failed to self-restore state from disk: {e}")

    @classmethod
    def load_state(cls, save_path: str) -> dict:
        """Load curriculum state from disk, fallback gracefully to Level 1 defaults if missing."""
        path = os.path.join(save_path, "auto_curriculum_state.json")
        if os.path.exists(path):
            with open(path, "r") as f:
                raw = json.load(f)
            raw["threshold_save_fired"] = set(raw.get("threshold_save_fired", []))
            print(f"[AutoCurriculum] Restored State -> Level {raw['current_level']} "
                  f"| Introduced: {len(raw.get('introduced_states', []))} | Steps: {raw['num_timesteps']:,}")
            return raw

        print("[AutoCurriculum] No saved auto-curriculum state found — starting fresh from Level 1.")
        return {
            "current_level": 1,
            "introduced_states": [],
            "stability_counter": 0,
            "num_timesteps": 0,
            "last_save_step": 0,
            "last_eval_step": 0,
            "state_name": None,
            "threshold_save_fired": set(),
            "state_win_buffers": {}
        }

    def _on_training_start(self) -> None:
        """Restore serialization vectors on training startup."""
        self._apply_level_hyperparams(self.current_level)
        self.update_environment_states()
        self._save_curriculum_state(force=True)

    def _safe_remove_model(self, base_path: str):
        if base_path:
            for ext in [".zip", "_vecnorm.pkl"]:
                full_path = base_path + ext
                try:
                    if os.path.exists(full_path):
                        os.remove(full_path)
                        if self.verbose:
                            print(f"[Prune] Removed old inferior best model: {os.path.basename(full_path)}")
                except Exception as e:
                    print(f"[Prune][WARN] Could not remove old model {full_path}: {e}")

    def _save_best_reward(self, mr: float):
        # Clean up previous inferior best reward model
        if hasattr(self, "_last_best_reward_path") and self._last_best_reward_path:
            self._safe_remove_model(self._last_best_reward_path)

        reward_val = int(round(mr))
        base_name = self._get_base_filename(f"Rew{reward_val}")
        full_base_path = os.path.join(self.save_path, base_name)
        
        self.model.save(full_base_path)
        self.training_env.save(os.path.join(self.save_path, f"{base_name}_vecnorm.pkl"))
        
        self._last_best_reward_path = full_base_path
        self._save_curriculum_state(force=True)
        if self.verbose:
            print(f"[Best-Reward *] {self.num_timesteps:,} steps | "
                  f"Lvl {self.current_level} | New best mean reward: {mr:.2f}")

    def _save_best_winrate(self, wr: float):
        # Clean up previous inferior best winrate model
        if hasattr(self, "_last_best_winrate_path") and self._last_best_winrate_path:
            self._safe_remove_model(self._last_best_winrate_path)

        winrate_pct = int(round(wr * 100))
        base_name = self._get_base_filename(f"WR{winrate_pct}pct")
        full_base_path = os.path.join(self.save_path, base_name)
        
        self.model.save(full_base_path)
        self.training_env.save(os.path.join(self.save_path, f"{base_name}_vecnorm.pkl"))
        
        self._last_best_winrate_path = full_base_path
        self._save_curriculum_state(force=True)
        if self.verbose:
            print(f"[Best-WinRate *] {self.num_timesteps:,} steps | "
                  f"Lvl {self.current_level} | New best win rate: {wr:.1%}")

    def _save_periodic_checkpoint(self, wr: float):
        winrate_pct = int(round(wr * 100))
        base_name = self._get_base_filename(f"WR{winrate_pct}pct_ckpt")
        path = os.path.join(self.save_path, base_name)
        vec_path = os.path.join(self.save_path, f"{base_name}_vecnorm.pkl")
        self.model.save(path)
        self.training_env.save(vec_path)
        self._checkpoint_registry.append((path + ".zip", vec_path))

        while len(self._checkpoint_registry) > self.MAX_CHECKPOINTS_TO_KEEP:
            old_model, old_vec = self._checkpoint_registry.pop(0)
            for filepath in (old_model, old_vec):
                try:
                    os.remove(filepath)
                    if self.verbose:
                        print(f"[Prune] Removed old checkpoint: {os.path.basename(filepath)}")
                except FileNotFoundError:
                    pass

        self.last_save_step = self.num_timesteps
        self._save_curriculum_state(force=True)

        if self.verbose:
            print(f"\n[Checkpoint] {self.num_timesteps:,} steps | Level {self.current_level} | "
                  f"Win Rate: {wr:.1%} | Saved Checkpoint: {base_name}")

    def _save_threshold_milestone(self, wr: float, label: str):
        """Creates a unique milestone save that will not be pruned."""
        winrate_pct = int(round(wr * 100))
        base_name = self._get_base_filename(f"WR{winrate_pct}pct_{label}_milestone")
        self.model.save(os.path.join(self.save_path, base_name))
        self.training_env.save(os.path.join(self.save_path, f"{base_name}_vecnorm.pkl"))
        self._save_curriculum_state(force=True)
        if self.verbose:
            print(f"\n{'*'*60}\n[THRESHOLD MILESTONE] {label} milestone reached! Winrate: {wr:.1%}\n{'*'*60}\n")

    def print_status(self, active_winrate: float, active_episodes: int, overall_winrate: float):
        """Print a detailed dashboard status of the progressive curriculum on demand."""
        print(f"\n{'-'*65}")
        print(f"  Auto-Curriculum Dashboard | Level: {self.current_level} / 8")
        print(f"  Timesteps             : {self.num_timesteps:,}")
        print(f"  Active States Count   : {len(config.DIFFICULTY_LEVELS.get(self.current_level, []))}")
        print(f"  Introduced L_next     : {len(self.introduced_states)} / 12 states")
        print(f"  Stability Counter     : {self.stability_counter} / {self.stability_threshold} consecutive evals")
        print(f"  Target Win Rate (Act) : {active_winrate:.1%} (Buffer size: {active_episodes} / min 100)")
        print(f"  Global Rolling WinRate: {overall_winrate:.1%}")
        print(f"{'-'*65}\n")

    def _on_step(self) -> bool:
        # Check for graceful file-based stop signal
        stop_file = os.path.join(config.PROJECT_ROOT, ".stop_training")
        if os.path.exists(stop_file):
            if self.verbose:
                print("\n[AutoCurriculum] Graceful file-based stop signal detected!")
            try:
                os.remove(stop_file)
            except Exception:
                pass
            raise KeyboardInterrupt

        # ---- 1. Telemetry Capture from Vectorized Environment workers ----
        for info in self.locals.get("infos", []):
            if "win" in info and "state_file" in info:
                state_file = info["state_file"]
                if state_file in self.state_win_buffers:
                    self.state_win_buffers[state_file].append(info["win"])

        # ---- 2. Periodic Evaluation Hook ----
        if self.num_timesteps - self.last_eval_step >= self.eval_interval:
            self.last_eval_step = self.num_timesteps

            # Compile "Active" and "New" States target set
            active_states = config.DIFFICULTY_LEVELS.get(self.current_level, []).copy()
            target_eval_states = active_states + self.introduced_states

            # Calculate isolated win rate on these target states
            total_wins = 0
            total_episodes = 0
            for state in target_eval_states:
                buf = self.state_win_buffers[state]
                total_wins += sum(buf)
                total_episodes += len(buf)


            # Ensure every active and introduced state has a minimum number of games played (e.g. 15 episodes)
            # to prevent past mastered states from masking un-trained or failing new states.
            sufficient_samples = True
            min_samples_per_state = 15  # Statistical confidence barrier
            missing_states_info = []
            
            for state in target_eval_states:
                n_eps = len(self.state_win_buffers[state])
                if n_eps < min_samples_per_state:
                    sufficient_samples = False
                    state_clean = state[4:] if state.startswith("RYU_") else state
                    missing_states_info.append(f"{state_clean.replace('_R1', '').replace('.State', '')} ({n_eps}/{min_samples_per_state})")

            # Global overall winrate for logging (across all buffers)
            global_wins = 0
            global_episodes = 0
            for state, buf in self.state_win_buffers.items():
                global_wins += sum(buf)
                global_episodes += len(buf)
            overall_wr = (global_wins / global_episodes) if global_episodes > 0 else 0.0

            active_wr = 0.0
            if total_episodes >= self.min_episodes_for_eval and sufficient_samples:
                active_wr = total_wins / total_episodes

                if active_wr >= self.win_rate_threshold:
                    self.stability_counter += 1
                else:
                    self.stability_counter = 0  # Broken streak (reset on deficit)
            else:
                self.stability_counter = 0  # Gated by sample size
                if self.verbose and len(missing_states_info) > 0:
                    print(f"[AutoCurriculum][Gating] Evaluation deferred. Waiting for state buffers: {', '.join(missing_states_info[:4])}...")

            if self.verbose:
                self.print_status(active_wr, total_episodes, overall_wr)

            # Periodically write current metrics to JSON to keep the live dashboard updated
            self._save_curriculum_state(force=False)

            # ---- 3. Promotion Logic (Threshold and Stability Cleared) ----
            if self.stability_counter >= self.stability_threshold:
                # Promotion Triggered!
                self.stability_counter = 0
                
                # Check if we have unintroduced states left in Level current_level + 1
                next_level = self.current_level + 1
                if next_level in config.DIFFICULTY_LEVELS:
                    all_next_states = sorted(config.DIFFICULTY_LEVELS[next_level])
                    unintroduced = [s for s in all_next_states if s not in self.introduced_states]

                    if len(unintroduced) > 0:
                        # Micro-Step: Introduce next 2 states from level current_level + 1
                        states_to_add = unintroduced[:2]
                        self.introduced_states.extend(states_to_add)
                        
                        # Initialize buffers for these new states explicitly
                        for state in states_to_add:
                            self.state_win_buffers[state].clear()

                        self._save_threshold_milestone(active_wr, f"intro{len(self.introduced_states)}")
                        self.update_environment_states()
                        self._save_curriculum_state(force=True)
                        
                        if self.verbose:
                            print(f"[AutoCurriculum] MICRO-STEP: Introduced {states_to_add} to pool.")
                    else:
                        # All 12 states of Lvl current_level + 1 are already introduced and fully mastered!
                        # We promote to Level current_level + 1
                        self.set_level(next_level)
                else:
                    # Already at Level 8 (hardest difficulty) and mastered it!
                    if self.current_level == 8 and 8 not in self._threshold_save_fired:
                        self._save_threshold_milestone(active_wr, "lvl8_mastered")
                        self._threshold_save_fired.add(8)
                        if self.verbose:
                            print(f"\n{'='*70}\n[AutoCurriculum] !!! CONGRATULATIONS: FULL CURRICULUM MASTERED !!!\n{'='*70}\n")

            # ---- 4. Save best models (Using global overall metrics) ----
            # Reward tracking fallbacks
            try:
                # Extract mean episode reward from Monitor logger if available
                ep_rewards = self.locals.get("infos", [{}])[0].get("episode", {}).get("r", -np.inf)
            except Exception:
                ep_rewards = -np.inf

            # Perform periodic saves based on global metrics
            # Note: We rely on global best saves to preserve the absolute strongest model overall
            # Best winrate checkpoint
            best_wr_key = f"best_wr_lvl{self.current_level}"
            if overall_wr > self.state_win_buffers.get(best_wr_key, deque([0.0]))[0] and global_episodes >= 50:
                # Store the float value inside our lookup dictionary safely
                # Reset best tracker
                self.state_win_buffers[best_wr_key] = deque([overall_wr], maxlen=1)
                self._save_best_winrate(overall_wr)

        # ---- 5. Periodic Save Checkpoint ----
        if self.num_timesteps - self.last_save_step >= self.save_interval:
            # Pull global winrate
            g_wins = 0
            g_episodes = 0
            for state, buf in self.state_win_buffers.items():
                if isinstance(buf, deque):
                    g_wins += sum(buf)
                    g_episodes += len(buf)
            checkpoint_wr = (g_wins / g_episodes) if g_episodes > 0 else 0.0
            self._save_periodic_checkpoint(checkpoint_wr)

        return True
