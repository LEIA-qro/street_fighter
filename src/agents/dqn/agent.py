import os
import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv

from core import config
from core.selective_norm import SelectiveVecNormalize
from core.env_tools import failsafe_env
from agents.manual_curriculum_callback import ManualCurriculumCallback
from agents.base_agent import BaseAgent
from agents.dqn.config import PHASE_HYPERPARAMS, BUFFER_SIZE, BATCH_SIZE, EXPLORATION_INITIAL_EPS, EXPLORATION_FINAL_EPS

class DQNAgent(BaseAgent):
    def train(self, env_fn, save_dir, steps, load_zip=None, load_pkl=None, start_phase="0", lr=0.0, ent_coef=0.0, clip_range=0.0, device="cuda", auto_curriculum=False):
        print(f"[Training] Initializing DQN Curriculum Production Training in {save_dir}...")
        print(f"[Training] Compute Device: {device}")
        
        if load_zip and load_zip != "None":
            print(f"[Training] Loading model from: {load_zip}")
        if load_pkl and load_pkl != "None":
            print(f"[Training] Loading normalization from: {load_pkl}")
            
        # Handle different phase types / auto curriculum pre-load
        if auto_curriculum:
            from agents.auto_curriculum_callback import AutoCurriculumCallback
            curr_state = AutoCurriculumCallback.load_state(save_dir)
            start_level = curr_state.get("current_level", 1)
            introduced = curr_state.get("introduced_states", [])
            
            # Reconstruct starting lottery pool for EmuHawk boot safety
            pool = []
            for lvl in range(1, start_level):
                if lvl in config.DIFFICULTY_LEVELS:
                    pool.extend(config.DIFFICULTY_LEVELS[lvl])
            if start_level in config.DIFFICULTY_LEVELS:
                pool.extend(config.DIFFICULTY_LEVELS[start_level] * 3)
            for s in introduced:
                pool.extend([s] * 5)
                
            config.TRAINING_STATES = pool
            active_phase_idx = start_level
            print(f"[Training] Auto-Curriculum active. Initialized starting lottery pool for Level {start_level} ({len(introduced)} introduced).")
        else:
            if start_phase == "RYU_ONLY":
                config.TRAINING_STATES = config.RYU_ONLY_STATES
                active_phase_idx = 0
                print("[Training] Using RYU_ONLY states.")
            elif start_phase == "CUSTOM":
                config.TRAINING_STATES = config.CUSTOM_STATES if config.CUSTOM_STATES else config.AVAILABLE_STATES
                active_phase_idx = 0
                print(f"[Training] Using CUSTOM states ({len(config.TRAINING_STATES)} found).")
            else:
                try:
                    active_phase_idx = int(start_phase)
                    config.TRAINING_STATES = config.CURRICULUM_PHASES[active_phase_idx]
                except (ValueError, IndexError):
                    print(f"[Training][WARN] Invalid phase {start_phase}. Defaulting to Phase 0.")
                    active_phase_idx = 0
                    config.TRAINING_STATES = config.CURRICULUM_PHASES[0]

        n_envs = config.N_ENVS
        
        # Base phase params
        phase_params = PHASE_HYPERPARAMS[active_phase_idx % len(PHASE_HYPERPARAMS)].copy() if not auto_curriculum else {}
        if auto_curriculum:
            # Under auto_curriculum, resolve hyperparams fallback
            phase_idx = (active_phase_idx - 1) // 2
            phase_params = PHASE_HYPERPARAMS.get(phase_idx, PHASE_HYPERPARAMS[0]).copy()
        
        active_lr = lr if lr > 0.0 else phase_params.get("lr", 1e-4)
        active_expl_frac = phase_params.get("exploration_fraction", 0.1)
        
        env = None
        model = None
        directories = config.get_directory()
        
        try:
            from gymnasium import ActionWrapper, spaces
            import numpy as np

            class DiscreteToMultiBinaryWrapper(ActionWrapper):
                """Convert DQN's Discrete output to MultiBinary or MultiDiscrete.

                Supports both action space types:
                - MultiBinary(n): Discrete(2^n) with binary string decode
                - MultiDiscrete(nvec): Discrete(prod(nvec)) with divmod decode
                """
                def __init__(self, env):
                    super().__init__(env)
                    raw_space = env.action_space
                    if isinstance(raw_space, spaces.MultiBinary):
                        self._mode = "multibinary"
                        self._n_buttons = raw_space.n
                        self.action_space = spaces.Discrete(2 ** self._n_buttons)
                    elif isinstance(raw_space, spaces.MultiDiscrete):
                        self._mode = "multidiscrete"
                        self._nvec = raw_space.nvec.copy()
                        self.action_space = spaces.Discrete(int(np.prod(self._nvec)))
                    else:
                        raise TypeError(
                            f"DQN wrapper: unsupported action space "
                            f"{type(raw_space).__name__}. Expected "
                            f"MultiBinary or MultiDiscrete."
                        )

                def action(self, action):
                    if self._mode == "multibinary":
                        binary_str = format(action, f'0{self._n_buttons}b')
                        return np.array([int(b) for b in binary_str], dtype=np.int8)
                    else:
                        # Decode flat index → MultiDiscrete via divmod
                        decoded = []
                        remaining = int(action)
                        for n in reversed(self._nvec):
                            decoded.append(remaining % n)
                            remaining //= n
                        return np.array(list(reversed(decoded)), dtype=np.int64)

            def make_dqn_env(rank):
                original_init = env_fn(rank)
                def _init():
                    return DiscreteToMultiBinaryWrapper(original_init())
                return _init

            env = SubprocVecEnv([make_dqn_env(i) for i in range(n_envs)])
            
            # Broadcast the target phase states to all workers (fixes multiprocessing inheritance bug)
            try:
                env.env_method("set_training_states", config.TRAINING_STATES)
                print(f"[Training] States broadcast to all {n_envs} workers.")
            except Exception as e:
                print(f"[Training][WARN] Could not broadcast states to workers: {e}")
                    
            if load_pkl and load_pkl != "None":
                env = SelectiveVecNormalize.load(os.path.join(config.PROJECT_ROOT, load_pkl), env)
                env.training = True
            else:
                env = SelectiveVecNormalize(env,
                                             n_continuous_dims=config.OBS_DIM, 
                                             n_frames=config.NUM_FRAMES)

            if load_zip and load_zip != "None":
                model = DQN.load(
                    os.path.join(config.PROJECT_ROOT, load_zip), 
                    env=env, 
                    device=device,
                    custom_objects={"learning_rate": active_lr, "exploration_fraction": active_expl_frac}
                )
            else:
                model = DQN(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=active_lr,
                    buffer_size=BUFFER_SIZE,
                    batch_size=BATCH_SIZE,
                    exploration_fraction=active_expl_frac,
                    exploration_initial_eps=EXPLORATION_INITIAL_EPS,
                    exploration_final_eps=EXPLORATION_FINAL_EPS,
                    gamma=0.99,
                    policy_kwargs=dict(net_arch=[512, 512, 256]),
                    verbose=1,
                    tensorboard_log=directories["logs"],
                    device=device
                )

            # Extract env_version and algo from save_dir safely
            normalized_path = os.path.normpath(save_dir)
            path_parts = normalized_path.split(os.sep)
            algo_part = "dqn"
            env_part = "v2"
            if len(path_parts) >= 2:
                algo_part = path_parts[-1]
                env_part = path_parts[-2]

            state_name = None
            if not auto_curriculum:
                if start_phase == "RYU_ONLY":
                    state_name = "ryu_only"
                elif start_phase == "CUSTOM":
                    state_name = "custom"

            if auto_curriculum:
                from agents.auto_curriculum_callback import AutoCurriculumCallback
                callback = AutoCurriculumCallback(
                    save_path=save_dir,
                    phase_hyperparams=PHASE_HYPERPARAMS,
                    start_level=active_phase_idx,
                    eval_interval=500,
                    save_interval=config.SAVE_FREQ_STEPS,
                    algo=algo_part,
                    env_version=env_part,
                    model_name=config.MODEL_NAME,
                    state_name=state_name
                )
            else:
                callback = ManualCurriculumCallback(
                    save_path=save_dir,
                    phase_hyperparams=PHASE_HYPERPARAMS,
                    start_phase=active_phase_idx,
                    eval_interval=500,
                    save_interval=config.SAVE_FREQ_STEPS,
                    algo=algo_part,
                    env_version=env_part,
                    model_name=config.MODEL_NAME,
                    state_name=state_name
                )
            
            print("[Training] Press Ctrl + C to stop the training. ")

            reset_timesteps = True
            if load_zip and load_zip != "None":
                reset_timesteps = False

            model.learn(
                total_timesteps=steps, 
                callback=callback,
                tb_log_name=f"{algo_part}_{env_part}_{config.MODEL_NAME}",
                reset_num_timesteps=reset_timesteps
            )
            
            # Dynamic Final Save
            win_rate_val = callback._win_rate() if hasattr(callback, "_win_rate") else 0.0
            winrate_pct = int(round(win_rate_val * 100))
            if hasattr(callback, "current_level"):
                state_tag = callback.state_name if callback.state_name is not None else f"lvl{callback.current_level}"
                if len(getattr(callback, "introduced_states", [])) > 0 and callback.state_name is None:
                    state_tag = f"lvl{callback.current_level}_plus{len(callback.introduced_states)}"
            else:
                state_tag = callback.state_name if callback.state_name is not None else f"phase{getattr(callback, 'current_phase', 0)}"
            
            final_base = f"{algo_part}_{env_part}_{config.MODEL_NAME}_{state_tag}_final_WR{winrate_pct}pct_{callback.num_timesteps}steps"
            
            model.save(os.path.join(save_dir, final_base))
            if hasattr(env, "save"): 
                env.save(os.path.join(save_dir, f"{final_base}_vecnorm.pkl"))
            print(f"\nProduction Training Complete! Saved final model as: {final_base}")
            
        except KeyboardInterrupt:
            print("\n[MANUAL OVERRIDE] Training forcefully interrupted by user. Saving Emergency Checkpoint...")
            if model is not None:
                emergency_base = os.path.join(save_dir, config.MODEL_NAME + "_EMERGENCY")
                model.save(emergency_base)
                print(f"[MANUAL OVERRIDE] Saved model weights -> {emergency_base}.zip")
            if env is not None and hasattr(env, "save"):
                vec_emergency = os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_EMERGENCY.pkl")
                env.save(vec_emergency)
                print(f"[MANUAL OVERRIDE] Saved normalizer stats -> {vec_emergency}")
            if 'callback' in locals():
                if hasattr(callback, "_save_curriculum_state"):
                    callback._save_curriculum_state(force=True)
                elif hasattr(callback, "_save_phase_state"):
                    callback._save_phase_state()
            raise

        except Exception as e:
            print(f"\n[CRITICAL ERROR] Training crashed: {e}")
            if model is not None: 
                model.save(os.path.join(save_dir, config.MODEL_NAME + "_CRASH_SAVE"))
                time.sleep(2) # Buffer for OS to finish disk write
            if env is not None and hasattr(env, "save"): env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_CRASH_SAVE.pkl"))
            if 'callback' in locals():
                if hasattr(callback, "_save_curriculum_state"):
                    callback._save_curriculum_state(force=True)
                elif hasattr(callback, "_save_phase_state"):
                    callback._save_phase_state()
            raise e

        finally:
            failsafe_env(env=env, model=model)

    def resume(self):
        pass

    def tune(self, env_fn, n_trials, study_name="dqn_sf2_tuning", load_zip=None, load_pkl=None, start_phase="0", timesteps=50000, device="cuda"):
        import optuna
        from agents.dqn.optuna_study import objective
        
        # Determine if we should treat start_phase as an index or a special keyword
        if start_phase not in ["RYU_ONLY", "CUSTOM"]:
            try:
                active_phase = int(start_phase)
            except (ValueError, TypeError):
                print(f"[Tuning][WARN] Non-integer start_phase '{start_phase}' detected. Falling back to Phase 0.")
                active_phase = 0
        else:
            active_phase = start_phase

        directories = config.get_directory()
        tuning_dir = os.path.join(directories["tuning"], "dqn")
        os.makedirs(tuning_dir, exist_ok=True)
        db_path = os.path.abspath(os.path.join(tuning_dir, "study.db"))
        storage_url = f"sqlite:///{db_path}"
        
        print(f"[Tuning] Starting Optuna Study: {study_name}")
        print(f"[Tuning] Storage: {storage_url}")
        print(f"[Tuning] Compute Device: {device}")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            load_if_exists=True,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=10000,
                max_resource=timesteps,
                reduction_factor=3
            )
        )
        
        study.optimize(
            lambda trial: objective(trial, env_fn, load_zip, load_pkl, active_phase, timesteps, device=device), 
            n_trials=n_trials,
            catch=(Exception,)
        )
        
        print("\n[Tuning] Complete!")
        print(f"Best Trial: {study.best_trial.number}")
        print(f"Best Value: {study.best_value}")
        print(f"Best Params: {study.best_params}")

        # Save Best Model and VecNorm
        import shutil
        import json
        
        best_trial_num = study.best_trial.number
        best_model_path = os.path.join(tuning_dir, f"trial_{best_trial_num}_model.zip")
        best_vec_path = os.path.join(tuning_dir, f"trial_{best_trial_num}_vecnormalize.pkl")
        
        final_model_path = os.path.join(tuning_dir, "best_model.zip")
        final_vec_path = os.path.join(tuning_dir, "best_vecnormalize.pkl")
        final_params_path = os.path.join(tuning_dir, "best_params.json")

        if os.path.exists(best_model_path):
            shutil.copy(best_model_path, final_model_path)
            print(f"[Tuning] Saved best model to: {final_model_path}")
        
        if os.path.exists(best_vec_path):
            shutil.copy(best_vec_path, final_vec_path)
            print(f"[Tuning] Saved best vecnormalize to: {final_vec_path}")

        with open(final_params_path, "w") as f:
            json.dump(study.best_params, f, indent=4)
            print(f"[Tuning] Saved best parameters to: {final_params_path}")

    def test(self):
        pass
