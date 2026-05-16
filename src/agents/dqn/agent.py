import os
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import SubprocVecEnv

from core import config
from core.selective_norm import SelectiveVecNormalize
from core.env_tools import failsafe_env
from manual_curriculum_callback import ManualCurriculumCallback
from agents.base_agent import BaseAgent
from agents.dqn.config import PHASE_HYPERPARAMS, BUFFER_SIZE, BATCH_SIZE, EXPLORATION_INITIAL_EPS, EXPLORATION_FINAL_EPS

class DQNAgent(BaseAgent):
    def train(self, env_fn, save_dir, steps, load_zip=None, load_pkl=None, start_phase="0", lr=0.0, ent_coef=0.0, clip_range=0.0):
        print(f"[Training] Initializing DQN Curriculum Production Training in {save_dir}...")
        
        if load_zip and load_zip != "None":
            print(f"[Training] Loading model from: {load_zip}")
        if load_pkl and load_pkl != "None":
            print(f"[Training] Loading normalization from: {load_pkl}")
            
        # Handle different phase types
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
        
        phase_params = PHASE_HYPERPARAMS[active_phase_idx].copy()
        
        active_lr = lr if lr > 0.0 else phase_params["lr"]
        active_expl_frac = phase_params["exploration_fraction"]
        
        env = None
        model = None
        directories = config.get_directory()
        
        try:
            from gymnasium import ActionWrapper, spaces
            import numpy as np

            class DiscreteToMultiBinaryWrapper(ActionWrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.action_space = spaces.Discrete(2 ** env.action_space.shape[0])

                def action(self, action):
                    binary_str = format(action, f'0{self.env.action_space.shape[0]}b')
                    return np.array([int(b) for b in binary_str], dtype=np.int8)

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
                    device="cuda",
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
                    device="cuda"
                )

            callback = ManualCurriculumCallback(
                save_path=save_dir,
                start_phase=active_phase_idx,
                eval_interval=500,
                save_interval=config.SAVE_FREQ_STEPS
            )
            
            print("[Training] Press Ctrl + C to stop the training. ")

            model.learn(
                total_timesteps=steps, 
                callback=callback,
                tb_log_name=config.MODEL_NAME
            )
            
            model.save(os.path.join(save_dir, config.MODEL_NAME + "_FINAL"))
            if hasattr(env, "save"): env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_FINAL.pkl"))
            print("\nProduction Training Complete!")
            
        except KeyboardInterrupt:
            print("\n[MANUAL OVERRIDE] Training forcefully interrupted by user.")
            if model is not None: model.save(os.path.join(save_dir, config.MODEL_NAME + "_EMERGENCY"))
            if env is not None and hasattr(env, "save"): env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_EMERGENCY.pkl"))

        except Exception as e:
            print(f"\n[CRITICAL ERROR] Training crashed: {e}")
            if model is not None: model.save(os.path.join(save_dir, config.MODEL_NAME + "_CRASH_SAVE"))
            if env is not None and hasattr(env, "save"): env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_CRASH_SAVE.pkl"))
            if 'callback' in locals(): callback._save_phase_state()
            raise e

        finally:
            failsafe_env(
                env=env if 'env' in dir() else None,
                model=model if 'model' in dir() else None
            )

    def resume(self):
        pass

    def tune(self, env_fn, n_trials, study_name="dqn_sf2_tuning", load_zip=None, load_pkl=None, start_phase="0", timesteps=50000):
        import optuna
        from agents.dqn.optuna_study import objective
        
        directories = config.get_directory()
        db_path = os.path.abspath(os.path.join(directories["tuning"], "dqn_study.db"))
        storage_url = f"sqlite:///{db_path}"
        
        print(f"[Tuning] Starting Optuna Study: {study_name}")
        print(f"[Tuning] Storage: {storage_url}")

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            direction="maximize",
            load_if_exists=True
        )
        
        study.optimize(lambda trial: objective(trial, env_fn, load_zip, load_pkl, start_phase, timesteps), n_trials=n_trials)
        
        print("\n[Tuning] Complete!")
        print(f"Best Trial: {study.best_trial.number}")
        print(f"Best Value: {study.best_value}")
        print(f"Best Params: {study.best_params}")

    def test(self):
        pass
