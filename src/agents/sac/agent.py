import os
import time
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv

from core import config
from core.selective_norm import SelectiveVecNormalize
from core.env_tools import failsafe_env
from manual_curriculum_callback import ManualCurriculumCallback
from agents.base_agent import BaseAgent
from agents.sac.config import PHASE_HYPERPARAMS, BUFFER_SIZE, BATCH_SIZE

class SACAgent(BaseAgent):
    def train(self, env_fn, save_dir, steps, load_zip=None, load_pkl=None, start_phase="0", lr=0.0, ent_coef=0.0, clip_range=0.0, device="cuda"):
        print(f"[Training] Initializing SAC Curriculum Production Training in {save_dir}...")
        print(f"[Training] Compute Device: {device}")
        
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
        active_ent = ent_coef if ent_coef > 0.0 else "auto"
        active_tau = phase_params["tau"]
        
        env = None
        model = None
        directories = config.get_directory()
        
        try:
            from gymnasium import ActionWrapper, spaces
            import numpy as np

            class ContinuousToMultiBinaryWrapper(ActionWrapper):
                def __init__(self, env):
                    super().__init__(env)
                    self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(env.action_space.shape[0],), dtype=np.float32)

                def action(self, action):
                    return (action > 0.0).astype(np.int8)

            def make_sac_env(rank):
                original_init = env_fn(rank)
                def _init():
                    return ContinuousToMultiBinaryWrapper(original_init())
                return _init

            env = SubprocVecEnv([make_sac_env(i) for i in range(n_envs)])
            
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
                model = SAC.load(
                    os.path.join(config.PROJECT_ROOT, load_zip), 
                    env=env, 
                    device=device,
                    custom_objects={"learning_rate": active_lr, "tau": active_tau, "ent_coef": active_ent}
                )
            else:
                model = SAC(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=active_lr,
                    buffer_size=BUFFER_SIZE,
                    batch_size=BATCH_SIZE,
                    tau=active_tau,
                    ent_coef=active_ent,
                    gamma=0.99,
                    policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], qf=[512, 512, 256])),
                    verbose=1,
                    tensorboard_log=directories["logs"],
                    device=device
                )

            callback = ManualCurriculumCallback(
                save_path=save_dir,
                phase_hyperparams=PHASE_HYPERPARAMS,
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
            raise

        except Exception as e:
            print(f"\n[CRITICAL ERROR] Training crashed: {e}")
            if model is not None: 
                model.save(os.path.join(save_dir, config.MODEL_NAME + "_CRASH_SAVE"))
                time.sleep(2) # Buffer for OS to finish disk write
            if env is not None and hasattr(env, "save"): env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_CRASH_SAVE.pkl"))
            if 'callback' in locals(): callback._save_phase_state()
            raise e

        finally:
            failsafe_env(env=env, model=model)

    def resume(self):
        pass

    def tune(self, env_fn, n_trials, study_name="sac_sf2_tuning", load_zip=None, load_pkl=None, start_phase="0", timesteps=50000, device="cuda"):
        import optuna
        from agents.sac.optuna_study import objective
        
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
        tuning_dir = os.path.join(directories["tuning"], "sac")
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
