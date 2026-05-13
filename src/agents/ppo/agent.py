import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

from core import config
from core.selective_norm import SelectiveVecNormalize
from core.env_tools import failsafe_env
from manual_curriculum_callback import ManualCurriculumCallback
from agents.base_agent import BaseAgent
from agents.ppo.config import PHASE_HYPERPARAMS, N_STEPS, BATCH_SIZE

class PPOAgent(BaseAgent):
    def train(self, env_fn, save_dir, steps, load_zip=None, load_pkl=None, start_phase=0, lr=0.0, ent_coef=0.0, clip_range=0.0):
        print(f"[Training] Initializing Curriculum Production Training in {save_dir}...")
        
        if load_zip and load_zip != "None":
            print(f"[Training] Loading model from: {load_zip}")
        if load_pkl and load_pkl != "None":
            print(f"[Training] Loading normalization from: {load_pkl}")
            
        config.TRAINING_STATES = config.CURRICULUM_PHASES[start_phase]
        n_envs = config.N_ENVS
        
        # Base phase params
        phase_params = PHASE_HYPERPARAMS[start_phase].copy()
        
        # Apply Overrides from Dashboard if provided (> 0.0)
        active_lr = lr if lr > 0.0 else phase_params["lr"]
        active_ent = ent_coef if ent_coef > 0.0 else phase_params["ent_coef"]
        active_clip = clip_range if clip_range > 0.0 else phase_params["clip"]
        
        env = None
        model = None
        directories = config.get_directory()
        
        try:
            env = SubprocVecEnv([env_fn(i) for i in range(n_envs)])
            
            if load_pkl and load_pkl != "None":
                env = SelectiveVecNormalize.load(os.path.join(config.PROJECT_ROOT, load_pkl), env)
                env.training = True
            else:
                env = SelectiveVecNormalize(env,
                                             n_continuous_dims=config.OBS_DIM, 
                                             n_frames=config.NUM_FRAMES)

            if load_zip and load_zip != "None":
                model = PPO.load(
                    os.path.join(config.PROJECT_ROOT, load_zip), 
                    env=env, 
                    device="cuda",
                    custom_objects={"learning_rate": active_lr, "clip_range": active_clip, "ent_coef": active_ent}
                )
            else:
                model = PPO(
                    policy="MlpPolicy",
                    env=env,
                    learning_rate=active_lr,
                    n_steps=N_STEPS,
                    batch_size=BATCH_SIZE,
                    ent_coef=active_ent,
                    clip_range=active_clip,
                    n_epochs=10,
                    gamma=0.99,
                    target_kl=0.03,
                    policy_kwargs=dict(net_arch=dict(pi=[512, 512, 256], vf=[512, 512, 256])),
                    verbose=1,
                    tensorboard_log=directories["logs"],
                    device="cuda"
                )

            callback = ManualCurriculumCallback(
                save_path=save_dir,
                start_phase=start_phase,
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
            env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_FINAL.pkl"))
            print("\nProduction Training Complete!")
            
        except KeyboardInterrupt:
            print("\n[MANUAL OVERRIDE] Training forcefully interrupted by user.")
            if model is not None: model.save(os.path.join(save_dir, config.MODEL_NAME + "_EMERGENCY"))
            if env is not None: env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_EMERGENCY.pkl"))

        except Exception as e:
            print(f"\n[CRITICAL ERROR] Training crashed: {e}")
            if model is not None: model.save(os.path.join(save_dir, config.MODEL_NAME + "_CRASH_SAVE"))
            if env is not None: env.save(os.path.join(save_dir, config.MODEL_NAME + "_vecnormalize_CRASH_SAVE.pkl"))

        finally:
            failsafe_env(
                env=env if 'env' in dir() else None,
                model=model if 'model' in dir() else None
            )

    def resume(self):
        pass

    def tune(self, env_fn, n_trials, study_name="ppo_sf2_tuning", load_zip=None, load_pkl=None, start_phase=0, timesteps=50000):
        import optuna
        from agents.ppo.optuna_study import objective
        
        directories = config.get_directory()
        db_path = os.path.abspath(os.path.join(directories["tuning"], "ppo_study.db"))
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
