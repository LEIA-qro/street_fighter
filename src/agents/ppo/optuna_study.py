import os
import optuna
import traceback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from core import config
from core.selective_norm import SelectiveVecNormalize
from core.env_tools import failsafe_env

def objective(trial, env_fn, load_zip=None, load_pkl=None, start_phase=0, tuning_timesteps=50000, device="cpu"):
    # --- Expanded Hyperparameter Search Space ---
    lr = trial.suggest_float("lr", 1e-5, 7e-4, log=True)
    ent_coef = trial.suggest_float("ent_coef", 1e-8, 0.05, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
    n_epochs = trial.suggest_int("n_epochs", 3, 20)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 1.0)
    
    # SAFETY CONSTRAINT: Architecture and Buffer sizes can ONLY be tuned if starting from scratch.
    # Changing them on a loaded model destroys the rollout buffer and weights.
    if load_zip is None or load_zip == "None":
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        net_width = trial.suggest_categorical("net_width", [256, 512, 1024])
        net_arch = dict(pi=[net_width, net_width, 256], vf=[net_width, net_width, 256])
    else:
        n_steps = None
        batch_size = None
        net_arch = None
    
    n_envs = config.N_ENVS

    # Bug Fix: Correctly apply RYU_ONLY and CUSTOM phases during Optuna tuning
    if start_phase == "RYU_ONLY":
        config.TRAINING_STATES = config.RYU_ONLY_STATES
    elif start_phase == "CUSTOM":
        config.TRAINING_STATES = config.CUSTOM_STATES if config.CUSTOM_STATES else config.AVAILABLE_STATES
    else:
        config.TRAINING_STATES = config.CURRICULUM_PHASES[int(start_phase)]

    # Setup Paths
    directories = config.get_directory()
    trial_log_dir = os.path.join(directories["tuning_logs"], f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)

    env = None
    model = None

    try:
        # 1. Initialize Parallel Environments
        env = SubprocVecEnv([env_fn(i) for i in range(n_envs)])

        # Broadcast the target phase states to all workers (fixes multiprocessing inheritance bug)
        try:
            env.env_method("set_training_states", config.TRAINING_STATES)
            print(f"[Optuna Trial {trial.number}] States broadcast to all {n_envs} workers.")
        except Exception as e:
            print(f"[Optuna Trial {trial.number}][WARN] Could not broadcast states: {e}")


        if load_pkl and load_pkl != "None":
            env = SelectiveVecNormalize.load(os.path.join(config.PROJECT_ROOT, load_pkl), env)
            # Ensure it continues to update normalization during tuning
            env.training = True
        else:
            env = SelectiveVecNormalize(env, n_continuous_dims=config.OBS_DIM, n_frames=config.NUM_FRAMES)

        # 2. Build Model
        if load_zip and load_zip != "None":
            model = PPO.load(
                os.path.join(config.PROJECT_ROOT, load_zip),
                env=env,
                device=device,
                custom_objects={
                    "learning_rate": lr,
                    "ent_coef": ent_coef,
                    "clip_range": clip_range,
                    "n_epochs": n_epochs,
                    "gamma": gamma,
                    "gae_lambda": gae_lambda
                }
            )
        else:
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=lr,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=n_epochs,
                ent_coef=ent_coef,
                clip_range=clip_range,
                gamma=gamma,
                gae_lambda=gae_lambda,
                policy_kwargs=dict(net_arch=net_arch),
                verbose=0,
                tensorboard_log=trial_log_dir,
                device=device
            )

        # 3. Train
        print(f"[Optuna] Trial {trial.number} started: lr={lr:.6f}, phase={start_phase}, timesteps={tuning_timesteps}")
        model.learn(total_timesteps=tuning_timesteps)

        # 4. Evaluate
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        
        # 5. Save Trial Model
        trial_model_path = os.path.join(directories["tuning"], f"trial_{trial.number}_model")
        model.save(trial_model_path)
        
        return mean_reward

    except optuna.exceptions.TrialPruned:
        raise
    except Exception as e:
        print(f"[Optuna] Trial {trial.number} failed with error: {e}")
        traceback.print_exc()
        raise e # Mark trial as FAIL instead of penalizing it

    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            failsafe_env()
        except Exception:
            pass
