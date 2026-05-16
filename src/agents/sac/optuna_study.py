import os
import optuna
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from core import config
from core.selective_norm import SelectiveVecNormalize
from core.env_tools import failsafe_env

def objective(trial, env_fn, load_zip=None, load_pkl=None, start_phase=0, tuning_timesteps=50000):
    # Ensure start_phase is an integer for list indexing
    start_phase = int(start_phase)

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    tau = trial.suggest_float("tau", 0.001, 0.05)
    gamma = trial.suggest_float("gamma", 0.95, 0.9999, log=True)
    
    if load_zip is None or load_zip == "None":
        buffer_size = trial.suggest_categorical("buffer_size", [50000, 100000, 200000])
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        net_width = trial.suggest_categorical("net_width", [256, 512, 1024])
        net_arch = dict(pi=[net_width, net_width, 256], qf=[net_width, net_width, 256])
    else:
        buffer_size = None
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
    
    directories = config.get_directory()
    trial_log_dir = os.path.join(directories["tuning_logs"], f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)

    env = None
    model = None
    
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
            print(f"[Optuna Trial {trial.number}] States broadcast to all {n_envs} workers.")
        except Exception as e:
            print(f"[Optuna Trial {trial.number}][WARN] Could not broadcast states: {e}")

        if load_pkl and load_pkl != "None":
            env = SelectiveVecNormalize.load(os.path.join(config.PROJECT_ROOT, load_pkl), env)
            env.training = True
        else:
            env = SelectiveVecNormalize(env, n_continuous_dims=config.OBS_DIM, n_frames=config.NUM_FRAMES)

        if load_zip and load_zip != "None":
            model = SAC.load(
                os.path.join(config.PROJECT_ROOT, load_zip),
                env=env,
                device="cuda",
                custom_objects={
                    "learning_rate": lr,
                    "tau": tau,
                    "gamma": gamma
                }
            )
        else:
            model = SAC(
                "MlpPolicy",
                env,
                learning_rate=lr,
                buffer_size=buffer_size,
                batch_size=batch_size,
                tau=tau,
                gamma=gamma,
                ent_coef="auto",
                policy_kwargs=dict(net_arch=net_arch),
                verbose=0,
                tensorboard_log=trial_log_dir,
                device="cuda"
            )

        print(f"[Optuna] Trial {trial.number} started: lr={lr:.6f}, phase={start_phase}, timesteps={tuning_timesteps}")
        model.learn(total_timesteps=tuning_timesteps)

        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        
        trial_model_path = os.path.join(directories["tuning"], f"trial_{trial.number}_model")
        model.save(trial_model_path)
        
        return mean_reward

    except Exception as e:
        print(f"[Optuna] Trial {trial.number} failed with error: {e}")
        return -99999.0

    finally:
        if env is not None:
            env.close()
        failsafe_env()
