import os
import optuna
import traceback
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from core import config
from core.selective_norm import SelectiveVecNormalize
from core.env_tools import failsafe_env
from agents.sac.agent import ContinuousToDiscreteSoftRelaxationWrapper

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque

class OptunaPruningCallback(BaseCallback):
    def __init__(self, trial, report_interval=25000, pruning_threshold=0.05):
        super().__init__()
        self.trial = trial
        self.report_interval = report_interval
        self.pruning_threshold = pruning_threshold
        self.win_buffer = deque(maxlen=250)
        self.last_report_step = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "win" in info:
                self.win_buffer.append(info["win"])
        
        if self.num_timesteps - self.last_report_step >= self.report_interval:
            win_rate = sum(self.win_buffer) / len(self.win_buffer) if len(self.win_buffer) > 0 else 0.0
            self.trial.report(win_rate, self.num_timesteps)
            
            if self.trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
            self.last_report_step = self.num_timesteps
        return True

def objective(trial, env_fn, load_zip=None, load_pkl=None, start_phase=0, tuning_timesteps=50000, device="cuda"):
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
    trial_log_dir = os.path.join(directories["tuning_logs"], "sac", f"trial_{trial.number}")
    os.makedirs(trial_log_dir, exist_ok=True)

    env = None
    model = None
    
    try:
        def make_sac_env(rank):
            original_init = env_fn(rank)
            def _init():
                return ContinuousToDiscreteSoftRelaxationWrapper(original_init())
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
                device=device,
                tensorboard_log=trial_log_dir,
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
                device=device
            )

        print(f"[Optuna] Trial {trial.number} started: lr={lr:.6f}, phase={start_phase}, timesteps={tuning_timesteps}")
        
        pruning_callback = OptunaPruningCallback(trial)
        model.learn(total_timesteps=tuning_timesteps, callback=pruning_callback)

        # Temporarily disable training and reward normalization during evaluation to prevent statistics contamination and reward compression
        old_training = env.training
        old_norm_reward = env.norm_reward
        env.training = False
        env.norm_reward = False
        
        # Disable wrapper training mode in all parallel sub-environments during evaluation
        try:
            env.env_method("set_attr", "training", False)
        except Exception:
            pass
            
        try:
            mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
        finally:
            env.training = old_training
            env.norm_reward = old_norm_reward
            try:
                env.env_method("set_attr", "training", True)
            except Exception:
                pass
        
        # 5. Save Trial Model and VecNorm
        trial_dir = os.path.join(directories["tuning"], "sac")
        os.makedirs(trial_dir, exist_ok=True)
        
        trial_model_path = os.path.join(trial_dir, f"trial_{trial.number}_model")
        model.save(trial_model_path)
        
        if hasattr(env, "save"):
            env.save(os.path.join(trial_dir, f"trial_{trial.number}_vecnormalize.pkl"))
        
        return mean_reward

    except optuna.exceptions.TrialPruned:
        raise
    except KeyboardInterrupt:
        print(f"\n[Optuna] Trial {trial.number} forcefully interrupted by user. Saving EMERGENCY files...")
        trial_dir = os.path.join(directories["tuning"], "sac")
        os.makedirs(trial_dir, exist_ok=True)
        if model is not None:
            model.save(os.path.join(trial_dir, f"trial_{trial.number}_EMERGENCY_model"))
        if hasattr(env, "save"):
            env.save(os.path.join(trial_dir, f"trial_{trial.number}_EMERGENCY_vecnormalize.pkl"))
        raise
    except Exception as e:
        print(f"[Optuna] Trial {trial.number} failed with error: {e}")
        traceback.print_exc()
        raise e

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
