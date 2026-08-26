import os
import random
import shutil
import tempfile
import atexit
from agents.pbt.pbt_config import (
    POPULATION_SIZE, STEPS_PER_EXPLOIT, PB2_HYPERPARAM_SPACE, FIXED_PARAMS
)
from core.config import get_directory

# Fix 2: Persistent temp dir for Ray checkpoints
_checkpoint_temp_dir = None

def _cleanup_checkpoint_dir():
    global _checkpoint_temp_dir
    if _checkpoint_temp_dir is not None:
        try:
            shutil.rmtree(_checkpoint_temp_dir, ignore_errors=True)
        except Exception:
            pass

atexit.register(_cleanup_checkpoint_dir)

def run_agent_worker(config_dict):
    import sys, os
    import tempfile
    import shutil
    from stable_baselines3.common.callbacks import BaseCallback
    from collections import deque
    
    # Fix 4: Local class definition for Windows Ray spawn
    class PBTWorkerCallback(BaseCallback):
        def __init__(self, rank):
            super().__init__()
            self.rank = rank
            self.win_buffer = deque(maxlen=250)

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                if "win" in info:
                    self.win_buffer.append(info["win"])
            return True
            
    src_path = config_dict["src_path"]
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
        
    from core import config as core_config
    from core.env_tools import SFv2_make_env, failsafe_env
    from core.selective_norm import SelectiveVecNormalize
    from core.rl_constants import AGENT_GAMMA
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from ray import tune as ray_tune
    import torch
    
    import re
    # Fix 5: Derive unique rank from Trial ID to prevent "Rank Theft" after cloning
    # Ray Tune clones the entire config_dict during exploit, which causes port collisions
    # if rank is stored in the config. Trial IDs are like 'a855e_00001'.
    trial_id = ray_tune.get_context().get_trial_id()
    try:
        # Safely extract trailing digits from the trial ID
        match = re.search(r'(\d+)$', trial_id)
        if match:
            rank = int(match.group(1))
        else:
            rank = config_dict["rank"]
    except Exception:
        rank = config_dict["rank"] # Fallback
    
    # Performance Optimization: Restrict each parallel PBT Ray worker to exactly 1 CPU thread
    # This prevents logical core thrashing when up to 10-16 workers run in parallel.
    torch.set_num_threads(1)

    if torch.cuda.is_available():
        # Limit VRAM to prevent out-of-memory when running many parallel workers
        torch.cuda.set_per_process_memory_fraction(0.09)
    
    steps_per_exploit = config_dict.get("steps_per_exploit", STEPS_PER_EXPLOIT)
    total_steps = config_dict.get("total_steps", 5000000)
    start_phase = str(config_dict.get("start_phase", "0"))
    
    # Phase Setup
    if start_phase == "RYU_ONLY":
        target_states = core_config.RYU_ONLY_STATES
    elif start_phase == "CUSTOM":
        target_states = core_config.CUSTOM_STATES if core_config.CUSTOM_STATES else core_config.AVAILABLE_STATES
    else:
        try:
            target_states = core_config.CURRICULUM_PHASES[int(start_phase)]
        except (ValueError, IndexError):
            target_states = core_config.CURRICULUM_PHASES[0]
            
    checkpoint = ray_tune.get_checkpoint()
    checkpoint_dir = checkpoint.to_directory() if checkpoint else None
    
    base_zip = config_dict.get("base_zip")
    base_pkl = config_dict.get("base_pkl")

    # Use DummyVecEnv because Ray already provides process isolation.
    # SubprocVecEnv on Windows is unstable when nested inside Ray workers.
    envs_per_worker = config_dict.get("envs_per_worker", 1)
    env_fns = [SFv2_make_env(rank * 10 + i, version=config_dict["env_version"]) for i in range(envs_per_worker)]
    env = DummyVecEnv(env_fns)
    try:
        env.env_method("set_training_states", target_states)
    except Exception as e:
        print(f"[PBT Worker {rank}] Could not broadcast states: {e}")

    
    # VecNormalize Loading
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "vecnorm.pkl")):
        vec_path = os.path.join(checkpoint_dir, "vecnorm.pkl")
        env = SelectiveVecNormalize.load(vec_path, env)
        env.training = True
        # Fix 1: Correct VecNormalize Count Decay
        if hasattr(env, "count"):
            env.count = min(env.count, 5000.0)
    elif base_pkl and base_pkl != "None":
        vec_path = os.path.join(core_config.PROJECT_ROOT, base_pkl)
        env = SelectiveVecNormalize.load(vec_path, env)
        env.training = True
    else:
        env = SelectiveVecNormalize(env, n_continuous_dims=core_config.OBS_DIM, n_frames=core_config.NUM_FRAMES)
    
    # Model Loading
    active_lr = config_dict["lr"]
    active_ent = config_dict["ent_coef"]
    active_clip = config_dict["clip_range"]
    
    # Enable TensorBoard logging within the trial directory
    trial_log_dir = ray_tune.get_context().get_trial_dir()
    custom_objs = {
        "learning_rate": active_lr, 
        "ent_coef": active_ent, 
        "clip_range": active_clip,
        "tensorboard_log": trial_log_dir
    }
    
    if checkpoint_dir and os.path.exists(os.path.join(checkpoint_dir, "model.zip")):
        model_path = os.path.join(checkpoint_dir, "model.zip")
        model = PPO.load(model_path, env=env, device="cuda", tensorboard_log=trial_log_dir, custom_objects=custom_objs)
    elif base_zip and base_zip != "None":
        model_path = os.path.join(core_config.PROJECT_ROOT, base_zip)
        model = PPO.load(model_path, env=env, device="cuda", tensorboard_log=trial_log_dir, custom_objects=custom_objs)
    else:
        model = PPO("MlpPolicy", env=env, learning_rate=active_lr, n_steps=FIXED_PARAMS["n_steps"], batch_size=FIXED_PARAMS["batch_size"],
                    ent_coef=active_ent, clip_range=active_clip, n_epochs=4, gamma=AGENT_GAMMA, target_kl=None,
                    policy_kwargs=dict(net_arch=dict(pi=[512,512,256], vf=[512,512,256])), 
                    verbose=0, device="cuda", tensorboard_log=trial_log_dir)
    
    callback = PBTWorkerCallback(rank=rank)
    
    try:
        global _checkpoint_temp_dir
        while model.num_timesteps < total_steps:
            model.learn(total_timesteps=steps_per_exploit, callback=callback, reset_num_timesteps=False)
            
            # Clean up previous checkpoint dir
            if _checkpoint_temp_dir and os.path.exists(_checkpoint_temp_dir):
                shutil.rmtree(_checkpoint_temp_dir, ignore_errors=True)

            _checkpoint_temp_dir = tempfile.mkdtemp()
            model.save(os.path.join(_checkpoint_temp_dir, "model"))
            env.save(os.path.join(_checkpoint_temp_dir, "vecnorm.pkl"))
            
            from ray.train import Checkpoint
            final_checkpoint = Checkpoint.from_directory(_checkpoint_temp_dir)
            win_rate = sum(callback.win_buffer) / len(callback.win_buffer) if len(callback.win_buffer) > 0 else 0.0
            ray_tune.report(metrics={"win_rate": win_rate, "timesteps": model.num_timesteps, "rank": rank}, checkpoint=final_checkpoint)
    finally:
        failsafe_env(env=env, model=model)

class PBTOrchestrator:
    def run(self, algo, env_version, total_steps, population_size, max_concurrent=None, steps_per_exploit=STEPS_PER_EXPLOIT, start_phase="0", base_zip=None, base_pkl=None, model_name="PBT_BEST_model", resume=False, envs_per_worker=1):
        import ray
        from ray import tune
        from ray.tune.schedulers.pb2 import PB2
        from ray import train as ray_train
        from ray.tune import RunConfig, CheckpointConfig
        
        src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        directories = get_directory()

        # Inject PYTHONPATH so Windows workers can unpickle dependencies
        ray.init(ignore_reinit_error=True, runtime_env={"env_vars": {"PYTHONPATH": src_path}})
        
        pb2_scheduler = PB2(time_attr="timesteps", metric="win_rate", mode="max", perturbation_interval=steps_per_exploit, hyperparam_bounds=PB2_HYPERPARAM_SPACE, synch=True)
        
        initial_configs = []
        for rank in range(population_size):
            initial_configs.append({
                "rank": rank, "env_version": env_version, "algo": algo, "src_path": src_path,
                "start_phase": start_phase, "steps_per_exploit": steps_per_exploit,
                "total_steps": total_steps, "envs_per_worker": envs_per_worker,
                "base_zip": base_zip, "base_pkl": base_pkl,
                "lr": random.uniform(*PB2_HYPERPARAM_SPACE["lr"]),
                "ent_coef": random.uniform(*PB2_HYPERPARAM_SPACE["ent_coef"]),
                "clip_range": random.uniform(*PB2_HYPERPARAM_SPACE["clip_range"]),
            })
        
        exp_dir = os.path.join(directories["tuning"], "pbt", "pbt_sf2")
        
        if resume and tune.Tuner.can_restore(exp_dir):
            print(f"Resuming PBT experiment from {exp_dir}...")
            tuner = tune.Tuner.restore(exp_dir, trainable=run_agent_worker, resume_unfinished=True, resume_errored=True)
        else:
            tuner = tune.Tuner(
                trainable=run_agent_worker,
                param_space=tune.grid_search(initial_configs),
                tune_config=tune.TuneConfig(
                    scheduler=pb2_scheduler, 
                    num_samples=1,
                    max_concurrent_trials=max_concurrent,
                    trial_name_creator=lambda trial: f"trial_{trial.trial_id}",
                    trial_dirname_creator=lambda trial: f"trial_{trial.trial_id}"
                ),
                run_config=RunConfig(
                    name="pbt_sf2", 
                    storage_path=os.path.join(directories["tuning"], "pbt"),
                    verbose=1,
                    checkpoint_config=CheckpointConfig(
                        num_to_keep=3, 
                        checkpoint_score_attribute="win_rate", 
                        checkpoint_score_order="max"
                    )
                )
            )
        
        results = tuner.fit()
        best_result = results.get_best_result(metric="win_rate", mode="max")
        best_config = best_result.config
        best_checkpoint = best_result.checkpoint
        
        print(f"Best Agent Config: {best_config}")
        
        if best_checkpoint:
            checkpoint_path = best_checkpoint.to_directory()
            print(f"Best Checkpoint: {checkpoint_path}")
            shutil.copy(os.path.join(checkpoint_path, "model.zip"), os.path.join(directories["production"], f"{model_name}.zip"))
            shutil.copy(os.path.join(checkpoint_path, "vecnorm.pkl"), os.path.join(directories["production"], f"{model_name}_vecnormalize.pkl"))
        
        ray.shutdown()
        return best_config
