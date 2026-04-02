import os
import subprocess
import time
import multiprocessing
import gc
import torch
from typing import Callable 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import BaseCallback

import config
from env_sf2_v2 import StreetFighterEnvV2
from selective_norm import SelectiveVecNormalize
from manual_curriculum_callback import CurriculumCallback, PhaseStoppingCallback
from env_nuclear_cleanup import _nuclear_cleanup


directories = config.get_directory()

# --- NEW: The Linear Schedule Function ---
def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Progressively drops the learning rate.
    progress_remaining starts at 1.0 and goes to 0.0.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_env(rank):
    def _init():
        env = StreetFighterEnvV2(rank=rank)
        env = Monitor(env)
        return env
    return _init

def resume_training(model_path, vec_path,
                    callback_class=None,
                    start_phase: int = None) -> dict:
    """
    Returns a result dict:
      {"success": True,  "reason": "completed"}
      {"success": True,  "reason": "phase_advanced", "new_phase": int}
      {"success": True,  "reason": "interrupted"}
      {"success": False, "reason": "crash"}
    """
    if callback_class is None:
        callback_class = CurriculumCallback
    
    print(f"Initializing {config.N_ENVS}-Core Resume Environment...")
    
    # --- RESTORE CURRICULUM STATE ---
    phase_state = CurriculumCallback.load_phase_state(directories["production"])
    restored_phase = start_phase if start_phase is not None else phase_state["current_phase"]
    phase_params   = config.PHASE_HYPERPARAMS[restored_phase]

    # Point training states to the restored phase (not config default)
    config.TRAINING_STATES = config.CURRICULUM_PHASES[restored_phase]
    print(f"[Resume] Restoring to Phase {restored_phase + 1} "
          f"with states: {config.TRAINING_STATES}")
    
    # 1. Boot Parallel Emulators
    print(f"Initializing {config.N_ENVS}-Core Resume Environment...")
    n_envs = config.N_ENVS
    env = SubprocVecEnv([make_env(i) for i in range(n_envs)])
    
    # 2. Load the VecNormalize Math
    print(f"Loading normalization stats from {vec_path}...")
    env = SelectiveVecNormalize.load(vec_path, env)
    
    # CRITICAL: Ensure the environment continues to update its normalization math
    env.training = True
    env.norm_reward = True

    # 3. Load the Brain (PPO)
    print(f"Loading neural network weights from {model_path}...")
    model = PPO.load(
        model_path, 
        env=env, 
        device="cuda", 
        tensorboard_log=directories["logs"],
        custom_objects={
            "learning_rate": phase_params["lr"],
            "ent_coef":      phase_params["ent_coef"],
            "clip_range":    phase_params["clip"],
            # n_steps and batch_size intentionally omitted —
            # they are structural and cannot change mid-training in SB3.
            # They remain as saved in the model zip.
        }
    )

    # 5. Restore Golden Architecture Constraints
    # model.target_kl = 0.03   # THE FIX: Restored to standard 0.03 
    # model.n_epochs = 10      
    callback = callback_class(
        save_path=directories["production"],
        verbose=1,
        start_phase=restored_phase   # <-- The fix
    )
    
    callback._phase_bests = phase_state.get("phase_bests", {})

    try:
        # reset_num_timesteps=False ensures TensorBoard continues the graph smoothly
        model.learn(
            total_timesteps=config.RESUME_PRODUCTION_TIMESTEPS, 
            callback=callback,
            tb_log_name=config.MODEL_NAME,
            reset_num_timesteps=False 
        )

        # Check if we stopped due to a phase transition (PhaseStoppingCallback)
        if hasattr(callback, "phase_just_advanced") and callback.phase_just_advanced:
            new_phase = callback.current_phase
            return {"success": True, "reason": "phase_advanced", "new_phase": new_phase}
        
        # Save Final Grandmaster
        model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_FINAL"))
        env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_FINAL.pkl"))
        print("\nProduction Training Complete!")
        return {"success": True, "reason": "completed"} # Signals the Supervisor that we finished successfully
        
    except KeyboardInterrupt:
        print("\n[MANUAL OVERRIDE] Training forcefully interrupted by user.")
        model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_EMERGENCY"))
        env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_EMERGENCY.pkl"))
        return {"success": True, "reason": "interrupted"} # Signals the Supervisor to stop
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed: {e}")
        model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_CRASH_SAVE"))
        env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_CRASH_SAVE.pkl"))
        return {"success": False, "reason": "crash"}

        
    finally:
        _nuclear_cleanup(model=model, env=env)

if __name__ == "__main__":
    current_model_path = os.path.join(directories["project_root"], config.TRAINING_ZIP_FILE)
    current_vec_path = os.path.join(directories["project_root"], config.TRAINING_PKL_FILE)

    restart_count = 0
    
    while True:
        result = resume_training(current_model_path, current_vec_path)

        if result["success"]:
            print(f"Training session ended: {result['reason']}")
            break
        else:
            restart_count += 1
            print(f"\n--- AUTO-RESTART #{restart_count} ---")
            current_model_path = os.path.join(directories["production"], f"{config.MODEL_NAME}_CRASH_SAVE.zip")
            current_vec_path   = os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_CRASH_SAVE.pkl")