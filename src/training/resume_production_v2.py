import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

#from typing import Callable 
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.callbacks import BaseCallback

import config
from selective_norm import SelectiveVecNormalize
from manual_curriculum_callback import ManualCurriculumCallback 
from env_tools import failsafe_env, SFv2_make_env

directories = config.get_directory()


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
        callback_class = ManualCurriculumCallback
    
    print(f"Initializing {config.N_ENVS}-Core Resume Environment...")
    
    # --- RESTORE CURRICULUM STATE ---
    phase_state = ManualCurriculumCallback.load_state(directories["production"])
    restored_phase = start_phase if start_phase is not None else phase_state["current_phase"]
    phase_params   = config.PHASE_HYPERPARAMS[restored_phase]

    # Point training states to the restored phase (not config default)
    config.TRAINING_STATES = config.CURRICULUM_PHASES[restored_phase]
    print(f"[Resume] Restoring to Phase {restored_phase + 1} "
          f"with states: {config.TRAINING_STATES}")
    
    # 1. Boot Parallel Emulators
    print(f"Initializing {config.N_ENVS}-Core Resume Environment...")
    n_envs = config.N_ENVS
    
    env, model = None, None # Placeholders for the finally block

    try:
        env = SubprocVecEnv([SFv2_make_env(i) for i in range(n_envs)])
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

        # model.target_kl = 0.03   # THE FIX: Restored to standard 0.03 
        # model.n_epochs = 10      
        callback = callback_class(
            save_path=directories["production"],
            start_phase=restored_phase,   # <-- The fix
            eval_interval=500,
            save_interval=config.SAVE_FREQ_STEPS
        )
        
        # In resume_production_v2.py — after restoring phase_bests:
        callback._phase_bests = phase_state.get("phase_bests", {})
        callback._threshold_save_fired = phase_state.get("threshold_save_fired", set())
        callback.last_save_step        = phase_state.get("last_save_step", 0)   # ADD
        callback.last_eval_step        = phase_state.get("last_eval_step", 0)   # ADD
        if start_phase is not None and start_phase != phase_state.get("current_phase", 0):
            print(f"[Resume] start_phase override detected. "
                f"Clearing phase {start_phase} bests for fresh tracking.")
            callback._phase_bests.pop(start_phase, None)  # Remove stale threshold for this phase
        # ← ADD THIS: Force-broadcast the correct states to all subprocess envs
        # The callback's __init__ sets self.current_phase but never calls env_method.
        # Fix 1 handles the initial spawn, but this handles any edge case where
        # env workers miss the config mutation (e.g., different Python interpreter state).
        if restored_phase > 0:
            try:
                env.env_method("set_training_states", config.CURRICULUM_PHASES[restored_phase])
                print(f"[Resume] States broadcast to all {config.N_ENVS} workers → Phase {restored_phase + 1}")
            except Exception as e:
                print(f"[Resume][WARN] Could not broadcast states to workers: {e}")
                
        model.learn(
            total_timesteps=config.RESUME_PRODUCTION_TIMESTEPS, 
            callback=callback,
            tb_log_name=config.MODEL_NAME,
            reset_num_timesteps=False # reset_num_timesteps=False ensures TensorBoard continues the graph smoothly
        )
        
        # Save Final Grandmaster
        model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_FINAL"))
        env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_FINAL.pkl"))
        print("\nProduction Training Complete!")
        return {"success": True, "reason": "completed"}
        
    except KeyboardInterrupt:
        print("\n[MANUAL OVERRIDE] Training forcefully interrupted by user.")
        if model is not None: model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_EMERGENCY"))
        if env is not None: env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_EMERGENCY.pkl"))
        return {"success": True, "reason": "interrupted"} # Signals the Supervisor to stop
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training crashed: {e}")
        if model is not None: model.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_CRASH_SAVE"))
        if env is not None: env.save(os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_CRASH_SAVE.pkl"))
        return {"success": False, "reason": "crash"}

        
    finally:
        failsafe_env(env=env, model=model)

if __name__ == "__main__":
    current_model_path = os.path.join(directories["project_root"], config.TRAINING_ZIP_FILE)
    current_vec_path = os.path.join(directories["project_root"], config.TRAINING_PKL_FILE)

    restart_count = 0
    phase_state = 1 # Placeholder for future phase advancement
    while True:
        result = resume_training(current_model_path, current_vec_path, start_phase=phase_state)
        

        if result["success"]:
            print(f"Training session ended: {result['reason']}")
            break
        else:
            restart_count += 1
             # phase_state never updated — always None
            # This is fine because load_state() reads curriculum_state.json
            # BUT only if ManualCurriculumCallback._save_phase_state() was called before the crash
            print(f"\n--- AUTO-RESTART #{restart_count} ---")
            current_model_path = os.path.join(directories["production"], f"{config.MODEL_NAME}_CRASH_SAVE.zip")
            current_vec_path   = os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_CRASH_SAVE.pkl")