import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from core import config
from core.selective_norm import SelectiveVecNormalize
from agents.manual_curriculum_callback import ManualCurriculumCallback
from core.env_tools import failsafe_env, SFv2_make_env, prevent_sleep

# Register SIGBREAK handler on Windows to catch CTRL_BREAK as KeyboardInterrupt
import signal
def _sigbreak_handler(sig, frame):
    raise KeyboardInterrupt

if hasattr(signal, "SIGBREAK"):
    try:
        signal.signal(signal.SIGBREAK, _sigbreak_handler)
    except Exception:
        pass

directories = config.get_directory()


from agents.ppo.config import PHASE_HYPERPARAMS, N_STEPS, BATCH_SIZE

def resume_training(model_path, vec_path,
                    callback_class=None,
                    start_phase: int = None,
                    device: str = "auto") -> dict:
    """
    Returns a result dict:
      {"success": True,  "reason": "completed"}
      {"success": True,  "reason": "phase_advanced", "new_phase": int}
      {"success": True,  "reason": "interrupted"}
      {"success": False, "reason": "crash"}
    """
    prevent_sleep()
    if callback_class is None:
        callback_class = ManualCurriculumCallback
    
    import torch
    # Performance Optimization: Restrict PyTorch to 2 CPU threads during training
    # This prevents it from hijacking logical cores alongside active EmuHawk emulators.
    torch.set_num_threads(2)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Initializing {config.N_ENVS}-Core Resume Environment (Device: {device})...")
    
    algo_part = "ppo"
    env_part = "v2"
    
    # Try parsing from model_path
    normalized_model_path = os.path.normpath(model_path)
    path_parts = normalized_model_path.split(os.sep)
    
    if "production" in path_parts:
        idx = path_parts.index("production")
        if len(path_parts) > idx + 2:
            env_part = path_parts[idx + 1]
            algo_part = path_parts[idx + 2]
    elif len(path_parts) >= 3:
        algo_part = path_parts[-2]

    # --- RESTORE CURRICULUM STATE ---
    is_auto = os.path.exists(os.path.join(directories["production"], "auto_curriculum_state.json"))
    
    if is_auto:
        from agents.auto_curriculum_callback import AutoCurriculumCallback
        phase_state = AutoCurriculumCallback.load_state(directories["production"])
        restored_phase = start_phase if start_phase is not None else phase_state["current_level"]
        
        # Resolve hyperparams fallback
        phase_idx = (restored_phase - 1) // 2
        phase_params = PHASE_HYPERPARAMS.get(phase_idx, PHASE_HYPERPARAMS[0])
        state_name = phase_state.get("state_name", None)
        
        # Point training states to the restored level's lottery pool (not config default)
        start_level = restored_phase
        introduced = phase_state.get("introduced_states", [])
        pool = []
        for lvl in range(1, start_level):
            if lvl in config.DIFFICULTY_LEVELS:
                pool.extend(config.DIFFICULTY_LEVELS[lvl])
        if start_level in config.DIFFICULTY_LEVELS:
            pool.extend(config.DIFFICULTY_LEVELS[start_level] * 3)
        for s in introduced:
            pool.extend([s] * 5)
            
        config.TRAINING_STATES = pool
        print(f"[Resume] Restoring to Auto-Curriculum Level {restored_phase} "
              f"with {len(introduced)} introduced states. Pool size: {len(config.TRAINING_STATES)}")
    else:
        phase_state = ManualCurriculumCallback.load_state(directories["production"])
        restored_phase = start_phase if start_phase is not None else phase_state["current_phase"]
        phase_params   = PHASE_HYPERPARAMS[restored_phase]
        state_name = phase_state.get("state_name", None)

        # Point training states to the restored phase (not config default)
        config.TRAINING_STATES = config.CURRICULUM_PHASES[restored_phase]
        print(f"[Resume] Restoring to Phase {restored_phase} "
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
            device=device, 
            tensorboard_log=directories["logs"],
            custom_objects={
                "learning_rate": phase_params["lr"],
                "ent_coef":      phase_params["ent_coef"],
                "clip_range":    phase_params["clip"],
            }
        )

        if is_auto:
            from agents.auto_curriculum_callback import AutoCurriculumCallback
            callback = callback_class if callback_class != ManualCurriculumCallback else AutoCurriculumCallback
            callback = callback(
                save_path=directories["production"],
                phase_hyperparams=PHASE_HYPERPARAMS,
                start_level=restored_phase,
                eval_interval=500,
                save_interval=config.SAVE_FREQ_STEPS,
                algo=algo_part,
                env_version=env_part,
                model_name=config.MODEL_NAME,
                state_name=state_name
            )
            # Restore buffers
            from collections import deque
            state_wins_raw = phase_state.get("state_win_buffers", {})
            for s_name, lst in state_wins_raw.items():
                if s_name in callback.state_win_buffers:
                    callback.state_win_buffers[s_name] = deque(lst, maxlen=100)
            callback.introduced_states = phase_state.get("introduced_states", [])
            callback.stability_counter = phase_state.get("stability_counter", 0)
        else:
            callback = callback_class(
                save_path=directories["production"],
                phase_hyperparams=PHASE_HYPERPARAMS,
                start_phase=restored_phase,
                eval_interval=500,
                save_interval=config.SAVE_FREQ_STEPS,
                algo=algo_part,
                env_version=env_part,
                model_name=config.MODEL_NAME,
                state_name=state_name
            )
            callback._phase_bests = phase_state.get("phase_bests", {})
        
        callback._threshold_save_fired = phase_state.get("threshold_save_fired", set())
        callback.last_save_step        = phase_state.get("last_save_step", 0)  
        callback.last_eval_step        = phase_state.get("last_eval_step", 0)   
        
        if not is_auto and start_phase is not None and start_phase != phase_state.get("current_phase", 0):
            print(f"[Resume] start_phase override detected. "
                f"Clearing phase {start_phase} bests for fresh tracking.")
            callback._phase_bests.pop(start_phase, None)  # Remove stale threshold for this phase
       
        if is_auto or restored_phase > 0:
            try:
                env.env_method("set_training_states", config.TRAINING_STATES)
                print(f"[Resume] States broadcast to all {config.N_ENVS} workers -> Level/Phase {restored_phase}")
            except Exception as e:
                print(f"[Resume][WARN] Could not broadcast states to workers: {e}")
                
        model.learn(
            total_timesteps=config.RESUME_PRODUCTION_TIMESTEPS, 
            callback=callback,
            tb_log_name=f"{algo_part}_{env_part}_{config.MODEL_NAME}",
            reset_num_timesteps=False
        )
        
        # Save Final Grandmaster (Dynamic Final Save)
        win_rate_val = callback._win_rate() if hasattr(callback, "_win_rate") else 0.0
        winrate_pct = int(round(win_rate_val * 100))
        if hasattr(callback, "current_level"):
            state_tag = callback.state_name if callback.state_name is not None else f"lvl{callback.current_level}"
            if len(getattr(callback, "introduced_states", [])) > 0 and callback.state_name is None:
                state_tag = f"lvl{callback.current_level}_plus{len(callback.introduced_states)}"
        else:
            state_tag = callback.state_name if callback.state_name is not None else f"phase{getattr(callback, 'current_phase', 0)}"
        
        final_base = f"{algo_part}_{env_part}_{config.MODEL_NAME}_{state_tag}_final_WR{winrate_pct}pct_{callback.num_timesteps}steps"
        
        model.save(os.path.join(directories["production"], final_base))
        env.save(os.path.join(directories["production"], f"{final_base}_vecnorm.pkl"))
        print(f"\nProduction Training Complete! Saved final model as: {final_base}")
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
        if 'callback' in locals():
            if hasattr(callback, "_save_curriculum_state"):
                callback._save_curriculum_state(force=True)
            elif hasattr(callback, "_save_phase_state"):
                callback._save_phase_state()
        return {"success": False, "reason": "crash"}

        
    finally:
        failsafe_env(env=env, model=model)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Street Fighter II RL Automated Crash-Recovery Supervisor")
    parser.add_argument("--load_zip", type=str, default=None, help="Path to model .zip file to resume")
    parser.add_argument("--load_pkl", type=str, default=None, help="Path to VecNormalize .pkl file")
    parser.add_argument("--phase", type=str, default=None, help="Optional starting phase or level override")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cuda, cpu, auto)")
    args = parser.parse_args()

    current_model_path = args.load_zip if args.load_zip else os.path.join(directories["project_root"], config.TRAINING_ZIP_FILE)
    current_vec_path = args.load_pkl if args.load_pkl else os.path.join(directories["project_root"], config.TRAINING_PKL_FILE)

    if not os.path.isabs(current_model_path):
        current_model_path = os.path.join(config.PROJECT_ROOT, current_model_path)
    if not os.path.isabs(current_vec_path):
        current_vec_path = os.path.join(config.PROJECT_ROOT, current_vec_path)

    phase_override = int(args.phase) if (args.phase is not None and args.phase.isdigit()) else None
    restart_count = 0
    try:
        while True:
            result = resume_training(current_model_path, current_vec_path, start_phase=phase_override, device=args.device)

            if result["success"]:
                print(f"Training session ended: {result['reason']}")
                break
            else:
                restart_count += 1
                print(f"\n--- AUTO-RESTART #{restart_count} ---")
                current_model_path = os.path.join(directories["production"], f"{config.MODEL_NAME}_CRASH_SAVE.zip")
                current_vec_path   = os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_CRASH_SAVE.pkl")
    finally:
        failsafe_env()

if __name__ == "__main__":
    main()