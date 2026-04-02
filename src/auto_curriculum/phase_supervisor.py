# phase_supervisor.py — NEW FILE
"""
Supervisor that runs the full curriculum pipeline:
  Phase N training → auto-stop at phase transition → Optuna → resume Phase N+1

Usage: python phase_supervisor.py
"""

import os
import config
import json
from curriculum_callback import CurriculumCallback, PhaseStoppingCallback
from env_nuclear_cleanup import _nuclear_cleanup
from resume_production_v2 import resume_training
from transfer_optuna_v2 import run_study

directories = config.get_directory()

OPTUNA_MEMORY_FILE = os.path.join(directories["production"], "optuna_phase_memory.json")

def _save_optuna_memory():
    with open(OPTUNA_MEMORY_FILE, "w") as f:
        json.dump(config.PHASE_HYPERPARAMS, f, indent=4)

def _get_phase_checkpoint_paths(phase_idx: int) -> tuple[str, str]:
    """Returns (model_path, vec_path) for a given phase entry checkpoint."""
    tag        = f"phase{phase_idx + 1}_entry"
    model_path = os.path.join(directories["production"], f"{config.MODEL_NAME}_{tag}.zip")
    vec_path   = os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnorm_{tag}.pkl")
    return model_path, vec_path

# phase_supervisor.py

MAX_CONSECUTIVE_CRASHES = 3
_REQUIRED_HYPERPARAM_KEYS = {"lr", "ent_coef", "clip"}

def _load_optuna_memory():
    if os.path.exists(OPTUNA_MEMORY_FILE):
        with open(OPTUNA_MEMORY_FILE, "r") as f:
            saved_params = json.load(f)
        
        for k, v in saved_params.items():
            phase_idx = int(k)
            # --- Schema validation before mutation ---
            if not isinstance(v, dict):
                print(f"[Supervisor] WARNING: Malformed entry for phase {phase_idx}, skipping.")
                continue
            missing_keys = _REQUIRED_HYPERPARAM_KEYS - set(v.keys())
            if missing_keys:
                print(f"[Supervisor] WARNING: Phase {phase_idx} missing keys {missing_keys}, skipping.")
                continue
            config.PHASE_HYPERPARAMS[phase_idx] = v
        
        print(f"[Supervisor] Restored persistent Optuna parameters from disk.")


def run_supervised_curriculum(initial_model_path: str, initial_vec_path: str,
                               optuna_trials_per_phase: int = config.N_HYPERPARAMETER_TRIALS):  # FIX: use config
    _load_optuna_memory()

    model_path = initial_model_path
    vec_path   = initial_vec_path
    consecutive_crashes = 0  # THE GUARD

    while True:
        result = resume_training(
            model_path=model_path,
            vec_path=vec_path,
            callback_class=PhaseStoppingCallback,
        )

        if result["reason"] == "interrupted":
            print("[Supervisor] Manual interrupt — exiting supervisor.")
            break

        if result["reason"] == "crash":
            consecutive_crashes += 1
            print(f"[Supervisor] Crash #{consecutive_crashes} detected.")
            
            # --- THE GUARD: hard stop after N consecutive crashes ---
            if consecutive_crashes >= MAX_CONSECUTIVE_CRASHES:
                print(f"[Supervisor] FATAL: {consecutive_crashes} consecutive crashes. "
                      f"Halting supervisor. Manual intervention required.")
                break
            
            model_path = os.path.join(directories["production"], f"{config.MODEL_NAME}_CRASH_SAVE.zip")
            vec_path   = os.path.join(directories["production"], f"{config.MODEL_NAME}_vecnormalize_CRASH_SAVE.pkl")
            continue

        # Any successful result resets the crash counter
        consecutive_crashes = 0

        if result["reason"] == "completed":
            print("[Supervisor] Full curriculum complete.")
            break

        if result["reason"] == "phase_advanced":
            completed_phase = result["new_phase"] - 1
            next_phase      = result["new_phase"]

            print(f"\n[Supervisor] Phase {completed_phase + 1} → {next_phase + 1}")
            print(f"[Supervisor] Running Optuna ({optuna_trials_per_phase} trials)...")

            # The transition checkpoint is what was saved by _advance_phase()
            ckpt_model, ckpt_vec = _get_phase_checkpoint_paths(next_phase)

            if next_phase < len(config.CURRICULUM_PHASES):
                next_states = config.CURRICULUM_PHASES[next_phase]
            else:
                next_states = config.CURRICULUM_PHASES[-1]

            try:
                best_params = run_study(
                    model_path=ckpt_model,
                    vec_path=ckpt_vec,
                    next_phase_states=next_states,
                    n_trials=optuna_trials_per_phase,
                    study_name=f"phase_{next_phase + 1}_tuning",
                )
                config.PHASE_HYPERPARAMS[next_phase].update({
                    "lr":       best_params.get("learning_rate", config.PHASE_HYPERPARAMS[next_phase]["lr"]),
                    "ent_coef": best_params.get("ent_coef",      config.PHASE_HYPERPARAMS[next_phase]["ent_coef"]),
                    "clip":     best_params.get("clip_range",    config.PHASE_HYPERPARAMS[next_phase]["clip"]),
                })
                _save_optuna_memory()
                print(f"[Supervisor] Phase {next_phase + 1} hyperparams updated: {config.PHASE_HYPERPARAMS[next_phase]}")
            except Exception as e:
                print(f"[Supervisor] WARNING: Optuna study failed — {e}. Proceeding with default hyperparams.")

            # Next iteration of resume_training() reads the updated config + JSON state
            optuna_best_model = os.path.join(directories["optuna"], "best_transfer_ppo.zip")
            optuna_best_vec   = os.path.join(directories["optuna"], "best_transfer_vecnorm.pkl")

            if os.path.exists(optuna_best_model) and os.path.exists(optuna_best_vec):
                print(f"[Supervisor] Using Optuna warm-start model for Phase {next_phase + 1}.")
                model_path = optuna_best_model
                vec_path   = optuna_best_vec
            else:
                print(f"[Supervisor] Optuna best model not found — using phase entry checkpoint.")
                model_path = ckpt_model
                vec_path   = ckpt_vec

if __name__ == "__main__":
    initial_model = os.path.join(directories["project_root"], config.TRAINING_ZIP_FILE)
    initial_vec   = os.path.join(directories["project_root"], config.TRAINING_PKL_FILE)

    run_supervised_curriculum(
        initial_model_path=initial_model,
        initial_vec_path=initial_vec,
        optuna_trials_per_phase=20,
    )
