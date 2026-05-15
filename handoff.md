# Project Handoff: Street Fighter II RL Pipeline

## 🎯 Goal
Build a robust, end-to-end Reinforcement Learning pipeline for Street Fighter II (Sega Genesis) using a custom Python-BizHawk TCP bridge. The system is centrally managed by a Gradio web dashboard that orchestrates Optuna tuning, PPO production training, and AI vs AI matchmaking.

## 📈 Current State of the Code
- The **PPO Agent** implementation is stable and integrates well with the multiprocessing `SubprocVecEnv`.
- **Gradio Dashboard (`web_dashboard.py`)** has been significantly polished:
  - Supports robust JSON-based Hyperparameter import/export for seamless resumption.
  - Features real-time UI auto-refreshing for dropdowns when models finish training or new external files are uploaded.
  - Allows direct uploading of external `.zip` models and `.pkl` normalizers into the production environment.
- **Curriculum & Logging:**
  - `ManualCurriculumCallback` tracks per-phase bests, issues milestone checkpoints, and correctly saves/restores state via a JSON state file.
  - Logging is clean, utilizing 0-based phase indices to match the dashboard, and outputs debug steps sparsely (every 10,000 steps).

## 📂 Files Actively Edited
- `src/scripts/web_dashboard.py` (UI layout, JSON import/export logic, external upload handlers, auto-refresh events)
- `src/manual_curriculum_callback.py` (Phase indexing fixes, removed Windows-incompatible unicode characters)
- `src/agents/ppo/agent.py` & `src/agents/ppo/optuna_study.py` (Fixed multiprocessing state inheritance via direct state broadcasting)
- `src/scripts/resume.py` (Phase logging alignment)
- `src/core/bizhawk_base.py` (Adjusted step debug interval to reduce spam)

## 🛑 What Was Tried That Failed (and How It Was Fixed)
1. **Multiprocessing Phase Inheritance Bug:** When resuming or starting training at a phase `> 0`, the parallel workers failed to update their internal phase states. This happened because Windows uses `spawn` for multiprocessing, meaning each worker re-imported the default `config.py` instead of inheriting the dynamically changed states from the main thread.
   - *Fix:* Added an explicit `env.env_method("set_training_states", ...)` broadcast immediately after initializing `SubprocVecEnv` in the main thread.
2. **Windows Unicode Encoding Crash (`CP1252`):** The training loop would crash upon successfully hitting a new reward/win-rate best because the Python `print()` statement attempted to output a unicode checkmark (`✓`), which is unsupported by default Windows console encodings.
   - *Fix:* Replaced unicode checkmarks with asterisks (`*`) in all print statements within the callback.

## ⏭ Next Steps
1. **Algorithm Expansion:** Implement the agent logic for Soft Actor-Critic (SAC) and Deep Q-Network (DQN) in `src/agents/` to fully support the dashboard's algorithm dropdown options.
2. **Curriculum Refinement:** Monitor the agent's performance in later curriculum phases and potentially adjust the reward shaping logic or the `WIN_RATE_THRESHOLD` as opponents become harder.
3. **End-to-End Verification:** Conduct a full, uninterrupted run (Optuna -> Train -> Test) using the polished dashboard.