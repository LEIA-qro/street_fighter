# Project Handoff: Street Fighter II RL (Ryu Specialist)

## 📝 Project Summary
This project implements a production-grade Reinforcement Learning (RL) pipeline for Street Fighter II' - Special Champion Edition on the Sega Genesis. Using a custom lock-step TCP bridge between Python (Stable Baselines3) and BizHawk (Lua), we train a Ryu specialist through a manual curriculum. The architecture is algorithm-agnostic (supporting PPO, SAC, DQN) and optimized for hardware acceleration (CUDA) and reproducible hyperparameter tuning via Optuna.

## 🎯 Goal
Current focus: **Reliability, Performance, and Hyperparameter Optimization.**
We are streamlining the transition from Optuna tuning to production training by ensuring tuning trials are isolated, normalization statistics (VecNorm) are preserved, and the system can be gracefully stopped without losing progress.

## 🚀 Current State
The project is now hardware-optimized and tuning-resilient:
1.  **Hardware Efficiency**: All agents (PPO, DQN, SAC) target the Dedicated GPU (CUDA) by default. Inference is throttled to 1 CPU thread to prevent spikes.
2.  **Tuning Isolation**: Optuna trials are namespaced into subdirectories (e.g., `models/tuning/ppo/`).
3.  **State Persistence**: `SelectiveVecNormalize` statistics are saved for every tuning trial and production checkpoint.
4.  **Graceful Emergency Lifecycle**: The dashboard "Stop" command triggers an `_EMERGENCY` save for both models and VecNorm files, with a 15s buffer to ensure disk write completion.
5.  **Clean UI**: Dashboard dropdowns are strictly filtered by algorithm to prevent loading incompatible models.
6.  **Performance Profiling**: Integrated `cProfile` support in testing scripts to identify bottlenecks.

## 📂 Files Actively Edited
- `src/agents/{ppo,dqn,sac}/optuna_study.py`: Tuning isolation, VecNorm saving, and interrupt handling.
- `src/agents/{ppo,dqn,sac}/agent.py`: Best-model export and re-raising interrupts for reliability.
- `src/scripts/web_dashboard.py`: UI filtering, shutdown logic, and performance toggles.
- `src/core/config.py`: Configuration persistence and Lua bridge management.

## ❌ Failed Attempts (Archive)
- **Regex Backreferences**: Previously used `\1` in config updates; switched to `\g<1>` to prevent numerical corruption (e.g., Group 110 errors).
- **CPU Multi-threading**: Initial PPO CPU training caused 90%+ lag; resolved by forcing CUDA and `set_num_threads(1)`.
- **Global Tuning Folder**: Trial 0 for PPO used to overwrite Trial 0 for DQN; fixed with subdirectory namespacing.
- **Swallowing Interrupts**: Agent loops used to catch `KeyboardInterrupt` without re-raising; fixed to allow `train.py` retry loops to stop definitively.
- **PowerShell Syntax**: Attempted `&&` for command chaining; switched to `;` for compatibility.

## ⏭️ Next Steps
1.  **Config Sync**: Manually update `PHASE_HYPERPARAMS` in `src/agents/{algo}/config.py` using values from the newly generated `best_params.json` files.
2.  **Extended Tuning**: Run a high-trial tuning session (50+ trials) on Phase 3/4 to find optimal coefficients for late-game matchups.
3.  **Validation**: Test the `best_model.zip` (exported after tuning) in a Matchup to confirm it outperforms the baseline.
