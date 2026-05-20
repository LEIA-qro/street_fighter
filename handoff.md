# Project Handoff: Street Fighter II RL (Ryu Specialist)

## 📝 Project Summary
This project implements a production-grade Reinforcement Learning (RL) pipeline for Street Fighter II' - Special Champion Edition on the Sega Genesis. Using a custom lock-step TCP bridge between Python (Stable Baselines3) and BizHawk (Lua), we train a Ryu specialist through a manual curriculum. The architecture is algorithm-agnostic (supporting PPO, SAC, DQN) and optimized for hardware acceleration (CUDA) and reproducible hyperparameter tuning via Optuna.

## 🎯 Goal
Current focus: **Solving the Entropy Plateau and Stabilizing Convergence.**
We have identified that the original `MultiBinary(10)` action space (1,024 combinations) was too sparse and invalid for the policy gradient to effectively explore, leading to stagnant learning. The goal is to transition to the `v3` architecture to accelerate win rate improvement.

## 🚀 Current State
The project has undergone a major architectural stabilization phase:
1.  **Translation Invariance (Fix C)**: `v2` and `v3` environments now use relative $X/Y$ coordinates and wall distance instead of absolute RAM values, enabling faster spatial generalization.
2.  **Reward Normalization (Fix A)**: Enhanced `SelectiveVecNormalize` with a Welford online algorithm for rewards, preventing Value Function explosion while protecting one-hot encoded observations.
3.  **v3 Architecture (Fix B)**: Implemented a new `MultiDiscrete([9, 7])` environment. This reduces the exploration space from **1,024 to 63** valid combinations and corrects button mapping for Ryu's full moveset (6 individual buttons).
4.  **Hardware Resilience (Fix D)**: Socket deaths now return a `0.0` reward and a `socket_death` flag, preventing rollout buffer poisoning.
5.  **Ecosystem Compatibility**: Dashboard and AI-vs-AI evaluation scripts now fully support mixed-version matchups and `v3` specific logic.

## 📂 Files Actively Edited
- `src/envs/sf2_v3.py`: Implementation of the MultiDiscrete environment.
- `src/core/selective_norm.py`: Welford reward normalization and state persistence.
- `src/envs/sf2_v2.py`: Relative coordinate updates and socket error handling.
- `src/scripts/web_dashboard.py`: UI integration for v3 and mixed-version matchups.
- `src/scripts/test_ai_vs_ai_v2.py` & `test_agent_v2.py`: Cross-version evaluation support.
- `src/core/config.py`: `OBS_DIM` adjustment (11 to 10).

## ❌ Failed Attempts (Compressed)
- **Absolute Coordinates**: Caused the network to learn identical interactions multiple times based on screen position.
- **MultiBinary(10) Exploration**: Confirmed via telemetry (entropy loss -6.83 at 675K steps) to be statistically intractable for fast convergence.
- **-50.0 Socket Penalty**: Poisoned the training buffer with artificial hardware-failure data.
- **Fuzzy Dashboard Matches**: Initial UI replacements for Player 2 rows caused syntax errors due to duplicate lines; resolved via targeted surgical edits.
- **Regex Backreferences**: Switched `\1` to `\g<1>` for numerical config safety.

## ⏭️ Next Steps
1.  **v3 Optimization**: Run a short Optuna study (20 trials, 75K steps) specifically for the `v3` environment (`python src/scripts/tune.py --algo ppo --env v3`).
2.  **Entropy Monitoring**: Verify that `v3` entropy loss reaches -6.50 or below within the first 200K steps.
3.  **Production Migration**: Once hyperparameters are tuned, launch the first `v3` production run against the baseline curriculum.
