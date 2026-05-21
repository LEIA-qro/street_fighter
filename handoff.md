# Project Handoff: Street Fighter II RL (PBT Scaling & Stability)

## 📝 Project Summary
This project trains a Ryu specialist for Street Fighter II' SCE using a custom TCP bridge between SB3 and BizHawk. The current focus is a high-throughput Population Based Training (PBT/PB2) pipeline using Ray Tune to automate hyperparameter scheduling (LR, Entropy, Clip Range).

## 🎯 Goal
**Maximize PBT Throughput and Scaling Stability.**
Enable training of large populations (12+ agents) using "Time-Multiplexing" (Synchronous PBT) and multiple emulator instances per agent to accelerate exploration without overwhelming system resources.

## 🚀 Current State
The PBT architecture is now robust and high-throughput:
1.  **Synchronous PBT (Time-Multiplexing)**: Enabled `synch=True` in PB2. When `population > max_concurrent`, Ray Tune now pauses active trials at the exploit milestone to let pending trials run. This prevents agent starvation.
2.  **Multi-Env per Agent**: PBT workers now support `envs_per_worker > 1`. Each worker uses a `DummyVecEnv` to manage multiple parallel BizHawk instances, significantly increasing sample throughput per Python process.
3.  **Regex-Based Rank Safety**: Fixed a critical "Rank Theft" bug where cloned agents copied ports from their donors. Ranks are now derived from the unique `trial_id` suffix via Regex, ensuring immutable port assignments (`rank * 10 + i`).
4.  **TensorBoard Unified Logging**: The dashboard's "Launch TensorBoard" now uses `--logdir_spec` to monitor both standard `logs/` and PBT tuning directories (`models/tuning/pbt`).
5.  **Dashboard UI**: Added "Envs per Worker" slider and updated PBT logic to pass throughput parameters to the orchestrator.

## 📂 Files Actively Edited
- `src/agents/pbt/pbt_orchestrator.py`: Synchronous PB2 logic, Regex-rank derivation, and Multi-Env support.
- `src/scripts/train_pbt.py`: CLI support for `--envs_per_worker`.
- `src/scripts/web_dashboard.py`: Unified TensorBoard logging and PBT throughput UI.

## ❌ Failed Attempts (Summarized)
- **Asynchronous PBT Scaling**: Using `max_concurrent < population` in async mode caused "Starvation"; trials never relinquished slots, preventing pending trials from ever starting. (Fixed by `synch=True`).
- **Config-Based Rank Storage**: Storing `rank` in the config dictionary caused port collisions during PBT "Exploit" phases because cloning copied the port assignments. (Fixed by Trial ID derivation).
- **String-Split Rank Extraction**: Simple `split("_")[-1]` logic was unreliable during cloning; replaced with robust Regex extraction.
- **Global `taskkill`**: Caused cascade failures in parallel mode. (Fixed by isolated PID-based `failsafe_env`).

## ⏭️ Next Steps
1.  **Time-Multiplexing Validation**: Verify that when Trial 0 hits 50k steps, it suspends and allows Trial 5 (Pending) to start.
2.  **Hardware Profiling**: Test the limit of `envs_per_worker`. (e.g., 4 concurrent agents x 4 envs each = 16 BizHawk instances).
3.  **PBT Phase Transitions**: Observe if synchronous PBT correctly synchronizes 12 agents before performing the first global PB2 exploit/explore step.
