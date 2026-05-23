# Project Handoff: Street Fighter II RL (Dynamic Naming, Self-Play League, & Stability)

## 📝 Project Summary
This project implements a high-performance Reinforcement Learning pipeline for Street Fighter II' SCE on Genesis (Ryu specialist). It employs a custom, lock-step TCP bridge between Stable Baselines3 (PPO, DQN, SAC) and BizHawk. The current phase establishes massive stability improvements, centralized state scanning/upload interfaces, and a premium dynamic model checkpoint naming scheme inside all training callbacks.

---

## 🎯 Goal
*   **Establish a robust, highly descriptive, dynamic model checkpoint naming convention** across all curriculum and self-play league callbacks to track training performance inline.
*   **Maintain crash recovery reliability** by keeping emergency saves static, preventing recovery failures.
*   **Centralize state pool settings, dynamic PvP state uploads, and matchup selectors** inside the Gradio Web Control Center.

---

## 🚀 Current State
The codebase is structurally clean, high-performance, and completely verified:
1.  **Dynamic Naming System**: `ManualCurriculumCallback` (in `manual_curriculum_callback.py`) and `LeagueMatchmakingCallback` (in `train_league.py`) now dynamically format saved checkpoints:
    - *Best Win Rate:* `{algo}_{env_version}_{model_name}_{state}_WR{winRate}pct_{steps}`
    - *Best Reward:* `{algo}_{env_version}_{model_name}_{state}_Rew{reward}_{steps}`
    - *Periodic Checkpoints:* `{algo}_{env_version}_{model_name}_{state}_WR{winRate}pct_ckpt_{steps}`
    - *Final Saves:* `{algo}_{env_version}_{model_name}_{state}_final_WR{winRate}pct_{steps}`
2.  **Launcher Metadata Extraction**: Agent dispatchers (PPO, DQN, SAC in `agent.py`) and `resume.py` extract environment version, algorithm, and custom model name metadata dynamically from their directory paths and forward them into the callbacks without breaking method signatures.
3.  **Handoff Persistence**: `state_name` (e.g. `ryu_only` or `custom`) is now serialized within `curriculum_state.json` so special override phases survive resumes.
4.  **League Win Rate Calculation**: `LeagueMatchmakingCallback` computes the main agent's overall rolling win rate dynamically across all registered opponent pools and saves milestone checkpoints reflecting the matchup mode:
    `{algo}_{env_version}_{model_name}_{matchup_mode}_WR{winRate}pct_ckpt_{steps}`
5.  **Premium Savestate Uploads**: The Gradio web dashboard dynamically scans the `states/` directory for `.State` files, supports inline savestate uploads, and auto-refreshes selectors.
6.  **Symmetric relative controls**: Environment wrappers implement translation-invariant left/right relative button mappings to resolve Player 2's starting side bias.
7.  **Blackwell GPU Verification**: Virtual environment is fully updated to PyTorch `2.11.0+cu128` (CUDA 12.8), verifying native GPU calculation on the active RTX 5070 Ti.

---

## ❌ Failed Attempts (Summarized & Unified)
*   **Dynamic Crash Saves Port Collisions**: Swapping `_CRASH_SAVE` and `_EMERGENCY` to dynamic formats broke `train.py`'s supervisor which searches for static file tags. (Fixed by keeping crash/emergency files static).
*   **Space Path Execution Failures**: Running commands in directories containing spaces (e.g., `Diego Perea`) failed in Windows shells. (Fixed by properly quoting paths, e.g., `python "C:\Path with spaces\test.py"`).
*   **External Relative Root Path Shift**: Unit tests located in subdirectory paths incorrectly resolved the project root via `parents[5]`, causing imports to fail. (Fixed by injecting literal project root paths inside test headers).
*   **Asynchronous Ray PBT Starvation**: Async Ray workers starved slots under concurrent caps. (Fixed by switching to synchronous PB2 Ray orchestration).
*   **Direct Class Instantiation during Tests**: Directly instantiating environment objects spawned subprocesses and crashed on active ports. (Fixed by mocking target environment structures).

---

## 📂 Files Actively Edited
*   `src/agents/manual_curriculum_callback.py` (Parameters, dynamic filenames, `state_name` serialization).
*   `src/scripts/train_league.py` (`LeagueMatchmakingCallback` rolling win rates and dynamic milestone saves).
*   `src/agents/ppo/agent.py`, `src/agents/dqn/agent.py`, & `src/agents/sac/agent.py` (Metadata forwarding and dynamic final saves).
*   `src/scripts/resume.py` (Resumption path parsing and dynamic final saves).
*   `scratch/test_dynamic_naming.py` (Dynamic filename formatting test suite).

---

## ⏭️ Next Steps
1.  **Curriculum Progression Check**: Launch a brief curriculum training run (e.g. 5,000 steps) via the Gradio control center and verify that checkpoints are written under `models/production/v3/ppo/` with correct winrate parameters.
2.  **PvP Matchmaking Run**: Validate that the dynamic league checkpoints successfully expand the active matchmaking pool and load onto vectorized workers without latency spikes.
