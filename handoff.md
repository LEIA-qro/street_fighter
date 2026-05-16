# Project Handoff: Street Fighter II RL Pipeline

## 🎯 Goal
Build a robust, end-to-end Reinforcement Learning pipeline for Street Fighter II (Sega Genesis) using a custom Python-BizHawk TCP bridge. The system is centrally managed by a Gradio web dashboard that orchestrates Optuna tuning, PPO, SAC, and DQN production training, and AI vs AI matchmaking.

## 📈 Current State of the Code
- **Algorithms:** PPO, SAC, and DQN are fully implemented and integrated.
- **Compatibility:** SAC (Continuous) and DQN (Discrete) action spaces are now dynamically normalized to `MultiBinary(10)` via wrapping in training and inference logic.
- **Model Management:** 
    - Dropdowns are context-aware, filtering models based on the selected algorithm.
    - Tuning storage is organized per-algorithm (`models/tuning/{algo}/`).
    - Localized file uploaders are integrated into the Training and Matchup tabs for fast iteration.
- **Matchup System:**
    - Unified matchup selector: Removed manual "Match Mode" radio buttons.
    - Added "Human Player" and "CPU" options to P1/P2 algorithm selectors.
    - Dynamic routing: `run_matchup` automatically switches between `test_ai_vs_ai_v2.py` and `test_agent_v2.py`.
    - Failsafe integration: Added "Terminate Match" button in the Matchups tab.
- **Stability:** Robust safeguards added to `test_ai_vs_ai_v2.py` and `test_agent_v2.py` to prevent crashes on algorithm-model mismatches and memory issues (via `buffer_size=1` for inference).

## 📂 Files Actively Edited
- `src/scripts/web_dashboard.py`: Refactored UI logic, dynamic filtering, and button handling.
- `src/scripts/test_ai_vs_ai_v2.py`: Upgraded with robust loading/normalization logic.
- `src/scripts/test_agent_v2.py`: Upgraded with robust loading/normalization logic.
- `src/core/selective_norm.py`: Fixed Unicode encoding issues on Windows.
- `src/envs/sf2_v2.py`: Added input "ignore" protocol to support Human/CPU control.
- `lua/v2.0/match_test_env_client.lua`: Integrated player-specific input handling.

## 🛑 What Was Tried That Failed (and How It Was Fixed)
1. **Unicode Crash:** Windows console crashed when printing unicode arrows. *Fixed by using ASCII arrows.*
2. **Dashboard Syntax Errors:** Repeated syntax/naming errors during UI refactoring. *Fixed by standardizing component IDs and fixing launch syntax.*
3. **Algorithm Mismatches:** DQN models loaded as PPO crashed with cryptic `TypeError`s. *Fixed by implementing `load_model_safely` with mismatch detection.*
4. **Memory Warning:** DQN/SAC loading giant replay buffers during testing. *Fixed by setting `buffer_size=1` during inference.*
5. **Human Control Block:** Lua script blocked Human/CPU input when AI was in control. *Fixed by implementing `ignore` string protocol.*

## ⏭ Next Steps
1. **Curriculum Refinement:** Validate if reward shaping logic holds up across all curriculum phases as opponents become harder.
2. **Dashboard Performance:** If the dashboard dropdown refresh becomes slow as more models are added, implement an asynchronous file indexing cache.
3. **End-to-End Verification:** Conduct full, uninterrupted training/tuning runs for all algorithms to ensure long-term stability.
