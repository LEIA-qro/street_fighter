# Project Handoff: Street Fighter II RL

## 🎯 Current Goal
The objective is to train a Ryu specialist capable of defeating all opponents on the hardest difficulty using a custom RL pipeline (PPO/SAC/DQN) built on a lock-step TCP bridge between Python and BizHawk (Lua). We are currently hardening the "v2.0" environment, which features a 555-dimensional hybrid observation space (continuous RAM + One-Hot Actions/Characters) and a manual curriculum.

## 📊 Current State of the Code
- **Hardware Integration:** Stable. Staggered booting is implemented to prevent CPU/IO bottlenecks during parallel initialization of 10+ emulators.
- **Environment (v2.0):** 
    - Observations include universal relative distance (RAM 0x834C) to fix character-specific hitbox discrepancies.
    - Velocity clipping widened to `[-100, 100]` to capture high-speed movement without signal saturation.
    - Self-healing `reset()` logic implemented to automatically respawn crashed emulator ranks without killing the whole training session.
- **Web Dashboard:** Updated to Gradio 6.0 standards. Features a dedicated "Copy Logs" button (JS-based) and a Compute Device selector (`auto`, `cpu`, `cuda`).
- **Optuna Tuning:** Bulletproofed. Crashed trials are now marked as `FAIL` (preserving params but ignoring them in the math) rather than being penalized with bad scores.
- **Performance:** `auto` device logic implemented (PPO on CPU for speed, SAC/DQN on CUDA for heavy sampling).

## 🛠 Active Files
- `src/envs/sf2_v2.py`: Core RL environment and reward logic.
- `src/core/env_tools.py`: Boot orchestration and staggered initialization.
- `src/core/bizhawk_base.py`: Low-level TCP socket bridge.
- `src/scripts/web_dashboard.py`: Gradio control center.
- `lua/v2.0/training_env_client.lua`: Headless training loop.
- `src/agents/*/agent.py`: Algorithm implementations and parameter propagation.

## ❌ What Failed / Was Fixed
- **Penalizing Crashed Trials:** Returning `-99999.0` for crashed Optuna trials was poisoning the sampler. Fixed by raising exceptions and using Optuna's native `FAIL` state.
- **Parallel Boot Deadlock:** Launching 10 emulators at once caused `[WinError 10054]` resets. Fixed by increasing Lua timeouts to 600s and staggering Python-side boots (3.5s delay per rank).
- **Gradio 6.0 Warnings:** `theme`/`css` moved from constructor to `launch()`, and `show_copy_button` was removed. Fixed via refactoring and custom JS button.
- **Incorrect RAM Width:** Universal distance was initially read as `u16_be`, which picked up garbage from adjacent bytes. Fixed by switching to `u8`.

## ⏭ Next Steps
1. **Validation Run:** Start a fresh training session using the `auto` device setting via the dashboard to verify the PPO-on-CPU performance speedup.
2. **Fresh Models:** Since the observation dimension changed from 554 to 555, all existing `.zip` models are incompatible. A new "Phase 0" baseline needs to be trained.
3. **Reward Tuning:** Monitor the new exponentially decaying footsie reward (`FOOTSIE_DECAY_RATE`). Ensure it effectively prevents "camping" without discouraging safe spacing.
4. **Interactive Testing:** Use the "AI vs AI" tab to verify the perspective-isolated parsers correctly handle the new 13-item payload for both players.
