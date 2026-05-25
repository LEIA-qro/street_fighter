# Project Handoff: Street Fighter II RL (Documentation Overhaul & Matchup Process Failsafes)

## 📝 Project Summary
This project implements a high-performance Reinforcement Learning pipeline for Street Fighter II' SCE on Genesis (Ryu specialist). It employs a custom, lock-step TCP bridge between Stable Baselines3 (PPO, DQN, SAC) and BizHawk. The current phase establishes a complete production-grade documentation overhaul, rigorous algorithmic/systems justifications, and absolute process containment to eliminate zombie grandchild emulators during interactive matchup testing.

---

## 🎯 Goal
*   **Production-Grade Documentation Overhaul:** Overwrite the root `README.md` (master specify and developer manual) and `doc/README.md` (algorithmic/systems justification guide) to align perfectly with the active codebase, completely omitting theoretical concepts (such as `eggroll`).
*   **Orphaned Process Containment:** Investigate and resolve the process leak where clicking "Terminate Match" in the Gradio dashboard leaves orphaned `EmuHawk.exe` grandchild processes spinning in the background.

---

## 🚀 Current State
The codebase is clean, high-performance, and fully containment-secured:
1.  **Overhauled `README.md` (Master Spec):** Up-to-date directory layout map, exhaustive **Gradio Web Control Center** configuration/running guide (including live curriculum HTML visualizer, model uploader, and matchup testers), dependencies setup, and comprehensive CLI execution commands (`train.py`, `resume.py`, `tune.py`, `test_agent_v2.py`, `test_ai_vs_ai_v2.py`).
2.  **Overhauled `doc/README.md` (Justification Guide):** Deep-dive mathematical and bare-metal systems justifications:
    - *Lock-Step TCP Bridge:* Deterministic synchronization protocol, 10ms Lua timeouts, 5.0s Python socket timeouts, and stream buffer slicing to prevent partial packet corruptions.
    - *Motorola 68000 WRAM Mapping:* Big-Endian memory reads (`read_u16_be`) and critical **Data Leakage Defenses** (excluding player button inputs `0x81E2` from P1 observations to prevent policy network identity loops).
    - *Observation Stack:* 2216-dim observation space ($554 \times 4$ frames) and HP safety clamps.
    - *Category-Weighted Rehearsal Lottery:* Mathematical proof showing the weighted lottery (`past: 12`, `mastered: 24`, `active: 36`, `weakness: 60`, `new: 48`) yields a stable **41.7% active weakness selection probability** at Level 2 to target combat bottlenecks.
    - *Gating Thresholds:* Statistical gating (`min_samples_per_state = 15`, `min_episodes_for_eval = 100`, `stability_threshold = 3`) to prevent promotion due to statistical noise.
    - *Interactive Performance:* CPU unthrottling, audio disabling, display VSync turning off, and throttled disk operations to once every 30 frames inside matchup testing to eliminate micro-stutters.
3.  **Matchup Termination process sniper:** Fixed the emulator leak by updating `stop_active_process()` inside `src/scripts/web_dashboard.py`. The supervisor dashboard now **always triggers the project process sniper (`failsafe_env()`)** synchronously at the end of process terminations. This guarantees that all orphaned `EmuHawk.exe` grandchild instances are swept and terminated, even if the parent test process exits abruptly.

---

## ❌ Failed Attempts (Summarized & Updated)
*   **Lua Socket Reads in PAUSE Mode:** Attempting to exit BizHawk gracefully by sending an `"EXIT"` socket command failed because `match_test_env_client.lua` completely bypasses socket checks when the matchup is paused to allow manual human menu navigation. (Fixed by triggering the dashboard process sniper failsafe).
*   **SIGBREAK Cleanup Bypass:** Terminating parent runner processes via Windows console events sometimes exits Python abruptly before executing `finally: env.close()` blocks, leaving the emulator orphaned. (Fixed by direct dashboard process sniper calls).
*   **Dynamic Crash Saves Port Collisions:** Swapping `_CRASH_SAVE` and `_EMERGENCY` to dynamic formats broke supervisor scripts that search for static tags. (Fixed by keeping crash/emergency files static).
*   **Space Path Execution Failures:** Running scripts in directories containing spaces (e.g. `Diego Perea`) failed in Windows shells. (Fixed by quoting paths).

---

## 📂 Files Actively Edited
*   `README.md` (Overhauled master developer manual)
*   `doc/README.md` (Overhauled algorithmic and systems justification guide)
*   `src/scripts/web_dashboard.py` (Overhauled matchup termination process sniper failsafe)
*   `handoff.md` (This project handoff file)

---

## ⏭️ Next Steps
1.  **Matchup Process Containment Verification:** Launch the Gradio Web Control Center (`python src/scripts/web_dashboard.py`), open the Matchup Test tab, start a matchup, and then click "Terminate Match" to visually verify that EmuHawk exits cleanly and CPU/VRAM usage drops back to idle immediately.
2.  **Advanced Projectile Vector Tracking:** Extend the Gymnasium observation space wrapper in `envs/sf2_v3.py` to refine projectile state mapping (e.g. tracking vector trajectories) to further improve Ryu's defensive play against projectile spammers.
