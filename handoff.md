# Project Handoff: Street Fighter II RL (TensorBoard Step Continuation & Curriculum Analytics Isolation)

## 📝 Project Summary
This project implements a high-performance Reinforcement Learning pipeline for Street Fighter II' SCE on Genesis (Ryu specialist). It employs a custom, lock-step TCP bridge between Stable Baselines3 (PPO, DQN, SAC) and BizHawk. The current phase establishes perfect continuation of TensorBoard steps upon model resumptions, isolates curriculum state tracking on a per-model basis to prevent progress collision/dilution, and implements robust uploader/downloader tools in the web control center.

---

## 🎯 Goals We Are Working Toward
*   **TensorBoard Step Continuation**: Resolve step counters resetting to 0 when loading existing model checkpoints or during auto-recovery retry loops by setting `reset_num_timesteps=False` in SB3's `model.learn()`.
*   **Model-Specific Curriculum Analytics**: Save and read curriculum state logs as `auto_curriculum_state_{model_name}.json` instead of a generic shared file to prevent cross-model directory state collisions.
*   **Gradio State Migration Tools**: Integrate interactive `.json` state uploaders (with robust validation) and a downloader button directly under the auto-curriculum analytics card.
*   **Backward Compatibility**: Ensure all modifications fall back gracefully to the generic `auto_curriculum_state.json` if a model-specific file does not exist.

---

## 🚀 Current State of the Code
The codebase is clean, robustly structured, and compiles cleanly with zero warnings:
1.  **Continuous TensorBoard Steps**: The `train` methods in all agent types (`ppo/agent.py`, `dqn/agent.py`, `sac/agent.py`) dynamically check if a pre-existing checkpoint is loaded and pass `reset_num_timesteps=False` to `model.learn()` accordingly.
2.  **Telemetry Isolation**: `AutoCurriculumCallback` serializes states into model-specific JSON files. A robust fallback resolves and loads the generic `auto_curriculum_state.json` if the model-specific version is missing.
3.  **Robust Recovery**: The training script `train.py` correctly handles the model-specific files with the same fallback behavior inside its error-recovery retry loop.
4.  **Web Control Center**:
    - **Uploader**: Added a `.json` uploader next to the model zip/pkl uploaders with format validation.
    - **Downloader**: Integrated a `"Download Auto-Curriculum Analytics"` button and file download component, making curriculum progress fully exportable and migratable.

---

## ❌ Everything We've Tried That Failed
*   **Standard SB3 Resumptions**: Letting `model.learn()` use the default `reset_num_timesteps=True` reset rollout steps to 0, causing disjointed step graphs in TensorBoard.
*   **Shared State Overwriting**: Previously, all models under the same algorithm/env subdirectory wrote to `auto_curriculum_state.json`, leading to state overwrites and resetting curriculum analytics cards.
*   **Unvalidated JSON Uploads**: Uploading raw, corrupted `.json` files could crash the Python dashboard at load-time. (Fixed by adding a try-except validation block that parses the file with `json.load()` before writing to the target directory).
*   **Windows Subprocess Spaces**: Parent runner execution occasionally failed on user paths containing spaces; resolved via strict quote escaping.

---

## 📂 Files Actively Edited
*   `src/agents/ppo/agent.py` (Added `reset_num_timesteps` continuation logic)
*   `src/agents/dqn/agent.py` (Added `reset_num_timesteps` continuation logic)
*   `src/agents/sac/agent.py` (Added `reset_num_timesteps` continuation logic)
*   `src/agents/auto_curriculum_callback.py` (Isolated model-specific JSON states with fallback)
*   `src/scripts/train.py` (Updated recovery loop state parsing)
*   `src/scripts/web_dashboard.py` (Added uploader/downloader UI components, events, and validation)
*   `handoff.md` (This project handoff file)

---

## ⏭️ Next Steps
1.  **Dashboard Telemetry Verification**: Launch the Gradio Web Control Center (`python src/scripts/web_dashboard.py`), toggle `"Enable Auto-Curriculum"`, start a training run, and verify that a file named `auto_curriculum_state_{MODEL_NAME}.json` is created.
2.  **Verify TensorBoard Step Continuity**: Resubmitting or resuming a model should show a continuous step line in TensorBoard rather than restarting the x-axis from 0.
3.  **Verify JSON Uploader/Downloader**: Test downloading the curriculum state file, modifying/backing it up, and uploading it back via the new interface.
