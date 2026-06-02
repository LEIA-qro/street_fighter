# Street Fighter II Reinforcement Learning (RL) Pipeline

[![Python 3.13.12](https://img.shields.io/badge/python-3.13.12-blue.svg)](https://www.python.org/downloads/release/python-31312/)
[![Lua 5.4.6](https://img.shields.io/badge/lua-5.4.6-orange.svg)](https://www.lua.org/)
[![Emulator BizHawk 2.8](https://img.shields.io/badge/emulator-BizHawk_2.8-red.svg)](https://tasvideos.org/Bizhawk/PreviousReleaseHistory)
[![Stable-Baselines3](https://img.shields.io/badge/RL_Framework-Stable--Baselines3-green.svg)](https://stable-baselines3.readthedocs.io/)

A production-grade, highly optimized Reinforcement Learning (RL) pipeline for *Street Fighter II' - Special Champion Edition (USA)* on the Sega Genesis. Using a custom **lock-step TCP bridge** to replace the out-of-date `gym-retro` library, this system enables training autonomous specialists (e.g., Ryu) through an automated curriculum. The project features hyperparameter tuning via **Optuna**, self-play/league play modes, and a robust multi-process manager that runs up to 12 parallel emulators with automatic crash recovery.

---

## System Architecture Overview

The core innovation is a synchronized, lock-step communication bridge. Instead of relying on fragile hooks or multi-threaded timing, the system utilizes a strict **1-send / 1-receive** communication cycle.

```
┌────────────────────────────────┐                 ┌───────────────────────────────┐
│     BizHawk Emulator (Lua)     │                 │     Python RL Server (SB3)    │
│  1. Read big-endian WRAM data  │                 │  1. Receive observation vector │
│  2. Send structured string     ├─[ TCP Socket ]─>│  2. Compute policy action     │
│  3. Spinlock / Block execution │                 │  3. Format & send input keys  │
│  4. Read socket & inject keys  │<─[ TCP Socket ]─┤  4. Rollout updates / SGD     │
└────────────────────────────────┘                 └───────────────────────────────┘
```

Every frame step is deterministic. The emulator blocks until Python transmits the controller state, preventing data drift or skipped frames during heavy backpropagation spikes.

---

## 📂 Modern Directory Layout

The codebase enforces a **strict isolation architecture** to decouple the emulator interface, gymnasium environment wrappers, agent logic, and scripting runners.

```
street_fighter/
├── roms/                    # Read-only. Sega Genesis game ROMs (never modify)
├── states/                  # Savestates (.State files) mapped to curriculum difficulty levels 1-8
├── logs/                    # Training metrics, tensorboard run files, and tuning summaries
├── models/                  # Production milestones and checkpoints (.zip and vecnorm .pkl)
│   └── production/          # Model checkpoint directory structured by [env_version]/[algo]/
├── lua/                     # Lua scripting engine matching BizHawk environment
│   └── v2.0/                # Active lock-step client loops
│       ├── training_env_client.lua   # Headless emulator loop optimized for training
│       ├── match_test_env_client.lua  # Interactive loop with throttled disk I/O for testing
│       └── generated_config.lua      # Configured dynamically by Python (do not edit)
└── src/                     # Core Python codebase
    ├── core/                # Emulator lifecycle, socket bridge, selective normalizer, and global config
    │   ├── config.py             # Active directory definitions, states list, and hyperparameters
    │   ├── bizhawk_base.py       # Bare-metal TCP server socket and payload parsing
    │   ├── env_tools.py          # Process snipers, clean teardown registration, and VRAM purging
    │   └── selective_norm.py     # Custom running statistics normalizer for observations
    ├── envs/                # Gymnasium environment implementations (Zero RL knowledge)
    │   ├── base_env.py           # Base Gym wrapper mapping observations and rewards
    │   ├── sf2_v1.py             # Early experimental observation spaces
    │   ├── sf2_v2.py             # Enhanced 554-dim/stacked reward tracking
    │   └── sf2_v3.py             # Fully-integrated current Gymnasium environment
    ├── agents/              # Core RL algorithms and callback systems
    │   ├── ppo/                  # PPO-specific modular agent logic
    │   ├── sac/                  # SAC-specific modular agent logic
    │   ├── dqn/                  # DQN-specific modular agent logic
    │   ├── base_agent.py         # Abstract base class enforcing Train/Resume/Tune/Test interface
    │   ├── manual_curriculum_callback.py  # Manual training progression checkpoints
    │   └── auto_curriculum_callback.py    # Rehearsal lottery, gating, and micro-step evaluator
    └── scripts/             # Execution CLI and Gradio Web Dashboard
        ├── train.py              # CLI training initializer
        ├── resume.py             # CLI model resuming and curriculum recoverer
        ├── tune.py               # CLI Optuna optimization runner
        ├── test_agent_v2.py      # Interactive matchup tester (P1 agent vs CPU or Human)
        ├── test_ai_vs_ai_v2.py   # Agent vs Agent matchup battles
        └── web_dashboard.py      # Gradio Web Control Center (Visual management suite)
```

---

## 🖥️ Gradio Web Control Center

The project features a premium, centralized control dashboard built on Gradio. This web interface consolidates training execution, real-time logging, hyperparameter overrides, curriculum monitoring, matchup testing, and model uploads into a unified, visual application.

### Launching the Dashboard

Ensure your virtual environment is active, then execute:

```powershell
python src/scripts/web_dashboard.py
```

The server will initialize on **`http://127.0.0.1:7860`**. Open this URL in any web browser.

### Key Features and Panels

| Panel Tab | Description | Features Included |
| :--- | :--- | :--- |
| **🚀 Single Training & Tuning** | Configure, initialize, and optimize training models or Optuna trials. | - Swap algorithms dynamically (PPO, SAC, DQN)<br>- Toggle **Auto-Curriculum Learning**<br>- Adjust device, learning rates, entropy, and timesteps |
| **📊 League & PBT Training** | Conduct multi-agent training pipelines (self-play). | - Population-Based Training controls<br>- League pool manager and matchup mode selectors |
| **⚔️ Matchup Testing** | Evaluate trained models in real-time inside BizHawk. | - Load custom `.zip` and `.pkl` configurations for Player 1 and Player 2<br>- Select Player 1 Mode (Agent vs CPU, Agent vs Human, or AI vs AI)<br>- Real-time agent status toggle and performance profiling |
| **📈 Live Curriculum Status** | Graphical status card updated every 5 seconds. | - Table showing all 96 states across levels 1-8<br>- Live rolling win rate and episode count per state<br>- Visual highlight of current Level, Active States, and introduced Weakness states |
| **⚙️ Global Config** | Centralized server and emulator optimization controls. | - Adjust the number of parallel BizHawk instances (`N_ENVS`) <br>- Toggle Emulator Input Display and Visual Rendering overlays |
| **📥 Model Uploader** | Drag-and-drop system for model integration. | - Automatically parses file extensions (`.zip` for networks, `.pkl` for normalizers)<br>- Saves files under correct paths and automatically auto-selects them in active selectors |

---

## Prerequisites & Installation

This project is built to run natively on **Windows 10/11** using **BizHawk 2.8**.

### 1. System Requirements & Emulators
*   **Python:** Install [Python 3.13.12](https://www.python.org/downloads/release/python-31312/).
*   **BizHawk Emulator:** Download [BizHawk 2.8](https://github.com/TASEmulators/BizHawk/releases/tag/2.8). Extract the emulator to your local filesystem.
*   **ROM File:** Obtain *Street Fighter II' - Special Champion Edition (USA)* in Genesis ROM format (`.md` or `.bin`). Name the file precisely `Street Fighter II' - Special Champion Edition (USA).md` and place it in the `roms/` directory.

### 2. Isolated Workspace Setup
For optimal organization, **clone or move this `street_fighter` project directory inside your main BizHawk folder**. This allows the scripts to dynamically map relative directories safely.

### 3. Virtual Environment (venv) Setup
Initialize the isolated environment from the `street_fighter/` root directory:

```powershell
# Create the environment
python -m venv .venv

# Activate the environment
.venv\Scripts\Activate.ps1
```

Confirm that your terminal prompt is now prefixed with `(.venv)`.

### 4. Dependency Installation
Install all production requirements directly:

```powershell
pip install stable_baselines3 gymnasium torch optuna numpy pandas tensorboard gradio
```

Verify GPU availability for PyTorch training:

```powershell
python verify_gpu.py
```

---

## Developer CLI Guide

For scripting, automation, or remote environments, all systems can be operated directly through terminal commands.

### 1. Launch Training from Scratch
To start a new Ryu training specialist using standard hyperparameter configurations:

```powershell
python src/scripts/train.py --algo ppo --model_name ryu_specialist --timesteps 10000000 --device cuda
```

Add `--auto_curriculum` to enable the dynamic rehearsal lottery and gating callback:

```powershell
python src/scripts/train.py --algo ppo --model_name ryu_specialist --timesteps 10000000 --device cuda --auto_curriculum
```

### 2. Resume Previous Training Run
Resume training a saved model, safely picking up checkpoints, normalization parameters, and the active curriculum phase:

```powershell
python src/scripts/resume.py --algo ppo --model_name ryu_specialist --device cuda --auto_curriculum
```

### 3. Hyperparameter Tuning via Optuna
Run parallel Optuna trials across 12 EmuHawk instances to find the mathematically optimal policy parameters:

```powershell
python src/scripts/tune.py --algo ppo --timesteps 500000 --trials 50 --device cuda
```

### 4. Interactive Matchup Testing
Launch a model in interactive mode to test its capabilities against CPU opponents, run profiling, or play against it yourself:

```powershell
python src/scripts/test_agent_v2.py --algo ppo --load_zip models/production/v3/ppo/ppo_model.zip --load_pkl models/production/v3/ppo/ppo_model_vecnorm.pkl --player 1 --opponent_type cpu --device cuda
```
*Options:*
*   `--player 1` or `2`: Assigns the agent's controller side.
*   `--opponent_type`: `cpu` (runs the emulator's campaign AI) or `human` (binds keys to Player 2 inputs).

### 5. AI vs AI Matchup Battles
Load two separate models to fight against each other to evaluate policy strengths, exploit weaknesses, or test versions:

```powershell
python src/scripts/test_ai_vs_ai_v2.py --p1_algo ppo --p1_zip models/production/v3/ppo/p1.zip --p2_algo ppo --p2_zip models/production/v3/ppo/p2.zip --device cuda
```

---

## Monitoring & Visualizing Training

Use TensorBoard to monitor rewards, actor/critic losses, explained variance, entropy, and evaluation metrics:

```powershell
tensorboard --logdir=logs/
```

Open `http://localhost:6006/` in your browser.

### Key Metrics to Track:
*   `ep_len_mean`: Average length of matches (in steps). A step represents a frame skipping block of 4.
*   `ep_rew_mean`: Rolling average of rewards. Positive trajectories signify superior spatial awareness and damage output.
*   `win_rate`: Rolling win rate (window size = 250 rounds) indicating performance against current curriculum opponents.
*   `train/approx_kl`: Divergence between the old and new policies. Stabilizing near `0.03` is optimal.

---

## Failsafes and Systems Engineering

Training multiple simultaneous instances of emulation is resource-intensive. The pipeline implements three core failsafes to guarantee stability:
1.  **Multi-Process Thread Sniper:** Terminating Python training forces a cleanup callback via `atexit`. It triggers a PowerShell command query (`Get-CimInstance Win32_Process`) that automatically sniper-terminates grandchild `EmuHawk.exe` instances matching this specific project's path, completely avoiding zombie CPU hogs.
2.  **RAM Delta Safety Clamping:** Raw RAM readings for HP are strictly clamped (threshold = 100 HP) inside `sf2_v3.py` to prevent emulator memory glitches from generating synthetic, corrupted gradient spikes.
3.  **Graceful Stopping Signal:** Writing a `.stop_training` file in the project root triggers the active callbacks to serialize model states, dump the current curriculum registry to JSON, and execute a clean process exit without corrupting rollout buffers.

---

*For detailed explanations of the mathematical formulations, category lottery pool probability derivations, Big-Endian Motorola memory address layouts, and deep-dive algorithmic selections, refer to the [Architectural & Algorithmic Justification Guide](doc/README.md).*
