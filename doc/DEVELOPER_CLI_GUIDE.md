# Street Fighter II RL — Developer CLI Guide

This guide provides a comprehensive reference for executing, configuring, optimizing, and evaluating the Reinforcement Learning pipeline via the command-line interface (CLI).

All commands should be executed from the `street_fighter/` project root directory with your virtual environment activated:

```powershell
# Activate the virtual environment (Windows PowerShell)
.venv\Scripts\Activate.ps1
```

---

## Table of Contents

1. [Hardware & Environment Diagnostics](#1-hardware--environment-diagnostics)
2. [Single-Agent Training (`train.py`)](#2-single-agent-training-trainpy)
3. [Crash Recovery & Supervised Resumption (`resume.py`)](#3-crash-recovery--supervised-resumption-resumepy)
4. [Hyperparameter Optimization via Optuna (`tune.py`)](#4-hyperparameter-optimization-via-optuna-tunepy)
5. [Interactive Matchup Testing (`test_agent_v2.py`)](#5-interactive-matchup-testing-test_agent_v2py)
6. [AI vs AI Matchup Battles (`test_ai_vs_ai_v2.py`)](#6-ai-vs-ai-matchup-battles-test_ai_vs_ai_v2py)
7. [Self-Play League Training (`train_league.py`)](#7-self-play-league-training-train_leaguepy)
8. [Adversarial Exploiter Training (`train_exploiter.py`)](#8-adversarial-exploiter-training-train_exploiterpy)
9. [Population-Based Training / PB2 (`train_pbt.py`)](#9-population-based-training--pb2-train_pbtpy)
10. [Gradio Web Control Center (`web_dashboard.py`)](#10-gradio-web-control-center-web_dashboardpy)
11. [Monitoring, Signals & Failsafes](#11-monitoring-signals--failsafes)

---

## 1. Hardware & Environment Diagnostics

Verify PyTorch CUDA acceleration, GPU device indexing, available VRAM, and driver compatibility before starting training:

```powershell
python code_testing/verify_gpu.py
```

### Specific GPU Selection
To target a specific NVIDIA GPU on multi-GPU systems:

```powershell
# PowerShell
$env:CUDA_VISIBLE_DEVICES = "0"; python src/scripts/train.py --algo ppo

# Windows CMD
set CUDA_VISIBLE_DEVICES=0 && python src/scripts/train.py --algo ppo
```

---

## 2. Single-Agent Training (`train.py`)

`train.py` is the primary entry point for training single agents across PPO, SAC, and DQN algorithms.

### Basic Usage

```powershell
# Train standard PPO specialist with Auto-Curriculum enabled
python src/scripts/train.py --algo ppo --env v2 --steps 10000000 --device cuda --auto_curriculum
```

### Resuming From Checkpoints via CLI

```powershell
# Resume an existing checkpoint with specific phase
python src/scripts/train.py --algo ppo --env v2 --steps 5000000 --load_zip models/production/v2/ppo/ppo_model.zip --load_pkl models/production/v2/ppo/ppo_model_vecnorm.pkl --phase 2 --auto_curriculum --device cuda
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--algo` | `str` | *Required* | Algorithm to train: `ppo`, `sac`, or `dqn`. |
| `--env` | `str` | `v2` | Environment version: `v1`, `v2`, or `v3`. |
| `--steps` | `int` | `50000000` | Target total training timesteps. |
| `--load_zip` | `str` | `None` | Path to existing model `.zip` file to load. |
| `--load_pkl` | `str` | `None` | Path to existing `VecNormalize` `.pkl` file to load. |
| `--phase` | `str` | `"0"` | Starting phase number (`"0"`, `"1"`, ...) or phase label (`RYU_ONLY`, `CUSTOM`). |
| `--device` | `str` | `auto` | Compute device (`cuda`, `cpu`, or `auto`). |
| `--auto_curriculum` | `flag` | `False` | Enables dynamic rehearsal lottery and gating callback. |
| `--lr` | `float` | `0.0` | Learning rate override (active if `> 0.0`). |
| `--ent_coef` | `float` | `0.0` | Entropy coefficient override (active if `> 0.0`). |
| `--clip_range` | `float` | `0.0` | PPO clipping range override (active if `> 0.0`). |

---

## 3. Crash Recovery & Supervised Resumption (`resume.py`)

`resume.py` runs a fully automated supervisor loop that reads model paths and curriculum state files directly from `core/config.py` (`TRAINING_ZIP_FILE`, `TRAINING_PKL_FILE`, `curriculum_state.json` / `auto_curriculum_state.json`). It automatically catches socket exceptions, recovers from `_CRASH_SAVE.zip` / `_CRASH_SAVE.pkl`, and restarts training seamlessly.

### Usage

```powershell
python src/scripts/resume.py
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--load_zip` | `str` | `config.TRAINING_ZIP_FILE` | Path to starting model `.zip` file. |
| `--load_pkl` | `str` | `config.TRAINING_PKL_FILE` | Path to starting `VecNormalize` `.pkl` file. |
| `--phase` | `str` | `None` (auto-detected) | Starting curriculum phase or level override. |
| `--device` | `str` | `auto` | Compute device (`cuda`, `cpu`, or `auto`). |

> [!TIP]
> Use `resume.py` for long unattended background runs where automated restart loops and crash-save recovery are needed. For targeted manual resumes with custom file paths, use `train.py --load_zip ... --load_pkl ...`.

---

## 4. Hyperparameter Optimization via Optuna (`tune.py`)

Executes parallel Optuna trials across multi-core EmuHawk instances to find the mathematically optimal hyperparameter configuration for the selected policy.

### Usage

```powershell
# Run 50 Optuna trials for PPO with 500,000 steps per trial
python src/scripts/tune.py --algo ppo --env v2 --trials 50 --timesteps 500000 --study_name ppo_sf2_optimization --device cuda
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--algo` | `str` | *Required* | Algorithm to tune: `ppo`, `sac`, or `dqn`. |
| `--env` | `str` | `v2` | Environment version: `v1`, `v2`, or `v3`. |
| `--trials` | `int` | `2` | Number of Optuna trials to run. |
| `--study_name`| `str` | `ppo_sf2_tuning` | Unique Optuna study name stored in logs/database. |
| `--timesteps` | `int` | `50000` | Total environment steps allocated per trial. |
| `--load_zip` | `str` | `None` | Starting model `.zip` file for fine-tuning. |
| `--load_pkl` | `str` | `None` | Starting `VecNormalize` `.pkl` file. |
| `--phase` | `str` | `"0"` | Target curriculum phase for evaluation. |
| `--device` | `str` | `auto` | Compute device (`cuda`, `cpu`, or `auto`). |

---

## 5. Interactive Matchup Testing (`test_agent_v2.py`)

Launches a single interactive BizHawk window with visual rendering to evaluate a trained model against CPU opponents or a local human player on the keyboard.

### Usage

```powershell
# Test PPO model as Player 1 against emulator CPU
python src/scripts/test_agent_v2.py --algo ppo --env v2 --load_zip models/production/v2/ppo/ppo_model.zip --load_pkl models/production/v2/ppo/ppo_model_vecnorm.pkl --player 1 --opponent_type cpu --device cuda

# Play against the trained agent as Human (Player 2 inputs)
python src/scripts/test_agent_v2.py --algo ppo --env v2 --load_zip models/production/v2/ppo/ppo_model.zip --load_pkl models/production/v2/ppo/ppo_model_vecnorm.pkl --player 1 --opponent_type human --device cuda
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--load_zip` | `str` | *Required* | Path to agent model `.zip`. |
| `--load_pkl` | `str` | *Required* | Path to agent `VecNormalize` `.pkl`. |
| `--algo` | `str` | `ppo` | Algorithm type (`ppo`, `sac`, `dqn`). Auto-detected from path if present. |
| `--env` | `str` | `v2` | Environment version: `v2` or `v3`. |
| `--player` | `int` | `1` | Controller assigned to the agent (`1` or `2`). |
| `--opponent_type` | `str` | `human` | Opponent type: `cpu` (emulator AI) or `human` (keyboard controls). |
| `--device` | `str` | `auto` | Inference device (`cuda`, `cpu`, `auto`). |
| `--profile` | `flag` | `False` | Runs `cProfile` performance analysis on step inference. |
| `--infinite_match` | `flag` | `False` | Automatically reset and start rematches on KO. |
| `--rematch_delay` | `float` | `2.0` | Delay in seconds before triggering auto-rematch. |
| `--cpu_level_cap` | `int` | `5` | Maximum CPU difficulty level cap (1-8) for infinite matchups. |

---

## 6. AI vs AI Matchup Battles (`test_ai_vs_ai_v2.py`)

Loads two separate trained models simultaneously into Player 1 and Player 2 controllers to evaluate head-to-head performance, exploit weaknesses, or benchmark model checkpoints.

### Usage

```powershell
python src/scripts/test_ai_vs_ai_v2.py `
  --algo_p1 ppo --env_p1 v2 --load_zip_p1 models/production/v2/ppo/p1_model.zip --load_pkl_p1 models/production/v2/ppo/p1_vecnorm.pkl `
  --algo_p2 dqn --env_p2 v2 --load_zip_p2 models/production/v2/dqn/p2_model.zip --load_pkl_p2 models/production/v2/dqn/p2_vecnorm.pkl `
  --device_p1 cuda --device_p2 cuda
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--load_zip_p1` | `str` | *Required* | Path to Player 1 model `.zip`. |
| `--load_pkl_p1` | `str` | *Required* | Path to Player 1 `VecNormalize` `.pkl`. |
| `--algo_p1` | `str` | `ppo` | Player 1 algorithm (`ppo`, `sac`, `dqn`). |
| `--env_p1` | `str` | `v2` | Player 1 environment version (`v2`, `v3`). |
| `--device_p1` | `str` | `auto` | Compute device for Player 1 policy inference. |
| `--load_zip_p2` | `str` | *Required* | Path to Player 2 model `.zip`. |
| `--load_pkl_p2` | `str` | *Required* | Path to Player 2 `VecNormalize` `.pkl`. |
| `--algo_p2` | `str` | `ppo` | Player 2 algorithm (`ppo`, `sac`, `dqn`). |
| `--env_p2` | `str` | `v2` | Player 2 environment version (`v2`, `v3`). |
| `--device_p2` | `str` | `auto` | Compute device for Player 2 policy inference. |
| `--profile` | `flag` | `False` | Enables `cProfile` performance telemetry. |
| `--infinite_match` | `flag` | `False` | Automatically reset and start rematches on KO. |
| `--rematch_delay` | `float` | `2.0` | Delay in seconds before triggering auto-rematch. |

---

## 7. Self-Play League Training (`train_league.py`)

Trains a Main Agent through an automated self-play league matchmaking pool. Checkpoints are automatically serialized and inserted into the opponent pool every 500,000 steps with dynamic Elo/win-rate tracking.

### Usage

```powershell
# Train Ryu vs Ryu PvP League
python src/scripts/train_league.py --env_version v2 --steps 5000000 --matchup_mode ryu_vs_ryu --model_name ryu_league_main --device cuda

# Resume existing League run
python src/scripts/train_league.py --env_version v2 --steps 5000000 --matchup_mode ryu_vs_ryu --model_name ryu_league_main --resume --device cuda

# Train Ryu against all character matchups in PvP
python src/scripts/train_league.py --env_version v2 --steps 10000000 --matchup_mode ryu_vs_all --model_name ryu_pvp_grandmaster --device cuda
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--steps` | `int` | `5000000` | Total timesteps for league self-play training. |
| `--env_version` | `str` | `v2` | Environment version: `v2` or `v3`. |
| `--matchup_mode` | `str` | `ryu_vs_ryu` | Matchmaking pool mode: `ryu_vs_ryu`, `ryu_vs_all`, or `custom`. |
| `--custom_state` | `str` | `None` | Specific `.State` file for `custom` matchup mode. |
| `--model_name` | `str` | `league` | Custom model name for checkpoints and logs. |
| `--resume` | `flag` | `False` | Resumes from existing active league model checkpoint. |
| `--device` | `str` | `auto` | Compute device (`cuda`, `cpu`, `auto`). |

---

## 8. Adversarial Exploiter Training (`train_exploiter.py`)

Trains specialized adversarial agents with shaped reward objectives against the active League model to expose and patch blind spots.

### Archetypes:
*   `rusher`: Hyper-aggressive close-range pressure and rushdown tactics.
*   `spammer`: Projectile-zoning, fireball traps, and keep-away play.
*   `turtle`: Defensive counter-poking, baiting, and high block ratios.

### Usage

```powershell
# Train a close-range Rushdown exploiter against the active league model
python src/scripts/train_exploiter.py --type rusher --env_version v2 --steps 1000000 --model_name ryu_league_main --device cuda

# Train a Projectile Spammer exploiter
python src/scripts/train_exploiter.py --type spammer --env_version v2 --steps 1000000 --model_name ryu_league_main --device cuda
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--type` | `str` | `rusher` | Specialized archetype: `rusher`, `spammer`, or `turtle`. |
| `--steps` | `int` | `1000000` | Training timesteps for the exploiter. |
| `--env_version` | `str` | `v2` | Environment version: `v2` or `v3`. |
| `--matchup_mode` | `str` | `ryu_vs_ryu` | Matchup mode: `ryu_vs_ryu`, `ryu_vs_all`, or `custom`. |
| `--custom_state` | `str` | `None` | Custom fight state filename. |
| `--model_name` | `str` | `league` | Name of the active target league model to exploit. |
| `--device` | `str` | `auto` | Compute device (`cuda`, `cpu`, `auto`). |

---

## 9. Population-Based Training / PB2 (`train_pbt.py`)

Implements Population-Based Training using Population-Based Bandits (PB2) with Gaussian Process regression to dynamically adapt learning rates, entropy coefficients, and clip ranges throughout training.

### Usage

> [!NOTE]
> Population-Based Training uses Ray Tune. Install the optional dependency before running:
> ```powershell
> pip install "ray[tune]"
> ```

```powershell
# Run PBT with 10 agents, 1 environment per worker, exploring every 500k steps
python src/scripts/train_pbt.py --algo ppo --env v2 --population 10 --steps 5000000 --steps_per_exploit 500000 --model_name PBT_PPO_v2
```

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--algo` | `str` | `ppo` | RL algorithm (`ppo`, `sac`, `dqn`). |
| `--env` | `str` | `v2` | Environment version (`v1`, `v2`, `v3`). |
| `--model_name` | `str` | `PBT_BEST_model`| Base output name for the optimal survivor model. |
| `--load_zip` | `str` | `None` | Base model `.zip` checkpoint to seed population. |
| `--load_pkl` | `str` | `None` | Base `VecNormalize` `.pkl` checkpoint. |
| `--phase` | `str` | `"0"` | Curriculum starting phase. |
| `--steps` | `int` | `5000000` | Total training timesteps across all generations. |
| `--steps_per_exploit` | `int` | `500000` | Timesteps per exploitation/mutation generation. |
| `--population` | `int` | `10` | Population size (must be `>= 4` for PB2 Gaussian Process). |
| `--max_concurrent` | `int` | `population` | Maximum concurrent worker trials. |
| `--envs_per_worker` | `int` | `1` | BizHawk environments allocated per PBT agent. |
| `--resume` | `flag` | `False` | Resumes an existing PBT run. |

---

## 10. Gradio Web Control Center (`web_dashboard.py`)

Launches the visual web management suite containing training controls, real-time telemetry, model uploaders, live curriculum tables, and match testers.

### Usage

```powershell
python src/scripts/web_dashboard.py
```

Access the UI in your web browser at: **`http://127.0.0.1:7860`**

### Parameter Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--host`, `--server_name` | `str` | `0.0.0.0` | Host IP to bind server to. |
| `--port`, `--server_port` | `int` | `7860` | Network port for dashboard UI. |
| `--share` | `flag` | `False` | Generates a temporary public Gradio URL. |

---

## 11. Monitoring, Signals & Failsafes

### TensorBoard Visual Metrics
Launch TensorBoard to track reward curves, policy/value losses, entropy, episode length, and win rates in real time:

```powershell
tensorboard --logdir=logs/
```

Access the dashboard at: **`http://localhost:6006/`**

### Graceful Stop Signal (`.stop_training`)
To gracefully stop an active training run without corrupting model weights or curriculum progression state, create a `.stop_training` file in the project root:

```powershell
# PowerShell
New-Item -ItemType File -Name ".stop_training" -Force
```

The running callback will detect this trigger file, serialize current model weights, export the curriculum progress JSON, remove the signal file, and exit cleanly.

### Automatic Zombie Process Cleanup
Every script registers the `failsafe_env()` utility in its `finally:` block. If training is interrupted (e.g. `Ctrl + C` or terminal closure), `failsafe_env()` executes a PowerShell process query to terminate orphaned `EmuHawk.exe` instances and purges PyTorch CUDA VRAM cache:

```python
from core.env_tools import failsafe_env
failsafe_env(env=env, model=model)
```
