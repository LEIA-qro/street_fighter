import gradio as gr
import os
import subprocess
import threading
import sys
import re
import importlib
import glob
import webbrowser
import time
import signal
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parents[1]))
from core import config

# Virtual Environment Detection / Setup
VENV_PYTHON = os.path.join(config.PROJECT_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = os.path.join(config.PROJECT_ROOT, ".venv", "bin", "python")
    if not os.path.exists(VENV_PYTHON):
        VENV_PYTHON = sys.executable 

# Global state for background processes
class GlobalState:
    active_process = None
    stop_event = threading.Event()

state = GlobalState()

# --- Utility Functions ---

def refresh_dropdowns():
    z, p = get_model_files()
    upd_z = gr.update(choices=z, value="None")
    upd_p = gr.update(choices=p, value="None")
    return upd_z, upd_p, upd_z, upd_p, upd_z, upd_p, upd_z, upd_p, upd_z, upd_p

def load_hyperparams_from_json(file_path):
    if file_path is None:
        return 0.0, 0.0, 0.0, {}
    import json
    try:
        with open(file_path.name if hasattr(file_path, "name") else file_path, "r") as f:
            data = json.load(f)
        
        lr = data.pop("lr", 0.0)
        ent = data.pop("ent_coef", 0.0)
        clip = data.pop("clip_range", 0.0)
        
        return lr, ent, clip, data
    except Exception as e:
        return 0.0, 0.0, 0.0, {"error": f"Failed to parse JSON: {e}"}

def get_model_files(algo=None):
    """Scans models directory recursively for zip and pkl files, filtering by algorithm if provided."""
    models_dir = os.path.join(config.PROJECT_ROOT, "models")
    if not os.path.exists(models_dir):
        return ["None"], ["None"]
    
    zip_files = []
    pkl_files = []
    
    if algo:
        # Search recursively for algorithm subfolders (e.g. models/production/v2/ppo/ or models/production/ppo/)
        for category in ["production", "tuning"]:
            cat_dir = os.path.join(models_dir, category)
            if os.path.exists(cat_dir):
                for root, dirs, files in os.walk(cat_dir):
                    for d in dirs:
                        if d.lower() == algo.lower():
                            target_path = os.path.join(root, d)
                            zip_files.extend(glob.glob(os.path.join(target_path, "**/*.zip"), recursive=True))
                            pkl_files.extend(glob.glob(os.path.join(target_path, "**/*.pkl"), recursive=True))
    else:
        zip_files = glob.glob(os.path.join(models_dir, "**/*.zip"), recursive=True)
        pkl_files = glob.glob(os.path.join(models_dir, "**/*.pkl"), recursive=True)
        
    # Remove duplicates and return relative paths with forward slashes
    zips = sorted(list(set([os.path.relpath(f, config.PROJECT_ROOT).replace("\\", "/") for f in zip_files])))
    pkls = sorted(list(set([os.path.relpath(f, config.PROJECT_ROOT).replace("\\", "/") for f in pkl_files])))
    
    return ["None"] + zips, ["None"] + pkls

def get_all_state_files():
    """Scans STATES_DIR for all available .State or .state files dynamically."""
    states_dir = config.STATES_DIR
    if not os.path.exists(states_dir):
        return ["None"]
    state_files = glob.glob(os.path.join(states_dir, "*.State"))
    state_files.extend(glob.glob(os.path.join(states_dir, "*.state")))
    names = sorted(list(set([os.path.basename(f) for f in state_files])))
    return ["None"] + names

def stream_logs(cmd):
    """Executes a command and yields output for Gradio."""
    if state.active_process:
        yield "Error: A process is already running!"
        return

    state.stop_event.clear()
    full_output = f"Executing: {' '.join(cmd)}\n{'-'*50}\n"
    yield full_output

    try:
        # 1. Use CREATE_NEW_PROCESS_GROUP to allow sending CTRL_C_EVENT on Windows
        # 2. shell=False is required to send signals directly to the process
        state.active_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            shell=False,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
        
        # Local reference to prevent NoneType race condition during stop
        proc = state.active_process
        
        for line in proc.stdout:
            if state.stop_event.is_set():
                break
            full_output += line
            yield full_output
            
        proc.wait()
        full_output += f"\n{'-'*50}\nProcess finished with exit code {proc.returncode}"
        yield full_output
    except Exception as e:
        yield full_output + f"\n[ERROR] {str(e)}"
    finally:
        state.active_process = None

def stop_active_process():
    """Gracefully stops the active process by writing a stop trigger file and sending CTRL_BREAK, allowing a model save."""
    if state.active_process:
        proc = state.active_process
        state.stop_event.set()
        
        # 1. Write the file-based stop trigger to the project root
        stop_file = os.path.join(config.PROJECT_ROOT, ".stop_training")
        try:
            with open(stop_file, "w") as f:
                f.write("STOP")
            print(f"[Dashboard] Graceful stop file written to {stop_file}")
        except Exception as e:
            print(f"[Dashboard] Error writing stop trigger file: {e}")

        # 2. Send CTRL_BREAK_EVENT as a backup signal (highly robust on Windows for process groups)
        try:
            print(f"[Dashboard] Sending backup Graceful Stop (CTRL_BREAK) to PID {proc.pid}...")
            os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
        except Exception as e:
            print(f"[Dashboard] Signal error (ignored): {e}")
            
        # 3. Wait up to 15 seconds for the agent to finish its EMERGENCY save logic
        for _ in range(15):
            if proc.poll() is not None:
                break
            time.sleep(1)
        
        # Clean up the stop file if it wasn't already consumed by the callback
        if os.path.exists(stop_file):
            try:
                os.remove(stop_file)
            except Exception:
                pass
            
        # 4. Final Tree-Kill Failsafe: Ensure BizHawk and Lua are definitely dead
        if proc.poll() is None:
            print(f"[Dashboard] Process {proc.pid} timed out during stop. Force killing...")
            subprocess.run(f"taskkill /F /T /PID {proc.pid}", shell=True, capture_output=True)
        else:
            print(f"[Dashboard] Process {proc.pid} gracefully stopped.")
            
        state.active_process = None
        return "🛑 Process stopped. Weights should be saved in the production folder as '_EMERGENCY.zip'."
    
    from core.env_tools import failsafe_env
    threading.Thread(target=failsafe_env).start()
    return "Global Failsafe triggered: Killing all EmuHawk instances."

def update_config_var(key, value):
    """Updates a single variable in config.py using regex."""
    config_path = os.path.join(config.SRC_DIR, "core", "config.py")
    with open(config_path, "r") as f:
        content = f.read()
    
    if isinstance(value, str) and not (value.replace('.', '', 1).isdigit() or value.lower() in ["true", "false"] or "[" in value):
        if not (value.startswith('"') or value.startswith("'")):
            formatted_value = f'"{value}"'
        else:
            formatted_value = value
    else:
        formatted_value = str(value)

    pattern = rf"^({key}\s*=\s*)(.*?)(\s*(?:#.*)?)$"
    if re.search(pattern, content, flags=re.MULTILINE):
        # Escape backslashes for the replacement string to prevent regex backreference corruption (Bug 7)
        safe_value = formatted_value.replace("\\", "\\\\")
        content = re.sub(pattern, rf"\g<1>{safe_value}\g<3>", content, flags=re.MULTILINE)
        with open(config_path, "w") as f:
            f.write(content)
        return True
    return False

# --- Dashboard Tab Handlers ---

def run_tuning(algo, env, study_name, load_zip, load_pkl, phase, timesteps, trials, device):
    # Store in models/tuning/{env}/{algo}/
    tuning_dir = os.path.join(config.PROJECT_ROOT, "models", "tuning", env, algo)
    os.makedirs(tuning_dir, exist_ok=True)
    
    cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "tune.py"), 
           "--algo", algo, "--env", env, "--study_name", study_name, 
           "--trials", str(trials), "--phase", str(phase), "--timesteps", str(timesteps),
           "--device", device]
    
    if load_zip != "None": cmd += ["--load_zip", load_zip]
    if load_pkl != "None": cmd += ["--load_pkl", load_pkl]
    
    for log in stream_logs(cmd):
        yield log

def get_best_tuning_params(algo, env, study_name):
    # Resolve storage path based on env and algorithm
    tuning_dir = os.path.join(config.get_directory()["tuning"], env, algo)
    os.makedirs(tuning_dir, exist_ok=True)
    db_path = os.path.abspath(os.path.join(tuning_dir, "study.db")).replace("\\", "/")
    json_path = os.path.abspath(os.path.join(tuning_dir, f"best_params_{study_name}.json")).replace("\\", "/")
    
    script = f"""import optuna, json
try:
    study = optuna.load_study(study_name='{study_name}', storage='sqlite:///{db_path}')
    print(f'Best Trial: {{study.best_trial.number}}')
    print(f'Value: {{study.best_value}}')
    print(f'Params: {{study.best_params}}')
    with open('{json_path}', 'w') as f:
        json.dump(study.best_params, f, indent=4)
except Exception as e:
    print(f'Error: {{e}}')"""
    
    try:
        result = subprocess.check_output([VENV_PYTHON, "-c", script], text=True, stderr=subprocess.STDOUT)
        if os.path.exists(json_path):
            return result, json_path
        return result, None
    except Exception as e:
        return f"Subprocess execution error: {e}", None

def run_training(algo, env, model_name, load_zip, load_pkl, phase, timesteps, lr, ent_coef, clip_range, device, auto_curriculum):
    update_config_var("MODEL_NAME", model_name)
    
    cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "train.py"), 
           "--algo", algo, "--env", env, "--steps", str(timesteps), "--phase", str(phase),
           "--device", device]
           
    if load_zip != "None": cmd += ["--load_zip", load_zip]
    if load_pkl != "None": cmd += ["--load_pkl", load_pkl]
    if lr > 0.0: cmd += ["--lr", str(lr)]
    if ent_coef > 0.0: cmd += ["--ent_coef", str(ent_coef)]
    if clip_range > 0.0: cmd += ["--clip_range", str(clip_range)]
    if auto_curriculum: cmd += ["--auto_curriculum"]
    
    for log in stream_logs(cmd):
        yield log

def launch_tb():
    pbt_log_dir = os.path.join(config.get_directory()["tuning"], "pbt")
    # Use logdir_spec to monitor multiple directories
    log_spec = f'logs:"{config.LOG_DIR}",pbt_tuning:"{pbt_log_dir}"'
    cmd = f'"{VENV_PYTHON}" -m tensorboard.main --logdir_spec {log_spec} --port 6006'
    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    webbrowser.open("http://localhost:6006")
    return "TensorBoard launched at http://localhost:6006"

def run_matchup(p1_algo, p1_env, p1_zip, p1_pkl, p1_device, p2_algo, p2_env, p2_zip, p2_pkl, p2_device, profile_enabled):
    ai_algos = ["ppo", "sac", "dqn"]
    p1_is_ai = p1_algo in ai_algos
    p2_is_ai = p2_algo in ai_algos

    if p1_is_ai or p2_is_ai:
        # Initialize the agent state to PAUSE for interactive control
        state_file = os.path.join(config.PROJECT_ROOT, ".agent_state")
        with open(state_file, "w") as f:
            f.write("PAUSE")

    if p1_is_ai and p2_is_ai:
        cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "test_ai_vs_ai_v2.py"),
               "--algo_p1", p1_algo, "--env_p1", p1_env, "--load_zip_p1", p1_zip, "--load_pkl_p1", p1_pkl, "--device_p1", p1_device,
               "--algo_p2", p2_algo, "--env_p2", p2_env, "--load_zip_p2", p2_zip, "--load_pkl_p2", p2_pkl, "--device_p2", p2_device]
    elif p1_is_ai:
        # P1 is AI, P2 is Player or CPU
        opp_type = "cpu" if p2_algo == "CPU (Built-in AI)" else "human"
        cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "test_agent_v2.py"),
               "--algo", p1_algo, "--env", p1_env, "--load_zip", p1_zip, "--load_pkl", p1_pkl, 
               "--player", "1", "--opponent_type", opp_type, "--device", p1_device]
    elif p2_is_ai:
        # P2 is AI, P1 is Player
        opp_type = "cpu" if p1_algo == "CPU (Built-in AI)" else "human"
        cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "test_agent_v2.py"),
               "--algo", p2_algo, "--env", p2_env, "--load_zip", p2_zip, "--load_pkl", p2_pkl, 
               "--player", "2", "--opponent_type", opp_type, "--device", p2_device]
    else:
        yield "Invalid Matchup: At least one player must be an AI model (PPO, SAC, or DQN)."
        return

    if profile_enabled:
        cmd += ["--profile"]
        
    for log in stream_logs(cmd):
        yield log

def toggle_agent_state():
    state_file = os.path.join(config.PROJECT_ROOT, ".agent_state")
    current_state = "PAUSE"
    if os.path.exists(state_file):
        with open(state_file, "r") as f:
            current_state = f.read().strip()
    
    new_state = "PLAY" if current_state == "PAUSE" else "PAUSE"
    
    try:
        with open(state_file, "w") as f:
            f.write(new_state)
        return f"Agent State: **{new_state}**"
    except Exception as e:
        return f"❌ Error toggling state: {e}"

def stop_match_process():
    log_msg = stop_active_process()
    state_file = os.path.join(config.PROJECT_ROOT, ".agent_state")
    try:
        with open(state_file, "w") as f:
            f.write("PAUSE")
    except Exception:
        pass
    return log_msg, "Agent State: **PAUSED** (Default)"

def save_all_config(n_envs, win_rate, steps, port, input_display, activate_viz):
    updates = {
        "N_ENVS": int(n_envs),
        "WIN_RATE_THRESHOLD": win_rate,
        "STARTING_TOTAL_TIMESTEPS": int(steps),
        "PORT": int(port),
        "ENABLE_INPUT_DISPLAY": input_display,
        "ACTIVATE_VISUALIZATION": activate_viz
    }
    success = True
    for k, v in updates.items():
        if not update_config_var(k, v):
            success = False
    
    if success:
        importlib.reload(config)
        gr.Info("Configuration saved and environment reloaded!")
        return "✅ Configuration saved successfully!"
    return "❌ Error: Some variables could not be found in config.py"

def update_config_list(key, new_values):
    """Updates a list variable in config.py."""
    config_path = os.path.join(config.SRC_DIR, "core", "config.py")
    with open(config_path, "r") as f:
        content = f.read()

    # Format list: ["a", "b", "c"]
    formatted_list = "[" + ", ".join([f'"{v}"' for v in new_values]) + "]"
    
    pattern = rf"^({key}\s*=\s*)(.*?)(\s*(?:#.*)?)$"
    if re.search(pattern, content, flags=re.MULTILINE):
        content = re.sub(pattern, rf"\1{formatted_list}\3", content, flags=re.MULTILINE)
        with open(config_path, "w") as f:
            f.write(content)
        return True
    return False

def handle_model_upload(file_obj, algo, env):
    if file_obj is None:
        return "Please select a file.", gr.update(), gr.update()
    try:
        import shutil
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        filename = os.path.basename(file_path)
        target_dir = os.path.join(config.PROJECT_ROOT, "models", "production", env, algo)
        os.makedirs(target_dir, exist_ok=True)
        
        target_path = os.path.join(target_dir, filename)
        shutil.copy2(file_path, target_path)
        
        # Calculate relative path of saved file
        rel_path = os.path.relpath(target_path, config.PROJECT_ROOT).replace("\\", "/")
        
        # Scan updated files lists
        z, p = get_model_files(algo)
        
        status = f"**Success:** Saved `{filename}` to `models/production/{env}/{algo}/`"
        
        # Auto-select the newly uploaded file based on its extension
        if filename.endswith(".zip"):
            return status, gr.update(choices=z, value=rel_path), gr.update(choices=p)
        elif filename.endswith(".pkl"):
            return status, gr.update(choices=z), gr.update(choices=p, value=rel_path)
            
        return status, gr.update(choices=z), gr.update(choices=p)
    except Exception as e:
        return f"**Error:** {e}", gr.update(), gr.update()

def run_pbt(algo, env, model_name, load_zip, load_pkl, phase, total_steps, exploit_steps, population, max_concurrent, resume, envs_per_worker):
    cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "train_pbt.py"), 
           "--algo", algo, "--env", env, "--model_name", model_name,
           "--steps", str(total_steps), "--population", str(population),
           "--max_concurrent", str(max_concurrent),
           "--steps_per_exploit", str(exploit_steps), "--phase", str(phase),
           "--envs_per_worker", str(envs_per_worker)]
    
    if load_zip != "None": cmd += ["--load_zip", load_zip]
    if load_pkl != "None": cmd += ["--load_pkl", load_pkl]
    if resume: cmd += ["--resume"]
    
    for log in stream_logs(cmd):
        yield log

def run_league(model_name, steps, env_version, matchup_mode, custom_state, resume, device):
    cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "train_league.py"), 
           "--model_name", model_name,
           "--steps", str(int(steps)), "--env_version", env_version, "--device", device]
           
    mode_map = {
        "Ryu vs. Ryu (Strict Self-Play)": "ryu_vs_ryu",
        "Ryu vs. All (12 Characters)": "ryu_vs_all",
        "Custom Savestate (Uploaded)": "custom"
    }
    mode_val = mode_map.get(matchup_mode, "ryu_vs_ryu")
    cmd += ["--matchup_mode", mode_val]
    
    if mode_val == "custom" and custom_state and custom_state != "None":
        cmd += ["--custom_state", custom_state]
        
    if resume:
        cmd += ["--resume"]
        
    for log in stream_logs(cmd):
        yield log

def run_exploiter(model_name, exploiter_type, steps, env_version, matchup_mode, custom_state, device):
    cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "train_exploiter.py"), 
           "--model_name", model_name,
           "--type", exploiter_type, "--steps", str(int(steps)), "--env_version", env_version, "--device", device]
           
    mode_map = {
        "Ryu vs. Ryu (Strict Self-Play)": "ryu_vs_ryu",
        "Ryu vs. All (12 Characters)": "ryu_vs_all",
        "Custom Savestate (Uploaded)": "custom"
    }
    mode_val = mode_map.get(matchup_mode, "ryu_vs_ryu")
    cmd += ["--matchup_mode", mode_val]
    
    if mode_val == "custom" and custom_state and custom_state != "None":
        cmd += ["--custom_state", custom_state]
        
    for log in stream_logs(cmd):
        yield log

def get_league_pool_status_html():
    from agents.league.pool_manager import LeaguePoolManager
    try:
        pool_manager = LeaguePoolManager()
        past_self, exploiters = pool_manager.scan_pool()
        
        n_checkpoints = len(past_self)
        n_exploiters = len(exploiters)
        
        html = f"""
        <div style='background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); padding: 24px; font-family: system-ui, -apple-system, sans-serif; color: #fff;'>
            <h3 style='margin-top: 0; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; font-size: 1.35rem; font-weight: 600; color: #3b82f6;'>
                🏆 League Pool Status & Analytics
            </h3>
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;'>
                <div style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 16px; text-align: center;'>
                    <div style='font-size: 1.75rem; font-weight: 700; color: #3b82f6;'>{n_checkpoints}</div>
                    <div style='font-size: 0.85rem; color: #93c5fd; margin-top: 4px; font-weight: 500;'>Self Checkpoints</div>
                </div>
                <div style='background: rgba(168, 85, 247, 0.1); border: 1px solid rgba(168, 85, 247, 0.2); border-radius: 12px; padding: 16px; text-align: center;'>
                    <div style='font-size: 1.75rem; font-weight: 700; color: #a855f7;'>{n_exploiters}</div>
                    <div style='font-size: 0.85rem; color: #d8b4fe; margin-top: 4px; font-weight: 500;'>Active Exploiters</div>
                </div>
            </div>
            
            <h4 style='margin-top: 0; margin-bottom: 12px; font-size: 0.95rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>
                📊 Matchup Win Rates (Weakness Patching)
            </h4>
        """
        
        opponents = list(pool_manager.win_buffers.keys())
        if not opponents:
            html += """
            <div style='font-size: 0.9rem; color: #94a3b8; font-style: italic; text-align: center; padding: 16px;'>
                No matches played yet. Start League training to populate analytics!
            </div>
            """
        else:
            for opp_id in sorted(opponents):
                wr = pool_manager.get_win_rate(opp_id)
                pct = int(wr * 100)
                
                if wr < 0.50:
                    color = "#ef4444"
                    badge = "CRITICAL WEAKNESS"
                    bg_color = "rgba(239, 68, 68, 0.15)"
                elif wr < 0.75:
                    color = "#f59e0b"
                    badge = "CONTESTED"
                    bg_color = "rgba(245, 158, 11, 0.15)"
                else:
                    color = "#22c55e"
                    badge = "MASTERED"
                    bg_color = "rgba(34, 197, 94, 0.15)"
                    
                display_name = opp_id.replace("past_self_", "Checkpt: ").replace("exploiter_", "Exploiter: ").replace("current_self", "Current Self").replace(".zip", "")
                
                html += f"""
                <div style='margin-bottom: 16px;'>
                    <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.9rem;'>
                        <span style='font-weight: 500; color: #e2e8f0;'>{display_name}</span>
                        <span style='font-size: 0.75rem; font-weight: 600; padding: 2px 8px; border-radius: 9999px; color: {color}; background: {bg_color}; border: 1px solid {color}33;'>{badge} ({pct}%)</span>
                    </div>
                    <div style='width: 100%; height: 8px; background: rgba(255, 255, 255, 0.05); border-radius: 9999px; overflow: hidden;'>
                        <div style='width: {pct}%; height: 100%; background: {color}; border-radius: 9999px; transition: width 0.3s ease;'></div>
                    </div>
                </div>
                """
                
        html += "</div>"
        return html
    except Exception as e:
        return f"<div style='color: red; padding: 12px;'>Error reading pool analytics: {e}</div>"

def get_auto_curriculum_status_html(algo, env):
    """Parses auto_curriculum_state.json and renders a premium live progress and analytics card."""
    try:
        target_dir = os.path.join(config.PROJECT_ROOT, "models", "production", env, algo)
        state_path = os.path.join(target_dir, "auto_curriculum_state.json")
        
        if not os.path.exists(state_path):
            return """
            <div style='background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); padding: 24px; font-family: system-ui, -apple-system, sans-serif; color: #fff;'>
                <h3 style='margin-top: 0; margin-bottom: 12px; display: flex; align-items: center; gap: 8px; font-size: 1.35rem; font-weight: 600; color: #3b82f6;'>
                    📈 Auto-Curriculum Analytics
                </h3>
                <div style='font-size: 0.9rem; color: #94a3b8; font-style: italic; text-align: center; padding: 16px;'>
                    No active auto-curriculum session found for this algorithm/environment. Start auto-curriculum training to view real-time metrics!
                </div>
            </div>
            """
            
        import json
        with open(state_path, "r") as f:
            state_data = json.load(f)
            
        current_level = state_data.get("current_level", 1)
        stability_counter = state_data.get("stability_counter", 0)
        introduced = state_data.get("introduced_states", [])
        steps = state_data.get("num_timesteps", 0)
        state_wins = state_data.get("state_win_buffers", {})
        
        # Formulate consecutive stability blocks e.g. [🟩][🟩][⬜]
        stability_blocks = ""
        for i in range(3):
            if i < stability_counter:
                stability_blocks += "<span style='font-size: 1.2rem; margin-right: 4px;'>🟩</span>"
            else:
                stability_blocks += "<span style='font-size: 1.2rem; margin-right: 4px;'>⬜</span>"
                
        level_pct = int((current_level / 8) * 100)
        
        html = f"""
        <div style='background: rgba(30, 41, 59, 0.7); backdrop-filter: blur(12px); border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); padding: 24px; font-family: system-ui, -apple-system, sans-serif; color: #fff;'>
            <h3 style='margin-top: 0; margin-bottom: 16px; display: flex; align-items: center; gap: 8px; font-size: 1.35rem; font-weight: 600; color: #3b82f6;'>
                📈 Auto-Curriculum Analytics
            </h3>
            
            <div style='margin-bottom: 20px;'>
                <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px; font-size: 0.9rem;'>
                    <span style='font-weight: 500; color: #94a3b8;'>Master Level</span>
                    <span style='font-weight: 700; color: #3b82f6; font-size: 1rem;'>Lvl {current_level} / 8</span>
                </div>
                <div style='width: 100%; height: 10px; background: rgba(255, 255, 255, 0.05); border-radius: 9999px; overflow: hidden;'>
                    <div style='width: {level_pct}%; height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); border-radius: 9999px; transition: width 0.4s ease;'></div>
                </div>
            </div>
            
            <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 24px;'>
                <div style='background: rgba(59, 130, 246, 0.1); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 12px; text-align: center;'>
                    <div style='font-size: 1.25rem; font-weight: 700; color: #3b82f6;'>{steps:,}</div>
                    <div style='font-size: 0.8rem; color: #93c5fd; margin-top: 4px; font-weight: 500;'>Steps Completed</div>
                </div>
                <div style='background: rgba(34, 197, 94, 0.1); border: 1px solid rgba(34, 197, 94, 0.2); border-radius: 12px; padding: 12px; text-align: center;'>
                    <div style='display: flex; justify-content: center; align-items: center; height: 1.25rem;'>{stability_blocks}</div>
                    <div style='font-size: 0.8rem; color: #86efac; margin-top: 4px; font-weight: 500;'>Stability Streak</div>
                </div>
            </div>
            
            <h4 style='margin-top: 0; margin-bottom: 12px; font-size: 0.85rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;'>
                🥋 Matchup Mastery (Active & New Pool)
            </h4>
        """
        
        active_states = config.DIFFICULTY_LEVELS.get(current_level, []).copy()
        target_states = active_states + introduced
        
        if not state_wins:
            html += """
            <div style='font-size: 0.85rem; color: #94a3b8; font-style: italic; text-align: center; padding: 8px;'>
                Waiting for first episode telemetry to gather win rates...
            </div>
            """
        else:
            found_any = False
            for state in sorted(target_states):
                if state in state_wins:
                    found_any = True
                    buf = state_wins[state]
                    wr = sum(buf) / len(buf) if len(buf) > 0 else 0.0
                    pct = int(wr * 100)
                    
                    if wr < 0.50:
                         color = "#ef4444"
                         badge = "WEAKNESS"
                         bg_color = "rgba(239, 68, 68, 0.15)"
                    elif wr < 0.75:
                         color = "#f59e0b"
                         badge = "CONTESTED"
                         bg_color = "rgba(245, 158, 11, 0.15)"
                    else:
                         color = "#22c55e"
                         badge = "MASTERED"
                         bg_color = "rgba(34, 197, 94, 0.15)"
                        
                    is_introduced = state in introduced
                    role_prefix = "New: " if is_introduced else "Act: "
                    state_clean = state[4:] if state.startswith("RYU_") else state
                    display_name = role_prefix + state_clean.replace("_R1", "").replace(".State", "")
                    
                    html += f"""
                    <div style='margin-bottom: 12px;'>
                        <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; font-size: 0.85rem;'>
                            <span style='font-weight: 500; color: #e2e8f0; font-size: 0.8rem;'>{display_name}</span>
                            <span style='font-size: 0.7rem; font-weight: 600; padding: 1px 6px; border-radius: 9999px; color: {color}; background: {bg_color}; border: 1px solid {color}33;'>{badge} ({pct}%)</span>
                        </div>
                        <div style='width: 100%; height: 6px; background: rgba(255, 255, 255, 0.05); border-radius: 9999px; overflow: hidden;'>
                            <div style='width: {pct}%; height: 100%; background: {color}; border-radius: 9999px; transition: width 0.3s ease;'></div>
                        </div>
                    </div>
                    """
            if not found_any:
                html += """
                <div style='font-size: 0.85rem; color: #94a3b8; font-style: italic; text-align: center; padding: 8px;'>
                    No active state buffers recorded yet. Play a round!
                </div>
                """
                
        html += "</div>"
        return html
    except Exception as e:
        return f"<div style='color: red; padding: 12px;'>Error reading auto-curriculum analytics: {e}</div>"

def refresh_league_status():
    importlib.reload(config)
    all_states = get_all_state_files()
    html = get_league_pool_status_html()
    return html, gr.update(choices=all_states), gr.update(choices=all_states)

def toggle_league_matchup_mode(mode):
    is_custom = (mode == "Custom Savestate (Uploaded)")
    return gr.update(visible=is_custom), gr.update(visible=is_custom), gr.update(visible=is_custom)

def toggle_exploiter_matchup_mode(mode):
    is_custom = (mode == "Custom Savestate (Uploaded)")
    return gr.update(visible=is_custom), gr.update(visible=is_custom), gr.update(visible=is_custom)

def handle_league_state_upload(file_obj):
    if file_obj is None: 
        return gr.update(), "❌ No file selected."
    try:
        import shutil
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        name = os.path.basename(file_path)
        target = os.path.join(config.STATES_DIR, name)
        shutil.copy2(file_path, target)
        
        # Update config.py CUSTOM_STATES for backward compatibility
        current_custom = list(config.CUSTOM_STATES)
        if name not in current_custom:
            new_custom = list(set(current_custom + [name]))
            update_config_list("CUSTOM_STATES", new_custom)
            importlib.reload(config)
            
        all_states = get_all_state_files()
        return gr.update(choices=all_states, value=name), f"✅ Uploaded `{name}` successfully!"
    except Exception as e:
        return gr.update(), f"❌ Upload error: {e}"

# --- UI Construction ---

zips_init, pkls_init = get_model_files("ppo")

with gr.Blocks(title="Street Fighter II RL Dashboard") as demo:
    gr.Markdown("# 🕹️ Street Fighter II RL Control Center")
    
    with gr.Tabs():
        # --- TAB 1: UNIFIED TRAINING & TUNING ---
        with gr.Tab("🏋️‍♂️ Training & Tuning"):
            gr.Markdown("### Global Settings")
            with gr.Row():
                with gr.Column(scale=1):
                    algo_sel = gr.Dropdown(label="Algorithm", choices=["ppo", "sac", "dqn"], value="ppo")
                with gr.Column(scale=1):
                    env_sel = gr.Dropdown(label="Environment", choices=["v1", "v2", "v3"], value="v2")
                with gr.Column(scale=1):
                    tb_main_btn = gr.Button("📈 Launch TensorBoard", variant="secondary")
            
            with gr.Row():
                with gr.Column(scale=1):
                    with gr.Tabs():
                        # Section A: Production
                        with gr.Tab("🚀 Production Training"):
                            model_name_input = gr.Textbox(label="New Model Name", value=config.MODEL_NAME)
                            
                            with gr.Row():
                                train_zip_drop = gr.Dropdown(label="Base Model (.zip)", choices=zips_init, value="None")
                                train_pkl_drop = gr.Dropdown(label="Base Norm (.pkl)", choices=pkls_init, value="None")
                            
                            with gr.Row():
                                ext_zip_upload = gr.File(label="Upload Model (.zip)", file_types=[".zip"])
                                ext_pkl_upload = gr.File(label="Upload Normalization (.pkl)", file_types=[".pkl"])
                            upload_status = gr.Markdown("")
                            
                            with gr.Row():
                                auto_curr_check = gr.Checkbox(label="Enable Auto-Curriculum (Progressive 8-Level)", value=False)
                                train_phase_drop = gr.Dropdown(label="Start Phase (Manual)", choices=[0, 1, 2, 3, "RYU_ONLY", "CUSTOM"], value=0)
                                train_steps = gr.Number(label="Total Timesteps", value=1000000, precision=0)
                                train_device = gr.Dropdown(label="Compute Device", choices=["auto", "cpu", "cuda"], value="auto")
                            
                            with gr.Accordion("Advanced Hyperparameters (Overrides Config)", open=False):
                                gr.Markdown("*(Set values > 0.0 to override defaults)*")
                                train_lr = gr.Number(label="Learning Rate Override", value=0.0)
                                train_ent = gr.Number(label="Entropy Coef Override", value=0.0)
                                train_clip = gr.Number(label="Clip Range Override", value=0.0)
                                
                                upload_json = gr.File(label="Upload Hyperparameters JSON", file_types=[".json"])
                                readonly_params = gr.JSON(label="Fixed / Read-Only Hyperparameters")
                            
                            start_train_btn = gr.Button("▶ Start Training", variant="primary")
                            
                            gr.Markdown("---")
                            with gr.Row():
                                refresh_curr_btn = gr.Button("🔄 Refresh Auto-Curriculum Stats", variant="secondary")
                            auto_curr_card = gr.HTML(value=get_auto_curriculum_status_html("ppo", "v2"))
                        
                        # Section B: Optuna
                        with gr.Tab("🧪 Optuna Tuning"):
                            study_name_input = gr.Textbox(label="Study Name", value="ppo_sf2_tuning")
                            with gr.Row():
                                tune_zip_drop = gr.Dropdown(label="Base Model (.zip) [Optional]", choices=zips_init, value="None")
                                tune_pkl_drop = gr.Dropdown(label="Base Norm (.pkl) [Optional]", choices=pkls_init, value="None")
                            
                            with gr.Row():
                                tune_phase_drop = gr.Dropdown(label="Start Phase (States)", choices=[0, 1, 2, 3, "RYU_ONLY", "CUSTOM"], value=0)
                                tune_steps = gr.Number(label="Timesteps per Trial", value=50000, precision=0)
                                tune_device = gr.Dropdown(label="Compute Device", choices=["auto", "cpu", "cuda"], value="auto")
                            trials_input = gr.Number(label="Number of Trials", value=10, precision=0)
                            
                            with gr.Row():
                                start_tune_btn = gr.Button("🚀 Start Tuning", variant="primary")
                                get_results_btn = gr.Button("🔍 Fetch Best Results")
                            best_params_output = gr.Textbox(label="Best Hyperparameters", interactive=False)
                            download_json = gr.File(label="Download Best Hyperparameters", interactive=False)
                        
                        # Section C: PBT
                        with gr.Tab("🧬 PBT Training"):
                            gr.Markdown("Population Based Training (PB2) for automatic hyperparameter scheduling.")
                            pbt_model_name_input = gr.Textbox(label="Output Model Name", value="PBT_BEST_model")
                            
                            with gr.Row():
                                pbt_zip_drop = gr.Dropdown(label="Base Model to Seed Population (.zip)", choices=zips_init, value="None")
                                pbt_pkl_drop = gr.Dropdown(label="Base Norm to Seed Population (.pkl)", choices=pkls_init, value="None")
                            
                            with gr.Row():
                                pbt_phase_drop = gr.Dropdown(label="Start Phase (States)", choices=[0, 1, 2, 3, "RYU_ONLY", "CUSTOM"], value=0)
                                pbt_steps = gr.Number(label="Total Timesteps", value=5000000, precision=0)
                                pbt_exploit_steps = gr.Number(label="Steps per Exploit", value=500000, precision=0)
                            
                            with gr.Row():
                                pbt_pop = gr.Slider(label="Population Size", minimum=4, maximum=16, value=10, step=1)
                                pbt_concurrent = gr.Slider(label="Max Concurrent Trials", minimum=1, maximum=16, value=4, step=1)
                                pbt_envs = gr.Slider(label="Envs per Worker", minimum=1, maximum=8, value=1, step=1)
                                pbt_resume = gr.Checkbox(label="Resume existing PBT run (loads from Ray Tuner cache)", value=False)
                            
                            start_pbt_btn = gr.Button("🧬 Launch PBT", variant="primary")

                    gr.Markdown("---")
                    with gr.Row():
                        stop_btn = gr.Button("🛑 Stop All Processes", variant="stop")
                        refresh_files_btn = gr.Button("🔄 Refresh Dropdown Models")
                    stop_status = gr.Markdown("")

                # RIGHT: Terminal
                with gr.Column(scale=2):
                    unified_logs = gr.Textbox(label="Console Output", lines=35, max_lines=45, interactive=False, elem_id="terminal")
                    copy_btn = gr.Button("📋 Copy Logs", size="sm")

        # --- TAB 1.5: AUTO-LEARNING LEAGUE ---
        with gr.Tab("🏆 Auto-Learning League"):
            gr.Markdown("### 🏆 Street Fighter II' Auto-Learning League Control Panel")
            
            with gr.Row():
                # LEFT COLUMN: Controls & Logs
                with gr.Column(scale=7):
                    with gr.Tabs():
                        # Sub-tab 1: Self-Play League
                        with gr.Tab("🎯 Self-Play League Training"):
                            gr.Markdown("Orchestrate active Main Agent training against the dynamic historical matchmaking pool.")
                            
                            with gr.Row():
                                league_model_name = gr.Textbox(label="League Model Name", value="league")
                                league_steps = gr.Number(label="Total Timesteps", value=5000000, precision=0)
                                league_env = gr.Dropdown(label="Environment Version", choices=["v2", "v3"], value="v2")
                                league_device = gr.Dropdown(label="Compute Device", choices=["auto", "cpu", "cuda"], value="auto")
                                
                            all_states = get_all_state_files()
                            with gr.Row():
                                league_matchup_mode = gr.Dropdown(
                                    label="Matchup Mode", 
                                    choices=["Ryu vs. Ryu (Strict Self-Play)", "Ryu vs. All (12 Characters)", "Custom Savestate (Uploaded)"], 
                                    value="Ryu vs. Ryu (Strict Self-Play)"
                                )
                                league_custom_state = gr.Dropdown(
                                    label="Select Custom Fight Savestate", 
                                    choices=all_states, 
                                    value="None", 
                                    visible=False
                                )
                                league_resume = gr.Checkbox(label="Resume from previous active League model", value=True)
                            
                            with gr.Row():
                                league_state_upload = gr.File(
                                    label="Upload Custom Savestate (.State)", 
                                    file_types=[".State"], 
                                    visible=False
                                )
                                league_upload_status = gr.Markdown("", visible=False)
                                
                            start_league_btn = gr.Button("▶ Launch League Training", variant="primary")
                            
                        # Sub-tab 2: Specialized Exploiters
                        with gr.Tab("⚔️ Specialized Exploiter Training"):
                            gr.Markdown("Train a dedicated agent to search for and exploit weaknesses in the current Main Agent.")
                            
                            with gr.Row():
                                exploiter_model_name = gr.Textbox(label="Target League Model Name", value="league")
                                exploiter_type = gr.Dropdown(label="Exploiter Archetype", choices=["rusher", "spammer", "turtle"], value="rusher")
                                exploiter_steps = gr.Number(label="Timesteps", value=1000000, precision=0)
                                
                            with gr.Row():
                                exploiter_env = gr.Dropdown(label="Environment Version", choices=["v2", "v3"], value="v2")
                                exploiter_device = gr.Dropdown(label="Compute Device", choices=["auto", "cpu", "cuda"], value="auto")
                                exploiter_matchup_mode = gr.Dropdown(
                                    label="Matchup Mode", 
                                    choices=["Ryu vs. Ryu (Strict Self-Play)", "Ryu vs. All (12 Characters)", "Custom Savestate (Uploaded)"], 
                                    value="Ryu vs. Ryu (Strict Self-Play)"
                                )
                                
                            with gr.Row():
                                exploiter_custom_state = gr.Dropdown(
                                    label="Select Custom Fight Savestate", 
                                    choices=all_states, 
                                    value="None", 
                                    visible=False
                                )
                                exploiter_state_upload = gr.File(
                                    label="Upload Custom Savestate (.State)", 
                                    file_types=[".State"], 
                                    visible=False
                                )
                                exploiter_upload_status = gr.Markdown("", visible=False)
                                
                            start_exploiter_btn = gr.Button("⚔️ Launch Exploiter Training", variant="primary")
                    
                    gr.Markdown("---")
                    league_logs = gr.Textbox(label="League Console Output", lines=20, max_lines=25, interactive=False, elem_id="terminal")
                    
                    with gr.Row():
                        refresh_league_btn = gr.Button("🔄 Refresh Pool Status & States")
                        stop_league_btn = gr.Button("🛑 Stop Active League Runs", variant="stop")
                        copy_league_logs_btn = gr.Button("📋 Copy League Logs", size="sm")
                        
                # RIGHT COLUMN: Pool Analytics Card
                with gr.Column(scale=4):
                    league_analytics_card = gr.HTML(value=get_league_pool_status_html())

        # --- TAB 2: MATCHUPS ---
        with gr.Tab("🎮 Model Testing & Matchups"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Player 1 (Ryu)")
                    with gr.Row():
                        p1_algo = gr.Dropdown(label="P1 Algorithm", choices=["ppo", "sac", "dqn", "Human Player"], value="ppo")
                        p1_env = gr.Dropdown(label="P1 Environment", choices=["v2", "v3"], value="v2")
                    p1_device = gr.Dropdown(label="P1 Compute Device", choices=["auto", "cpu", "cuda"], value="auto")
                    
                    with gr.Column(visible=True) as p1_model_group:
                        with gr.Row():
                            p1_zip = gr.Dropdown(label="P1 Model (.zip)", choices=zips_init, value="None")
                            p1_pkl = gr.Dropdown(label="P1 Normalization (.pkl)", choices=pkls_init, value="None")
                        with gr.Row():
                            p1_zip_upload = gr.File(label="Upload P1 Model (.zip)", file_types=[".zip"])
                            p1_pkl_upload = gr.File(label="Upload P1 Normalization (.pkl)", file_types=[".pkl"])
                    
                    gr.Markdown("### Player 2 (Opponent)")
                    with gr.Row():
                        p2_algo = gr.Dropdown(label="P2 Algorithm", choices=["ppo", "sac", "dqn", "Human Player", "CPU (Built-in AI)"], value="ppo")
                        p2_env = gr.Dropdown(label="P2 Environment", choices=["v2", "v3"], value="v2")
                    p2_device = gr.Dropdown(label="P2 Compute Device", choices=["auto", "cpu", "cuda"], value="auto")
                    
                    with gr.Column(visible=True) as p2_model_group:
                        with gr.Row():
                            p2_zip = gr.Dropdown(label="P2 Model (.zip)", choices=zips_init, value="None")
                            p2_pkl = gr.Dropdown(label="P2 Normalization (.pkl)", choices=pkls_init, value="None")
                        with gr.Row():
                            p2_zip_upload = gr.File(label="Upload P2 Model (.zip)", file_types=[".zip"])
                            p2_pkl_upload = gr.File(label="Upload P2 Normalization (.pkl)", file_types=[".pkl"])
                    
                    with gr.Row():
                        launch_match_btn = gr.Button("⚔️ Launch Match", variant="primary")
                        stop_match_btn = gr.Button("🛑 Terminate Match", variant="stop")
                    
                    with gr.Row():
                        match_profile_checkbox = gr.Checkbox(label="Enable Performance Profiling", value=False)
                        toggle_agent_btn = gr.Button("⏯️ Toggle Agent (Play/Pause)", variant="secondary")
                    
                    agent_state_status = gr.Markdown("Agent State: **PAUSED** (Default)")
                    
                    match_upload_status = gr.Markdown("")
                
                with gr.Column():
                    match_logs = gr.Textbox(label="Match Console", lines=25, max_lines=35, interactive=False, elem_id="terminal")
                    copy_match_btn = gr.Button("📋 Copy Match Logs", size="sm")
            
            # Interactive visibility and filtering toggles
            def update_match_ui(algo):
                is_ai = algo in ["ppo", "sac", "dqn"]
                if not is_ai:
                    return gr.update(visible=False), gr.update(), gr.update()
                
                z, p = get_model_files(algo)
                return (
                    gr.update(visible=True),
                    gr.update(choices=z, value="None"),
                    gr.update(choices=p, value="None")
                )

            p1_algo.change(update_match_ui, inputs=[p1_algo], outputs=[p1_model_group, p1_zip, p1_pkl])
            p2_algo.change(update_match_ui, inputs=[p2_algo], outputs=[p2_model_group, p2_zip, p2_pkl])

            # Link matchup uploaders
            p1_zip_upload.upload(handle_model_upload, inputs=[p1_zip_upload, p1_algo, p1_env], outputs=[match_upload_status, p1_zip, p1_pkl])
            p1_pkl_upload.upload(handle_model_upload, inputs=[p1_pkl_upload, p1_algo, p1_env], outputs=[match_upload_status, p1_zip, p1_pkl])
            p2_zip_upload.upload(handle_model_upload, inputs=[p2_zip_upload, p2_algo, p2_env], outputs=[match_upload_status, p2_zip, p2_pkl])
            p2_pkl_upload.upload(handle_model_upload, inputs=[p2_pkl_upload, p2_algo, p2_env], outputs=[match_upload_status, p2_zip, p2_pkl])

            launch_match_btn.click(run_matchup, inputs=[p1_algo, p1_env, p1_zip, p1_pkl, p1_device, p2_algo, p2_env, p2_zip, p2_pkl, p2_device, match_profile_checkbox], outputs=[match_logs])

            stop_match_btn.click(stop_match_process, outputs=[match_logs, agent_state_status])
            toggle_agent_btn.click(toggle_agent_state, outputs=[agent_state_status])

        # --- TAB 3: CONFIG ---
        with gr.Tab("⚙️ Core Config Editor"):
            with gr.Row():
                with gr.Column():
                    cfg_n_envs = gr.Number(label="N_ENVS (Parallel Instances)", value=config.N_ENVS, precision=0)
                    cfg_win_rate = gr.Slider(label="WIN_RATE_THRESHOLD (Phase Advance)", minimum=0.5, maximum=0.95, value=config.WIN_RATE_THRESHOLD, step=0.01)
                    cfg_steps = gr.Number(label="Default Training Steps", value=config.STARTING_TOTAL_TIMESTEPS, precision=0)
                    cfg_port = gr.Number(label="Base Socket Port", value=config.PORT, precision=0)
                    cfg_input_display = gr.Checkbox(label="Enable Input Display in Match Tests", value=getattr(config, 'ENABLE_INPUT_DISPLAY', True))
                    cfg_activate_visualization = gr.Checkbox(label="Enable Training Visualization", value=getattr(config, 'ACTIVATE_VISUALIZATION', True))
                    
                    save_cfg_btn = gr.Button("💾 Save Configuration", variant="primary")
                    cfg_status = gr.Markdown("")

                with gr.Column():
                    gr.Markdown("### 📂 State Management")
                    state_upload = gr.File(label="Upload Custom Savestates (.State)", file_types=[".State"], file_count="multiple")
                    state_upload_status = gr.Markdown("")
            
            save_cfg_btn.click(save_all_config, inputs=[cfg_n_envs, cfg_win_rate, cfg_steps, cfg_port, cfg_input_display, cfg_activate_visualization], outputs=[cfg_status])

    # --- GLOBAL EVENT HANDLERS ---
    
    # Algorithm change logic (Training Tab)
    def update_ui_on_algo(algo):
        # Update zips and pkls
        zips, pkls = get_model_files(algo)
        
        return (
            gr.update(choices=zips, value="None"), 
            gr.update(choices=pkls, value="None"),
            gr.update(choices=zips, value="None"), 
            gr.update(choices=pkls, value="None"),
            gr.update(value=f"{algo}_sf2_tuning"),
            gr.update(value=f"{algo}_sf2_production")
        )
    
    algo_sel.change(update_ui_on_algo, inputs=[algo_sel], outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, study_name_input, model_name_input]).then(
        get_auto_curriculum_status_html, inputs=[algo_sel, env_sel], outputs=[auto_curr_card]
    )
    env_sel.change(get_auto_curriculum_status_html, inputs=[algo_sel, env_sel], outputs=[auto_curr_card])

    # Link uploaders (Training Tab)
    ext_zip_upload.upload(handle_model_upload, inputs=[ext_zip_upload, algo_sel, env_sel], outputs=[upload_status, train_zip_drop, train_pkl_drop])
    ext_pkl_upload.upload(handle_model_upload, inputs=[ext_pkl_upload, algo_sel, env_sel], outputs=[upload_status, train_zip_drop, train_pkl_drop])

    upload_json.upload(load_hyperparams_from_json, inputs=[upload_json], outputs=[train_lr, train_ent, train_clip, readonly_params])

    # Dynamic Auto-Curriculum UI overrides
    def toggle_auto_curriculum_ui(is_enabled):
        if is_enabled:
            return gr.update(label="Start Level (Auto)", choices=[1, 2, 3, 4, 5, 6, 7, 8], value=1)
        else:
            return gr.update(label="Start Phase (Manual)", choices=[0, 1, 2, 3, "RYU_ONLY", "CUSTOM"], value=0)

    auto_curr_check.change(toggle_auto_curriculum_ui, inputs=[auto_curr_check], outputs=[train_phase_drop])

    # Auto-Curriculum Live Card Timer Updates (runs every 5 seconds)
    gr.Timer(5).tick(fn=get_auto_curriculum_status_html, inputs=[algo_sel, env_sel], outputs=[auto_curr_card])

    # Global Process Handlers
    start_train_btn.click(run_training, inputs=[algo_sel, env_sel, model_name_input, train_zip_drop, train_pkl_drop, train_phase_drop, train_steps, train_lr, train_ent, train_clip, train_device, auto_curr_check], outputs=[unified_logs]).then(
        refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, pbt_zip_drop, pbt_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl]
    )
    
    start_tune_btn.click(run_tuning, inputs=[algo_sel, env_sel, study_name_input, tune_zip_drop, tune_pkl_drop, tune_phase_drop, tune_steps, trials_input, tune_device], outputs=[unified_logs]).then(
        refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, pbt_zip_drop, pbt_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl]
    )
    
    start_pbt_btn.click(run_pbt, inputs=[algo_sel, env_sel, pbt_model_name_input, pbt_zip_drop, pbt_pkl_drop, pbt_phase_drop, pbt_steps, pbt_exploit_steps, pbt_pop, pbt_concurrent, pbt_resume, pbt_envs], outputs=[unified_logs]).then(
        refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, pbt_zip_drop, pbt_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl]
    )
    
    get_results_btn.click(get_best_tuning_params, inputs=[algo_sel, env_sel, study_name_input], outputs=[best_params_output, download_json])
    
    copy_btn.click(None, inputs=[unified_logs], js="(text) => { navigator.clipboard.writeText(text); alert('Logs copied to clipboard!'); return []; }")
    copy_match_btn.click(None, inputs=[match_logs], js="(text) => { navigator.clipboard.writeText(text); alert('Match logs copied to clipboard!'); return []; }")

    stop_btn.click(stop_active_process, outputs=[stop_status])
    refresh_files_btn.click(refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, pbt_zip_drop, pbt_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl])
    refresh_curr_btn.click(get_auto_curriculum_status_html, inputs=[algo_sel, env_sel], outputs=[auto_curr_card])
    tb_main_btn.click(launch_tb, outputs=[gr.Textbox(visible=False)])

    # League Tab Event Bindings
    league_matchup_mode.change(
        toggle_league_matchup_mode, 
        inputs=[league_matchup_mode], 
        outputs=[league_custom_state, league_state_upload, league_upload_status]
    )
    
    exploiter_matchup_mode.change(
        toggle_exploiter_matchup_mode, 
        inputs=[exploiter_matchup_mode], 
        outputs=[exploiter_custom_state, exploiter_state_upload, exploiter_upload_status]
    )
    
    league_state_upload.upload(
        handle_league_state_upload, 
        inputs=[league_state_upload], 
        outputs=[league_custom_state, league_upload_status]
    )
    
    exploiter_state_upload.upload(
        handle_league_state_upload, 
        inputs=[exploiter_state_upload], 
        outputs=[exploiter_custom_state, exploiter_upload_status]
    )

    start_league_btn.click(
        run_league, 
        inputs=[league_model_name, league_steps, league_env, league_matchup_mode, league_custom_state, league_resume, league_device], 
        outputs=[league_logs]
    ).then(
        refresh_league_status, 
        outputs=[league_analytics_card, league_custom_state, exploiter_custom_state]
    )
    
    start_exploiter_btn.click(
        run_exploiter, 
        inputs=[exploiter_model_name, exploiter_type, exploiter_steps, exploiter_env, exploiter_matchup_mode, exploiter_custom_state, exploiter_device], 
        outputs=[league_logs]
    ).then(
        refresh_league_status, 
        outputs=[league_analytics_card, league_custom_state, exploiter_custom_state]
    )
    
    refresh_league_btn.click(
        refresh_league_status, 
        outputs=[league_analytics_card, league_custom_state, exploiter_custom_state]
    )
    
    stop_league_btn.click(
        stop_active_process, 
        outputs=[league_logs]
    )
    
    copy_league_logs_btn.click(
        None, 
        inputs=[league_logs], 
        js="(text) => { navigator.clipboard.writeText(text); alert('League logs copied to clipboard!'); return []; }"
    )

    def handle_state_upload(file_objs):
        if not file_objs: return "No files selected."
        import shutil
        saved_names = []
        for f in file_objs:
            name = os.path.basename(f.name)
            target = os.path.join(config.STATES_DIR, name)
            shutil.copy2(f.name, target)
            saved_names.append(name)
        
        # Update config.py
        current_custom = list(config.CUSTOM_STATES)
        new_custom = list(set(current_custom + saved_names))
        if update_config_list("CUSTOM_STATES", new_custom):
            importlib.reload(config)
            return f"✅ Uploaded {len(saved_names)} states and updated CUSTOM_STATES registry."
        return "❌ Error updating config.py CUSTOM_STATES list."

    state_upload.upload(handle_state_upload, inputs=[state_upload], outputs=[state_upload_status])

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0", 
        server_port=7860, 
        theme=gr.themes.Soft(primary_hue="blue"), 
        css="#terminal textarea { font-family: monospace; }"
    )
