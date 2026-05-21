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
    """Scans models directory for zip and pkl files, optionally filtering by algorithm."""
    models_dir = os.path.join(config.PROJECT_ROOT, "models")
    if not os.path.exists(models_dir):
        return ["None"], ["None"]
    
    # If algorithm is specified, narrow the search scope
    if algo:
        search_dirs = [
            os.path.join(models_dir, "production", algo),
            os.path.join(models_dir, "tuning", algo)
        ]
        
        zip_files = []
        pkl_files = []
        for d in search_dirs:
            if os.path.exists(d):
                # Look for files directly in the algorithm's dedicated directories
                zip_files.extend(glob.glob(os.path.join(d, "**/*.zip"), recursive=True))
                pkl_files.extend(glob.glob(os.path.join(d, "**/*.pkl"), recursive=True))
    else:
        zip_files = glob.glob(os.path.join(models_dir, "**/*.zip"), recursive=True)
        pkl_files = glob.glob(os.path.join(models_dir, "**/*.pkl"), recursive=True)
    
    # Remove duplicates and return relative paths with forward slashes
    zips = sorted(list(set([os.path.relpath(f, config.PROJECT_ROOT).replace("\\", "/") for f in zip_files])))
    pkls = sorted(list(set([os.path.relpath(f, config.PROJECT_ROOT).replace("\\", "/") for f in pkl_files])))
    
    return ["None"] + zips, ["None"] + pkls

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
    """Gracefully stops the active process by sending CTRL_C, allowing a model save."""
    if state.active_process:
        proc = state.active_process
        state.stop_event.set()
        try:
            # 1. Send CTRL_C_EVENT to the process group
            # This triggers KeyboardInterrupt in Python, causing the EMERGENCY save
            print(f"[Dashboard] Sending Graceful Stop (CTRL_C) to PID {proc.pid}...")
            os.kill(proc.pid, signal.CTRL_C_EVENT)
            
            # 2. Wait up to 15 seconds for the agent to finish its EMERGENCY save logic
            for _ in range(15):
                if proc.poll() is not None:
                    break
                time.sleep(1)
            
            # 3. Final Tree-Kill Failsafe: Ensure BizHawk and Lua are definitely dead
            if proc.poll() is None:
                print(f"[Dashboard] Process {proc.pid} timed out during stop. Force killing...")
                subprocess.run(f"taskkill /F /T /PID {proc.pid}", shell=True, capture_output=True)
            else:
                print(f"[Dashboard] Process {proc.pid} gracefully stopped.")
                
        except Exception as e:
            print(f"[Dashboard] Error during stop: {e}")
        
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
    # Store in models/tuning/{algo}/
    tuning_dir = os.path.join(config.PROJECT_ROOT, "models", "tuning", algo)
    os.makedirs(tuning_dir, exist_ok=True)
    
    cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "tune.py"), 
           "--algo", algo, "--env", env, "--study_name", study_name, 
           "--trials", str(trials), "--phase", str(phase), "--timesteps", str(timesteps),
           "--device", device]
    
    if load_zip != "None": cmd += ["--load_zip", load_zip]
    if load_pkl != "None": cmd += ["--load_pkl", load_pkl]
    
    for log in stream_logs(cmd):
        yield log

def get_best_tuning_params(algo, study_name):
    # Resolve storage path based on algorithm
    tuning_dir = os.path.join(config.get_directory()["tuning"], algo)
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

def run_training(algo, env, model_name, load_zip, load_pkl, phase, timesteps, lr, ent_coef, clip_range, device):
    update_config_var("MODEL_NAME", model_name)
    
    cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "train.py"), 
           "--algo", algo, "--env", env, "--steps", str(timesteps), "--phase", str(phase),
           "--device", device]
           
    if load_zip != "None": cmd += ["--load_zip", load_zip]
    if load_pkl != "None": cmd += ["--load_pkl", load_pkl]
    if lr > 0.0: cmd += ["--lr", str(lr)]
    if ent_coef > 0.0: cmd += ["--ent_coef", str(ent_coef)]
    if clip_range > 0.0: cmd += ["--clip_range", str(clip_range)]
    
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

def handle_upload(file_obj, algo):
    if file_obj is None: return "Please select a file."
    try:
        import shutil
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        filename = os.path.basename(file_path)
        target_dir = os.path.join(config.PROJECT_ROOT, "models", "production", algo)
        os.makedirs(target_dir, exist_ok=True)
        shutil.copy2(file_path, os.path.join(target_dir, filename))
        return f"**Success:** Saved `{filename}` to `models/production/{algo}/`"
    except Exception as e: return f"**Error:** {e}"

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
                                train_phase_drop = gr.Dropdown(label="Start Phase (States)", choices=[0, 1, 2, 3, "RYU_ONLY", "CUSTOM"], value=0)
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
            p1_zip_upload.upload(handle_upload, inputs=[p1_zip_upload, p1_algo], outputs=[match_upload_status]).then(
                lambda algo: get_model_files(algo), inputs=[p1_algo], outputs=[p1_zip, p1_pkl]
            )
            p1_pkl_upload.upload(handle_upload, inputs=[p1_pkl_upload, p1_algo], outputs=[match_upload_status]).then(
                lambda algo: get_model_files(algo), inputs=[p1_algo], outputs=[p1_zip, p1_pkl]
            )
            p2_zip_upload.upload(handle_upload, inputs=[p2_zip_upload, p2_algo], outputs=[match_upload_status]).then(
                lambda algo: get_model_files(algo), inputs=[p2_algo], outputs=[p2_zip, p2_pkl]
            )
            p2_pkl_upload.upload(handle_upload, inputs=[p2_pkl_upload, p2_algo], outputs=[match_upload_status]).then(
                lambda algo: get_model_files(algo), inputs=[p2_algo], outputs=[p2_zip, p2_pkl]
            )

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
    
    algo_sel.change(update_ui_on_algo, inputs=[algo_sel], outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, study_name_input, model_name_input])

    # Link uploaders (Training Tab)
    ext_zip_upload.upload(handle_upload, inputs=[ext_zip_upload, algo_sel], outputs=[upload_status]).then(
        lambda algo: get_model_files(algo), inputs=[algo_sel], outputs=[train_zip_drop, train_pkl_drop]
    )
    ext_pkl_upload.upload(handle_upload, inputs=[ext_pkl_upload, algo_sel], outputs=[upload_status]).then(
        lambda algo: get_model_files(algo), inputs=[algo_sel], outputs=[train_zip_drop, train_pkl_drop]
    )

    upload_json.upload(load_hyperparams_from_json, inputs=[upload_json], outputs=[train_lr, train_ent, train_clip, readonly_params])

    # Global Process Handlers
    start_train_btn.click(run_training, inputs=[algo_sel, env_sel, model_name_input, train_zip_drop, train_pkl_drop, train_phase_drop, train_steps, train_lr, train_ent, train_clip, train_device], outputs=[unified_logs]).then(
        refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl]
    )
    
    start_tune_btn.click(run_tuning, inputs=[algo_sel, env_sel, study_name_input, tune_zip_drop, tune_pkl_drop, tune_phase_drop, tune_steps, trials_input, tune_device], outputs=[unified_logs]).then(
        refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl]
    )
    
    start_pbt_btn.click(run_pbt, inputs=[algo_sel, env_sel, pbt_model_name_input, pbt_zip_drop, pbt_pkl_drop, pbt_phase_drop, pbt_steps, pbt_exploit_steps, pbt_pop, pbt_concurrent, pbt_resume, pbt_envs], outputs=[unified_logs]).then(
        refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, pbt_zip_drop, pbt_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl]
    )
    
    get_results_btn.click(get_best_tuning_params, inputs=[algo_sel, study_name_input], outputs=[best_params_output, download_json])
    
    copy_btn.click(None, inputs=[unified_logs], js="(text) => { navigator.clipboard.writeText(text); alert('Logs copied to clipboard!'); return []; }")
    copy_match_btn.click(None, inputs=[match_logs], js="(text) => { navigator.clipboard.writeText(text); alert('Match logs copied to clipboard!'); return []; }")

    stop_btn.click(stop_active_process, outputs=[stop_status])
    refresh_files_btn.click(refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl])
    tb_main_btn.click(launch_tb, outputs=[gr.Textbox(visible=False)])

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
