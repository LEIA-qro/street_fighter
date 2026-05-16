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
    return upd_z, upd_p, upd_z, upd_p, upd_z, upd_p, upd_z, upd_p

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
        # Also include root models directory if we want a global fallback
        search_dirs.append(models_dir)
        
        zip_files = []
        pkl_files = []
        for d in search_dirs:
            if os.path.exists(d):
                # Look for files matching the algo in name or directory
                all_zips = glob.glob(os.path.join(d, "**/*.zip"), recursive=True)
                all_pkls = glob.glob(os.path.join(d, "**/*.pkl"), recursive=True)
                
                zip_files.extend([f for f in all_zips if algo.lower() in f.lower()])
                pkl_files.extend([f for f in all_pkls if algo.lower() in f.lower()])
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
        # Use shell=True for better Windows command resolution
        state.active_process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT, 
            text=True, 
            bufsize=1, 
            shell=True
        )
        
        for line in state.active_process.stdout:
            if state.stop_event.is_set():
                break
            full_output += line
            yield full_output
            
        state.active_process.wait()
        full_output += f"\n{'-'*50}\nProcess finished with exit code {state.active_process.returncode}"
        yield full_output
    except Exception as e:
        yield full_output + f"\n[ERROR] {str(e)}"
    finally:
        state.active_process = None

def stop_active_process():
    """Forcefully stops any running script."""
    if state.active_process:
        state.stop_event.set()
        try:
            subprocess.run(f"taskkill /F /T /PID {state.active_process.pid}", shell=True)
        except:
            pass
        state.active_process = None
        return "Process stopped."
    
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
        content = re.sub(pattern, rf"\1{formatted_value}\3", content, flags=re.MULTILINE)
        with open(config_path, "w") as f:
            f.write(content)
        return True
    return False

# --- Dashboard Tab Handlers ---

def run_tuning(algo, env, study_name, load_zip, load_pkl, phase, timesteps, trials):
    # Store in models/tuning/{algo}/
    tuning_dir = os.path.join(config.PROJECT_ROOT, "models", "tuning", algo)
    os.makedirs(tuning_dir, exist_ok=True)
    
    cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "tune.py"), 
           "--algo", algo, "--env", env, "--study_name", study_name, 
           "--trials", str(trials), "--phase", str(phase), "--timesteps", str(timesteps)]
    
    if load_zip != "None": cmd += ["--load_zip", load_zip]
    if load_pkl != "None": cmd += ["--load_pkl", load_pkl]
    
    for log in stream_logs(cmd):
        yield log

def get_best_tuning_params(algo, study_name):
    # Resolve storage path based on algorithm
    db_path = os.path.join(config.PROJECT_ROOT, "models", "tuning", algo, "study.db").replace("\\", "/")
    json_path = os.path.join(config.PROJECT_ROOT, "models", "tuning", algo, f"best_params_{study_name}.json").replace("\\", "/")
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

def run_training(algo, env, model_name, load_zip, load_pkl, phase, timesteps, lr, ent_coef, clip_range):
    update_config_var("MODEL_NAME", model_name)
    
    cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "train.py"), 
           "--algo", algo, "--env", env, "--steps", str(timesteps), "--phase", str(phase)]
           
    if load_zip != "None": cmd += ["--load_zip", load_zip]
    if load_pkl != "None": cmd += ["--load_pkl", load_pkl]
    if lr > 0.0: cmd += ["--lr", str(lr)]
    if ent_coef > 0.0: cmd += ["--ent_coef", str(ent_coef)]
    if clip_range > 0.0: cmd += ["--clip_range", str(clip_range)]
    
    for log in stream_logs(cmd):
        yield log

def launch_tb():
    cmd = f'"{VENV_PYTHON}" -m tensorboard.main --logdir "{config.LOG_DIR}" --port 6006'
    subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(2)
    webbrowser.open("http://localhost:6006")
    return "TensorBoard launched at http://localhost:6006"

def run_matchup(p1_algo, p1_zip, p1_pkl, p2_algo, p2_zip, p2_pkl):
    ai_algos = ["ppo", "sac", "dqn"]
    p1_is_ai = p1_algo in ai_algos
    p2_is_ai = p2_algo in ai_algos

    if p1_is_ai and p2_is_ai:
        cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "test_ai_vs_ai_v2.py"),
               "--algo_p1", p1_algo, "--load_zip_p1", p1_zip, "--load_pkl_p1", p1_pkl,
               "--algo_p2", p2_algo, "--load_zip_p2", p2_zip, "--load_pkl_p2", p2_pkl]
    elif p1_is_ai:
        # P1 is AI, P2 is Player or CPU
        cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "test_agent_v2.py"),
               "--algo", p1_algo, "--load_zip", p1_zip, "--load_pkl", p1_pkl, "--player", "1"]
    elif p2_is_ai:
        # P2 is AI, P1 is Player
        cmd = [VENV_PYTHON, os.path.join(config.SRC_DIR, "scripts", "test_agent_v2.py"),
               "--algo", p2_algo, "--load_zip", p2_zip, "--load_pkl", p2_pkl, "--player", "2"]
    else:
        yield "Invalid Matchup: At least one player must be an AI model (PPO, SAC, or DQN)."
        return
        
    for log in stream_logs(cmd):
        yield log

def save_all_config(n_envs, win_rate, steps, port):
    updates = {
        "N_ENVS": n_envs,
        "WIN_RATE_THRESHOLD": win_rate,
        "STARTING_TOTAL_TIMESTEPS": steps,
        "PORT": port
    }
    success = True
    for k, v in updates.items():
        if not update_config_var(k, v):
            success = False
    
    if success:
        importlib.reload(config)
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

# --- UI Construction ---

zips_init, pkls_init = get_model_files()

with gr.Blocks(title="Street Fighter II RL Dashboard", theme=gr.themes.Soft(primary_hue="blue"), css="#terminal textarea { font-family: monospace; }") as demo:
    gr.Markdown("# 🕹️ Street Fighter II RL Control Center")
    
    with gr.Tabs():
        # --- TAB 1: UNIFIED TRAINING & TUNING ---
        with gr.Tab("🏋️‍♂️ Training & Tuning"):
            gr.Markdown("### Global Settings")
            with gr.Row():
                with gr.Column(scale=1):
                    algo_sel = gr.Dropdown(label="Algorithm", choices=["ppo", "sac", "dqn"], value="ppo")
                with gr.Column(scale=1):
                    env_sel = gr.Dropdown(label="Environment", choices=["v1", "v2"], value="v2")
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
                            trials_input = gr.Number(label="Number of Trials", value=10, precision=0)
                            
                            with gr.Row():
                                start_tune_btn = gr.Button("🚀 Start Tuning", variant="primary")
                                get_results_btn = gr.Button("🔍 Fetch Best Results")
                            best_params_output = gr.Textbox(label="Best Hyperparameters", interactive=False)
                            download_json = gr.File(label="Download Best Hyperparameters", interactive=False)

                    gr.Markdown("---")
                    with gr.Row():
                        stop_btn = gr.Button("🛑 Stop All Processes", variant="stop")
                        refresh_files_btn = gr.Button("🔄 Refresh Dropdown Models")

                # RIGHT: Terminal
                with gr.Column(scale=2):
                    unified_logs = gr.Textbox(label="Console Output", show_copy_button=True, lines=35, max_lines=45, interactive=False, elem_id="terminal")

        # --- TAB 2: MATCHUPS ---
        with gr.Tab("🎮 Model Testing & Matchups"):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Player 1 (Ryu)")
                    p1_algo = gr.Dropdown(label="P1 Algorithm", choices=["ppo", "sac", "dqn", "Human Player"], value="ppo")
                    
                    with gr.Column(visible=True) as p1_model_group:
                        with gr.Row():
                            p1_zip = gr.Dropdown(label="P1 Model (.zip)", choices=zips_init, value="None")
                            p1_pkl = gr.Dropdown(label="P1 Normalization (.pkl)", choices=pkls_init, value="None")
                        with gr.Row():
                            p1_zip_upload = gr.File(label="Upload P1 Model (.zip)", file_types=[".zip"])
                            p1_pkl_upload = gr.File(label="Upload P1 Normalization (.pkl)", file_types=[".pkl"])
                    
                    gr.Markdown("### Player 2 (Opponent)")
                    p2_algo = gr.Dropdown(label="P2 Algorithm", choices=["ppo", "sac", "dqn", "Human Player", "CPU (Built-in AI)"], value="ppo")
                    
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
                    
                    match_upload_status = gr.Markdown("")
                
                with gr.Column():
                    match_logs = gr.Textbox(label="Match Console", show_copy_button=True, lines=25, max_lines=35, interactive=False, elem_id="terminal")
            
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

            launch_match_btn.click(run_matchup, inputs=[p1_algo, p1_zip, p1_pkl, p2_algo, p2_zip, p2_pkl], outputs=[match_logs])
            stop_match_btn.click(stop_active_process, outputs=[match_logs])

        # --- TAB 3: CONFIG ---
        with gr.Tab("⚙️ Core Config Editor"):
            with gr.Row():
                with gr.Column():
                    cfg_n_envs = gr.Number(label="N_ENVS (Parallel Instances)", value=config.N_ENVS, precision=0)
                    cfg_win_rate = gr.Slider(label="WIN_RATE_THRESHOLD (Phase Advance)", minimum=0.5, maximum=0.95, value=config.WIN_RATE_THRESHOLD, step=0.01)
                    cfg_steps = gr.Number(label="Default Training Steps", value=config.STARTING_TOTAL_TIMESTEPS, precision=0)
                    cfg_port = gr.Number(label="Base Socket Port", value=config.PORT, precision=0)
                    
                    save_cfg_btn = gr.Button("💾 Save Configuration", variant="primary")
                    cfg_status = gr.Markdown("")

                with gr.Column():
                    gr.Markdown("### 📂 State Management")
                    state_upload = gr.File(label="Upload Custom Savestates (.State)", file_types=[".State"], file_count="multiple")
                    state_upload_status = gr.Markdown("")
            
            save_cfg_btn.click(save_all_config, inputs=[cfg_n_envs, cfg_win_rate, cfg_steps, cfg_port], outputs=[cfg_status])

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
    start_train_btn.click(run_training, inputs=[algo_sel, env_sel, model_name_input, train_zip_drop, train_pkl_drop, train_phase_drop, train_steps, train_lr, train_ent, train_clip], outputs=[unified_logs]).then(
        refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl]
    )
    
    start_tune_btn.click(run_tuning, inputs=[algo_sel, env_sel, study_name_input, tune_zip_drop, tune_pkl_drop, tune_phase_drop, tune_steps, trials_input], outputs=[unified_logs]).then(
        refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl]
    )
    get_results_btn.click(get_best_tuning_params, inputs=[algo_sel, study_name_input], outputs=[best_params_output, download_json])
    
    stop_btn.click(stop_active_process, outputs=[unified_logs])
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
    demo.queue().launch(server_name="0.0.0.0", server_port=7860)
