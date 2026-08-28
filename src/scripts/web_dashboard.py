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
from envs.action_macros import N_ACTIONS as STAND_N_ACTIONS
from scripts.stand_leia import (
    DEFAULT_CHECKPOINT as STAND_DEFAULT_CHECKPOINT,
    OPPONENTS as STAND_OPPONENTS,
)

# Virtual Environment Detection / Setup
VENV_PYTHON = os.path.join(config.PROJECT_ROOT, ".venv", "Scripts", "python.exe")
if not os.path.exists(VENV_PYTHON):
    VENV_PYTHON = os.path.join(config.PROJECT_ROOT, ".venv", "bin", "python")
    if not os.path.exists(VENV_PYTHON):
        VENV_PYTHON = sys.executable 

# Global state for background processes
class GlobalState:
    def __init__(self):
        self.active_process = None
        self.launch_token = None
        self.cleanup_in_progress = False
        self.stop_event = threading.Event()
        self.process_lock = threading.Lock()

state = GlobalState()

DASHBOARD_BUILD_ID = "v1404-additive-r3"

# Una pestaña Gradio abierta conserva el schema de componentes aunque el
# proceso de Python se reinicie. Si el app_id cambia, navegar con una query
# nueva fuerza a descargar el frontend correspondiente al backend vigente.
_DASHBOARD_RELOAD_HEAD = r'''<script>
(() => {
  const cfg = window.gradio_config || {};
  let loadedAppId = cfg.app_id;
  const root = (cfg.root || window.location.origin).replace(/\/$/, "");
  const prefix = cfg.api_prefix || "/gradio_api";
  const endpoint = `${root}${prefix}/app_id`;
  document.documentElement.dataset.leiaWatcherApp = String(loadedAppId || "pending");
  console.info(`[LEIA] reload watcher active for app ${loadedAppId || "pending"}`);
  const timer = window.setInterval(async () => {
    try {
      const response = await fetch(`${endpoint}?_=${Date.now()}`, {
        cache: "no-store"
      });
      if (!response.ok) return;
      const {app_id: liveAppId} = await response.json();
      if (!loadedAppId) {
        loadedAppId = liveAppId;
        document.documentElement.dataset.leiaWatcherApp = String(liveAppId);
        return;
      }
      if (liveAppId && liveAppId !== loadedAppId) {
        console.info(`[LEIA] app changed ${loadedAppId} -> ${liveAppId}; reloading`);
        window.clearInterval(timer);
        const fresh = new URL(window.location.href);
        fresh.searchParams.set("_app", String(liveAppId));
        window.location.replace(fresh.toString());
      }
    } catch (_) {
      // Un restart breve puede rechazar una consulta; el siguiente tick reintenta.
    }
  }, 3000);
})();
</script>'''


def _finish_emulator_cleanup(proc=None):
    try:
        from core.env_tools import failsafe_env
        failsafe_env(ignore_gate=True)
    except Exception:
        pass
    finally:
        with state.process_lock:
            if proc is not None and state.active_process is proc:
                state.active_process = None
            state.cleanup_in_progress = False


def _drain_detached_process(proc):
    """Reapea un hijo cuyo cliente Gradio cerró sin perder el botón Stop."""
    try:
        if proc.stdout:
            for _line in iter(proc.stdout.readline, ""):
                pass
        proc.wait()
    except Exception:
        pass
    finally:
        if proc.stdout:
            try:
                proc.stdout.close()
            except Exception:
                pass
        with state.process_lock:
            if state.active_process is proc:
                state.active_process = None


def _clear_stale_stop_marker():
    """Un lanzamiento nuevo no debe heredar el Stop de una sesión anterior."""
    stop_file = os.path.join(config.PROJECT_ROOT, ".stop_training")
    try:
        os.remove(stop_file)
    except FileNotFoundError:
        pass


STAND_CHECKPOINT_DIRS = (
    Path(config.PROJECT_ROOT) / "benchmarks" / "apex_milestones",
    Path(config.PROJECT_ROOT) / "models" / "rainbow_apex",
)
_stand_checkpoint_meta_cache = {}
_stand_checkpoint_meta_lock = threading.Lock()

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


def _load_stand_checkpoint_meta(file_path):
    """Carga segura y valida arquitectura + pesos del QR-DQN de exhibición."""
    from collections.abc import Mapping

    import torch
    from agents.rainbow import QRDuelingNet
    from es.policy import OBS_DIM, ONEHOT_OBS_DIM

    path = Path(file_path).resolve(strict=True)
    stat = path.stat()
    cache_key = (str(path), stat.st_mtime_ns, stat.st_size)
    with _stand_checkpoint_meta_lock:
        cached = _stand_checkpoint_meta_cache.get(cache_key)
    if cached is not None:
        return dict(cached)

    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(ckpt, dict) or not isinstance(ckpt.get("meta"), dict):
        raise ValueError("checkpoint sin metadata")
    if not isinstance(ckpt.get("state_dict"), Mapping):
        raise ValueError("checkpoint sin state_dict")

    meta = dict(ckpt["meta"])
    try:
        in_dim = int(meta["in_dim"])
        n_actions = int(meta["n_actions"])
        n_quantiles = int(meta["quantiles"])
        hidden = int(meta["hidden"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"metadata arquitectónica incompleta: {exc}") from None

    if not meta.get("macros", False) or n_actions != STAND_N_ACTIONS:
        raise ValueError(
            f"checkpoint incompatible: requiere macros y {STAND_N_ACTIONS} acciones")
    expected_in_dim = ONEHOT_OBS_DIM if bool(meta.get("onehot", True)) else OBS_DIM
    if in_dim != expected_in_dim:
        raise ValueError(
            f"entrada incompatible: {in_dim}; esperaba {expected_in_dim}")
    if n_quantiles <= 0 or hidden <= 0:
        raise ValueError("quantiles y hidden deben ser positivos")

    net = QRDuelingNet(
        in_dim,
        n_actions=n_actions,
        n_quantiles=n_quantiles,
        hidden=hidden,
    )
    try:
        net.load_state_dict(ckpt["state_dict"], strict=True)
    except RuntimeError as exc:
        raise ValueError(f"pesos incompatibles con QRDuelingNet: {exc}") from None

    with _stand_checkpoint_meta_lock:
        for old_key in list(_stand_checkpoint_meta_cache):
            if old_key[0] == str(path) and old_key != cache_key:
                del _stand_checkpoint_meta_cache[old_key]
        _stand_checkpoint_meta_cache[cache_key] = dict(meta)
    return meta


def _stand_sidecar_metrics(checkpoint_path):
    import json

    sidecar = Path(str(checkpoint_path) + ".json")
    if not sidecar.is_file():
        return {}
    try:
        with sidecar.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def get_stand_checkpoint_files(search_roots=None):
    """Descubre solo checkpoints QR-DQN compatibles con el stand.

    Es un inventario separado del flujo SB3 (.zip/.pkl): estos modelos son
    QRDuelingNet de Ape-X/Rainbow en .pt, con observacion v4 y macro-acciones.
    """
    project_root = Path(config.PROJECT_ROOT).resolve()
    roots = STAND_CHECKPOINT_DIRS if search_roots is None else search_roots
    compatible = []
    seen = set()

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        candidates = [root_path] if root_path.is_file() else root_path.rglob("*.pt")
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
                relative = resolved.relative_to(project_root).as_posix()
            except (OSError, ValueError):
                continue
            if relative in seen:
                continue
            seen.add(relative)
            try:
                _load_stand_checkpoint_meta(resolved)
            except Exception:
                continue
            compatible.append(relative)

    return sorted(compatible)


def get_stand_default_checkpoint(checkpoints=None):
    """Elige el alias vigente; si falta, prioriza cobertura y WR del sidecar."""
    choices = list(get_stand_checkpoint_files() if checkpoints is None else checkpoints)
    default_rel = Path(STAND_DEFAULT_CHECKPOINT).as_posix()
    if default_rel in choices:
        return default_rel
    if not choices:
        return None

    def score(relative):
        path = Path(config.PROJECT_ROOT) / relative
        metrics = _stand_sidecar_metrics(path)
        levels = sum(1 for key in metrics if re.fullmatch(r"wr_lvl[1-8]", key))
        try:
            win_rate = float(metrics.get("wr_media", -1.0))
        except (TypeError, ValueError):
            win_rate = -1.0
        try:
            modified = path.stat().st_mtime_ns
        except OSError:
            modified = 0
        return levels, win_rate, modified

    return max(choices, key=score)


def _resolve_stand_checkpoint(checkpoint):
    if checkpoint in (None, "", "None"):
        raise ValueError("selecciona un checkpoint .pt")

    project_root = Path(config.PROJECT_ROOT).resolve()
    candidate = Path(checkpoint)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(project_root).as_posix()
    except (OSError, ValueError):
        raise ValueError("el checkpoint debe existir dentro del proyecto") from None
    if resolved.suffix.lower() != ".pt":
        raise ValueError("el checkpoint Ape-X debe terminar en .pt")
    allowed = False
    for root in STAND_CHECKPOINT_DIRS:
        try:
            resolved.relative_to(Path(root).resolve())
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise ValueError("el checkpoint no pertenece al inventario QR-DQN permitido")
    meta = _load_stand_checkpoint_meta(resolved)
    return resolved, relative, meta


def get_stand_checkpoint_status(checkpoint):
    """Resumen legible de arquitectura y banco para la tarjeta del dashboard."""
    try:
        path, relative, meta = _resolve_stand_checkpoint(checkpoint)
    except Exception as exc:
        return f"⚠️ **Checkpoint no disponible:** {exc}"

    metrics = _stand_sidecar_metrics(path)
    title = "Campeón vigente de la escalera" if (
        relative == Path(STAND_DEFAULT_CHECKPOINT).as_posix()
    ) else "Checkpoint compatible"
    lines = [
        f"✅ **{title}:** `{relative}`",
        (f"QR-DQN Ape-X · entrada `{meta.get('in_dim', '?')}` · "
         f"`{meta.get('n_actions', '?')}` acciones (63 primitivas + 9 macros) · "
         f"`{meta.get('quantiles', '?')}` cuantiles"),
    ]

    reported_levels = [
        level for level in range(1, 9) if f"wr_lvl{level}" in metrics]
    if "wr_media" in metrics:
        try:
            wr_label = (
                "WR del selector robusto (8 niveles, desfase ≤30)"
                if len(reported_levels) == 8 else
                f"WR reportado en sidecar ({len(reported_levels)} niveles)"
                if reported_levels else
                "WR reportado en sidecar"
            )
            lines.append(
                f"**{wr_label}:** "
                f"{100.0 * float(metrics['wr_media']):.1f}%")
        except (TypeError, ValueError):
            pass
    if "weights_version" in metrics:
        lines.append(f"**Versión de pesos:** {metrics['weights_version']}")
    level_rates = []
    for level in reported_levels:
        key = f"wr_lvl{level}"
        if key in metrics:
            try:
                level_rates.append(f"L{level} {100.0 * float(metrics[key]):.1f}%")
            except (TypeError, ValueError):
                pass
    if level_rates:
        lines.append("**Escalera:** " + " · ".join(level_rates))
    return "  \n".join(lines)


def refresh_stand_checkpoints(current=None):
    choices = get_stand_checkpoint_files()
    selected = current if current in choices else get_stand_default_checkpoint(choices)
    return gr.update(choices=choices, value=selected), get_stand_checkpoint_status(selected)

def get_all_state_files():
    """Scans STATES_DIR for all available .State or .state files dynamically."""
    states_dir = config.STATES_DIR
    if not os.path.exists(states_dir):
        return ["None"]
    state_files = glob.glob(os.path.join(states_dir, "*.State"))
    state_files.extend(glob.glob(os.path.join(states_dir, "*.state")))
    names = sorted(list(set([os.path.basename(f) for f in state_files])))
    return ["None"] + names

def stream_logs(cmd, before_start=None):
    """Executes a command and yields output live for Gradio with unbuffered I/O."""
    launch_token = object()
    proc = None
    preparation_error = None
    with state.process_lock:
        busy = (state.active_process is not None
                or state.launch_token is not None
                or state.cleanup_in_progress)
        if not busy:
            # Reservar antes del primer yield cierra la carrera entre dos
            # botones Launch y permite que Stop cancele un arranque pendiente.
            state.launch_token = launch_token
            state.stop_event.clear()
            if before_start is not None:
                try:
                    # Efectos de preparación (p. ej. .agent_state) ocurren
                    # solo después de reservar el slot. Un segundo Launch
                    # rechazado nunca debe alterar el combate ya activo.
                    before_start()
                except Exception as exc:
                    state.launch_token = None
                    preparation_error = exc

    if busy:
        yield "Error: A process is already running!"
        return
    if preparation_error is not None:
        yield f"Error preparing process: {preparation_error}"
        return

    # Ensure -u unbuffered flag is present if python command
    if len(cmd) > 0 and "python" in os.path.basename(cmd[0]).lower() and "-u" not in cmd:
        cmd = [cmd[0], "-u"] + cmd[1:]

    full_output = f"Executing: {' '.join(cmd)}\n{'-'*50}\n"
    try:
        yield full_output

        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        process_group_args = (
            {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
            if os.name == "nt" else
            {"start_new_session": True}
        )

        with state.process_lock:
            if state.launch_token is not launch_token or state.stop_event.is_set():
                if state.launch_token is launch_token:
                    state.launch_token = None
                cancelled = True
            else:
                # Se hace inmediatamente antes de Popen y bajo el mismo lock:
                # cualquier Stop posterior pertenece inequívocamente al hijo.
                _clear_stale_stop_marker()
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    bufsize=1,
                    shell=False,
                    env=env,
                    **process_group_args,
                )
                state.active_process = proc
                state.launch_token = None
                cancelled = False

        if cancelled:
            yield full_output + "\n🛑 Launch cancelled before the process started."
            return
        
        for line in iter(proc.stdout.readline, ''):
            if not line:
                break
            full_output += line
            yield full_output
            
        proc.wait()
        if state.stop_event.is_set():
            full_output += f"\n{'-'*50}\n🛑 Process stopped by user."
        else:
            full_output += f"\n{'-'*50}\nProcess finished with exit code {proc.returncode}"
        yield full_output
    except Exception as e:
        yield full_output + f"\n[ERROR] {str(e)}"
    finally:
        detached_live_process = proc is not None and proc.poll() is None
        if detached_live_process:
            # Un reload/desconexión cierra el generador, no necesariamente el
            # combate. Drenar evita bloquear el pipe y conservar active_process
            # permite que el siguiente cliente todavía use Terminate Match.
            threading.Thread(
                target=_drain_detached_process, args=(proc,), daemon=True).start()
        else:
            if proc and proc.stdout:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
            with state.process_lock:
                if proc is not None and state.active_process is proc:
                    state.active_process = None
        with state.process_lock:
            if state.launch_token is launch_token:
                state.launch_token = None

def graceful_stop_process():
    """Gracefully stops the active process by writing .stop_training and waiting for emergency model save."""
    with state.process_lock:
        if state.cleanup_in_progress:
            return "Emulator cleanup is already in progress."
        state.stop_event.set()
        proc = state.active_process
        launch_pending = proc is None and state.launch_token is not None
        if launch_pending:
            # Invalida solamente esa reserva; un launch posterior tendrá otro
            # token y el finally viejo no podrá borrar su proceso.
            state.launch_token = None
        else:
            state.cleanup_in_progress = True

    if launch_pending:
        return "🛑 Pending process launch cancelled before startup."
    if proc is None:
        threading.Thread(target=_finish_emulator_cleanup, daemon=True).start()
        return "No active process was running. Cleaned up any lingering emulator instances."
    
    # 1. Write the file-based stop trigger to the project root
    stop_file = os.path.join(config.PROJECT_ROOT, ".stop_training")
    try:
        with open(stop_file, "w") as f:
            f.write("STOP")
        print(f"[Dashboard] Graceful stop signal written to {stop_file}")
    except Exception as e:
        print(f"[Dashboard] Error writing stop trigger file: {e}")

    # 2. Poll and wait up to 30 seconds for the training process to consume the file and save _EMERGENCY.zip
    emergency_saved = False
    for elapsed in range(30):
        if proc.poll() is not None:
            break
        # After 5 seconds, if the process is blocked on a long socket wait, send backup CTRL_BREAK
        if elapsed == 5 and proc.poll() is None:
            try:
                print(f"[Dashboard] Sending backup Graceful Stop signal (CTRL_BREAK) to PID {proc.pid}...")
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
            except Exception as e:
                print(f"[Dashboard] Backup signal note: {e}")
        time.sleep(1)
    
    # Clean up the stop file if still present
    if os.path.exists(stop_file):
        try:
            os.remove(stop_file)
        except Exception:
            pass
        
    # Check if this process was a match evaluation or model test (inference only)
    cmd_str = " ".join(proc.args) if hasattr(proc, "args") and proc.args else ""
    is_match_evaluation = any(marker in cmd_str for marker in (
        "test_agent", "test_ai_vs_ai", "stand_leia"))

    # Check if emergency files exist on disk in candidate production directories (training runs only)
    model_name = getattr(config, "MODEL_NAME", "model")
    candidate_dirs = [
        os.path.join(config.PROJECT_ROOT, "models", "production", "v3", "ppo"),
        os.path.join(config.PROJECT_ROOT, "models", "production", "v2", "ppo"),
        os.path.join(config.PROJECT_ROOT, "models", "production", "v3", "sac"),
        os.path.join(config.PROJECT_ROOT, "models", "production", "v2", "sac"),
        os.path.join(config.PROJECT_ROOT, "models", "production", "v3", "dqn"),
        os.path.join(config.PROJECT_ROOT, "models", "production", "v2", "dqn"),
    ]
    saved_file_name = None
    if not is_match_evaluation:
        for c_dir in candidate_dirs:
            e_path = os.path.join(c_dir, f"{model_name}_EMERGENCY.zip")
            if os.path.exists(e_path):
                # Check if modified within the last 60 seconds
                if time.time() - os.path.getmtime(e_path) < 60:
                    saved_file_name = f"{model_name}_EMERGENCY.zip"
                    emergency_saved = True
                    break
        
    # 3. If still running after 30s timeout, force terminate
    if proc.poll() is None:
        print(f"[Dashboard] Process {proc.pid} timed out after 30s during graceful stop. Force terminating...")
        subprocess.run(f"taskkill /F /T /PID {proc.pid}", shell=True, capture_output=True)
        msg = f"⚠️ Graceful stop timed out after 30s. Process {proc.pid} force terminated."
    else:
        print(f"[Dashboard] Process {proc.pid} gracefully stopped.")
        if is_match_evaluation:
            msg = "🛑 **Match Test Stopped**: Process exited cleanly. No model weights were modified."
        elif emergency_saved:
            msg = f"✅ **Graceful Stop Complete**: Emergency model successfully saved to disk as `{saved_file_name}`."
        else:
            msg = f"🛑 **Process Stopped**: Process {proc.pid} exited. Check terminal output for checkpoint details."
        
    # El slot sigue reservado durante el sniper: un relanzamiento no puede ser
    # eliminado por el cleanup del proceso anterior.
    _finish_emulator_cleanup(proc)
        
    return msg

def force_kill_process():
    """Immediately force-kills all active Python and BizHawk processes without saving."""
    with state.process_lock:
        if state.cleanup_in_progress:
            return "Emulator cleanup is already in progress."
        state.stop_event.set()
        proc = state.active_process
        launch_pending = proc is None and state.launch_token is not None
        if launch_pending:
            state.launch_token = None
        state.cleanup_in_progress = True
    pid_str = "None"
    
    if proc is not None:
        pid_str = str(proc.pid)
        try:
            print(f"[Dashboard] Force killing process tree for PID {proc.pid}...")
            subprocess.run(f"taskkill /F /T /PID {proc.pid}", shell=True, capture_output=True)
        except Exception as e:
            print(f"[Dashboard] Error force-killing PID {proc.pid}: {e}")
    # Trigger global process sniper; mantiene bloqueado Launch hasta terminar.
    threading.Thread(
        target=_finish_emulator_cleanup, args=(proc,), daemon=True).start()
        
    return f"⚡ **Force Kill Executed**: Terminated process (PID: {pid_str}) and all BizHawk emulator instances immediately without saving."

def stop_active_process():
    """Default stop handler (aliases to graceful_stop_process)."""
    return graceful_stop_process()

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

def run_matchup(p1_algo, p1_env, p1_zip, p1_pkl, p1_device, p2_algo, p2_env, p2_zip, p2_pkl, p2_device, profile_enabled, infinite_match_enabled=False, rematch_delay=2.0, cpu_level_cap=5):
    ai_algos = ["ppo", "sac", "dqn"]
    p1_is_ai = p1_algo in ai_algos
    p2_is_ai = p2_algo in ai_algos

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
    if infinite_match_enabled:
        cmd += ["--infinite_match", "--rematch_delay", str(float(rematch_delay))]
        if (p1_is_ai and not p2_is_ai and p2_algo == "CPU (Built-in AI)") or (p2_is_ai and not p1_is_ai and p1_algo == "CPU (Built-in AI)"):
            cmd += ["--cpu_level_cap", str(int(cpu_level_cap))]
        
    def initialize_agent_state():
        # PLAY para rematch automático; PAUSE para navegar el menú manual.
        state_file = os.path.join(config.PROJECT_ROOT, ".agent_state")
        initial_mode = "PLAY" if infinite_match_enabled else "PAUSE"
        with open(state_file, "w") as f:
            f.write(initial_mode)

    for log in stream_logs(cmd, before_start=initialize_agent_state):
        yield log


def run_stand(checkpoint, opponent, rematch_delay, device):
    """Lanza humano-vs-QR-DQN sin pasar el .pt por el loader SB3."""
    try:
        _path, relative, _meta = _resolve_stand_checkpoint(checkpoint)
        opponent = str(opponent).upper()
        if opponent not in ("RANDOM",) + tuple(STAND_OPPONENTS):
            raise ValueError(f"rival no válido: {opponent}")
        rematch_delay = float(rematch_delay)
        if rematch_delay < 0:
            raise ValueError("el rematch delay no puede ser negativo")
        device = str(device).lower()
        if device not in ("cpu", "cuda"):
            raise ValueError("el dispositivo de inferencia debe ser cpu o cuda")
        if device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise ValueError("CUDA no está disponible en esta máquina")
    except Exception as exc:
        yield f"Error de configuración del modelo Ape-X: {exc}"
        return

    cmd = [
        VENV_PYTHON,
        os.path.join(config.SRC_DIR, "scripts", "stand_leia.py"),
        "--ckpt", relative,
        "--opponent", opponent,
        "--rematch-delay", str(rematch_delay),
        "--device", device,
    ]
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

def save_all_config(n_envs, win_rate, steps, port, input_display, activate_viz, enable_throttling, throttle_speed):
    updates = {
        "N_ENVS": int(n_envs),
        "WIN_RATE_THRESHOLD": win_rate,
        "STARTING_TOTAL_TIMESTEPS": int(steps),
        "PORT": int(port),
        "ENABLE_INPUT_DISPLAY": input_display,
        "ACTIVATE_VISUALIZATION": activate_viz,
        "ENABLE_THROTTLING": enable_throttling,
        "THROTTLE_SPEED": int(throttle_speed)
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
        import json
        file_path = file_obj.name if hasattr(file_obj, "name") else file_obj
        filename = os.path.basename(file_path)
        target_dir = os.path.join(config.PROJECT_ROOT, "models", "production", env, algo)
        os.makedirs(target_dir, exist_ok=True)
        
        target_path = os.path.join(target_dir, filename)
        
        # If it's a JSON file, validate format first
        if filename.endswith(".json"):
            try:
                with open(file_path, "r") as test_f:
                    json.load(test_f)
            except Exception as json_err:
                return f"**Error: Invalid JSON format:** {json_err}", gr.update(), gr.update()

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

def _resolve_active_curriculum_path(algo, env):
    """Dynamically resolves the newest active auto_curriculum_state JSON file by mtime."""
    # Check selected env/algo directory first, then fallback across v3/v2
    candidate_dirs = [
        os.path.join(config.PROJECT_ROOT, "models", "production", env, algo),
        os.path.join(config.PROJECT_ROOT, "models", "production", "v3", algo),
        os.path.join(config.PROJECT_ROOT, "models", "production", "v2", algo),
        os.path.join(config.PROJECT_ROOT, "models", "production", algo),
    ]
    
    # Also check if config.MODEL_NAME points directly to a file
    model_name = getattr(config, "MODEL_NAME", None)
    
    for c_dir in candidate_dirs:
        if not os.path.exists(c_dir):
            continue
        
        # 1. Search for all auto_curriculum_state JSON files
        json_files = [
            os.path.join(c_dir, f)
            for f in os.listdir(c_dir)
            if f.startswith("auto_curriculum_state") and f.endswith(".json")
        ]
        
        if json_files:
            # Sort by newest modification timestamp to automatically latch onto the live active run
            json_files.sort(key=os.path.getmtime, reverse=True)
            return json_files[0]
            
    return None

def get_auto_curriculum_file(algo, env):
    """Resolves and returns the curriculum JSON file path for downloading."""
    try:
        state_path = _resolve_active_curriculum_path(algo, env)
        if not state_path or not os.path.exists(state_path):
            return gr.update(value=None, visible=False)
        return gr.update(value=state_path, visible=True)
    except Exception as e:
        print(f"[Dashboard][Error] Failed to resolve auto-curriculum file: {e}")
        return gr.update(value=None, visible=False)

def get_auto_curriculum_status_html(algo, env):
    """Parses auto_curriculum_state.json and renders a premium live progress and analytics card."""
    try:
        state_path = _resolve_active_curriculum_path(algo, env)
        
        if not state_path or not os.path.exists(state_path):
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
            
        # Extract model tag from file name
        base_filename = os.path.basename(state_path)
        model_display = base_filename.replace("auto_curriculum_state_", "").replace(".json", "")
        if model_display == "auto_curriculum_state":
            model_display = "Active"
            
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
            <div style='display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;'>
                <h3 style='margin: 0; display: flex; align-items: center; gap: 8px; font-size: 1.35rem; font-weight: 600; color: #3b82f6;'>
                    📈 Auto-Curriculum Analytics
                </h3>
                <span style='font-size: 0.85rem; font-weight: 500; color: #94a3b8; background: rgba(59, 130, 246, 0.1); padding: 4px 10px; border-radius: 8px; border: 1px solid rgba(59, 130, 246, 0.2);'>
                    Tracking: <strong style='color: #60a5fa;'>{model_display}</strong>
                </span>
            </div>
            
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

def compute_fighter_visual_coords(rel_x: int, rel_y: int, corner_dist: int, arena_width: int = 500) -> tuple[float, int, float, int]:
    """Computes faithful visual coordinates (percentages for X, pixels for Y) for P1 and P2.
    
    Uses smooth engagement-centered coordinate mapping (C-infinity continuous)
    to eliminate visual threshold popping while preserving true relative distance.
    
    Returns:
        (p1_x_pct, p1_y_px, p2_x_pct, p2_y_px)
    """
    # Vertical jump calculation (Floor = 24px, max jump elevation = 50px)
    # rel_y = p2_y - p1_y. SF2 jump delta is typically 60-120 units.
    y_scale = 50.0 / 120.0
    floor_px = 24
    
    if rel_y > 0:
        # P1 is airborne higher than P2
        p1_y_px = int(floor_px + min(50.0, rel_y * y_scale))
        p2_y_px = floor_px
    elif rel_y < 0:
        # P2 is airborne higher than P1
        p1_y_px = floor_px
        p2_y_px = int(floor_px + min(50.0, -rel_y * y_scale))
    else:
        p1_y_px = floor_px
        p2_y_px = floor_px

    # Engagement-Centered Horizontal Mapping:
    # Midpoint of the combatants is anchored at arena center (250px / 50%).
    # This guarantees smooth, continuous rendering with zero snapping or jitter.
    half_dist = float(rel_x) / 2.0
    p1_stage_x = 250.0 - half_dist
    p2_stage_x = 250.0 + half_dist

    # Scale to percentage with safe padding [4%, 96%]
    scale = 100.0 / float(arena_width)
    p1_x_pct = max(4.0, min(96.0, p1_stage_x * scale))
    p2_x_pct = max(4.0, min(96.0, p2_stage_x * scale))

    return p1_x_pct, p1_y_px, p2_x_pct, p2_y_px

_cached_telemetry_html = None
_cached_telemetry_time = 0.0

def get_live_telemetry_html():
    global _cached_telemetry_html, _cached_telemetry_time
    import json
    import os
    import time
    from core import config
    
    target_path = os.path.join(config.PROJECT_ROOT, ".telemetry.json")
    now = time.time()
    
    # Render Premium Standby UI if file is missing or stale
    if not os.path.exists(target_path):
        if _cached_telemetry_html and (now - _cached_telemetry_time) < 1.5:
            return _cached_telemetry_html
        return """
        <div style='background: #101626; border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 16px; padding: 48px; text-align: center; font-family: system-ui, sans-serif; color: #fff;'>
            <h3 style='margin: 0 0 8px 0; color: #3b82f6; font-size: 1.4rem;'>🔮 Standby Mode: Telemetry Offline</h3>
            <p style='color: #94a3b8; font-size: 0.95rem; margin: 0;'>Launch an interactive <strong>Match Test</strong> from the matchup panel to stream live agent observations and network activations.</p>
        </div>
        """
        
    try:
        with open(target_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        # If transient read collision occurs, seamlessly return recent cached HTML
        if _cached_telemetry_html and (now - _cached_telemetry_time) < 2.0:
            return _cached_telemetry_html
        return f"<div style='color: red; padding: 12px;'>Error reading telemetry data: {e}</div>"

    model_name = data.get("model_name", "Unknown")
    env_version = data.get("env_version", "v2")
    status = data.get("status", "PLAYING")
    value_est = data.get("value_estimate", 0.0)
    dist = data.get("policy_distributions", {})
    frames = data.get("frames", [])
    
    # Build 2x2 stacked frames grid
    frames_html = ""
    for idx, f in enumerate(frames):
        # SB3 stacks in oldest-to-newest. We reversed list so frames[0] is the current frame
        # Let's render frame index cleanly
        if idx == 0:
            card_title = "Frame t (Latest / Active)"
            card_class = "frame-card active-frame"
            border_color_style = "border-color: rgba(59, 130, 246, 0.45);"
        elif idx == 1:
            card_title = "Frame t-1"
            card_class = "frame-card"
            border_color_style = ""
        elif idx == 2:
            card_title = "Frame t-2"
            card_class = "frame-card"
            border_color_style = ""
        else:
            card_title = "Frame t-3 (Oldest)"
            card_class = "frame-card"
            border_color_style = ""

        p1_hp = f.get("p1_hp", 176)
        p2_hp = f.get("p2_hp", 176)
        p1_hp_pct = int((p1_hp / 176.0) * 100)
        p2_hp_pct = int((p2_hp / 176.0) * 100)

        # Stage coordinates translation logic using verified visual coordinate mapper
        rel_x = f.get("rel_x", 80)
        rel_y = f.get("rel_y", 0)
        p1_corner_dist = f.get("p1_corner_dist", 120)
        p1_proj = f.get("p1_proj", -1)
        p2_proj = f.get("p2_proj", -1)
        
        p1_pct, p1_bottom, p2_pct, p2_bottom = compute_fighter_visual_coords(
            rel_x=rel_x, rel_y=rel_y, corner_dist=p1_corner_dist
        )
        
        scale = 100.0 / 500.0
        proj1_html = ""
        if p1_proj > 0:
            p1_proj_pct = max(2, min(98, int(p1_proj * scale)))
            proj1_html = f"<div class='projectile p1-proj' style='position: absolute; bottom: {p1_bottom + 20}px; width: 10px; height: 10px; border-radius: 50%; background: #60a5fa; box-shadow: 0 0 12px #3b82f6, 0 0 6px #fff; left: {p1_proj_pct}%; transform: translateX(-50%);'></div>"

        proj2_html = ""
        if p2_proj > 0:
            p2_proj_pct = max(2, min(98, int(p2_proj * scale)))
            proj2_html = f"<div class='projectile p2-proj' style='position: absolute; bottom: {p2_bottom + 20}px; width: 10px; height: 10px; border-radius: 50%; background: #f87171; box-shadow: 0 0 12px #ef4444, 0 0 6px #fff; left: {p2_proj_pct}%; transform: translateX(-50%);'></div>"

        p1_char_name = f.get("p1_char_name", "AI")
        p2_char_name = f.get("p2_char_name", "OPP")

        frames_html += f"""
        <div class='{card_class}' style='background: rgba(15, 23, 42, 0.85); border: 1px solid rgba(255, 255, 255, 0.06); {border_color_style} border-radius: 14px; padding: 20px; position: relative;'>
            <div class='frame-label' style='font-size: 0.85rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.12em; margin-bottom: 12px; font-weight: 800; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 6px;'>{card_title}</div>
            
            <div class='hp-bar-container' style='display: flex; justify-content: space-between; font-size: 0.75rem; color: #94a3b8; margin-bottom: 6px; font-weight: bold;'>
                <span>{p1_char_name} ({p1_hp})</span>
                <span>{p2_char_name} ({p2_hp})</span>
            </div>
            <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
                <div class='hp-bar' style='width: 46%; height: 8px; background: #111827; border-radius: 4px; overflow: hidden;'><div class='hp-fill p1' style='height:100%; width: {p1_hp_pct}%; background: linear-gradient(90deg, #2563eb, #3b82f6);'></div></div>
                <div class='hp-bar' style='width: 46%; height: 8px; background: #111827; border-radius: 4px; overflow: hidden;'><div class='hp-fill p2' style='height:100%; width: {p2_hp_pct}%; background: linear-gradient(90deg, #dc2626, #ef4444);'></div></div>
            </div>

            <!-- Arena Vector View -->
            <div class='arena-view' style='width: 100%; height: 140px; background: #04060b; border-radius: 10px; position: relative; border: 1px solid rgba(255, 255, 255, 0.08); margin: 12px 0; overflow: hidden;'>
                <div class='corner-line left' style='position: absolute; top: 0; bottom: 24px; width: 3px; left: 12px; border-left: 2px dashed rgba(59, 130, 246, 0.4);'></div>
                <div class='corner-line right' style='position: absolute; top: 0; bottom: 24px; width: 3px; right: 12px; border-right: 2px dashed rgba(239, 68, 68, 0.4);'></div>
                <!-- Fighter P1 -->
                <div class='fighter p1' style='position: absolute; bottom: {p1_bottom}px; width: 26px; height: 56px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: 800; left: {p1_pct:.1f}%; transform: translateX(-50%); background: linear-gradient(180deg, rgba(59, 130, 246, 0.85), rgba(29, 78, 216, 0.85)); border: 1.5px solid #60a5fa; color: #fff; transition: bottom 0.05s ease, left 0.05s ease;' title='{p1_char_name} (AI)'>{p1_char_name[:3].upper()}</div>
                <!-- Fighter P2 -->
                <div class='fighter p2' style='position: absolute; bottom: {p2_bottom}px; width: 26px; height: 56px; border-radius: 6px; display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: 800; left: {p2_pct:.1f}%; transform: translateX(-50%); background: linear-gradient(180deg, rgba(239, 68, 68, 0.85), rgba(185, 28, 28, 0.85)); border: 1.5px solid #f87171; color: #fff; transition: bottom 0.05s ease, left 0.05s ease;' title='{p2_char_name} (OPP)'>{p2_char_name[:3].upper()}</div>
                {proj1_html}
                {proj2_html}
                <div class='arena-floor' style='position: absolute; bottom: 0; left: 0; width: 100%; height: 24px; background: repeating-linear-gradient(45deg, #131a2b, #131a2b 12px, #080b13 12px, #080b13 24px); border-top: 2px solid rgba(255, 255, 255, 0.12);'></div>
            </div>

            <!-- ALL POSSIBLE OBSERVATION FEATURE VALUES -->
            <div class='detailed-obs-grid' style='display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 0.75rem; text-align: left; margin-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 12px;'>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>P1 (AI) HP</span><span class='obs-val' style='color: #60a5fa; font-weight: bold;'>{p1_hp}</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>P2 (OPP) HP</span><span class='obs-val' style='color: #f87171; font-weight: bold;'>{p2_hp}</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>Relative X</span><span class='obs-val' style='color: #fff;'>{rel_x} px</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>Relative Y</span><span class='obs-val' style='color: #fff;'>{f.get("rel_y", 0)} px</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>AI Wall Dist</span><span class='obs-val' style='color: #fff;'>{p1_corner_dist} px</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>AI Proj X</span><span class='obs-val' style='color: #fff;'>{p1_proj if p1_proj > 0 else 'None'}</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>OPP Proj X</span><span class='obs-val' style='color: #fff;'>{p2_proj if p2_proj > 0 else 'None'}</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>AI Velocity X</span><span class='obs-val' style='color: #fff;'>{f.get("p1_vel_x", 0)} px/f</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>OPP Velocity X</span><span class='obs-val' style='color: #fff;'>{f.get("p2_vel_x", 0)} px/f</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>Absolute Dist</span><span class='obs-val' style='color: #fff;'>{f.get("rel_dist", 0)} px</span></div>
                
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>AI Action ID</span><span class='obs-val' style='color: #60a5fa;'>{f.get("p1_action_name", "N/A")}</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>OPP Action ID</span><span class='obs-val' style='color: #f87171;'>{f.get("p2_action_name", "N/A")}</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>AI Char ID</span><span class='obs-val' style='color: #60a5fa;'>{f.get("p1_char_name", "N/A")}</span></div>
                <div class='obs-item' style='display: flex; justify-content: space-between; background: rgba(0, 0, 0, 0.25); padding: 4px 8px; border-radius: 6px;'><span class='obs-label' style='color: #94a3b8;'>OPP Char ID</span><span class='obs-val' style='color: #f87171;'>{f.get("p2_char_name", "N/A")}</span></div>
            </div>
        </div>
        """

    # Build Neural Activation Panels
    activations_html = ""

    # Value Gauge render
    val_clamped = max(-50.0, min(60.0, value_est))
    val_pct = int(((val_clamped + 50.0) / 110.0) * 100)
    
    val_class = "positive" if value_est >= 0.0 else "negative"
    val_sign = "+" if value_est >= 0.0 else ""
    val_style_color = "#22c55e" if value_est >= 0.0 else "#ef4444"

    value_gauge_html = f"""
    <div class='value-gauge-container' style='background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(59, 130, 246, 0.2); border-radius: 12px; padding: 18px; text-align: center; margin-top: 12px;'>
        <div class='value-header' style='display: flex; justify-content: space-between; font-size: 0.8rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em; margin-bottom: 8px;'>
            <span>🧠 Value Function Estimate</span>
            <span class='char-badge' style='background: rgba(34, 197, 94, 0.15); border: 1px solid rgba(34, 197, 94, 0.3); color: #86efac; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 700;'>Critic Head</span>
        </div>
        <div class='value-display {val_class}' style='font-size: 1.6rem; font-weight: 900; color: {val_style_color}; font-family: monospace; margin-bottom: 12px;'>{val_sign}{value_est:.4f}</div>
        <div class='gauge-track' style='width: 100%; height: 12px; background: linear-gradient(90deg, #ef4444 0%, #1e293b 50%, #22c55e 100%); border-radius: 6px; position: relative; border: 1px solid rgba(255, 255, 255, 0.08);'>
            <div class='gauge-pointer' style='position: absolute; top: -4px; width: 6px; height: 20px; background-color: #ffffff; border: 1px solid #000; border-radius: 3px; box-shadow: 0 0 8px #ffffff; left: {val_pct}%;'></div>
        </div>
        <div class='gauge-labels' style='display: flex; justify-content: space-between; font-size: 0.65rem; font-weight: bold; color: #94a3b8; margin-top: 6px;'>
            <span style='color: #ef4444;'>-50.0 (Danger)</span>
            <span>0.0</span>
            <span style='color: #22c55e;'>+60.0 (Advantage)</span>
        </div>
    </div>
    """

    # Probability Bars rendering
    if env_version == "v3" and "directions" in dist:
        dir_names = ["Neutral (0)", "Up (1)", "Down (2)", "Left / Retreat (3)", "Right / Advance (4)", "Up-Left (5)", "Up-Right (6)", "Down-Left (7)", "Down-Right (8)"]
        dir_bars = ""
        for idx, p in enumerate(dist["directions"]):
            pct = int(p * 100)
            label_color = "color: #60a5fa; font-weight: bold;" if p == max(dist["directions"]) else ""
            dir_bars += f"""
            <div class='probability-bar-row' style='margin-bottom: 10px;'>
                <div class='bar-label-row' style='display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600; margin-bottom: 4px; {label_color}'><span>{dir_names[idx]}</span><span>{pct}%</span></div>
                <div class='bar-bg' style='width: 100%; height: 8px; background: #0c0f17; border-radius: 4px; overflow: hidden;'><div class='bar-fill' style='height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); width: {pct}%'></div></div>
            </div>
            """

        btn_names = ["Idle / Wait (0)", "A - Light Kick (1)", "B - Med Kick (2)", "C - Hard Kick (3)", "X - Light Punch (4)", "Y - Med Punch (5)", "Z - Hard Punch (6)"]
        btn_bars = ""
        for idx, p in enumerate(dist["buttons"]):
            pct = int(p * 100)
            label_color = "color: #60a5fa; font-weight: bold;" if p == max(dist["buttons"]) else ""
            btn_bars += f"""
            <div class='probability-bar-row' style='margin-bottom: 10px;'>
                <div class='bar-label-row' style='display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600; margin-bottom: 4px; {label_color}'><span>{btn_names[idx]}</span><span>{pct}%</span></div>
                <div class='bar-bg' style='width: 100%; height: 8px; background: #0c0f17; border-radius: 4px; overflow: hidden;'><div class='bar-fill' style='height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); width: {pct}%'></div></div>
            </div>
            """

        activations_html += f"""
        <div class='probability-group' style='background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 16px; margin-bottom: 12px;'>
            <h4 style='margin: 0 0 10px 0; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; display: flex; justify-content: space-between;'>
                <span>🕹️ Direction Select (Discrete)</span>
                <span class='char-badge' style='background: rgba(59, 130, 246, 0.15); color: #60a5fa; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem;'>MultiDiscrete[0]</span>
            </h4>
            {dir_bars}
        </div>
        <div class='probability-group' style='background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 16px; margin-bottom: 12px;'>
            <h4 style='margin: 0 0 10px 0; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; display: flex; justify-content: space-between;'>
                <span>🎯 Action Button (Discrete)</span>
                <span class='char-badge' style='background: rgba(59, 130, 246, 0.15); color: #60a5fa; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem;'>MultiDiscrete[1]</span>
            </h4>
            {btn_bars}
        </div>
        """
    elif env_version == "v2" and "buttons" in dist:
        btn_names = ["Up", "Down", "Left", "Right", "A (LK)", "B (MK)", "C (HK)", "X (LP)", "Y (MP)", "Z (HP)"]
        bin_bars = ""
        for idx, p in enumerate(dist["buttons"]):
            pct = int(p * 100)
            bin_bars += f"""
            <div class='probability-bar-row' style='margin-bottom: 10px;'>
                <div class='bar-label-row' style='display: flex; justify-content: space-between; font-size: 0.75rem; font-weight: 600; margin-bottom: 4px;'><span>{btn_names[idx]} Bit</span><span>{pct}% Confidence</span></div>
                <div class='bar-bg' style='width: 100%; height: 8px; background: #0c0f17; border-radius: 4px; overflow: hidden;'><div class='bar-fill' style='height: 100%; background: linear-gradient(90deg, #3b82f6, #60a5fa); width: {pct}%'></div></div>
            </div>
            """
        activations_html += f"""
        <div class='probability-group' style='background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 16px; margin-bottom: 12px;'>
            <h4 style='margin: 0 0 10px 0; font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; display: flex; justify-content: space-between;'>
                <span>🕹️ 10-Bit Button Probabilities</span>
                <span class='char-badge' style='background: rgba(59, 130, 246, 0.15); color: #60a5fa; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem;'>MultiBinary[10]</span>
            </h4>
            {bin_bars}
        </div>
        """
    else:
        activations_html += """
        <div class='probability-group' style='background: rgba(15, 23, 42, 0.65); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 12px; padding: 24px; text-align: center; color: #94a3b8;'>
            No policy activations available for this environment version or algorithm.
        </div>
        """

    active_player = data.get("player", 1)

    html = f"""
    <div style='background-color: #080b13; font-family: "Outfit", "Inter", sans-serif; color: #fff;'>
        <div style='margin-bottom: 20px; background: rgba(30, 41, 59, 0.4); border: 1px solid rgba(255, 255, 255, 0.05); padding: 14px 18px; border-radius: 12px; display: flex; justify-content: space-between; align-items: center;'>
            <div>
                <span style='font-size: 0.85rem; font-weight: 700; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;'>Active Model:</span>
                <span style='font-size: 1rem; font-weight: bold; margin-left: 8px; color: #3b82f6;'>{model_name} (Player {active_player})</span>
            </div>
            <div style='font-size: 0.85rem; color: #94a3b8;'>Environment: <strong style='color:#fff;'>SF2 {env_version.upper()}</strong></div>
        </div>

        <div style='display: grid; grid-template-columns: 2.5fr 1fr; gap: 24px;'>
            <!-- Stacked Frames (Left) -->
            <div>
                <div class='frames-grid' style='display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;'>
                    {frames_html}
                </div>
            </div>
            
            <!-- Brain / Activation Panel (Right) -->
            <div style='background: #101626; border: 1px solid rgba(59, 130, 246, 0.25); border-radius: 16px; padding: 24px; box-shadow: 0 4px 25px rgba(0, 0, 0, 0.6); display: flex; flex-direction: column; gap: 20px;'>
                <h3 style='margin: 0; font-size: 1.2rem; display: flex; align-items: center; gap: 8px; border-bottom: 1px solid rgba(255, 255, 255, 0.05); padding-bottom: 8px;'>🧠 Actor/Critic Head Distributions</h3>
                {activations_html}
                {value_gauge_html}
            </div>
        </div>
    </div>
    """
    _cached_telemetry_html = html
    _cached_telemetry_time = time.time()
    return html

# --- UI Construction ---

zips_init, pkls_init = get_model_files("ppo")
stand_checkpoints_init = get_stand_checkpoint_files()
stand_default_init = get_stand_default_checkpoint(stand_checkpoints_init)

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
                                ext_json_upload = gr.File(label="Upload Curriculum State (.json)", file_types=[".json"])
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
                                download_curr_btn = gr.Button("📥 Download Auto-Curriculum Analytics", variant="secondary")
                            download_curr_file = gr.File(label="Downloadable Curriculum State File", visible=False, interactive=False)
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
                        graceful_stop_btn = gr.Button("🛑 Graceful Stop (Save Model)", variant="stop")
                        force_kill_btn = gr.Button("⚡ Force Kill (No Save)", variant="secondary")
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
                        stop_league_btn = gr.Button("🛑 Graceful Stop League", variant="stop")
                        kill_league_btn = gr.Button("⚡ Force Kill League", variant="secondary")
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
                        infinite_match_checkbox = gr.Checkbox(label="🔄 Infinite Matchups (Auto-Rematch)", value=False)
                        rematch_delay_slider = gr.Slider(label="Rematch Delay (seconds)", minimum=1.0, maximum=5.0, value=2.0, step=0.5)
                        cpu_level_cap_slider = gr.Slider(label="CPU Max Level Cap (Infinite Match)", minimum=1, maximum=8, value=5, step=1)
                    
                    with gr.Row():
                        toggle_agent_btn = gr.Button("⏯️ Toggle Agent (Play/Pause)", variant="secondary")
                    
                    agent_state_status = gr.Markdown("Agent State: **PAUSED** (Default)")
                    
                    match_upload_status = gr.Markdown("")
                
                with gr.Column():
                    match_logs = gr.Textbox(label="Match Console", lines=25, max_lines=35, interactive=False, elem_id="terminal")
                    copy_match_btn = gr.Button("📋 Copy Match Logs", size="sm")

            # El viewer QR-DQN es aditivo: vive dentro de Model Testing sin
            # cambiar defaults, controles ni eventos del matchup clásico.
            with gr.Accordion("Ape-X QR-DQN vs Human (Viewer)", open=False):
                gr.Markdown(
                    "Prueba el campeón `.pt` contra un retador con el control "
                    "físico configurado como Player 2 en BizHawk."
                )
                with gr.Row():
                    apex_checkpoint = gr.Dropdown(
                        label="Ape-X checkpoint (.pt)",
                        choices=stand_checkpoints_init,
                        value=stand_default_init,
                        scale=3,
                    )
                    apex_device = gr.Dropdown(
                        label="Compute Device",
                        choices=["cpu", "cuda"],
                        value="cpu",
                        scale=1,
                    )
                    apex_human_character = gr.Dropdown(
                        label="Human Character (P2)",
                        choices=["RANDOM"] + list(STAND_OPPONENTS),
                        value="RANDOM",
                        scale=1,
                    )
                apex_checkpoint_status = gr.Markdown(
                    get_stand_checkpoint_status(stand_default_init)
                )
                with gr.Row():
                    apex_rematch_delay = gr.Slider(
                        label="Rematch Delay (seconds)",
                        minimum=1.0,
                        maximum=5.0,
                        value=2.0,
                        step=0.5,
                    )
                    refresh_apex_btn = gr.Button(
                        "🔄 Refresh Ape-X checkpoints", variant="secondary")
                with gr.Row():
                    launch_apex_btn = gr.Button(
                        "🥊 Launch Ape-X vs Human", variant="primary")
                    stop_apex_btn = gr.Button(
                        "🛑 Terminate Ape-X Match", variant="stop")
                apex_logs = gr.Textbox(
                    label="Ape-X Viewer Console",
                    lines=15,
                    max_lines=25,
                    interactive=False,
                    elem_id="terminal",
                )
            
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

            def update_infinite_match_status(is_infinite):
                if is_infinite:
                    return "Agent State: **PLAYING** (Auto)"
                return "Agent State: **PAUSED** (Default)"

            infinite_match_checkbox.change(update_infinite_match_status, inputs=[infinite_match_checkbox], outputs=[agent_state_status])

            # Link matchup uploaders
            p1_zip_upload.upload(handle_model_upload, inputs=[p1_zip_upload, p1_algo, p1_env], outputs=[match_upload_status, p1_zip, p1_pkl])
            p1_pkl_upload.upload(handle_model_upload, inputs=[p1_pkl_upload, p1_algo, p1_env], outputs=[match_upload_status, p1_zip, p1_pkl])
            p2_zip_upload.upload(handle_model_upload, inputs=[p2_zip_upload, p2_algo, p2_env], outputs=[match_upload_status, p2_zip, p2_pkl])
            p2_pkl_upload.upload(handle_model_upload, inputs=[p2_pkl_upload, p2_algo, p2_env], outputs=[match_upload_status, p2_zip, p2_pkl])

            launch_match_btn.click(
                run_matchup, 
                inputs=[
                    p1_algo, p1_env, p1_zip, p1_pkl, p1_device, 
                    p2_algo, p2_env, p2_zip, p2_pkl, p2_device, 
                    match_profile_checkbox, infinite_match_checkbox, rematch_delay_slider,
                    cpu_level_cap_slider
                ], 
                outputs=[match_logs]
            )

            stop_match_btn.click(stop_match_process, outputs=[match_logs, agent_state_status])
            toggle_agent_btn.click(toggle_agent_state, outputs=[agent_state_status])

            apex_checkpoint.change(
                get_stand_checkpoint_status,
                inputs=[apex_checkpoint],
                outputs=[apex_checkpoint_status],
            )
            refresh_apex_btn.click(
                refresh_stand_checkpoints,
                inputs=[apex_checkpoint],
                outputs=[apex_checkpoint, apex_checkpoint_status],
            )
            launch_apex_btn.click(
                run_stand,
                inputs=[
                    apex_checkpoint, apex_human_character,
                    apex_rematch_delay, apex_device,
                ],
                outputs=[apex_logs],
            )
            stop_apex_btn.click(
                stop_match_process,
                outputs=[apex_logs, agent_state_status],
            )

        # --- TAB 2.5: TELEMETRY ---
        with gr.Tab("🔮 Observation Telemetry"):
            gr.Markdown("### Real-time Agent Observations & Network Activations")
            gr.Markdown("This live dashboard prints stacked chronological frames, relative bounding boxes on stage, actor log-probabilities, and state value gauges during active match tests.")
            
            telemetry_html = gr.HTML(value=get_live_telemetry_html())
            
            # Poll every 100ms
            telemetry_timer = gr.Timer(value=0.1, active=True)
            telemetry_timer.tick(get_live_telemetry_html, outputs=[telemetry_html])

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
                    cfg_enable_throttling = gr.Checkbox(label="Enable Emulator Speed Throttling (Limits CPU - Applies to Training Only)", value=getattr(config, 'ENABLE_THROTTLING', False))
                    cfg_throttle_speed = gr.Slider(label="Training Throttle Speed % (e.g. 100=Normal, 200=Double - Applies to Training Only)", minimum=50, maximum=1000, value=getattr(config, 'THROTTLE_SPEED', 200), step=10)
                    
                    save_cfg_btn = gr.Button("💾 Save Configuration", variant="primary")
                    cfg_status = gr.Markdown("")

                with gr.Column():
                    gr.Markdown("### 📂 State Management")
                    state_upload = gr.File(label="Upload Custom Savestates (.State)", file_types=[".State"], file_count="multiple")
                    state_upload_status = gr.Markdown("")
            
            save_cfg_btn.click(save_all_config, inputs=[cfg_n_envs, cfg_win_rate, cfg_steps, cfg_port, cfg_input_display, cfg_activate_visualization, cfg_enable_throttling, cfg_throttle_speed], outputs=[cfg_status])

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
    ext_json_upload.upload(handle_model_upload, inputs=[ext_json_upload, algo_sel, env_sel], outputs=[upload_status, train_zip_drop, train_pkl_drop])

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

    graceful_stop_btn.click(graceful_stop_process, outputs=[stop_status])
    force_kill_btn.click(force_kill_process, outputs=[stop_status])
    refresh_files_btn.click(refresh_dropdowns, outputs=[train_zip_drop, train_pkl_drop, tune_zip_drop, tune_pkl_drop, pbt_zip_drop, pbt_pkl_drop, p1_zip, p1_pkl, p2_zip, p2_pkl])
    refresh_curr_btn.click(get_auto_curriculum_status_html, inputs=[algo_sel, env_sel], outputs=[auto_curr_card])
    download_curr_btn.click(get_auto_curriculum_file, inputs=[algo_sel, env_sel], outputs=[download_curr_file])
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
        graceful_stop_process, 
        outputs=[league_logs]
    )
    kill_league_btn.click(
        force_kill_process,
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

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Street Fighter II RL Gradio Web Control Center")
    parser.add_argument("--host", "--server_name", dest="server_name", type=str, default="0.0.0.0", help="Server host address (default: 0.0.0.0)")
    parser.add_argument("--port", "--server_port", dest="server_port", type=int, default=7860, help="Server port number (default: 7860)")
    parser.add_argument("--share", action="store_true", help="Generate public shareable Gradio link")
    args = parser.parse_args()

    print(f"[Dashboard] build {DASHBOARD_BUILD_ID}", flush=True)
    demo.queue().launch(
        server_name=args.server_name, 
        server_port=args.server_port, 
        share=args.share,
        theme=gr.themes.Soft(primary_hue="blue"), 
        css="#terminal textarea { font-family: monospace; }",
        head=_DASHBOARD_RELOAD_HEAD,
    )

if __name__ == "__main__":
    main()
