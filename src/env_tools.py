import os, time, multiprocessing, gc, torch
from stable_baselines3.common.monitor import Monitor

from env_sf2_v2 import StreetFighterEnvV2
import config


def SFv2_make_env(rank):
    def _init():
        env = StreetFighterEnvV2(rank=rank)
        log_dir = os.path.join(config.LOG_DIR, f"monitor_rank_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        return Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    return _init

def failsafe_env(env=None, model=None):

    if env is not None:
        try:
            env.close()
            del env
        except UnboundLocalError:
            pass
        except Exception:
            pass

    if model is not None:
        try:
            del model
        except UnboundLocalError:
            pass

    print("[ENV]Executing Failsafe: Purging zombie instances and VRAM...")
    # 1. Kill Emulators
    os.system("taskkill /F /IM EmuHawk.exe >nul 2>&1")
    time.sleep(2)
    
    # 2. The Thread Sniper
    active_children = multiprocessing.active_children()
    if active_children:
        for child in active_children:
            try:
                child.kill()
            except Exception:
                pass
    
    # 3. The VRAM Purge
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    time.sleep(3)
    print("[ENV] Failsafe complete. All zombie processes terminated and VRAM cleared.")