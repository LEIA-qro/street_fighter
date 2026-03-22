import os
import time   
import multiprocessing
import gc
import torch 

def _nuclear_cleanup(model=None, env=None):
    print("Executing Failsafe: Purging zombie instances and VRAM...")
    """Reusable teardown: kill emulators, zombie workers, flush VRAM."""
    
    # 1. Kill Emulators
    os.system("taskkill /F /IM EmuHawk.exe >nul 2>&1")
    time.sleep(2)

    # 2. The Thread Sniper
    active_children = multiprocessing.active_children()
    if active_children:
        print(f"Force-killing {len(active_children)} zombie Python worker processes...")
        for child in active_children:
            try: child.kill()
            except Exception: pass

    # 3. The VRAM Purge
    try:
        if model is not None: del model
        if env is not None:   del env
    except Exception: pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    time.sleep(5)
