import os, time, multiprocessing, gc

import core.config as config

def SFv2_make_env(rank, **kwargs):
    from stable_baselines3.common.monitor import Monitor
    from envs.sf2_v2 import StreetFighterEnvV2
    
    version = kwargs.get("version", "v2")
    player = kwargs.get("player", 1)
    
    def _init():
        # STAGGERED BOOT: Delay starting the emulator based on rank
        # Prevents 10 instances from hammering the CPU/Disk simultaneously
        if rank > 0:
            delay = rank * 3.5 # Spread 10 boots over ~31 seconds
            print(f"[Rank {rank}] Staggering boot: waiting {delay:.1f}s...")
            time.sleep(delay)

        if version == "v3":
            from envs.sf2_v3 import StreetFighterEnvV3
            env = StreetFighterEnvV3(rank=rank, player=player, verbose=(rank == 0))
        else:
            from envs.sf2_v2 import StreetFighterEnvV2
            env = StreetFighterEnvV2(rank=rank, player=player, verbose=(rank == 0))
            
        log_dir = os.path.join(config.LOG_DIR, f"monitor_rank_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        return Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    return _init

def failsafe_env(env=None, model=None):
    if env is not None:
        try:
            env.close()
            del env
        except Exception:
            pass

    if model is not None:
        try:
            del model
        except Exception:
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
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
        
    time.sleep(3)
    print("[ENV] Failsafe complete. All zombie processes terminated and VRAM cleared.")
