import os, sys, time, multiprocessing, gc, atexit

import core.config as config

_failsafe_executed = False

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

        if version == "v4":
            from envs.sf2_v4 import StreetFighterEnvV4
            env = StreetFighterEnvV4(rank=rank, player=player, verbose=(rank == 0))
        elif version == "v3":
            from envs.sf2_v3 import StreetFighterEnvV3
            env = StreetFighterEnvV3(rank=rank, player=player, verbose=(rank == 0))
        else:
            from envs.sf2_v2 import StreetFighterEnvV2
            env = StreetFighterEnvV2(rank=rank, player=player, verbose=(rank == 0))

        if kwargs.get("macros", False):
            if version not in ("v3", "v4"):
                raise ValueError(
                    f"macros require the MultiDiscrete([9,7]) action space of env "
                    f"v3 or v4, got version={version!r}"
                )
            from envs.macro_wrapper import MacroActionWrapper
            from envs.sf2_v4 import V4_FRAME_DIM
            frame_size = V4_FRAME_DIM if version == "v4" else 554
            env = MacroActionWrapper(env, frame_size=frame_size)

        log_dir = os.path.join(config.LOG_DIR, f"monitor_rank_{rank}")
        os.makedirs(log_dir, exist_ok=True)
        return Monitor(env, filename=os.path.join(log_dir, "monitor.csv"))
    return _init

def failsafe_env(env=None, model=None, ignore_gate=False):
    global _failsafe_executed
    
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

    # Idempotent execution gate: Only run the global sniper, GC, and VRAM purge once per exit session
    if not ignore_gate:
        if _failsafe_executed and env is None:
            return
        _failsafe_executed = True

    print("[ENV] Executing Failsafe: Clearing VRAM and GC...")
    
    # 1. The Thread Sniper (Only kills child processes of THIS Python process)
    active_children = multiprocessing.active_children()
    if active_children:
        for child in active_children:
            try:
                child.kill()
            except Exception:
                pass
    
    # 2. Windows Workspace Sniper (Terminates orphaned grandchild EmuHawk.exe processes of this project)
    try:
        import subprocess
        import json
        
        # Use PowerShell to list all EmuHawk.exe processes and their command lines in JSON format
        cmd = [
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name = 'EmuHawk.exe'\" | "
            "Select-Object ProcessId, CommandLine | ConvertTo-Json -Compress"
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0 and result.stdout.strip():
            data = json.loads(result.stdout.strip())
            processes = data if isinstance(data, list) else [data]
            
            # Use the config directory name or project root path as our unique project fingerprint
            target_token = "street_fighter"
            
            for proc in processes:
                pid = proc.get("ProcessId")
                cmdline = proc.get("CommandLine") or ""
                if pid and target_token.lower() in cmdline.lower():
                    print(f"[Cleanup] Sniper-killing orphaned EmuHawk.exe process (PID: {pid})...")
                    subprocess.run(["taskkill", "/F", "/PID", str(pid)], capture_output=True)
    except Exception as e:
        pass
    
    # 3. The VRAM Purge
    gc.collect()
    if "torch" in sys.modules:
        try:
            torch_mod = sys.modules["torch"]
            if hasattr(torch_mod, "cuda") and torch_mod.cuda.is_available():
                torch_mod.cuda.empty_cache()
        except Exception:
            pass
        
    allow_sleep()
    print("[ENV] Failsafe complete.")

def prevent_sleep():
    """Prevent Windows from going to sleep or entering standby while training is active."""
    try:
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        ES_CONTINUOUS = 0x80000000
        ES_SYSTEM_REQUIRED = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
    except Exception:
        pass

def allow_sleep():
    """Restore default Windows power standby management."""
    try:
        import ctypes
        ES_CONTINUOUS = 0x80000000
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
    except Exception:
        pass

# Register failsafe_env to run automatically at exit on any process that imports env_tools
atexit.register(failsafe_env)

