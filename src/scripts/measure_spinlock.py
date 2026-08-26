# measure_spinlock.py
#
# Settles the roadmap's load-bearing unknown: does the Lua client's
# comm.socketServerResponse() spinlock burn a CPU core while waiting for
# Python, or does comm.socketServerSetTimeout(10) (training_env_client.lua:46)
# turn it into a ~100 Hz timed poll?
#
# If it burns: 16 workers waste up to 16 cores and replacing the spinlock
# with a blocking receive is a top-priority optimization. If it polls: strike
# "16 wasted cores" off the roadmap and stop optimizing a ghost.
#
# Method: boot ONE emulator, step it to mid-episode steady state, then hold
# (send nothing) while sampling the EmuHawk process's CPU%. During the hold
# Lua is inside its spinlock; the emulator advances no frames, so any CPU it
# uses is the spinlock itself.
#
#   .venv/Scripts/python.exe src/scripts/measure_spinlock.py --hold 15
#
# Interpretation on an idle machine:
#   ~100% of one core during hold  -> hot busy-wait; fix it.
#   <=~15% during hold             -> timed poll; the handoff's assumption was wrong.

import argparse
import time
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

from core import config
from core.env_tools import failsafe_env


def main():
    parser = argparse.ArgumentParser(description="Measure the Lua spinlock's CPU cost")
    parser.add_argument("--env", default="v3", choices=["v2", "v3"])
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--hold", type=float, default=15.0,
                        help="seconds to stall Python while sampling EmuHawk CPU%%")
    args = parser.parse_args()

    try:
        import psutil
    except ImportError:
        print("psutil is required: .venv/Scripts/python.exe -m pip install psutil")
        sys.exit(1)

    config.generate_lua_config()
    config.TRAINING_STATES = config.DIFFICULTY_LEVELS[1]

    if args.env == "v3":
        from envs.sf2_v3 import StreetFighterEnvV3 as EnvCls
    else:
        from envs.sf2_v2 import StreetFighterEnvV2 as EnvCls

    env = EnvCls(rank=0, verbose=True)
    try:
        proc = psutil.Process(env.emulator_process.pid)
        n_cores = psutil.cpu_count(logical=True)
        env.reset()

        # Steady-state stepping first, as the busy-emulating reference.
        proc.cpu_percent(None)  # prime the sampler
        t0 = time.perf_counter()
        for _ in range(args.warmup_steps):
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                env.reset()
        stepping_seconds = time.perf_counter() - t0
        stepping_cpu = proc.cpu_percent(None)

        # The hold: Python goes silent, Lua sits in its receive spinlock.
        # (Keep the hold far under the 600 s Lua dead-man's-switch and note
        # that Python's own 5 s recv timeout is not in play -- we simply do
        # not call receive_payload during the hold.)
        print(f"[spinlock] Holding for {args.hold:.0f}s while Lua waits on us...")
        samples = []
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < args.hold:
            samples.append(proc.cpu_percent(interval=0.25))
        samples = np.asarray(samples[1:])  # first sample covers the transition

        print("\n===== SPINLOCK MEASUREMENT =====")
        print(f"machine cores (logical):       {n_cores}")
        print(f"stepping: {args.warmup_steps} steps in {stepping_seconds:.1f}s "
              f"({args.warmup_steps / stepping_seconds:.1f} steps/s), "
              f"EmuHawk CPU {stepping_cpu:.0f}% (of one core)")
        print(f"holding:  EmuHawk CPU mean {samples.mean():.1f}%  "
              f"p95 {np.percentile(samples, 95):.1f}%  max {samples.max():.1f}%")
        if samples.mean() >= 50:
            print("VERDICT: HOT BUSY-WAIT -- the spinlock burns a core per worker. "
                  "Replacing it with a blocking receive (or a larger "
                  "socketServerSetTimeout) is a real optimization.")
        elif samples.mean() <= 15:
            print("VERDICT: TIMED POLL -- the spinlock is cheap. Remove "
                  "'16 wasted cores' from the optimization roadmap.")
        else:
            print("VERDICT: INTERMEDIATE -- partial spin. Worth a look at this "
                  "BizHawk build's socketServerResponse implementation.")
    finally:
        try:
            env.close()
        except Exception:
            pass
        failsafe_env(ignore_gate=True)


if __name__ == "__main__":
    main()
