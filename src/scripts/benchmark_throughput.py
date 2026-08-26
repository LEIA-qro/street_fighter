# benchmark_throughput.py
#
# Measures THE missing number: aggregate agent steps/s of the real environment
# stack (BizHawk + Lua bridge + SubprocVecEnv), swept over worker counts, on
# whatever machine it runs on. Every optimization decision in the roadmap is
# scored against this baseline, and every number recorded before 2026-08-25
# came from a laptop -- run this on the 13900K first.
#
# Usage (from the project root, inside the BizHawk folder):
#   .venv/Scripts/python.exe src/scripts/benchmark_throughput.py --env v3 --n_envs 1,8,16,24
#   .venv/Scripts/python.exe src/scripts/benchmark_throughput.py --env v4 --vec dummy --n_envs 16
#
# --vec dummy runs every env in-process (DummyVecEnv): comparing it against
# subproc at the same n_envs directly measures what the 16 worker processes
# and their pipe-pickling cost -- the Python side of each env is I/O-bound on
# its emulator socket, so in-process may well win.
#
# Results are printed as a table and appended as JSON to
# logs/throughput_bench.jsonl for later comparison across machines.

import argparse
import json
import os
import platform
import time
from pathlib import Path
import sys; sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np

from core import config
from core.env_tools import SFv2_make_env, failsafe_env


def bench_one(n_envs: int, version: str, macros: bool, vec_kind: str,
              steps: int, warmup: int) -> dict:
    from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

    env_fns = [SFv2_make_env(i, version=version, macros=macros) for i in range(n_envs)]
    vec_cls = SubprocVecEnv if vec_kind == "subproc" else DummyVecEnv

    print(f"\n[bench] Booting {n_envs} x {version} ({vec_kind}). "
          f"Staggered boot takes ~{max(0, (n_envs - 1)) * 3.5:.0f}s...")
    t_boot = time.perf_counter()
    venv = vec_cls(env_fns)
    try:
        venv.env_method("set_training_states", config.TRAINING_STATES)
        venv.reset()
        boot_seconds = time.perf_counter() - t_boot

        # Pre-sample action batches so sampling cost stays out of the loop.
        pool = [np.stack([venv.action_space.sample() for _ in range(n_envs)])
                for _ in range(256)]

        for i in range(warmup):
            venv.step(pool[i % len(pool)])

        times = np.empty(steps, dtype=np.float64)
        t0 = time.perf_counter()
        for i in range(steps):
            t = time.perf_counter()
            venv.step(pool[i % len(pool)])
            times[i] = time.perf_counter() - t
        total = time.perf_counter() - t0
    finally:
        try:
            venv.close()
        except Exception:
            pass

    agent_steps = steps * n_envs
    return {
        "n_envs": n_envs,
        "version": version,
        "macros": macros,
        "vec": vec_kind,
        "vec_steps": steps,
        "agent_steps_per_s": agent_steps / total,
        "vec_step_ms_mean": float(times.mean() * 1e3),
        "vec_step_ms_p50": float(np.percentile(times, 50) * 1e3),
        "vec_step_ms_p95": float(np.percentile(times, 95) * 1e3),
        "vec_step_ms_max": float(times.max() * 1e3),
        "boot_seconds": boot_seconds,
        "machine": platform.node(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    parser = argparse.ArgumentParser(description="Aggregate env throughput benchmark")
    parser.add_argument("--env", default="v3", choices=["v1", "v2", "v3", "v4"])
    parser.add_argument("--macros", action="store_true")
    parser.add_argument("--n_envs", default="1,8,16",
                        help="comma-separated worker counts to sweep")
    parser.add_argument("--vec", default="subproc", choices=["subproc", "dummy"])
    parser.add_argument("--steps", type=int, default=2000,
                        help="timed vec-steps per configuration")
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    config.generate_lua_config()
    config.TRAINING_STATES = config.DIFFICULTY_LEVELS[1]

    results = []
    for n in [int(x) for x in args.n_envs.split(",")]:
        result = bench_one(n, args.env, args.macros, args.vec,
                           args.steps, args.warmup)
        results.append(result)
        print(f"[bench] n_envs={n:>2}  {result['agent_steps_per_s']:>8.1f} agent steps/s  "
              f"vec-step p50 {result['vec_step_ms_p50']:.2f}ms  "
              f"p95 {result['vec_step_ms_p95']:.2f}ms  max {result['vec_step_ms_max']:.1f}ms")
        failsafe_env(ignore_gate=True)
        time.sleep(3)

    os.makedirs(config.LOG_DIR, exist_ok=True)
    out_path = os.path.join(config.LOG_DIR, "throughput_bench.jsonl")
    with open(out_path, "a") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"\n{'n_envs':>6} {'vec':>8} {'steps/s':>10} {'p50 ms':>8} {'p95 ms':>8} {'boot s':>7}")
    for r in results:
        print(f"{r['n_envs']:>6} {r['vec']:>8} {r['agent_steps_per_s']:>10.1f} "
              f"{r['vec_step_ms_p50']:>8.2f} {r['vec_step_ms_p95']:>8.2f} {r['boot_seconds']:>7.1f}")
    print(f"\n[bench] Appended to {out_path}")


if __name__ == "__main__":
    main()
