# worker.py -- stateless ES evaluation worker. Run one per machine.
#
#   python src/es/worker.py --coordinator http://192.168.1.10:8823 --procs 8
#
# Loop forever: GET /work, reconstruct each member's perturbed policy from
# (theta, pair_seed, sign) alone, evaluate it for the requested episodes on
# a stable-retro env, POST the fitnesses, repeat. All state of value lives
# on the coordinator: killing a worker at ANY point loses at most one
# chunk's lease, which expires and is re-served to someone else.
#
# Robustness contract: any network error backs off exponentially (capped)
# and retries -- an unreachable coordinator (restart, sleeping laptop,
# flaky LAN) is a wait, never a crash. SIGTERM/SIGINT finish cleanly.
#
# --procs N runs N persistent env processes: one stable-retro emulator per
# OS process is a hard API limit (retro.make in the same process twice
# errors), so parallelism is multiprocessing, never threads.

import argparse
import json
import multiprocessing as mp
import os
import signal
import socket
import sys
import time
import urllib.error
import urllib.request

if __package__ in (None, ""):  # `python src/es/worker.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from es import protocol
from es.openes import fitness_from_episode, perturbation
from es.policy import MLPPolicy, NUM_PARAMS

MAX_EPISODE_STEPS = 20000  # hard failsafe; the env's own truncation should fire first
_STOP = False

# one env per process, created lazily on the first evaluation in that process
_ENV = None
_ENV_KWARGS = None


def _make_env():
    """Import guarded so this module (and the test suite) imports without
    stable-retro/the retro_env track being present; only an evaluation
    actually needs the env."""
    from envs.retro_env import RetroSF2Env  # built by the retro-backend track
    try:
        return RetroSF2Env(**_ENV_KWARGS)
    except TypeError:  # interface drift tolerance: fall back to defaults
        return RetroSF2Env()


def _pool_init(env_kwargs):
    global _ENV_KWARGS
    _ENV_KWARGS = env_kwargs
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent coordinates shutdown


def _run_episode(env, policy):
    obs, _ = env.reset()
    steps, info = 0, {}
    while steps < MAX_EPISODE_STEPS:
        obs, _reward, terminated, truncated, info = env.step(policy.act(obs))
        steps += 1
        if terminated or truncated:
            break
    return fitness_from_episode(info, steps)


def evaluate_member(task):
    """(theta, sigma, seed, sign, episodes) -> mean fitness. Pool-callable."""
    global _ENV
    theta, sigma, seed, sign, episodes = task
    if _ENV is None:
        _ENV = _make_env()
    eps = perturbation(theta.shape[0], seed)
    policy = MLPPolicy(theta + np.float32(sign) * np.float32(sigma) * eps)
    return float(np.mean([_run_episode(_ENV, policy) for _ in range(episodes)]))


# ---------------------------------------------------------------------------
# HTTP with retry. urllib only -- same zero-dependency rule as the
# coordinator. Every call either returns parsed JSON or None (caller backs
# off); exceptions never escape to the main loop.
# ---------------------------------------------------------------------------

def _http_json(url, body=None, timeout=30):
    try:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"} if body else {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"[worker] {url}: {e}", flush=True)
        return None


class Backoff:
    """Exponential with cap; reset() on any success."""

    def __init__(self, base=1.0, cap=60.0):
        self.base, self.cap, self.n = base, cap, 0

    def sleep(self):
        delay = min(self.cap, self.base * (2 ** self.n))
        self.n += 1
        time.sleep(delay)

    def reset(self):
        self.n = 0


def fetch_theta(coordinator, version, cache):
    """cache is {version: theta}; only the newest version is kept."""
    if version in cache:
        return cache[version]
    payload = _http_json(f"{coordinator}/theta")
    if payload is None:
        return None
    got_version, theta = protocol.decode_theta(payload)
    if theta.shape[0] != NUM_PARAMS:
        raise SystemExit(f"[worker] theta has {theta.shape[0]} params, policy expects "
                         f"{NUM_PARAMS} -- worker code is stale, update this machine")
    cache.clear()
    cache[got_version] = theta
    # coordinator always serves current theta: if the /work lease named an
    # older version the lease is stale; returning None makes us re-poll
    return theta if got_version == version else None


def main():
    ap = argparse.ArgumentParser(description="OpenAI-ES evaluation worker")
    ap.add_argument("--coordinator", required=True, help="http://host:port")
    ap.add_argument("--procs", type=int, default=1, help="env processes on this machine")
    ap.add_argument("--name", default=None, help="worker name shown in /status")
    ap.add_argument("--state", default="Champion.Level1.RyuVsGuile",
                    help="savestate handed to RetroSF2Env")
    args = ap.parse_args()

    global _ENV_KWARGS
    name = args.name or f"{socket.gethostname()}-{os.getpid()}"
    coordinator = args.coordinator.rstrip("/")
    env_kwargs = {"state": args.state}
    _ENV_KWARGS = env_kwargs

    def _sigterm(_sig, _frame):
        global _STOP
        _STOP = True
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    pool = None
    if args.procs > 1:
        # spawn, not fork: an emulator's threads/fds do not survive fork on
        # macOS, and spawn keeps Windows/WSL2 behaviour identical
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.procs, initializer=_pool_init, initargs=(env_kwargs,))

    print(f"[worker] {name} -> {coordinator} ({args.procs} proc)", flush=True)
    backoff = Backoff()
    theta_cache = {}
    try:
        while not _STOP:
            msg = _http_json(f"{coordinator}/work?worker={name}")
            if msg is None:
                backoff.sleep()  # coordinator down/restarting: poll until it is back
                continue
            backoff.reset()
            work = msg.get("work")
            if work is None:
                time.sleep(float(msg.get("retry_in", 2.0)))
                continue

            theta = fetch_theta(coordinator, work["theta_version"], theta_cache)
            if theta is None:
                backoff.sleep()
                continue
            backoff.reset()

            tasks = [(theta, work["sigma"], seed, sign, work["episodes"])
                     for _idx, seed, sign in work["members"]]
            t0 = time.time()
            if pool is not None:
                fitnesses = pool.map(evaluate_member, tasks)
            else:
                fitnesses = [evaluate_member(t) for t in tasks]

            result = {"chunk_id": work["chunk_id"], "generation": work["generation"],
                      "worker": name,
                      "member_idx": [idx for idx, _s, _g in work["members"]],
                      "fitnesses": fitnesses}
            # result POSTs retry harder than /work: the evaluation is paid for
            for _ in range(5):
                resp = _http_json(f"{coordinator}/result", body=result)
                if resp is not None:
                    break
                backoff.sleep()
            print(f"[worker] {work['chunk_id']}: {len(tasks)} members "
                  f"in {time.time() - t0:.1f}s", flush=True)
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        print(f"[worker] {name} stopped", flush=True)


if __name__ == "__main__":
    main()
