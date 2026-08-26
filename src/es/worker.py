# worker.py -- stateless ES evaluation worker. Run one per machine.
#
#   python src/es/worker.py --coordinator http://192.168.1.10:8823          # auto-sized
#   python src/es/worker.py --coordinator http://madre:8080 --cpu-share 0.5 # donate half
#
# Loop forever: GET /work, reconstruct each member's perturbed policy from
# (theta, pair_seed, sign) alone, evaluate it for the requested episodes on
# a stable-retro env, POST the fitnesses, repeat. All state of value lives
# on the coordinator: killing a worker at ANY point loses at most one
# chunk's lease, which expires and is re-served to someone else. A lease
# that carries a "states" rotation picks each episode's savestate from the
# PAIR seed (openes.states_for_member), so a member and its antithetic twin
# always fight the same opponent sequence; without the key, every episode
# runs on this worker's --state default exactly as before. Rotation results
# echo protocol.states_fingerprint of the list actually evaluated -- the
# coordinator refuses rotation results without it, which is what keeps a
# stale pre-rotation worker from silently poisoning a rotation run.
#
# Robustness contract: any network error backs off exponentially (capped)
# and retries -- an unreachable coordinator (restart, sleeping laptop,
# flaky LAN) is a wait, never a crash. SIGTERM/SIGINT finish cleanly.
#
# --procs N runs N persistent env processes: one stable-retro emulator per
# OS process is a hard API limit (retro.make in the same process twice
# errors), so parallelism is multiprocessing, never threads. --procs auto
# (the default) asks es/resources.py to size the machine instead; the same
# module also drops the emulators to nice 10 so the box stays usable while
# it contributes. Every chunk's POST carries a `stats` block so /status can
# show what each machine is actually worth to the fleet.

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

from es import protocol, resources
from es.openes import (eval_rng_for_episode, fitness_from_episode, perturbation,
                       states_for_member)
from es.policy import DEFAULT_POLICY, POLICIES

NEUTRAL_ACTION = np.array([0, 0], dtype=np.int64)  # sin direccion, sin boton

# Capacidades que ESTE codigo sabe honrar, anunciadas en cada /work. La
# madre banca a los workers que no anuncien lo que su run requiere (codigo
# viejo = sin query param = banca): mejor un "espera y actualizate" que
# quemar CPU evaluando mal para ser rechazado por los fingerprints.
WORKER_CAPS = "states,eval"

MAX_EPISODE_STEPS = 20000  # hard failsafe; the env's own truncation should fire first
RATE_WINDOW_S = 120.0      # rolling window behind the steps/s we report
_STOP = False

# one env per process, created lazily on the first evaluation in that process
_ENV = None
_ENV_KWARGS = None


def _log(message):
    print(f"[worker] {message}", flush=True)


def _make_env():
    """Import guarded so this module (and the test suite) imports without
    stable-retro/the retro_env track being present; only an evaluation
    actually needs the env."""
    from envs.retro_env import RetroSF2Env  # built by the retro-backend track
    try:
        return RetroSF2Env(**_ENV_KWARGS)
    except TypeError:  # interface drift tolerance: fall back to defaults
        return RetroSF2Env()


def _pool_init(env_kwargs, nice_delta):
    global _ENV_KWARGS
    _ENV_KWARGS = env_kwargs
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # parent coordinates shutdown
    # Renice in the CHILD: this process is the one that will pin a core for
    # hours. The parent only polls HTTP and stays at normal priority so it
    # keeps answering the coordinator even when the machine is saturated.
    if nice_delta:  # 0 means "leave priority alone", not "renice by 0"
        resources.apply_nice(nice_delta, warn=_log)


def _run_episode(env, policy, state=None, perturb_rng=None, desync_max=0,
                 action_noise=0.0):
    # options={"state": name} pins THIS episode's savestate (RetroSF2Env's
    # per-reset override); None keeps the env's own default, which is the
    # entire pre-rotation behaviour.
    #
    # perturb_rng (run 3): el RNG PAREADO del episodio (eval_rng_for_episode
    # con el seed del par) -- sortea el desfase de arranque y el ruido de
    # acciones identicos para ambos gemelos antiteticos. None = episodio
    # limpio, byte a byte el comportamiento previo.
    obs, _ = env.reset(options={"state": state} if state else None)
    steps, info = 0, {}
    if perturb_rng is not None and desync_max:
        for _ in range(int(perturb_rng.integers(0, int(desync_max) + 1))):
            obs, _reward, terminated, truncated, info = env.step(NEUTRAL_ACTION)
            steps += 1
            if terminated or truncated:
                return fitness_from_episode(info, steps), steps
    while steps < MAX_EPISODE_STEPS:
        if (perturb_rng is not None and action_noise
                and perturb_rng.random() < action_noise):
            action = np.array([perturb_rng.integers(0, 9),
                               perturb_rng.integers(0, 7)], dtype=np.int64)
        else:
            action = policy.act(obs)
        obs, _reward, terminated, truncated, info = env.step(action)
        steps += 1
        if terminated or truncated:
            break
    return fitness_from_episode(info, steps), steps


def evaluate_member(task):
    """(theta, sigma, seed, sign, episodes, states) -> (mean fitness, agent steps).

    Pool-callable. The step count rides along because it is free here and is
    the only honest measure of how fast this machine emulates -- wall time per
    chunk conflates throughput with how long the episodes happened to last.

    `states` is the coordinator's rotation (or None = default state). The
    episode->state map comes from states_for_member(seed, ...) with the PAIR
    seed, so this member's antithetic twin -- evaluated who-knows-where on
    the fleet -- derives the identical opponent sequence from the same seed:
    the pair's fitness difference measures the perturbation, not the luck of
    the opponent draw. Fitness stays the plain per-member mean either way.
    """
    global _ENV
    # Longitud del task = version del wire: 6 (pre-registro, v4 limpio),
    # 7 (+ nombre de policy), 8 (+ dict de perturbaciones de evaluacion).
    eval_params = None
    if len(task) == 8:
        theta, sigma, seed, sign, episodes, states, policy_name, eval_params = task
    elif len(task) == 7:
        theta, sigma, seed, sign, episodes, states, policy_name = task
    else:
        theta, sigma, seed, sign, episodes, states = task
        policy_name = DEFAULT_POLICY
    if _ENV is None:
        _ENV = _make_env()
    eps = perturbation(theta.shape[0], seed)
    policy = POLICIES[policy_name](theta + np.float32(sign) * np.float32(sigma) * eps)
    desync = int(eval_params.get("desync_max", 0)) if eval_params else 0
    noise = float(eval_params.get("action_noise", 0.0)) if eval_params else 0.0

    def _episode(ep_idx, state_name):
        # RNG del seed del PAR + indice de episodio: el gemelo antitetico de
        # este miembro deriva EXACTAMENTE las mismas perturbaciones
        rng = (eval_rng_for_episode(seed, ep_idx)
               if (desync or noise) else None)
        return _run_episode(_ENV, policy, state=state_name, perturb_rng=rng,
                            desync_max=desync, action_noise=noise)

    if states:
        picks = states_for_member(seed, episodes, len(states))
        runs = [_episode(e, states[i]) for e, i in enumerate(picks)]
    else:
        runs = [_episode(e, None) for e in range(episodes)]
    return float(np.mean([f for f, _s in runs])), int(sum(s for _f, s in runs))


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
    """cache is {version: (theta, policy_name)}; only the newest version is kept.

    Returns (theta, policy_name) or None. The payload's "policy" key names
    the architecture theta parameterizes (absent = "v4", every pre-registry
    coordinator); an unknown name or a shape mismatch is a stale-code exit,
    never a silent wrong-architecture evaluation.
    """
    if version in cache:
        return cache[version]
    payload = _http_json(f"{coordinator}/theta")
    if payload is None:
        return None
    got_version, theta = protocol.decode_theta(payload)
    policy_name = str(payload.get("policy", DEFAULT_POLICY))
    policy_cls = POLICIES.get(policy_name)
    if policy_cls is None:
        raise SystemExit(f"[worker] coordinator serves policy '{policy_name}' which this "
                         f"worker does not know -- worker code is stale, update this machine")
    if theta.shape[0] != policy_cls.num_params():
        raise SystemExit(f"[worker] theta has {theta.shape[0]} params, policy "
                         f"'{policy_name}' expects {policy_cls.num_params()} -- worker "
                         f"code is stale, update this machine")
    cache.clear()
    cache[got_version] = (theta, policy_name)
    # coordinator always serves current theta: if the /work lease named an
    # older version the lease is stale; returning None makes us re-poll
    return (theta, policy_name) if got_version == version else None


# ---------------------------------------------------------------------------
# Machine sizing + self-reported throughput
# ---------------------------------------------------------------------------

def parse_procs_flag(value):
    """--procs 'auto' -> None (let resources size the machine). Anything else
    goes to plan_procs untouched, which is the single place that decides what
    a bad value means -- an autostarted worker must not die on a typo."""
    if value is None:
        return None
    return None if str(value).strip().lower() in ("", "auto") else value


def make_stats(procs, host, step_rate, ep_rate):
    """The `stats` block of a /result POST.

    The coordinator aggregates these per worker to show what each machine
    contributes, so the key names are a wire contract: do not rename them.
    """
    return {"procs": int(procs),
            "steps_per_s": round(step_rate.rate(), 1),
            "episodes_per_s": round(ep_rate.rate(), 3),
            "host": host}


def _power_label(battery):
    return {True: "battery", False: "ac"}.get(battery, "unknown")


def _sizing_label(args, requested, topo):
    """What actually decided `procs`, for the startup line. A --procs value
    plan_procs could not parse decided nothing, so it gets no credit here."""
    try:
        int(requested)
        return "--procs"
    except (TypeError, ValueError):
        pass
    share = args.cpu_share
    if share is not None and share == share:  # NaN parses as float but decides nothing
        return f"--cpu-share {share}"
    return f"auto: {topo.physical_cpus} cores - {args.reserve_cores} reserved"


def main():
    ap = argparse.ArgumentParser(description="OpenAI-ES evaluation worker")
    ap.add_argument("--coordinator", required=True, help="http://host:port")
    ap.add_argument("--procs", default="auto",
                    help="emulator processes: 'auto' (default) or an explicit count")
    ap.add_argument("--reserve-cores", type=int, default=2,
                    help="cores left free for the owner of the machine (auto sizing)")
    ap.add_argument("--cpu-share", type=float, default=None,
                    help="0.0-1.0: donate this fraction of the machine instead")
    ap.add_argument("--max-procs", type=int, default=None,
                    help="hard cap on processes, whatever the sizing says")
    ap.add_argument("--nice", type=int, default=10 if hasattr(os, "nice") else 0,
                    help="niceness added to each emulator process (POSIX; 0 disables)")
    ap.add_argument("--name", default=None, help="worker name shown in /status")
    ap.add_argument("--state", default="Champion.Level1.RyuVsGuile",
                    help="savestate handed to RetroSF2Env")
    args = ap.parse_args()

    global _ENV_KWARGS
    host = socket.gethostname()
    name = args.name or f"{host}-{os.getpid()}"
    coordinator = args.coordinator.rstrip("/")
    env_kwargs = {"state": args.state}
    _ENV_KWARGS = env_kwargs

    topo = resources.detect_topology()
    requested = parse_procs_flag(args.procs)
    procs = resources.plan_procs(topo, requested=requested,
                                 reserve_cores=args.reserve_cores,
                                 cpu_share=args.cpu_share, max_procs=args.max_procs,
                                 warn=_log)
    battery = resources.on_battery()

    def _sigterm(_sig, _frame):
        global _STOP
        _STOP = True
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    pool = None
    if procs > 1:
        # spawn, not fork: an emulator's threads/fds do not survive fork on
        # macOS, and spawn keeps Windows/WSL2 behaviour identical
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(procs, initializer=_pool_init, initargs=(env_kwargs, args.nice))
    else:
        # single-process worker: THIS process is the emulator, so it is the one
        # that needs the nice bump (there is no child to carry it)
        resources.apply_nice(args.nice, warn=_log)

    # one line, everything a student on a strange machine needs to see
    _log(f"{name} -> {coordinator} | {topo.platform} {topo.logical_cpus}cpu/"
         f"{topo.physical_cpus}core{'' if topo.physical_known else '?'} | "
         f"procs={procs} ({_sizing_label(args, requested, topo)}) | "
         f"nice={args.nice:+d} | power={_power_label(battery)}")
    if battery:
        _log("on battery: expect thermal/power throttling to roughly halve throughput "
             "-- plug this machine in, chunks it drops are re-leased anyway")

    backoff = Backoff()
    theta_cache = {}
    step_rate = resources.RollingRate(RATE_WINDOW_S)
    ep_rate = resources.RollingRate(RATE_WINDOW_S)
    last_reason = None
    try:
        while not _STOP:
            msg = _http_json(f"{coordinator}/work?worker={name}&caps={WORKER_CAPS}")
            if msg is None:
                backoff.sleep()  # coordinator down/restarting: poll until it is back
                continue
            backoff.reset()
            work = msg.get("work")
            if work is None:
                # "reason" = la madre nos banco (o informa algo); una vez por
                # razon distinta, no cada poll
                reason = msg.get("reason")
                if reason and reason != last_reason:
                    _log(f"madre dice: {reason}")
                    last_reason = reason
                time.sleep(float(msg.get("retry_in", 2.0)))
                continue
            last_reason = None

            fetched = fetch_theta(coordinator, work["theta_version"], theta_cache)
            if fetched is None:
                backoff.sleep()
                continue
            backoff.reset()
            theta, policy_name = fetched

            episodes = int(work.get("episodes", 1) or 1)
            # no "states" key (older coordinator, or a run without a rotation)
            # -> None -> every episode on this worker's default state
            states = work.get("states") or None
            # "eval" (run 3): perturbaciones de evaluacion; ausente = limpio
            eval_params = work.get("eval") or None
            tasks = [(theta, work["sigma"], seed, sign, episodes, states,
                      policy_name, eval_params)
                     for _idx, seed, sign in work["members"]]
            t0 = time.monotonic()
            if pool is not None:
                outcomes = pool.map(evaluate_member, tasks)
            else:
                outcomes = [evaluate_member(t) for t in tasks]
            elapsed = time.monotonic() - t0

            fitnesses = [f for f, _s in outcomes]
            steps = sum(s for _f, s in outcomes)
            step_rate.add(steps, elapsed)
            ep_rate.add(len(tasks) * episodes, elapsed)

            result = {"chunk_id": work["chunk_id"], "generation": work["generation"],
                      "worker": name,
                      "member_idx": [idx for idx, _s, _g in work["members"]],
                      "fitnesses": fitnesses,
                      "stats": make_stats(procs, host, step_rate, ep_rate)}
            if states:
                # prove which rotation these fitnesses were measured against:
                # computed from the list this loop actually evaluated, not
                # blindly acknowledged. The coordinator refuses rotation
                # results without a matching echo (see protocol.py).
                result["states_fingerprint"] = protocol.states_fingerprint(states)
            if eval_params:
                # misma prueba para las perturbaciones: el echo se calcula de
                # los parametros REALMENTE aplicados en este loop
                result["eval_fingerprint"] = protocol.eval_fingerprint(
                    eval_params.get("desync_max", 0),
                    eval_params.get("action_noise", 0.0))
            # result POSTs retry harder than /work: the evaluation is paid for
            for _ in range(5):
                resp = _http_json(f"{coordinator}/result", body=result)
                if resp is not None:
                    break
                backoff.sleep()
            _log(f"{work['chunk_id']}: {len(tasks)} members in {elapsed:.1f}s "
                 f"({steps} steps, {step_rate.rate():.0f} steps/s, "
                 f"{ep_rate.rate():.2f} ep/s)")
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
        _log(f"{name} stopped")


if __name__ == "__main__":
    main()
