# verify_states.py -- lint every savestate cataloged in states_manifest.json.
#
#     .venv/bin/python tools/verify_states.py [--write] [--seed N]
#
# For each entry in retro_integration/states_manifest.json this boots the state
# in RetroSF2Env (real emulator, headless) and measures: does it load, which
# characters are in the fight (p1_char/p2_char from RAM -- the ONLY trustworthy
# source; foreign state names lie), do both fighters start at full HP (176),
# and is the fight alive (30 random agent steps: HP or x-positions must move,
# no crash). Entries marked "players": 2 get load+settle only -- P2 is a human
# port there, so a liveness check driven from P1 alone proves nothing.
#
# Some foreign states (FightLadder's stars/*) boot into a screen transition
# where the fight RAM reads all zeros for a few dozen frames, and round-start
# states (FightLadder's curriculum/*) sit in the "ROUND 1 ... FIGHT!" intro
# freeze for ~100 frames where nothing can move yet. Measurement therefore
# settles in two phases under NOOP input: (A) both HP read full, then (B) the
# fight has actually begun -- the round timer ticks off its start value or
# something moves. Only then is the 30-step liveness window meaningful.
# States that never settle fail with hp_start as last seen.
#
# --write re-runs the same measurements and stores them under each entry's
# "verified" key, and fills "opponent" from p2_char_id when it is null.
# The manifest is shared with other state-farming tracks, so writing is
# read-merge-write: re-read right before writing, touch only our keys, write
# atomically. "difficulty" is NEVER touched here -- it cannot be measured from
# RAM with the documented variables, only sourced from provenance.
#
# Exit code: 0 if every state passes (loads, and fight_alive unless 2P), 1
# otherwise -- so CI or a fleet bring-up script can use this as a gate.

import argparse
import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import numpy as np  # noqa: E402

MANIFEST_PATH = os.path.join(REPO_ROOT, "retro_integration", "states_manifest.json")

# House opponent spellings (states/RYU_*.State convention), by p2_char id.
# Same id order as core/telemetry.CHAR_NAMES; spelling matches the BizHawk
# savestate filenames (EHONDA not E.Honda, CHUNLI not Chun-Li, ...).
OPPONENT_BY_ID = {
    0: "RYU", 1: "EHONDA", 2: "BLANKA", 3: "GUILE", 4: "KEN", 5: "CHUNLI",
    6: "ZANGIEF", 7: "DHALSIM", 8: "MBISON", 9: "SAGAT", 10: "BALROG",
    11: "VEGA",
}

FULL_HP = 176
NOOP = np.array([0, 0])   # DIRECTION_MAP[0] + BUTTON_MAP[0] == all bits off
SETTLE_MAX_STEPS = 120    # 120 agent steps * 4 frames = 8s -- generous
ALIVE_STEPS = 30
MOVE_EPS = 4              # pixels of x travel that count as "moved"


def read_manifest(path=MANIFEST_PATH):
    with open(path) as f:
        return json.load(f)


def write_manifest_merged(results, path=MANIFEST_PATH):
    """Read-merge-write: only replace 'verified' (and null 'opponent') on
    entries we measured; keep everything else -- including entries another
    track added since we read the file -- untouched."""
    manifest = read_manifest(path)
    states = manifest.setdefault("states", {})
    for name, res in results.items():
        entry = states.get(name)
        if entry is None:
            continue  # entry vanished under us; not ours to resurrect
        entry["verified"] = res["verified"]
        if entry.get("opponent") is None and res["opponent"] is not None:
            entry["opponent"] = res["opponent"]
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        os.replace(tmp, path)
    except BaseException:
        os.unlink(tmp)
        raise


def ram(env):
    return env._env.data.lookup_all()


def settle(env):
    """NOOP-step until the fight RAM is real AND the round has begun.
    Phase A: both HP read full (screen transitions zero the fight RAM).
    Phase B: the round timer ticks off its phase-A value, or HP/x moves --
    round-start states spend ~100 frames in the intro freeze where a liveness
    probe would measure a wall. Returns (settled, steps_taken)."""
    steps = 0
    while steps <= SETTLE_MAX_STEPS:
        r = ram(env)
        if int(r["p1_hp"]) == FULL_HP and int(r["p2_hp"]) == FULL_HP:
            break
        env.step(NOOP)
        steps += 1
    else:
        return False, SETTLE_MAX_STEPS
    t0, x0_p1, x0_p2 = int(r["round_timer"]), int(r["p1_x"]), int(r["p2_x"])
    while steps <= SETTLE_MAX_STEPS:
        r = ram(env)
        if (int(r["round_timer"]) != t0
                or int(r["p1_hp"]) < FULL_HP or int(r["p2_hp"]) < FULL_HP
                or int(r["p1_x"]) != x0_p1 or int(r["p2_x"]) != x0_p2):
            return True, steps
        env.step(NOOP)
        steps += 1
    return False, SETTLE_MAX_STEPS


def verify_state(env, name, entry, rng):
    res = {
        "opponent": None,
        "verified": {
            "loads": False, "p2_char_id": None,
            "hp_start": None, "fight_alive": None,
        },
        "detail": "",
    }
    two_player = entry.get("players") == 2
    try:
        env.reset(options={"state": name})
    except Exception as e:  # noqa: BLE001 -- a bad state can raise anything
        res["detail"] = f"reset: {type(e).__name__}: {e}"
        return res
    res["verified"]["loads"] = True

    try:
        settled, settle_steps = settle(env)
        r = ram(env)
        res["verified"]["p2_char_id"] = int(r["p2_char"])
        res["verified"]["hp_start"] = [int(r["p1_hp"]), int(r["p2_hp"])]
        res["opponent"] = OPPONENT_BY_ID.get(int(r["p2_char"]))
        res["p1_char_id"] = int(r["p1_char"])
        res["detail"] = f"settle={settle_steps}"
        if not settled:
            res["detail"] += " NEVER-SETTLED"
            return res
        if two_player:
            res["detail"] += " 2P:load-only"
            return res

        # Liveness: random P1 input for 30 agent steps. A live fight shows HP
        # damage and/or x movement from either side (P2 moving under zero P2
        # input additionally proves the game AI is driving it).
        x0_p1, x0_p2 = int(r["p1_x"]), int(r["p2_x"])
        hp_changed = p1_moved = p2_moved = False
        for _ in range(ALIVE_STEPS):
            action = np.array([rng.integers(0, 9), rng.integers(0, 7)])
            _obs, _rew, term, trunc, _info = env.step(action)
            r = ram(env)
            if int(r["p1_hp"]) < FULL_HP or int(r["p2_hp"]) < FULL_HP:
                hp_changed = True
            if abs(int(r["p1_x"]) - x0_p1) > MOVE_EPS:
                p1_moved = True
            if abs(int(r["p2_x"]) - x0_p2) > MOVE_EPS:
                p2_moved = True
            if term or trunc:
                break
        res["verified"]["fight_alive"] = hp_changed or p1_moved or p2_moved
        res["detail"] += (f" hp_chg={int(hp_changed)}"
                          f" p1_mv={int(p1_moved)} p2_mv={int(p2_moved)}")
    except Exception as e:  # noqa: BLE001
        res["detail"] += f" step: {type(e).__name__}: {e}"
    return res


def passed(entry, res):
    v = res["verified"]
    if not v["loads"]:
        return False
    if entry.get("players") == 2:
        return True  # 2P states only promise to load
    return bool(v["fight_alive"])


def main():
    parser = argparse.ArgumentParser(description="Lint cataloged savestates")
    parser.add_argument("--write", action="store_true",
                        help="store results into the manifest (read-merge-write)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    manifest = read_manifest()
    states = manifest.get("states", {})
    if not states:
        print("manifest has no states")
        return 1

    from envs.retro_env import RetroSF2Env  # noqa: E402 -- needs emulator
    env = RetroSF2Env(trainable=True)
    rng = np.random.default_rng(args.seed)

    results = {}
    n_fail = 0
    fmt = "{:<46} {:<11} {:<8} {:>4} {:<5} {:>3} {:<9} {:<5} {}"
    print(fmt.format("state", "source", "opp", "diff", "loads", "p2",
                     "hp_start", "alive", "detail"))
    print("-" * 118)
    for name in sorted(states):
        entry = states[name]
        res = verify_state(env, name, entry, rng)
        results[name] = res
        v = res["verified"]
        ok = passed(entry, res)
        n_fail += 0 if ok else 1
        diff = entry.get("difficulty")
        print(fmt.format(
            name[:46], entry.get("source", "?"),
            (entry.get("opponent") or res["opponent"] or "?")[:8],
            "-" if diff is None else diff,
            "yes" if v["loads"] else "NO",
            "-" if v["p2_char_id"] is None else v["p2_char_id"],
            "-" if v["hp_start"] is None else f"{v['hp_start'][0]},{v['hp_start'][1]}",
            "-" if v["fight_alive"] is None else ("yes" if v["fight_alive"] else "NO"),
            ("" if ok else "[FAIL] ") + res["detail"],
        ))
    env._env.close()

    print("-" * 118)
    print(f"{len(states)} states, {len(states) - n_fail} passed, {n_fail} failed")
    if args.write:
        write_manifest_merged(results)
        print(f"manifest updated: {MANIFEST_PATH}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
