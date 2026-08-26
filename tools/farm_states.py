# farm_states.py -- grow the retro backend its own savestate curriculum.
#
# The BizHawk rig has 96 curriculum states (RYU_{OPP}_R1_lvl{1..7} x 12
# opponents + HARD); stable-retro ships exactly ONE (Champion.Level1.RyuVsGuile,
# which is Guile at the DEFAULT difficulty -- its 0xFFFE45 byte reads 3, i.e.
# 4 stars, so even its "Level1" means ladder stage 1, not difficulty 1). This
# tool farms real equivalents by DRIVING THE GAME: boot from power-on, walk the
# menus to OPTION MODE, set DIFFICULTY (and prove it via the RAM byte below),
# enter CHAMPION mode as RYU, then let the trained champion policy win its way
# up the arcade ladder, snapshotting every new opponent's round-1 start before
# any input. A lost round reloads the fight-start snapshot and retries with a
# fresh torch seed (savestate scumming: legitimate here, the goal is to ADVANCE,
# not to measure the policy).
#
# DIFFICULTY RAM (proved by tools/menu_probe.py full-RAM diffs, 2026-08-26):
#   work-RAM offset 0xFE45 (68k 0xFFFE45), one byte, value == stars-1.
#   OPTION MODE default shows 4 stars -> byte 3; one RIGHT -> 4; RIGHT taps
#   clamp at 7 (8 stars lit); LEFT taps clamp at 0 (1 star). The byte is
#   near-stable across idle frames and persists unchanged into the fight
#   scene, so it can be read back out of any farmed state as evidence.
#   Curriculum mapping: lvl N (1..8) <-> byte N-1; lvl8 == the HARD setting.
#
# MENU MAP (discovered frame by frame with tools/menu_probe.py):
#   power-on -> ~1740 idle frames of logos -> title fully formed (pressing
#   START EARLIER, during the formation animation, lands the flow in a
#   different menu variant -- wait it out); START -> CHAMPION/HYPER/OPTIONS;
#   DOWN,DOWN,START(,START) -> OPTION MODE with the cursor already on
#   DIFFICULTY (LEFT/RIGHT edits it -- that responsiveness is also how we
#   programmatically confirm we are inside); START exits to the attract
#   intro. From there START-ONLY taps walk the rest: attract -> title ->
#   CHAMPION/HYPER/OPTIONS (cursor on CHAMPION) -> GAME START/V.S./GROUP
#   (cursor on GAME START) -> PLAYER SELECT (cursor on RYU, START confirms).
#   Never tap A in menus: it moves the mode-menu cursor onto HYPER. From
#   fight 1 onward the arcade flow auto-advances (score tally, bonus stages,
#   map) with no input at all, so between fights we only idle and poll.
#
# FIGHT-START SIGNATURE (logged through a real fight load): at scene load
# both HP words arm at 176, round_timer arms at 0x99 (BCD, reads 153) and
# holds for ~180 frames of walk-in before its first tick, and the round-win
# counters read 0. Snapshotting inside that window is exactly "round-1 start,
# before any input" -- the shipped retro state sits in the same window.
#
#     .venv/bin/python tools/farm_states.py --levels 1,3,5,7
#
# Offline unit tests (code_testing/pytest/test_state_manifest.py) cover the
# pure helpers: naming, manifest schema, read-merge-write. Everything that
# needs the emulator or the model is exercised by RUNNING this script.

import argparse
import fcntl
import gzip
import json
import os
import sys
import time
from collections import deque

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import numpy as np  # noqa: E402

from envs.retro_env import (  # noqa: E402
    FRAME_SKIP, RamTrack, apply_sticky, assemble_v4_frame,
    discrete_to_project_bits, project_bits_to_retro,
)
# Champion-policy machinery shared with the local ES fine-tune driver: the
# 39.7M-step v3 PPO checkpoint, the v4->v3 frame adapter and the frozen
# SelectiveVecNormalize stats. Importing the module is side-effect free.
from es_finetune_lastlayer import (  # noqa: E402
    DEFAULT_PKL, DEFAULT_ZIP, FrozenNorm, v4_frame_to_v3,
)

GAME = "StreetFighterIISpecialChampionEdition-Genesis-v0"
INTEGRATION_DIR = os.path.join(REPO_ROOT, "retro_integration", GAME)
MANIFEST_PATH = os.path.join(REPO_ROOT, "retro_integration", "states_manifest.json")
DEFAULT_WORKDIR = os.path.join(REPO_ROOT, "benchmarks", "state_farm_work")

# Char id -> curriculum name, same ids as core/telemetry.py CHAR_NAMES and the
# same spellings as the BizHawk .State files in states/ (EHONDA, CHUNLI,
# MBISON: uppercase, no punctuation). Duplicated, not imported: core.telemetry
# imports core.config, which raises at import time off-Windows (see the
# constants block in envs/retro_env.py for the precedent).
OPP_NAMES = {
    0: "RYU", 1: "EHONDA", 2: "BLANKA", 3: "GUILE", 4: "KEN", 5: "CHUNLI",
    6: "ZANGIEF", 7: "DHALSIM", 8: "MBISON", 9: "SAGAT", 10: "BALROG",
    11: "VEGA",
}

DIFFICULTY_RAM_OFFSET = 0xFE45  # work-RAM offset; 68k address 0xFFFE45
MIN_LEVEL, MAX_LEVEL = 1, 8     # lvl8 == all 8 stars == the HARD setting
TIMER_ARMED = 153               # round_timer var at round start: BCD 0x99

MANIFEST_SCHEMA = "sf2-retro-states/v1"

DIFFICULTY_RAM_EVIDENCE = {
    "work_ram_offset": DIFFICULTY_RAM_OFFSET,
    "m68k_address": "0xFFFE45",
    "encoding": "level_minus_1",
    "levels": {str(lvl): lvl - 1 for lvl in range(MIN_LEVEL, MAX_LEVEL + 1)},
    "evidence": {
        "probe": "tools/menu_probe.py full-RAM diff, 2026-08-26",
        "observations": [
            "OPTION MODE default (4 stars) -> byte 3; one RIGHT tap -> 4",
            "RIGHT taps clamp at 7 with all 8 stars lit; LEFT taps clamp at 0"
            " with 1 star lit (screenshots d_max/d_min in the probe workdir)",
            "byte stable across idle frames and unchanged into the fight scene",
            "shipped Champion.Level1.RyuVsGuile.state carries byte 3 (default"
            " difficulty): its 'Level1' is ladder stage 1, NOT difficulty 1",
        ],
    },
}


# --------------------------------------------------------------------------
# Pure helpers (unit-tested offline in test_state_manifest.py)
# --------------------------------------------------------------------------

def state_name(opp_id: int, level: int) -> str:
    """Char id + difficulty level -> curriculum state name (no extension).

    Mirrors the BizHawk naming (RYU_GUILE_R1_lvl1) so both curricula read as
    one family; the file on disk is this plus ".state".
    """
    if opp_id not in OPP_NAMES:
        raise ValueError(f"opp_id {opp_id} outside 0..11")
    if not MIN_LEVEL <= level <= MAX_LEVEL:
        raise ValueError(f"level {level} outside {MIN_LEVEL}..{MAX_LEVEL}")
    return f"RYU_{OPP_NAMES[opp_id]}_R1_lvl{level}"


def new_manifest() -> dict:
    return {"schema": MANIFEST_SCHEMA,
            "difficulty_ram": dict(DIFFICULTY_RAM_EVIDENCE),
            "states": {}}


def validate_entry(name: str, entry: dict) -> None:
    """Raises ValueError unless `entry` is a well-formed manifest state row."""
    for key in ("opponent", "difficulty", "source", "verified"):
        if key not in entry:
            raise ValueError(f"{name}: missing key '{key}'")
    if entry["opponent"] not in OPP_NAMES.values():
        raise ValueError(f"{name}: unknown opponent {entry['opponent']!r}")
    if entry["difficulty"] not in range(MIN_LEVEL, MAX_LEVEL + 1):
        raise ValueError(f"{name}: difficulty {entry['difficulty']!r} "
                         f"outside {MIN_LEVEL}..{MAX_LEVEL}")
    if not isinstance(entry["verified"], dict) or not entry["verified"]:
        raise ValueError(f"{name}: 'verified' must be a non-empty dict -- "
                         "an unverified state does not enter the manifest")


def validate_manifest(doc: dict) -> None:
    if "states" not in doc or not isinstance(doc["states"], dict):
        raise ValueError("manifest missing 'states' dict")
    for name, entry in doc["states"].items():
        validate_entry(name, entry)


def merge_manifest(base: dict, additions: dict) -> dict:
    """Pure read-merge-write core: fold `additions` state rows into `base`.

    Lossless: every key of `base` outside 'states' (and every state row not
    being added) survives untouched. Collision-safe: re-adding a name whose
    (opponent, difficulty) identity matches replaces that row (a re-farm
    refreshing its verification); a name collision with a DIFFERENT identity
    raises instead of silently clobbering someone else's state.
    """
    out = {k: v for k, v in base.items()}
    out.setdefault("schema", MANIFEST_SCHEMA)
    out.setdefault("difficulty_ram", dict(DIFFICULTY_RAM_EVIDENCE))
    out["states"] = dict(base.get("states", {}))
    for name, entry in additions.items():
        validate_entry(name, entry)
        old = out["states"].get(name)
        if old is not None and (old["opponent"] != entry["opponent"]
                                or old["difficulty"] != entry["difficulty"]):
            raise ValueError(
                f"{name}: collision with existing entry of different identity "
                f"({old['opponent']} lvl{old['difficulty']} vs "
                f"{entry['opponent']} lvl{entry['difficulty']})")
        out["states"][name] = entry
    return out


def write_manifest(additions: dict, path: str = MANIFEST_PATH) -> dict:
    """Locked read-merge-write so parallel per-level runs cannot drop rows."""
    lock_path = path + ".lock"
    with open(lock_path, "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            if os.path.exists(path):
                with open(path) as f:
                    base = json.load(f)
            else:
                base = new_manifest()
            merged = merge_manifest(base, additions)
            tmp = path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(merged, f, indent=2, sort_keys=True)
                f.write("\n")
            os.replace(tmp, path)
            return merged
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def is_fight_start(ram: dict, prev_opp) -> bool:
    """The round-1-start signature (see module docstring for the trace).

    prev_opp guards against bonus stages / stale scenes: their p2_char byte
    still holds the previous fight's opponent, and a CE ladder never repeats
    an opponent, so "same as last time" is never a new fight.
    """
    return (int(ram["p1_hp"]) == 176 and int(ram["p2_hp"]) == 176
            and int(ram["round_timer"]) == TIMER_ARMED
            and int(ram["matches_won"]) == 0
            and int(ram["enemy_matches_won"]) == 0
            and int(ram["p1_char"]) == 0
            and int(ram["p2_char"]) in OPP_NAMES
            and int(ram["p2_char"]) != prev_opp)


# --------------------------------------------------------------------------
# Emulator driving (needs stable-retro + ROM; exercised by running this tool)
# --------------------------------------------------------------------------

def make_poweron_env():
    """Raw retro env booted from power-on, all 12 pad buttons live.

    Actions.ALL is load-bearing: the default FILTERED mode masks actions
    against the scenario's combat combos, which silently drops START and
    makes every menu unnavigable.
    """
    import stable_retro as retro
    from stable_retro.data import Integrations

    integration_root = os.path.join(REPO_ROOT, "retro_integration")
    if integration_root not in Integrations.CUSTOM_ONLY.paths:
        Integrations.add_custom_path(integration_root)
    env = retro.make(
        game=GAME, state=retro.State.NONE, inttype=Integrations.CUSTOM,
        obs_type=retro.Observations.RAM, render_mode=None,
        use_restricted_actions=retro.Actions.ALL,
    )
    env.reset()
    return env


class Pad:
    """Frame-level input helper over a raw retro env."""

    def __init__(self, env):
        self.env = env
        self.nop = np.zeros(len(env.buttons), dtype=np.uint8)

    def idle(self, frames: int) -> None:
        for _ in range(frames):
            self.env.step(self.nop)

    def tap(self, button: str, hold: int = 8, release: int = 8) -> None:
        arr = self.nop.copy()
        arr[self.env.buttons.index(button)] = 1
        for _ in range(hold):
            self.env.step(arr)
        self.idle(release)

    def vars(self) -> dict:
        return self.env.data.lookup_all()

    def difficulty_byte(self) -> int:
        return int(self.env.get_ram()[DIFFICULTY_RAM_OFFSET])


def save_debug_shot(env, path: str) -> None:
    from PIL import Image
    Image.fromarray(env.em.get_screen()).save(path)


def set_difficulty(pad: Pad, level: int, shot_dir=None) -> None:
    """Power-on -> OPTION MODE -> difficulty set and PROVEN via the RAM byte.

    Leaves the game on the attract intro that follows the options exit. The
    assertions catch any menu desync loudly instead of farming states at a
    wrong or unproven setting.
    """
    pad.idle(1740)                       # logos -> title FULLY formed
    pad.tap("START"); pad.idle(120)      # CHAMPION / HYPER / OPTIONS
    pad.tap("DOWN"); pad.tap("DOWN")     # cursor -> OPTIONS
    pad.tap("START"); pad.idle(60)

    # In OPTION MODE the cursor starts on DIFFICULTY and LEFT/RIGHT edits it;
    # that responsiveness doubles as the "are we inside yet?" probe (the
    # entering START reliably needs a second tap: the first only latches the
    # highlight). LEFT probes when the byte can go down, RIGHT when it can't.
    for _attempt in range(4):
        before = pad.difficulty_byte()
        pad.tap("LEFT" if before > 0 else "RIGHT")
        if pad.difficulty_byte() != before:
            break
        pad.tap("START"); pad.idle(90)
    else:
        raise RuntimeError("never reached OPTION MODE: difficulty byte "
                           "unresponsive to LEFT/RIGHT taps")

    for _ in range(MAX_LEVEL + 1):       # clamp to 0 (1 star)
        pad.tap("LEFT")
    if pad.difficulty_byte() != 0:
        raise RuntimeError(f"difficulty byte {pad.difficulty_byte()} after "
                           "clamping LEFT; expected 0")
    for _ in range(level - 1):
        pad.tap("RIGHT")
    got = pad.difficulty_byte()
    if got != level - 1:
        raise RuntimeError(f"difficulty byte {got} after setting; "
                           f"expected {level - 1} for lvl{level}")
    if shot_dir:
        save_debug_shot(pad.env, os.path.join(shot_dir, f"lvl{level}_options.png"))
    print(f"[farm] OPTION MODE: difficulty byte 0x{DIFFICULTY_RAM_OFFSET:04X}"
          f" == {got} (lvl{level}) -- PROVEN", flush=True)
    pad.tap("START"); pad.idle(60)       # exit options -> attract intro


def seek_fight_start(pad: Pad, prev_opp, budget_frames: int,
                     confirm_taps: bool = False):
    """Idle frame by frame until the round-1-start signature appears.

    confirm_taps=True (first fight only) taps START every ~300 frames, which
    single-handedly walks attract -> title -> CHAMPION menu -> GAME START
    menu -> PLAYER SELECT -> confirm RYU: every cursor already rests on the
    row we want, so a confirm-only button is safe (A is NOT: it moves the
    mode-menu cursor onto HYPER). The per-frame poll exits ~180 frames before
    the round can go live, so a tap can never land as a PAUSE. From fight 2
    onward the arcade flow auto-advances and taps stay off entirely.
    """
    taps = 0
    for frame in range(budget_frames):
        pad.env.step(pad.nop)
        ram = pad.vars()
        if is_fight_start(ram, prev_opp):
            return ram, frame
        if confirm_taps and frame and frame % 300 == 0:
            pad.tap("START")
            taps += 1
    return None, budget_frames


class ChampionPolicy:
    """The frozen v3 PPO champion behind the same adapters ES uses."""

    def __init__(self, zip_path: str = DEFAULT_ZIP, pkl_path: str = DEFAULT_PKL):
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        self.torch = torch
        self.model = PPO.load(zip_path, device="cpu")
        self.norm = FrozenNorm(pkl_path)

    def seed(self, seed: int) -> None:
        self.torch.manual_seed(seed)

    def act(self, frames_v3) -> np.ndarray:
        stacked = self.norm(np.concatenate(frames_v3))
        action, _ = self.model.predict(stacked, deterministic=False)
        return action


def play_match(pad: Pad, policy: ChampionPolicy, seed: int,
               max_steps: int = 5000):
    """Policy-drive the CURRENT fight until the match resolves.

    Returns (outcome, agent_steps): 'won' when the round-win counter hits 2,
    'lost_round' the moment the enemy counter ticks (the caller reloads the
    fight-start snapshot -- round 2 losses restart the match too, per the
    scumming contract), 'stall' if nothing resolves inside max_steps
    (~5.5 game-minutes; treated as a burned retry).
    """
    policy.seed(seed)
    track = RamTrack()
    frames = deque(maxlen=4)
    frame, track, _, _ = assemble_v4_frame(pad.vars(), track, is_reset=True)
    for _ in range(4):
        frames.append(v4_frame_to_v3(frame))
    sticky_dir, sticky_ctr = None, 0

    for step in range(max_steps):
        bits = discrete_to_project_bits(policy.act(frames))
        bits, sticky_dir, sticky_ctr = apply_sticky(bits, sticky_dir, sticky_ctr)
        arr = project_bits_to_retro(bits, pad.env.buttons)
        for _ in range(FRAME_SKIP):
            pad.env.step(arr)
        ram = pad.vars()
        frame, track, _, _ = assemble_v4_frame(ram, track)
        frames.append(v4_frame_to_v3(frame))
        if int(ram["enemy_matches_won"]) > 0:
            return "lost_round", step + 1
        if int(ram["matches_won"]) >= 2:
            return "won", step + 1
    return "stall", max_steps


def farm_level(level: int, policy: ChampionPolicy, out_dir: str, workdir: str,
               max_retries: int, seek_budget: int, seed_offset: int = 0) -> dict:
    """One full ladder walk at one difficulty. Returns the run record."""
    os.makedirs(workdir, exist_ok=True)
    env = make_poweron_env()
    pad = Pad(env)
    t0 = time.time()
    run = {"level": level, "captured": [], "skipped_existing": [],
           "fights": [], "retries_burned": 0, "gap_after": None}
    try:
        set_difficulty(pad, level, shot_dir=workdir)
        prev_opp = None
        for fight_no in range(1, len(OPP_NAMES) + 1):
            ram, waited = seek_fight_start(
                pad, prev_opp, seek_budget, confirm_taps=(fight_no == 1))
            if ram is None:
                # No further fight materialized: the ladder is complete
                # (credits) -- or something unexpected; the screenshot tells.
                save_debug_shot(env, os.path.join(
                    workdir, f"lvl{level}_seek_timeout_f{fight_no}.png"))
                break
            pad.idle(8)                     # settle inside the walk-in window
            ram = pad.vars()
            if not is_fight_start(ram, prev_opp):
                raise RuntimeError(f"fight-start signature vanished during "
                                   f"settle (fight {fight_no})")
            opp = int(ram["p2_char"])
            name = state_name(opp, level)
            diff_byte = pad.difficulty_byte()
            if diff_byte != level - 1:
                raise RuntimeError(f"difficulty byte {diff_byte} inside "
                                   f"fight {fight_no}; expected {level - 1}")
            snapshot = env.em.get_state()
            save_debug_shot(env, os.path.join(workdir, f"{name}_capture.png"))
            path = os.path.join(out_dir, name + ".state")
            if os.path.exists(path):
                # Another track (e.g. a BizHawk raid) owns that file; leave it.
                run["skipped_existing"].append(name)
            else:
                with open(path, "wb") as f:
                    f.write(gzip.compress(snapshot))
                run["captured"].append(name)
            print(f"[farm] lvl{level} fight {fight_no:2d}: {name} captured "
                  f"(waited {waited} frames, diff_byte={diff_byte})", flush=True)

            fight = {"name": name, "opponent": OPP_NAMES[opp],
                     "retries": 0, "outcome": None}
            run["fights"].append(fight)
            for retry in range(max_retries + 1):
                # Deterministic but offsettable: a rerun with the same seeds
                # replays the exact same outcomes (torch sampling is the only
                # stochastic input), so supplemental higher-cap runs pass
                # --seed-offset to explore fresh action sequences.
                seed = seed_offset + 10_000 * level + 100 * fight_no + retry
                outcome, steps = play_match(pad, policy, seed)
                fight["outcome"], fight["retries"] = outcome, retry
                if outcome == "won":
                    print(f"[farm]   won in {steps} steps "
                          f"(retry {retry})", flush=True)
                    break
                run["retries_burned"] += 1
                print(f"[farm]   {outcome} after {steps} steps -> reload "
                      f"(retry {retry + 1}/{max_retries})", flush=True)
                env.em.set_state(snapshot)
            if fight["outcome"] != "won":
                run["gap_after"] = name
                print(f"[farm] lvl{level}: retry cap hit on {name}; "
                      f"stopping this walk", flush=True)
                break
            prev_opp = opp
    finally:
        env.close()
    run["wall_s"] = round(time.time() - t0, 1)
    return run


def verify_states(names, level: int, probe_frames: int = 900) -> dict:
    """Load each farmed state THROUGH the training path and check it.

    Same checks as the raid track (load, p2_char, HP, fight-alive) plus the
    difficulty byte read from inside the loaded state. Returns
    {name: verified_dict} for the passing states only.
    """
    import stable_retro as retro
    from stable_retro.data import Integrations

    env = make_poweron_env()
    nop = np.zeros(len(env.buttons), dtype=np.uint8)
    verified = {}
    try:
        for name in names:
            env.load_state(name, Integrations.CUSTOM)
            env.reset()
            ram = env.data.lookup_all()
            opp = int(ram["p2_char"])
            checks = {
                "p1_char_is_ryu": int(ram["p1_char"]) == 0,
                "p2_char_in_range": opp in OPP_NAMES,
                "name_matches_p2_char": name == state_name(opp, level),
                "hp_full": (int(ram["p1_hp"]) == 176
                            and int(ram["p2_hp"]) == 176),
                "timer_armed": int(ram["round_timer"]) == TIMER_ARMED,
            }
            diff_byte = int(env.get_ram()[DIFFICULTY_RAM_OFFSET])
            checks["difficulty_byte_matches"] = diff_byte == level - 1
            # Fight-alive: with NO input the round must go live on its own
            # (the armed clock starts ticking) and the cast must not mutate.
            ticked_at = None
            for frame in range(probe_frames):
                env.step(nop)
                ram = env.data.lookup_all()
                if int(ram["p2_char"]) != opp:
                    break
                if int(ram["round_timer"]) < TIMER_ARMED:
                    ticked_at = frame + 1
                    break
            checks["fight_alive"] = ticked_at is not None
            if all(checks.values()):
                # Key set = union of the raid track's verified schema
                # (tools/verify_states.py: loads / p2_char_id / hp_start /
                # fight_alive -- test_verify_states.py enforces it on every
                # manifest row) and this track's difficulty evidence.
                verified[name] = {
                    "loads": True,
                    "p2_char_id": opp,
                    "hp_start": [176, 176],
                    "fight_alive": True,
                    "round_timer_bcd": TIMER_ARMED,
                    "difficulty_byte": diff_byte,
                    "timer_ticked_after_frames": ticked_at,
                    "method": ("stable-retro load via Integrations.CUSTOM + "
                               f"{probe_frames}-frame no-input probe"),
                    "date": time.strftime("%Y-%m-%d"),
                }
                print(f"[verify] {name}: OK (diff_byte={diff_byte}, "
                      f"live after {ticked_at} frames)", flush=True)
            else:
                failed = [k for k, ok in checks.items() if not ok]
                print(f"[verify] {name}: FAILED {failed}", flush=True)
    finally:
        env.close()
    return verified


def main():
    ap = argparse.ArgumentParser(
        description="Farm verified R1-start savestates by playing the ladder")
    ap.add_argument("--levels", default="1,3,5,7",
                    help="comma-separated difficulty levels (1..8; 8 == HARD)")
    ap.add_argument("--max-retries", type=int, default=12,
                    help="snapshot reloads per fight before recording a gap")
    ap.add_argument("--seek-budget", type=int, default=36000,
                    help="frames to wait for a fight-start signature")
    ap.add_argument("--out-dir", default=INTEGRATION_DIR)
    ap.add_argument("--workdir", default=DEFAULT_WORKDIR)
    ap.add_argument("--zip", default=DEFAULT_ZIP)
    ap.add_argument("--pkl", default=DEFAULT_PKL)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="shift the deterministic retry seeds (supplemental "
                         "runs explore fresh action sequences)")
    ap.add_argument("--no-manifest", action="store_true",
                    help="farm + verify but skip the manifest merge")
    args = ap.parse_args()

    levels = [int(tok) for tok in args.levels.split(",") if tok.strip()]
    for lvl in levels:
        if not MIN_LEVEL <= lvl <= MAX_LEVEL:
            raise SystemExit(f"level {lvl} outside {MIN_LEVEL}..{MAX_LEVEL}")

    policy = ChampionPolicy(args.zip, args.pkl)
    summary = []
    for lvl in levels:
        print(f"\n[farm] ===== difficulty lvl{lvl} =====", flush=True)
        run = farm_level(lvl, policy, args.out_dir, args.workdir,
                         args.max_retries, args.seek_budget,
                         seed_offset=args.seed_offset)
        verified = verify_states(run["captured"], lvl)
        rejected = [n for n in run["captured"] if n not in verified]
        for name in rejected:
            # A state that fails verification must not sit in the search path
            # where a training run could pick it up.
            reject_dir = os.path.join(args.workdir, "rejected")
            os.makedirs(reject_dir, exist_ok=True)
            os.replace(os.path.join(args.out_dir, name + ".state"),
                       os.path.join(reject_dir, name + ".state"))
        if verified and not args.no_manifest:
            additions = {
                name: {"opponent": name.split("_")[1], "difficulty": lvl,
                       "source": "farmed", "verified": verified[name]}
                for name in verified
            }
            write_manifest(additions)
            print(f"[farm] manifest: +{len(additions)} entries", flush=True)
        run["verified"] = sorted(verified)
        run["rejected"] = rejected
        summary.append(run)

    print("\n[farm] ===== SUMMARY =====")
    for run in summary:
        print(json.dumps(run, indent=2))


if __name__ == "__main__":
    main()
