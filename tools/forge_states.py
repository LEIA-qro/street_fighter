# forge_states.py -- mint the missing difficulty tiers by poking the
# difficulty-dependent RAM inside donor snapshots.
#
# farm_states.py grows savestates by PLAYING the ladder, which means a tier
# only exists up to the fight the champion policy could reach: the farm's gaps
# are exactly the late-ladder bosses at lvl5+ and lvl8 almost everywhere
# (28 of the 96 RYU_{OPP}_R1_lvl{1..8} cells). This tool fills those cells
# WITHOUT playing: it loads an authentic donor state for the same opponent,
# rewrites every difficulty-dependent RAM byte to the target level's values
# inside the emulator's serialized snapshot, and writes the poked snapshot
# as the new .state.
#
# WHICH BYTES (the 2026-08-27 hunt): poking only the menu byte 0xFE45
# (== level-1, menu_probe 2026-08-26) FAILED behavioral validation flat --
# forged states kept playing exactly like their donors (p~0.5-0.97 vs donor
# across 3 opponents x 40 eps), so the fight scene copies difficulty into
# AI-local parameters at load. A full-blob diff across all 68 authentic
# farmed states (12 opponents, levels 1-8) then found EVERY byte that is a
# pure function of the level: the in-fight copy 0x97B2 (== level-1 in all
# 68), a nonlinear derived table 0x96B8, and a linear derivative
# tripled at 0xBA35/0xBA38/0xBA58 -- see DIFFICULTY_DEPENDENT_RAM below.
# Forging rewrites ALL of them, with a per-byte sanity check that the
# donor's current value matches its own level's table entry first.
#
# STILL A HYPOTHESIS UNTIL VALIDATED: parameters that depend on level AND
# opponent jointly would escape the pure-function filter. So the behavioral
# gate stands: tools/validate_forged_states.py forges tiers we ALSO have
# authentically, plays a fixed policy against both, and demands the forged
# copy is statistically indistinguishable from the authentic target while
# being distinguishable from its donor, in both directions. --fill refuses
# to run until that report says PASS (override: --force).
#
# Blob mechanics (probed 2026-08-27): em.get_state() serializes the core;
# the 68k work RAM sits in it as a contiguous byte-identical block (located
# by searching for a window of live RAM content, byte-swapped fallback kept
# for core updates). Poking the blob and em.set_state()-ing it back shows
# the new byte in get_ram() and it survives 900+ frames of live fight.
#
#     .venv/bin/python tools/forge_states.py --forge-validation
#     .venv/bin/python tools/validate_forged_states.py        # the science
#     .venv/bin/python tools/forge_states.py --fill           # gated on PASS
#     .venv/bin/python tools/forge_states.py --clean-validation
#
# Pure helpers are unit-tested offline in code_testing/pytest/
# test_forge_states.py; everything touching the emulator is exercised by
# running the tool.

import argparse
import glob
import gzip
import json
import os
import re
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
sys.path.insert(0, os.path.join(REPO_ROOT, "tools"))

import numpy as np  # noqa: E402

from farm_states import (  # noqa: E402
    DIFFICULTY_RAM_OFFSET, INTEGRATION_DIR, MANIFEST_PATH, MAX_LEVEL,
    MIN_LEVEL, OPP_NAMES, TIMER_ARMED, make_poweron_env, state_name,
    verify_states, write_manifest,
)

VALIDATION_REPORT = os.path.join(REPO_ROOT, "benchmarks", "forge_validation",
                                 "report.json")

# Validation matrix: opponents whose AUTHENTIC coverage is deep enough to
# forge a tier we can also measure for real, in both directions. Diverse
# archetypes on purpose (rushdown, grappler, zoner).
VALIDATION_OPPS = ("CHUNLI", "ZANGIEF", "DHALSIM")

# Work-RAM window anchors to try, in order, when locating the RAM block
# inside the serialized snapshot. Each is (ram_offset, width); a window must
# match at EXACTLY one blob position (direct or byte-swapped) to be trusted.
WINDOW_ANCHORS = ((0xFE00, 128), (0xFE00, 256), (0xFD00, 384))

# Every work-RAM byte that is a pure function of the difficulty level,
# measured empirically across all 68 authentic farmed states (12 opponents,
# levels 1-8, full-blob diff 2026-08-27). 0xFE45 is the OPTION-MODE setting
# (proved by menu_probe 2026-08-26) and is cosmetic in-fight; 0x97B2 is the
# fight scene's live copy; 0x96B8 and the 0xBA35/38/58 triplet are derived
# AI parameters. Forging rewrites all six so the forged state is
# byte-identical to an authentic capture at the target level everywhere
# difficulty shows up globally.
LEVEL_MINUS_1 = {lvl: lvl - 1 for lvl in range(MIN_LEVEL, MAX_LEVEL + 1)}
_BA_TABLE = {1: 239, 2: 255, 3: 15, 4: 31, 5: 47, 6: 63, 7: 79, 8: 95}
DIFFICULTY_DEPENDENT_RAM = {
    0x96B8: {1: 0, 2: 0, 3: 0, 4: 32, 5: 48, 6: 96, 7: 128, 8: 255},
    0x97B2: LEVEL_MINUS_1,
    0xBA35: _BA_TABLE,
    0xBA38: _BA_TABLE,
    0xBA58: _BA_TABLE,
    DIFFICULTY_RAM_OFFSET: LEVEL_MINUS_1,   # 0xFE45, the menu setting
}


# --------------------------------------------------------------------------
# Pure helpers (unit-tested offline)
# --------------------------------------------------------------------------

def authentic_levels(manifest: dict) -> dict:
    """{opponent: sorted authentic levels} -- rows NOT sourced 'forged'.

    Forged states never serve as donors: a re-run must not chain a forgery
    off a forgery and quietly launder the provenance.
    """
    out = {}
    for entry in manifest.get("states", {}).values():
        if entry.get("difficulty") is None or entry.get("source") == "forged":
            continue
        opp = entry.get("opponent")
        if opp in OPP_NAMES.values():
            out.setdefault(opp, set()).add(int(entry["difficulty"]))
    return {opp: sorted(lvls) for opp, lvls in out.items()}


def missing_pairs(manifest: dict) -> list:
    """[(opponent, level)] cells of the 12x{1..8} grid with no manifest row."""
    have = set(manifest.get("states", {}))
    out = []
    for opp_id in sorted(OPP_NAMES):
        for lvl in range(MIN_LEVEL, MAX_LEVEL + 1):
            if state_name(opp_id, lvl) not in have:
                out.append((OPP_NAMES[opp_id], lvl))
    return out


def pick_donor(levels: list, target: int) -> int:
    """Donor level for a target: the nearest authentic level, preferring the
    highest one BELOW the target (minimizes whatever difficulty-adjacent
    residue a snapshot might carry besides the byte)."""
    if not levels:
        raise ValueError("no authentic levels to donate from")
    below = [lvl for lvl in levels if lvl < target]
    return max(below) if below else min(lvl for lvl in levels if lvl > target)


def swap16(data: bytes) -> bytes:
    """Swap adjacent byte pairs (68k word order vs byte order)."""
    arr = bytearray(data)
    arr[0::2], arr[1::2] = data[1::2], data[0::2]
    return bytes(arr)


def locate_workram_base(blob: bytes, ram: np.ndarray):
    """-> (blob_pos_of_workram_byte_0, layout) or raises.

    Finds the work-RAM block inside the snapshot by searching for a window
    of the live RAM's actual content near the difficulty byte. Demands a
    unique hit; widens the window until it gets one.
    """
    for ram_off, width in WINDOW_ANCHORS:
        window = bytes(ram[ram_off:ram_off + width])
        for layout, pattern in (("direct", window), ("swapped", swap16(window))):
            hits, start = [], 0
            while len(hits) < 2:
                p = blob.find(pattern, start)
                if p < 0:
                    break
                hits.append(p)
                start = p + 1
            if len(hits) != 1:
                continue
            base = hits[0] - ram_off
            pos = blob_offset(base, DIFFICULTY_RAM_OFFSET, layout)
            if 0 <= base and blob[pos] == int(ram[DIFFICULTY_RAM_OFFSET]):
                return base, layout
    raise RuntimeError("could not locate the work-RAM block uniquely in the "
                       "snapshot (core serialization changed?)")


def blob_offset(base: int, ram_offset: int, layout: str) -> int:
    """Map a work-RAM offset to its blob position under the given layout."""
    return base + (ram_offset ^ 1 if layout == "swapped" else ram_offset)


def forged_entry(opp: str, level: int, donor_name: str, verified: dict) -> dict:
    """Manifest row for a forged state; provenance rides along."""
    verified = dict(verified)
    verified["forged_from"] = donor_name
    verified["forge_method"] = ("difficulty-dependent RAM set (0x96B8, "
                                "0x97B2, 0xBA35/38/58, 0xFE45) rewritten to "
                                f"lvl{level} values in the donor snapshot; "
                                "behavioral validation in "
                                "benchmarks/forge_validation/report.json")
    return {"opponent": opp, "difficulty": level, "source": "forged",
            "donor": donor_name, "verified": verified}


def validation_forge_plan(levels_by_opp: dict, opps=VALIDATION_OPPS) -> list:
    """[(forgeval_name, donor_name, target_level)] for both directions."""
    plan = []
    for opp in opps:
        levels = levels_by_opp.get(opp, [])
        if len(levels) < 2:
            raise ValueError(f"{opp}: needs >=2 authentic levels, has {levels}")
        lo, hi = levels[0], levels[-1]
        opp_id = next(i for i, n in OPP_NAMES.items() if n == opp)
        plan.append((f"FORGEVAL_{opp}_lvl{hi}_from{lo}",
                     state_name(opp_id, lo), hi))
        plan.append((f"FORGEVAL_{opp}_lvl{lo}_from{hi}",
                     state_name(opp_id, hi), lo))
    return plan


# --------------------------------------------------------------------------
# Emulator work (one emulator per process: a single env forges everything)
# --------------------------------------------------------------------------

FIGHT_START_KEYS = ("p1_hp", "p2_hp", "round_timer", "matches_won",
                    "enemy_matches_won", "p1_char", "p2_char")


def _fight_start_signature(env) -> dict:
    ram = env.data.lookup_all()
    sig = {k: int(ram[k]) for k in FIGHT_START_KEYS}
    if not (sig["p1_hp"] == 176 and sig["p2_hp"] == 176
            and sig["round_timer"] == TIMER_ARMED
            and sig["matches_won"] == 0 and sig["enemy_matches_won"] == 0
            and sig["p1_char"] == 0 and sig["p2_char"] in OPP_NAMES):
        raise RuntimeError(f"not a round-1-start state: {sig}")
    return sig


_DONOR_LVL_RE = re.compile(r"lvl(\d+)$")


def forge_one(env, donor_name: str, target_level: int) -> bytes:
    """Load donor -> rewrite the difficulty-dependent RAM set -> gzip bytes.

    Every step is asserted: donor must carry the round-1-start signature,
    every donor byte must match its OWN level's table entry before being
    rewritten (catches a wrong table, layout, or mislabeled donor), every
    poked byte must read back from the emulator after set_state, and the
    signature must survive the poke untouched.
    """
    from stable_retro.data import Integrations

    m = _DONOR_LVL_RE.search(donor_name)
    if not m:
        raise ValueError(f"donor {donor_name!r} does not end in lvlN")
    donor_level = int(m.group(1))

    env.load_state(donor_name, Integrations.CUSTOM)
    env.reset()
    sig0 = _fight_start_signature(env)
    ram = env.get_ram()
    blob = bytearray(env.em.get_state())
    base, layout = locate_workram_base(bytes(blob), ram)
    for off, table in sorted(DIFFICULTY_DEPENDENT_RAM.items()):
        pos = blob_offset(base, off, layout)
        if blob[pos] != table[donor_level]:
            raise RuntimeError(
                f"{donor_name}: byte 0x{off:04X} reads {blob[pos]}, expected "
                f"{table[donor_level]} for its lvl{donor_level} -- wrong "
                f"table, layout or donor label; refusing to forge")
        blob[pos] = table[target_level]

    env.em.set_state(bytes(blob))
    env.data.update_ram()
    ram_after = env.get_ram()
    for off, table in DIFFICULTY_DEPENDENT_RAM.items():
        got = int(ram_after[off])
        if got != table[target_level]:
            raise RuntimeError(f"poke did not land at 0x{off:04X}: byte {got},"
                               f" expected {table[target_level]} ({layout})")
    sig1 = _fight_start_signature(env)
    if sig1 != sig0:
        raise RuntimeError(f"poke disturbed the fight-start signature: "
                           f"{sig0} -> {sig1}")
    return gzip.compress(bytes(blob))


def read_manifest() -> dict:
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def cmd_forge_validation(args):
    manifest = read_manifest()
    plan = validation_forge_plan(authentic_levels(manifest),
                                 tuple(args.opps.split(",")))
    env = make_poweron_env()
    try:
        for name, donor, target in plan:
            payload = forge_one(env, donor, target)
            with open(os.path.join(INTEGRATION_DIR, name + ".state"), "wb") as f:
                f.write(payload)
            print(f"[forge] {name}: poked {donor} -> lvl{target}", flush=True)
    finally:
        env.close()
    print(f"[forge] {len(plan)} validation states in {INTEGRATION_DIR} "
          f"(NOT in the manifest: invisible to training). Next: "
          f".venv/bin/python tools/validate_forged_states.py")


def cmd_clean_validation(_args):
    victims = glob.glob(os.path.join(INTEGRATION_DIR, "FORGEVAL_*.state"))
    for path in victims:
        os.remove(path)
    print(f"[forge] removed {len(victims)} FORGEVAL states")


def validation_verdict() -> str:
    try:
        with open(VALIDATION_REPORT) as f:
            return str(json.load(f).get("verdict", "MISSING"))
    except (OSError, ValueError):
        return "MISSING"


def cmd_fill(args):
    verdict = validation_verdict()
    if verdict != "PASS" and not args.force:
        raise SystemExit(
            f"[forge] refusing --fill: validation verdict is {verdict!r} "
            f"(need PASS in {VALIDATION_REPORT}; run "
            f"tools/validate_forged_states.py, or --force to override)")

    manifest = read_manifest()
    levels_by_opp = authentic_levels(manifest)
    todo = missing_pairs(manifest)
    if not todo:
        print("[forge] the 12x8 grid is already complete")
        return
    print(f"[forge] {len(todo)} missing cells: "
          + " ".join(f"{o}:{l}" for o, l in todo))

    opp_ids = {n: i for i, n in OPP_NAMES.items()}
    forged = {}   # name -> (opp, level, donor_name, gzip payload)
    env = make_poweron_env()
    try:
        for opp, lvl in todo:
            name = state_name(opp_ids[opp], lvl)
            if os.path.exists(os.path.join(INTEGRATION_DIR, name + ".state")):
                # A file with no manifest row belongs to another track (a
                # farm --no-manifest run, a BizHawk raid): disk presence, not
                # manifest presence, marks ownership -- same guard as
                # farm_level. Verify/catalog it there instead of forging.
                print(f"[forge] SKIP {name}: .state on disk without a "
                      f"manifest row -- another track owns it", flush=True)
                continue
            donor_lvl = pick_donor(levels_by_opp.get(opp, []), lvl)
            donor = state_name(opp_ids[opp], donor_lvl)
            forged[name] = (opp, lvl, donor, forge_one(env, donor, lvl))
            print(f"[forge] {name}: poked from {donor}", flush=True)
    finally:
        env.close()

    # Files land in the search path only now, with every forge already
    # asserted: an abort mid-forge-loop strands nothing on disk, and any
    # failure below removes every file THIS run wrote before re-raising.
    written = []
    try:
        for name, (_opp, _lvl, _donor, payload) in forged.items():
            with open(os.path.join(INTEGRATION_DIR, name + ".state"), "wb") as f:
                f.write(payload)
            written.append(name)

        # Verify through the SAME gauntlet farmed states pass (load,
        # signature, difficulty byte, fight-alive) -- one env per level
        # batch, so the forge env above must already be closed.
        additions, rejected = {}, []
        by_level = {}
        for name, (_opp, lvl, _donor, _payload) in forged.items():
            by_level.setdefault(lvl, []).append(name)
        for lvl in sorted(by_level):
            verified = verify_states(sorted(by_level[lvl]), lvl)
            for name in by_level[lvl]:
                opp, _lvl, donor, _payload = forged[name]
                if name in verified:
                    additions[name] = forged_entry(opp, lvl, donor,
                                                   verified[name])
                else:
                    rejected.append(name)
                    os.remove(os.path.join(INTEGRATION_DIR, name + ".state"))
                    written.remove(name)
    except BaseException:
        for name in written:
            path = os.path.join(INTEGRATION_DIR, name + ".state")
            if os.path.exists(path):
                os.remove(path)
        print(f"[forge] aborted mid-verify: removed {len(written)} "
              f"unverified forgeries from the search path", flush=True)
        raise
    if rejected:
        print(f"[forge] REJECTED (verification failed, files removed): "
              f"{rejected}")
    if additions and not args.no_manifest:
        write_manifest(additions)
        print(f"[forge] manifest: +{len(additions)} forged entries")
    print(f"[forge] done: {len(additions)} forged, {len(rejected)} rejected")


def main():
    ap = argparse.ArgumentParser(
        description="Mint missing difficulty tiers by poking 0xFE45 in donors")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--forge-validation", action="store_true",
                      help="forge the FORGEVAL_* pairs the validator measures")
    mode.add_argument("--clean-validation", action="store_true",
                      help="delete FORGEVAL_* states from the integration dir")
    mode.add_argument("--fill", action="store_true",
                      help="forge every missing 12x8 cell (gated on the "
                           "validation report saying PASS)")
    ap.add_argument("--opps", default=",".join(VALIDATION_OPPS),
                    help="validation opponents (comma-separated)")
    ap.add_argument("--force", action="store_true",
                    help="--fill without a PASS verdict (know what you do)")
    ap.add_argument("--no-manifest", action="store_true",
                    help="--fill: write .state files but skip the manifest")
    args = ap.parse_args()
    if args.forge_validation:
        cmd_forge_validation(args)
    elif args.clean_validation:
        cmd_clean_validation(args)
    else:
        cmd_fill(args)


if __name__ == "__main__":
    main()
