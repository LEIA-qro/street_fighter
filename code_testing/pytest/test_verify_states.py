# test_verify_states.py
#
# Offline unit tests for tools/verify_states.py's pure parts and for the
# states_manifest.json catalog itself: schema shape, source vocabulary,
# manifest <-> integration-dir file agreement, the opponent-id table staying
# in sync with core/telemetry.CHAR_NAMES, pass/fail semantics, and the
# defensive read-merge-write. Runs with no emulator, no ROM, and no
# stable-retro -- verify_states imports the env lazily inside main().

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

_SPEC = importlib.util.spec_from_file_location(
    "verify_states", os.path.join(PROJECT_ROOT, "tools", "verify_states.py"))
verify_states = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verify_states)

MANIFEST_PATH = os.path.join(PROJECT_ROOT, "retro_integration", "states_manifest.json")
INTEGRATION_DIR = os.path.join(
    PROJECT_ROOT, "retro_integration",
    "StreetFighterIISpecialChampionEdition-Genesis-v0")

VALID_SOURCES = {"shipped", "fightladder", "farmed"}
REQUIRED_KEYS = {"opponent", "difficulty", "source", "verified"}
VERIFIED_KEYS = {"loads", "p2_char_id", "hp_start", "fight_alive"}


def load_manifest():
    with open(MANIFEST_PATH) as f:
        return json.load(f)


# ---------------------------------------------------------------- schema ----

def test_manifest_schema():
    manifest = load_manifest()
    assert set(manifest) >= {"states"}
    states = manifest["states"]
    assert isinstance(states, dict) and states
    for name, entry in states.items():
        assert REQUIRED_KEYS <= set(entry), f"{name}: missing {REQUIRED_KEYS - set(entry)}"
        assert entry["source"] in VALID_SOURCES, f"{name}: source {entry['source']}"
        diff = entry["difficulty"]
        assert diff is None or (isinstance(diff, int) and 1 <= diff <= 8), \
            f"{name}: difficulty {diff!r}"
        opp = entry["opponent"]
        assert opp is None or opp in verify_states.OPPONENT_BY_ID.values(), \
            f"{name}: opponent {opp!r}"
        v = entry["verified"]
        if v is not None:
            assert VERIFIED_KEYS <= set(v), f"{name}: verified missing keys"
            assert isinstance(v["loads"], bool)


def test_manifest_matches_state_files_on_disk():
    # Every cataloged state has a .state file, and every non-metadata .state
    # in the integration dir is cataloged -- the linter's premise.
    states = load_manifest()["states"]
    on_disk = {f[:-len(".state")] for f in os.listdir(INTEGRATION_DIR)
               if f.endswith(".state")}
    assert set(states) == on_disk


def test_shipped_state_cataloged_and_untouched():
    states = load_manifest()["states"]
    entry = states["Champion.Level1.RyuVsGuile"]
    assert entry["source"] == "shipped"
    assert entry["opponent"] == "GUILE"


def test_two_player_states_marked():
    states = load_manifest()["states"]
    for name, entry in states.items():
        if "2Player" in name:
            assert entry.get("players") == 2, f"{name} not marked 2P"


# ------------------------------------------------------- opponent tables ----

def test_opponent_table_matches_telemetry_char_names():
    from core.telemetry import CHAR_NAMES
    assert set(verify_states.OPPONENT_BY_ID) == set(CHAR_NAMES)
    for cid, name in CHAR_NAMES.items():
        house = re.sub(r"[^A-Za-z]", "", name).upper()  # E.Honda -> EHONDA
        assert verify_states.OPPONENT_BY_ID[cid] == house


# --------------------------------------------------------- pass semantics ---

def _res(loads=True, alive=True):
    return {"opponent": "GUILE",
            "verified": {"loads": loads, "p2_char_id": 3,
                         "hp_start": [176, 176], "fight_alive": alive}}


def test_passed_requires_load():
    assert not verify_states.passed({}, _res(loads=False))


def test_passed_1p_requires_alive():
    assert verify_states.passed({}, _res(alive=True))
    assert not verify_states.passed({}, _res(alive=False))


def test_passed_2p_load_only():
    assert verify_states.passed({"players": 2}, _res(alive=None))


# ------------------------------------------------------ read-merge-write ----

def test_write_manifest_merged_is_defensive(tmp_path):
    path = str(tmp_path / "manifest.json")
    seeded = {
        "states": {
            "mine": {"opponent": None, "difficulty": None,
                     "source": "fightladder", "verified": None},
            "preset": {"opponent": "KEN", "difficulty": 2,
                       "source": "farmed", "verified": None},
            "foreign": {"opponent": "VEGA", "difficulty": 7,
                        "source": "farmed", "verified": {"loads": True}},
        }
    }
    with open(path, "w") as f:
        json.dump(seeded, f)

    verify_states.write_manifest_merged(
        {"mine": _res(), "preset": _res(), "vanished": _res()}, path=path)

    merged = json.load(open(path))["states"]
    # measured entry: verified stored, null opponent filled from RAM read
    assert merged["mine"]["verified"]["loads"] is True
    assert merged["mine"]["opponent"] == "GUILE"
    # already-attributed opponent and difficulty are never overwritten
    assert merged["preset"]["opponent"] == "KEN"
    assert merged["preset"]["difficulty"] == 2
    # entries we did not measure stay byte-identical; unknown results dropped
    assert merged["foreign"] == seeded["states"]["foreign"]
    assert "vanished" not in merged
