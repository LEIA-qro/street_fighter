# test_state_manifest.py
#
# Offline unit tests for the pure helpers of tools/farm_states.py: curriculum
# state naming from char ids, manifest schema validation, and the
# read-merge-write manifest helper (lossless + collision-safe). Runs with no
# emulator, no ROM, no stable-retro and no torch -- farm_states imports its
# emulator and model machinery lazily, so importing the module is safe here.
# The emulator-driven farming/verification itself is covered by RUNNING
# tools/farm_states.py, not by pytest.

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
TOOLS_PATH = os.path.join(PROJECT_ROOT, "tools")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if TOOLS_PATH not in sys.path:
    sys.path.insert(0, TOOLS_PATH)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest

import farm_states
from farm_states import (
    DIFFICULTY_RAM_OFFSET, MANIFEST_SCHEMA, MAX_LEVEL, MIN_LEVEL, OPP_NAMES,
    is_fight_start, merge_manifest, new_manifest, state_name, validate_entry,
    validate_manifest, write_manifest,
)


def make_entry(opponent="GUILE", difficulty=1, **verified_extra):
    verified = {"p2_char": 3, "difficulty_byte": difficulty - 1}
    verified.update(verified_extra)
    return {"opponent": opponent, "difficulty": difficulty,
            "source": "farmed", "verified": verified}


# --------------------------------------------------------------------------
# Naming
# --------------------------------------------------------------------------

class TestStateName:
    def test_matches_bizhawk_spelling(self):
        # Same names as the states/ BizHawk curriculum files: uppercase, no
        # punctuation (E.Honda -> EHONDA, Chun-Li -> CHUNLI, M.Bison -> MBISON).
        assert state_name(3, 1) == "RYU_GUILE_R1_lvl1"
        assert state_name(1, 4) == "RYU_EHONDA_R1_lvl4"
        assert state_name(5, 7) == "RYU_CHUNLI_R1_lvl7"
        assert state_name(8, 8) == "RYU_MBISON_R1_lvl8"

    def test_mirror_match_is_a_valid_opponent(self):
        assert state_name(0, 2) == "RYU_RYU_R1_lvl2"

    def test_char_id_order_matches_telemetry_table(self):
        # Pinned copy of core/telemetry.py CHAR_NAMES ids (that module cannot
        # be imported off-Windows -- it drags in core.config, which raises
        # when EmuHawk.exe is absent; same duplication precedent as
        # envs/retro_env.py's constants block).
        expected = ["RYU", "EHONDA", "BLANKA", "GUILE", "KEN", "CHUNLI",
                    "ZANGIEF", "DHALSIM", "MBISON", "SAGAT", "BALROG", "VEGA"]
        assert [OPP_NAMES[i] for i in range(12)] == expected

    @pytest.mark.parametrize("opp_id", [-1, 12, 100])
    def test_rejects_bad_char_id(self, opp_id):
        with pytest.raises(ValueError):
            state_name(opp_id, 1)

    @pytest.mark.parametrize("level", [0, 9, -3])
    def test_rejects_bad_level(self, level):
        with pytest.raises(ValueError):
            state_name(0, level)


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------

class TestValidation:
    def test_good_entry_passes(self):
        validate_entry("RYU_GUILE_R1_lvl1", make_entry())

    @pytest.mark.parametrize("missing", ["opponent", "difficulty", "source",
                                         "verified"])
    def test_missing_key_rejected(self, missing):
        entry = make_entry()
        del entry[missing]
        with pytest.raises(ValueError, match=missing):
            validate_entry("RYU_GUILE_R1_lvl1", entry)

    def test_unknown_opponent_rejected(self):
        with pytest.raises(ValueError, match="opponent"):
            validate_entry("x", make_entry(opponent="AKUMA"))

    @pytest.mark.parametrize("level", [0, 9])
    def test_out_of_range_difficulty_rejected(self, level):
        entry = make_entry()
        entry["difficulty"] = level
        with pytest.raises(ValueError, match="difficulty"):
            validate_entry("x", entry)

    def test_empty_verification_rejected(self):
        # The whole point of the manifest: an unverified state never enters.
        entry = make_entry()
        entry["verified"] = {}
        with pytest.raises(ValueError, match="verified"):
            validate_entry("x", entry)

    def test_validate_manifest_walks_all_states(self):
        doc = new_manifest()
        doc["states"]["RYU_GUILE_R1_lvl1"] = make_entry()
        validate_manifest(doc)
        doc["states"]["RYU_KEN_R1_lvl1"] = {"opponent": "KEN"}
        with pytest.raises(ValueError):
            validate_manifest(doc)

    def test_new_manifest_carries_difficulty_ram_evidence(self):
        doc = new_manifest()
        assert doc["schema"] == MANIFEST_SCHEMA
        ram = doc["difficulty_ram"]
        assert ram["work_ram_offset"] == DIFFICULTY_RAM_OFFSET == 0xFE45
        # level N maps to byte N-1 for every supported level
        assert ram["levels"] == {str(lvl): lvl - 1
                                 for lvl in range(MIN_LEVEL, MAX_LEVEL + 1)}


# --------------------------------------------------------------------------
# merge_manifest: lossless + collision-safe
# --------------------------------------------------------------------------

class TestMerge:
    def test_addition_lands(self):
        merged = merge_manifest(new_manifest(),
                                {"RYU_GUILE_R1_lvl1": make_entry()})
        assert "RYU_GUILE_R1_lvl1" in merged["states"]
        validate_manifest(merged)

    def test_lossless_preserves_foreign_keys_and_rows(self):
        base = new_manifest()
        base["fleet"] = {"distribution": "rsync"}          # foreign top-level
        base["states"]["RYU_KEN_R1_lvl3"] = make_entry("KEN", 3)  # foreign row
        merged = merge_manifest(base, {"RYU_GUILE_R1_lvl1": make_entry()})
        assert merged["fleet"] == {"distribution": "rsync"}
        assert merged["states"]["RYU_KEN_R1_lvl3"] == make_entry("KEN", 3)
        # and the input dicts were not mutated
        assert "RYU_GUILE_R1_lvl1" not in base["states"]

    def test_same_identity_refarm_replaces(self):
        base = merge_manifest(new_manifest(),
                              {"RYU_GUILE_R1_lvl1": make_entry()})
        fresher = make_entry(timer_ticked_after_frames=181)
        merged = merge_manifest(base, {"RYU_GUILE_R1_lvl1": fresher})
        assert merged["states"]["RYU_GUILE_R1_lvl1"] == fresher

    def test_identity_collision_raises(self):
        base = merge_manifest(new_manifest(),
                              {"RYU_GUILE_R1_lvl1": make_entry()})
        with pytest.raises(ValueError, match="collision"):
            merge_manifest(base, {"RYU_GUILE_R1_lvl1": make_entry("KEN", 1)})
        with pytest.raises(ValueError, match="collision"):
            merge_manifest(base, {"RYU_GUILE_R1_lvl1": make_entry("GUILE", 2)})

    def test_invalid_addition_rejected_before_merge(self):
        with pytest.raises(ValueError):
            merge_manifest(new_manifest(), {"bad": {"opponent": "GUILE"}})


# --------------------------------------------------------------------------
# write_manifest: file-level read-merge-write round trip
# --------------------------------------------------------------------------

class TestWriteManifest:
    def test_creates_then_merges(self, tmp_path):
        path = str(tmp_path / "states_manifest.json")
        write_manifest({"RYU_GUILE_R1_lvl1": make_entry()}, path=path)
        write_manifest({"RYU_KEN_R1_lvl5": make_entry("KEN", 5)}, path=path)
        with open(path) as f:
            doc = json.load(f)
        validate_manifest(doc)
        assert set(doc["states"]) == {"RYU_GUILE_R1_lvl1", "RYU_KEN_R1_lvl5"}
        assert doc["schema"] == MANIFEST_SCHEMA

    def test_preserves_existing_document(self, tmp_path):
        path = str(tmp_path / "states_manifest.json")
        seed = new_manifest()
        seed["raid_notes"] = ["bizhawk import pending"]
        seed["states"]["RYU_VEGA_R1_lvl2"] = make_entry("VEGA", 2)
        with open(path, "w") as f:
            json.dump(seed, f)
        write_manifest({"RYU_GUILE_R1_lvl1": make_entry()}, path=path)
        with open(path) as f:
            doc = json.load(f)
        assert doc["raid_notes"] == ["bizhawk import pending"]
        assert "RYU_VEGA_R1_lvl2" in doc["states"]

    def test_collision_leaves_file_untouched(self, tmp_path):
        path = str(tmp_path / "states_manifest.json")
        write_manifest({"RYU_GUILE_R1_lvl1": make_entry()}, path=path)
        with open(path) as f:
            before = f.read()
        with pytest.raises(ValueError):
            write_manifest({"RYU_GUILE_R1_lvl1": make_entry("KEN", 1)},
                           path=path)
        with open(path) as f:
            assert f.read() == before


# --------------------------------------------------------------------------
# Fight-start signature (pure predicate over a RAM dict)
# --------------------------------------------------------------------------

def fight_start_ram(**overrides):
    ram = {"p1_hp": 176, "p2_hp": 176, "round_timer": 153,
           "matches_won": 0, "enemy_matches_won": 0,
           "p1_char": 0, "p2_char": 3}
    ram.update(overrides)
    return ram


class TestFightStartSignature:
    def test_round1_start_matches(self):
        assert is_fight_start(fight_start_ram(), None)

    def test_shipped_state_values_match(self):
        # The values RetroSF2Env reads right after loading the shipped
        # Champion.Level1.RyuVsGuile state (measured 2026-08-26).
        assert is_fight_start(fight_start_ram(p2_char=3), prev_opp=None)

    @pytest.mark.parametrize("overrides", [
        {"p1_hp": 0, "p2_hp": 0},          # loading screens: HP blanked
        {"round_timer": 152},              # clock already ticking: too late
        {"round_timer": 0},                # menus
        {"matches_won": 1},                # round 2 of a match
        {"enemy_matches_won": 1},
        {"p1_char": 1},                    # someone other than RYU
        {"p2_char": 12},                   # opponent id out of range
    ])
    def test_non_start_frames_rejected(self, overrides):
        assert not is_fight_start(fight_start_ram(**overrides), None)

    def test_stale_scene_rejected_via_prev_opp(self):
        # Bonus stages / victory screens keep the previous opponent's id in
        # p2_char; a CE ladder never repeats an opponent, so "same as last
        # fight" must never read as a new fight.
        assert not is_fight_start(fight_start_ram(p2_char=3), prev_opp=3)
        assert is_fight_start(fight_start_ram(p2_char=4), prev_opp=3)
