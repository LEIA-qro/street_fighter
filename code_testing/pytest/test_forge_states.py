# test_forge_states.py
#
# Offline unit tests for the pure helpers of tools/forge_states.py (grid
# inventory, donor choice, blob-level work-RAM location) and the statistical
# core of tools/validate_forged_states.py (permutation test + verdict
# logic). No emulator, no ROM, no torch: the poke-and-save path itself is
# exercised by running the tools, and its behavioral truth by the
# validation experiment those tools gate on.

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

import numpy as np
import pytest

import forge_states
from forge_states import (
    DIFFICULTY_DEPENDENT_RAM, DIFFICULTY_RAM_OFFSET, authentic_levels,
    blob_offset, forged_entry, locate_workram_base, missing_pairs,
    pick_donor, swap16, validation_forge_plan,
)
from farm_states import MAX_LEVEL, MIN_LEVEL, OPP_NAMES, validate_entry
from validate_forged_states import judge_global, judge_opponent, perm_pvalue


def manifest_with(rows):
    return {"schema": "sf2-retro-states/v1", "states": rows}


def farmed_row(opp, lvl):
    return {"opponent": opp, "difficulty": lvl, "source": "farmed",
            "verified": {"loads": True}}


# --------------------------------------------------------------------------
# Grid inventory
# --------------------------------------------------------------------------

def test_authentic_levels_sorts_and_skips_forged():
    m = manifest_with({
        "RYU_KEN_R1_lvl3": farmed_row("KEN", 3),
        "RYU_KEN_R1_lvl1": farmed_row("KEN", 1),
        "RYU_KEN_R1_lvl5": {"opponent": "KEN", "difficulty": 5,
                            "source": "forged", "verified": {"loads": True}},
        "FL_Level1.1": {"opponent": None, "difficulty": None,
                        "source": "fightladder", "verified": {"loads": True}},
    })
    assert authentic_levels(m) == {"KEN": [1, 3]}


def test_missing_pairs_covers_the_grid():
    rows = {}
    for opp_id, opp in OPP_NAMES.items():
        for lvl in range(MIN_LEVEL, MAX_LEVEL + 1):
            if not (opp == "VEGA" and lvl >= 7):
                rows[f"RYU_{opp}_R1_lvl{lvl}"] = farmed_row(opp, lvl)
    missing = missing_pairs(manifest_with(rows))
    assert missing == [("VEGA", 7), ("VEGA", 8)]


def test_missing_pairs_counts_source_agnostic():
    # A forged row still occupies its cell: missing == no row at all.
    rows = {"RYU_KEN_R1_lvl2": {"opponent": "KEN", "difficulty": 2,
                                "source": "forged",
                                "verified": {"loads": True}}}
    missing = missing_pairs(manifest_with(rows))
    assert ("KEN", 2) not in missing
    assert len(missing) == 12 * 8 - 1


def test_pick_donor_prefers_highest_below():
    assert pick_donor([1, 2, 3, 4], 6) == 4
    assert pick_donor([1, 7], 5) == 1
    assert pick_donor([4, 7], 2) == 4  # falls back to nearest above
    with pytest.raises(ValueError):
        pick_donor([], 5)


def test_validation_forge_plan_both_directions():
    plan = validation_forge_plan({"CHUNLI": [1, 4, 8]}, opps=("CHUNLI",))
    assert plan == [
        ("FORGEVAL_CHUNLI_lvl8_from1", "RYU_CHUNLI_R1_lvl1", 8),
        ("FORGEVAL_CHUNLI_lvl1_from8", "RYU_CHUNLI_R1_lvl8", 1),
    ]
    with pytest.raises(ValueError):
        validation_forge_plan({"CHUNLI": [1]}, opps=("CHUNLI",))


def test_forged_entry_is_manifest_valid_and_carries_provenance():
    entry = forged_entry("VEGA", 6, "RYU_VEGA_R1_lvl4", {"loads": True})
    validate_entry("RYU_VEGA_R1_lvl6", entry)  # farm's schema accepts it
    assert entry["source"] == "forged"
    assert entry["donor"] == "RYU_VEGA_R1_lvl4"
    assert entry["verified"]["forged_from"] == "RYU_VEGA_R1_lvl4"
    assert "0x97B2" in entry["verified"]["forge_method"]


def test_difficulty_dependent_ram_tables_are_complete():
    assert DIFFICULTY_RAM_OFFSET in DIFFICULTY_DEPENDENT_RAM
    assert 0x97B2 in DIFFICULTY_DEPENDENT_RAM  # the in-fight copy
    for off, table in DIFFICULTY_DEPENDENT_RAM.items():
        assert sorted(table) == list(range(1, 9)), hex(off)
        assert all(0 <= v <= 255 for v in table.values()), hex(off)
    # the two level-1 copies agree byte for byte
    assert (DIFFICULTY_DEPENDENT_RAM[0x97B2]
            == DIFFICULTY_DEPENDENT_RAM[DIFFICULTY_RAM_OFFSET])


# --------------------------------------------------------------------------
# Blob-level work-RAM location
# --------------------------------------------------------------------------

def synthetic_ram(seed=7):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=0x10000, dtype=np.uint8)


def test_locate_direct_layout():
    ram = synthetic_ram()
    blob = b"\x00" * 500 + bytes(ram) + b"\xff" * 300
    base, layout = locate_workram_base(blob, ram)
    assert (base, layout) == (500, "direct")
    for off in DIFFICULTY_DEPENDENT_RAM:
        assert blob[blob_offset(base, off, layout)] == ram[off]


def test_locate_swapped_layout():
    ram = synthetic_ram()
    blob = b"\x11" * 321 + swap16(bytes(ram)) + b"\x22" * 100
    base, layout = locate_workram_base(blob, ram)
    assert (base, layout) == (321, "swapped")
    for off in (0x96B8, 0x97B2, 0xBA35, DIFFICULTY_RAM_OFFSET):  # even & odd
        assert blob[blob_offset(base, off, layout)] == ram[off]


def test_locate_widens_past_duplicated_window():
    # First anchor's 128-byte window appears twice; the wider window is
    # unique, so location must survive by widening instead of guessing.
    ram = synthetic_ram()
    dup = bytes(ram[0xFE00:0xFE00 + 128])
    blob = dup + b"\x00" * 64 + bytes(ram)
    base, layout = locate_workram_base(blob, ram)
    assert (base, layout) == (128 + 64, "direct")


def test_locate_raises_when_absent():
    ram = synthetic_ram()
    with pytest.raises(RuntimeError):
        locate_workram_base(b"\x00" * 4096, ram)


def test_swap16_roundtrip():
    data = bytes(range(10))
    assert swap16(swap16(data)) == data
    assert swap16(b"\x01\x02") == b"\x02\x01"


# --------------------------------------------------------------------------
# Validator statistics + verdict logic
# --------------------------------------------------------------------------

def test_perm_pvalue_separates_and_matches():
    rng = np.random.default_rng(0)
    same_a = rng.normal(0.0, 1.0, 40)
    same_b = rng.normal(0.0, 1.0, 40)
    far_b = rng.normal(3.0, 1.0, 40)
    assert perm_pvalue(same_a, same_b, seed=1) > 0.05
    assert perm_pvalue(same_a, far_b, seed=1) < 0.001
    # deterministic under a fixed seed
    assert perm_pvalue(same_a, far_b, seed=1) == perm_pvalue(same_a, far_b, seed=1)


def comps(control_p=0.0001, equiv_p=0.5, discrim_p=0.0001,
          equiv_dwr=0.02, control_dwr=0.5):
    c = {"p_fitness": control_p, "p_wins": control_p,
         "d_win_rate": control_dwr, "d_fitness": 1.0}
    eq = {"p_fitness": equiv_p, "p_wins": equiv_p,
          "d_win_rate": equiv_dwr, "d_fitness": 0.0}
    di = {"p_fitness": discrim_p, "p_wins": discrim_p,
          "d_win_rate": control_dwr, "d_fitness": 1.0}
    return {"control": c, "equiv_hi": dict(eq), "discrim_hi": dict(di),
            "equiv_lo": dict(eq), "discrim_lo": dict(di)}


def test_judge_opponent_pass():
    verdict, _ = judge_opponent(comps(), control_wr_gap=0.5)
    assert verdict == "PASS"


def test_judge_opponent_inconclusive_without_control():
    verdict, why = judge_opponent(comps(control_p=0.4), control_wr_gap=0.02)
    assert verdict == "INCONCLUSIVE"
    assert "control" in why


def test_judge_opponent_fails_on_broken_equivalence():
    verdict, _ = judge_opponent(comps(equiv_p=0.001), control_wr_gap=0.5)
    assert verdict == "FAIL"
    # a big win-rate gap to the label fails even with a soft p
    verdict, _ = judge_opponent(comps(equiv_dwr=0.4), control_wr_gap=0.5)
    assert verdict == "FAIL"


def test_judge_opponent_inconclusive_when_forged_matches_donor():
    verdict, why = judge_opponent(comps(discrim_p=0.5), control_wr_gap=0.5)
    assert verdict == "INCONCLUSIVE"
    assert "donante" in why


def test_judge_global():
    assert judge_global({"A": ("PASS", ""), "B": ("PASS", ""),
                         "C": ("INCONCLUSIVE", "")}) == "PASS"
    assert judge_global({"A": ("PASS", ""), "B": ("FAIL", ""),
                         "C": ("PASS", "")}) == "FAIL"
    assert judge_global({"A": ("PASS", ""), "B": ("INCONCLUSIVE", ""),
                         "C": ("INCONCLUSIVE", "")}) == "INCONCLUSIVE"


def test_fill_gate_reads_verdict(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    monkeypatch.setattr(forge_states, "VALIDATION_REPORT", str(report))
    assert forge_states.validation_verdict() == "MISSING"
    report.write_text('{"verdict": "PASS"}')
    assert forge_states.validation_verdict() == "PASS"
    report.write_text("not json")
    assert forge_states.validation_verdict() == "MISSING"
