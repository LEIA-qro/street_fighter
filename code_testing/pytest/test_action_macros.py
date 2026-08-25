# test_action_macros.py

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import pytest

from envs.action_macros import (
    N_PRIMITIVES, N_ACTIONS, MACRO_NAMES, MACROS,
    mirror_direction, mirror_macro, decode,
)


def test_primitive_count_matches_the_v3_multidiscrete_space():
    assert N_PRIMITIVES == 9 * 7 == 63
    assert N_ACTIONS == N_PRIMITIVES + len(MACRO_NAMES)


def test_every_primitive_decodes_to_exactly_one_step():
    for a in range(N_PRIMITIVES):
        steps = decode(a, facing_right=True)
        assert len(steps) == 1
        direction, button = steps[0]
        assert 0 <= direction < 9
        assert 0 <= button < 7


def test_primitive_decode_is_the_v3_divmod_bijection():
    seen = set()
    for a in range(N_PRIMITIVES):
        seen.add(decode(a, facing_right=True)[0])
    assert len(seen) == 63  # every (direction, button) pair reachable exactly once


def test_hadouken_is_a_quarter_circle_forward_punch():
    # Facing right: Down(2), Down-Right(8), Right(4) + a punch button.
    steps = MACROS["hadouken_lp"]
    assert steps == [(2, 0), (8, 0), (4, 4)]


def test_shoryuken_is_forward_down_downforward_punch():
    assert MACROS["shoryuken_hp"] == [(4, 0), (2, 0), (8, 6)]


def test_tatsumaki_is_quarter_circle_back_kick():
    assert MACROS["tatsumaki_mk"] == [(2, 0), (7, 0), (3, 2)]


def test_mirror_direction_swaps_left_and_right_families():
    assert mirror_direction(3) == 4     # Left  <-> Right
    assert mirror_direction(4) == 3
    assert mirror_direction(5) == 6     # Up-Left <-> Up-Right
    assert mirror_direction(6) == 5
    assert mirror_direction(7) == 8     # Down-Left <-> Down-Right
    assert mirror_direction(8) == 7
    for neutral in (0, 1, 2):           # Neutral / Up / Down are unchanged
        assert mirror_direction(neutral) == neutral


def test_mirror_macro_preserves_buttons_and_length():
    original = MACROS["hadouken_lp"]
    mirrored = mirror_macro(original)
    assert len(mirrored) == len(original)
    assert [b for _, b in mirrored] == [b for _, b in original]
    assert mirrored == [(2, 0), (7, 0), (3, 4)]


def test_decode_mirrors_macros_when_facing_left():
    idx = N_PRIMITIVES + MACRO_NAMES.index("hadouken_lp")
    assert decode(idx, facing_right=True) == [(2, 0), (8, 0), (4, 4)]
    assert decode(idx, facing_right=False) == [(2, 0), (7, 0), (3, 4)]


def test_decode_rejects_out_of_range_actions():
    with pytest.raises(ValueError):
        decode(N_ACTIONS, facing_right=True)
    with pytest.raises(ValueError):
        decode(-1, facing_right=True)
