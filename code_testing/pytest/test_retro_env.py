# test_retro_env.py
#
# Offline unit tests for the pure parts of src/envs/retro_env.py: v4 frame
# assembly from a fake RAM dict, HP-sentinel handling, action translation to
# retro's Genesis button order, sticky movement, and the episode spacing
# aggregates. Runs with no emulator, no ROM, and no stable-retro installed --
# RetroSF2Env imports stable_retro lazily and is never instantiated here.

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pytest

from envs import retro_env
from envs.retro_env import (
    BUTTON_MAP, DIRECTION_MAP, PROJECT_BUTTON_ORDER, RamTrack,
    apply_sticky, assemble_v4_frame, attach_episode_spacing,
    discrete_to_project_bits, project_bits_to_retro,
    NUM_FRAMES, V4_FRAME_DIM, V4_SINGLE_HIGH, V4_SINGLE_LOW,
)

# env.buttons as stable-retro 1.0.1 reports it for Genesis, captured live.
# project_bits_to_retro resolves indices by name at runtime, so this constant
# only pins the test fixture, not the production mapping.
GENESIS_BUTTONS = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT",
                   "C", "Y", "X", "Z"]


def make_ram(**overrides) -> dict:
    """A plausible mid-round raw RAM dict (all 22 data.json variables)."""
    ram = dict(
        p1_hp=176, p2_hp=176, p1_x=100, p2_x=200, p1_y=40, p2_y=40,
        p1_state_word=0, p2_state_word=0, p1_proj_x=0, p2_proj_x=0,
        p1_char=0, p2_char=1, rel_dist=100, p1_btn=0, p2_btn=0,
        p1_air_raw=0, p2_air_raw=14, rel_y_dist=0,
        p1_chest=192, p1_head=192, p2_chest=192, p2_head=192,
    )
    ram.update(overrides)
    return ram


# ---------------------------------------------------------------------------
# Contract pins against the BizHawk-side modules (importable on the dev
# machines; retro_env deliberately does NOT import them at runtime because
# core.config raises without EmuHawk.exe).
# ---------------------------------------------------------------------------

def test_direction_and_button_maps_match_sf2_v3():
    from envs import sf2_v3
    assert retro_env.DIRECTION_MAP == sf2_v3.DIRECTION_MAP
    assert retro_env.BUTTON_MAP == sf2_v3.BUTTON_MAP


def test_duplicated_constants_match_their_sources():
    import core.config as config
    from envs import base_env, sf2_v4
    assert retro_env.NUM_FRAMES == config.NUM_FRAMES
    assert retro_env.MAX_STEPS_PER_ROUND == config.MAX_STEPS_PER_ROUND
    assert (retro_env.HP_SENTINEL_THRESHOLD
            == base_env.StreetFighterBaseEnv.HP_SENTINEL_THRESHOLD)
    assert retro_env.FAR_DIST_THRESHOLD == base_env.FAR_DIST_THRESHOLD
    assert retro_env.ACT_CATEGORIES == base_env.ACT_CATEGORIES
    assert retro_env.CHAR_CATEGORIES == base_env.CHAR_CATEGORIES
    assert retro_env.V4_FRAME_DIM == sf2_v4.V4_FRAME_DIM


def test_observation_bounds_match_the_v4_env():
    from fakes.fake_bizhawk import FakeBizHawkEnvV4, make_payload
    fake = FakeBizHawkEnvV4([make_payload(176, 176)])
    low = np.array(V4_SINGLE_LOW * NUM_FRAMES, dtype=np.float32)
    high = np.array(V4_SINGLE_HIGH * NUM_FRAMES, dtype=np.float32)
    np.testing.assert_array_equal(low, fake.observation_space.low)
    np.testing.assert_array_equal(high, fake.observation_space.high)


def test_module_imports_without_stable_retro(monkeypatch):
    # None in sys.modules makes any `import stable_retro` raise ImportError,
    # so a fresh exec of the module proves the dependency is truly lazy.
    monkeypatch.setitem(sys.modules, "stable_retro", None)
    monkeypatch.setitem(sys.modules, "stable_retro.data", None)
    path = os.path.join(SRC_PATH, "envs", "retro_env.py")
    spec = importlib.util.spec_from_file_location("retro_env_isolated", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.V4_FRAME_DIM == V4_FRAME_DIM


# ---------------------------------------------------------------------------
# Action translation
# ---------------------------------------------------------------------------

def test_neutral_action_presses_nothing():
    arr = project_bits_to_retro(discrete_to_project_bits([0, 0]), GENESIS_BUTTONS)
    assert arr.sum() == 0
    assert arr.shape == (len(GENESIS_BUTTONS),)


def test_up_right_medium_punch_lands_on_genesis_indices():
    # Direction 6 = Up+Right, button 5 = fifth project button = Y (MP).
    arr = project_bits_to_retro(discrete_to_project_bits([6, 5]), GENESIS_BUTTONS)
    pressed = {GENESIS_BUTTONS[i] for i in np.flatnonzero(arr)}
    assert pressed == {"UP", "RIGHT", "Y"}


def test_every_action_combo_translates_by_name_and_never_hits_mode_start():
    for d in DIRECTION_MAP:
        for b in BUTTON_MAP:
            bits = discrete_to_project_bits([d, b])
            arr = project_bits_to_retro(bits, GENESIS_BUTTONS)
            assert int(arr.sum()) == sum(bits)
            for name, bit in zip(PROJECT_BUTTON_ORDER, bits):
                assert arr[GENESIS_BUTTONS.index(name)] == bit
            assert arr[GENESIS_BUTTONS.index("MODE")] == 0
            assert arr[GENESIS_BUTTONS.index("START")] == 0


# ---------------------------------------------------------------------------
# Frame assembly
# ---------------------------------------------------------------------------

def test_frame_layout_and_lua_derivations():
    ram = make_ram(
        p1_state_word=0x1404, p2_state_word=0x0A02,  # hi=posture, lo=move id
        p2_y=0x0105,                                 # u16 read; Lua keeps the low byte (5)
        p1_btn=64, p2_btn=3, rel_y_dist=7, p1_head=100, p2_head=90,
        p1_char=2, p2_char=11, rel_dist=88,
    )
    frame, track, p1_sent, p2_sent = assemble_v4_frame(ram, RamTrack(), is_reset=True)

    assert frame.shape == (V4_FRAME_DIM,)
    assert frame.dtype == np.float32
    assert not p1_sent and not p2_sent
    assert (frame[0], frame[1]) == (176.0, 176.0)
    assert frame[2] == 100.0                       # rel_x = p2_x - p1_x
    assert frame[3] == pytest.approx(5 - 40)       # rel_y after the p2_y & 0xFF mask
    assert frame[4] == 100.0                       # corner dist = min(100, 500-100)
    assert (frame[5], frame[6]) == (-1.0, -1.0)    # both projectiles frozen at 0
    assert (frame[7], frame[8]) == (0.0, 0.0)      # is_reset zeroes velocities
    assert frame[9] == 88.0
    assert (frame[10], frame[11], frame[12]) == (7.0, 100.0, 90.0)
    assert (frame[13], frame[14]) == (0.0, 0.0)
    assert (frame[15], frame[17]) == (0x14, 0x04)  # p1 state word split hi/lo
    assert (frame[16], frame[18]) == (0x0A, 0x02)  # p2 state word split hi/lo
    assert (frame[19], frame[20]) == (64.0, 3.0)
    assert (frame[21], frame[22]) == (2.0, 11.0)
    assert (track.prev_p1_x, track.prev_p2_x) == (100, 200)


def test_air_flag_semantics():
    # 0x80C0: 0 = floor, nonzero (reads 257) = air. 0x86F4: airborne ONLY at 13.
    grounded, _, _, _ = assemble_v4_frame(make_ram(p1_air_raw=0, p2_air_raw=14),
                                          RamTrack(), is_reset=True)
    assert (grounded[13], grounded[14]) == (0.0, 0.0)
    airborne, _, _, _ = assemble_v4_frame(make_ram(p1_air_raw=257, p2_air_raw=13),
                                          RamTrack(), is_reset=True)
    assert (airborne[13], airborne[14]) == (1.0, 1.0)
    # p2's flag is equality, not nonzero-ness: 12 must NOT read as airborne.
    other, _, _, _ = assemble_v4_frame(make_ram(p1_air_raw=1, p2_air_raw=12),
                                       RamTrack(), is_reset=True)
    assert (other[13], other[14]) == (1.0, 0.0)


def test_projectile_freshness_moving_vs_frozen():
    frame, track, _, _ = assemble_v4_frame(make_ram(p1_proj_x=300, p2_proj_x=0),
                                           RamTrack(), is_reset=True)
    assert frame[5] == 300.0      # moved since the boot tracker (0) -> live
    assert frame[6] == -1.0       # frozen at 0 -> dead
    assert track.prev_p1_proj_raw == 300

    frame2, track, _, _ = assemble_v4_frame(make_ram(p1_proj_x=300, p2_proj_x=250),
                                            track)
    assert frame2[5] == -1.0      # same raw as last step -> frozen -> dead
    assert frame2[6] == 250.0     # moved -> live


def test_hp_sentinel_zeroes_only_the_sentinel_side():
    frame, _, p1_sent, p2_sent = assemble_v4_frame(make_ram(p1_hp=65535),
                                                   RamTrack(), is_reset=True)
    assert p1_sent and not p2_sent
    assert (frame[0], frame[1]) == (0.0, 176.0)

    frame, _, p1_sent, p2_sent = assemble_v4_frame(make_ram(p2_hp=201),
                                                   RamTrack(), is_reset=True)
    assert p2_sent and not p1_sent
    assert (frame[0], frame[1]) == (176.0, 0.0)

    # Exactly at the threshold is a real (if impossible) HP, not a sentinel.
    frame, _, p1_sent, p2_sent = assemble_v4_frame(make_ram(p1_hp=200),
                                                   RamTrack(), is_reset=True)
    assert not p1_sent and frame[0] == 200.0


def test_velocity_and_relative_position_clips():
    track = RamTrack(prev_p1_x=100, prev_p2_x=200)
    frame, _, _, _ = assemble_v4_frame(make_ram(p1_x=1000, p2_x=50), track)
    assert frame[7] == 100.0      # 1000-100 = 900 -> clip +100
    assert frame[8] == -100.0     # 50-200 = -150 -> clip -100
    assert frame[2] == -500.0     # 50-1000 = -950 -> clip -500


# ---------------------------------------------------------------------------
# Sticky movement (port of base_env.step's block)
# ---------------------------------------------------------------------------

def test_sticky_initiates_on_fresh_direction_and_holds_it():
    left = discrete_to_project_bits([3, 0])
    bits, direction, counter = apply_sticky(left, None, 0)
    assert (direction, counter) == ("L", 2)
    assert bits == left           # initiation does not modify the current input

    neutral = discrete_to_project_bits([0, 0])
    bits, direction, counter = apply_sticky(neutral, direction, counter)
    assert bits[2] == 1 and bits[3] == 0
    assert (direction, counter) == ("L", 1)
    assert neutral == discrete_to_project_bits([0, 0])  # pure: input untouched


def test_sticky_opposite_input_cancels_then_reinitiates():
    right = discrete_to_project_bits([4, 0])
    bits, direction, counter = apply_sticky(right, "L", 2)
    # The cancel clears L; the fresh Right press then starts its own sticky.
    assert (direction, counter) == ("R", 2)
    assert bits[3] == 1 and bits[2] == 0


def test_sticky_crouch_cancels_without_reinitiating():
    down = discrete_to_project_bits([2, 0])
    bits, direction, counter = apply_sticky(down, "L", 2)
    assert (direction, counter) == (None, 0)
    assert bits == down


# ---------------------------------------------------------------------------
# Episode spacing aggregates
# ---------------------------------------------------------------------------

def test_spacing_aggregates_match_base_env_keys_and_math():
    info = {}
    attach_episode_spacing(info, [70.0, 80.0, 90.0, 100.0])
    assert info["ep_rel_dist_mean"] == pytest.approx(85.0)
    assert info["ep_rel_dist_median"] == pytest.approx(85.0)
    assert info["ep_rel_dist_frac_far"] == pytest.approx(0.75)  # >= 80: three of four


def test_spacing_aggregates_absent_for_empty_episode():
    info = {}
    attach_episode_spacing(info, [])
    assert info == {}


# ---------------------------------------------------------------------------
# Bit-for-bit parity with the BizHawk v4 pipeline
# ---------------------------------------------------------------------------

def test_frame_parity_with_bizhawk_v4_pipeline():
    """assemble_v4_frame(raw RAM) must equal Lua-derivation -> v4 parse.

    The fake BizHawk env consumes the payload the Lua client WOULD have sent
    for the same raw RAM (hi/lo split, air normalization, p2_y mask and
    projectile freshness applied Lua-side); both pipelines must emit identical
    23-float frames, velocities included.
    """
    from fakes.fake_bizhawk import FakeBizHawkEnvV4, make_payload

    ram_a = make_ram(p1_y=0, p2_y=0x0100)  # p2_y masks to 0
    payload_a = make_payload(
        176, 176, p1_x=100, p2_x=200, p1_y=0, p2_y=0,
        p1_act=0, p2_act=0, p1_proj=-1, p2_proj=-1, p1_char=0, p2_char=1,
        rel_dist=100, extended=True, p1_act_lo=0, p2_act_lo=0,
        p1_btn=0, p2_btn=0, p1_air=0, p2_air=0, rel_y_dist=0,
    )
    ram_b = make_ram(
        p1_hp=150, p2_hp=140, p1_x=110, p2_x=190, p1_y=0, p2_y=0x0105,
        p1_state_word=0x1404, p2_state_word=0x0A02,
        p1_proj_x=300, p2_proj_x=0,           # p1 moved -> live; p2 frozen -> dead
        rel_dist=80, p1_btn=64, p1_air_raw=257, p2_air_raw=13,
        rel_y_dist=7, p1_head=100, p2_head=90,
    )
    payload_b = make_payload(
        150, 140, p1_x=110, p2_x=190, p1_y=0, p2_y=5,
        p1_act=0x14, p2_act=0x0A, p1_proj=300, p2_proj=-1,
        p1_char=0, p2_char=1, rel_dist=80, extended=True,
        p1_act_lo=0x04, p2_act_lo=0x02, p1_btn=64, p2_btn=0,
        p1_air=1, p2_air=1, rel_y_dist=7, p1_head=100, p2_head=90,
    )

    frame_a, track, _, _ = assemble_v4_frame(ram_a, RamTrack(), is_reset=True)
    frame_b, track, _, _ = assemble_v4_frame(ram_b, track)

    fake = FakeBizHawkEnvV4([payload_a])
    obs0, _ = fake.reset()
    np.testing.assert_array_equal(obs0[:V4_FRAME_DIM], frame_a)

    fake.queue([payload_b])
    obs1, _, _, _, _ = fake.step(np.array([0, 0]))
    np.testing.assert_array_equal(obs1[-V4_FRAME_DIM:], frame_b)
    assert (frame_b[7], frame_b[8]) == (10.0, -10.0)  # both rigs agree on velocity
