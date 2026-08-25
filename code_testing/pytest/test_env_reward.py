# test_env_reward.py
#
# Offline reward / termination / parsing tests for StreetFighterEnvV3.
# Runs with no emulator, no socket and no ROM.

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

from fakes.fake_bizhawk import FakeBizHawkEnv, make_payload
import core.config as config


def test_reset_fills_frame_stack():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    obs, info = env.reset()
    assert obs.shape == (554 * config.NUM_FRAMES,)
    # All four stacked frames are identical on reset.
    frame0 = obs[:554]
    for i in range(1, config.NUM_FRAMES):
        assert np.array_equal(obs[i * 554:(i + 1) * 554], frame0)


def test_damage_dealt_is_positive_reward():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(176, 156)])
    _, reward, terminated, truncated, _ = env.step(np.array([0, 0]))
    assert reward > 15.0
    assert not terminated and not truncated


def test_damage_taken_is_negative_reward():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(156, 176)])
    _, reward, _, _, _ = env.step(np.array([0, 0]))
    assert reward < 0.0


def test_ko_terminates_and_reports_win():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(120, 0)])
    _, reward, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["win"] == 1
    assert reward > 60.0


def test_action_string_is_ten_bits():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(176, 176)])
    env.step(np.array([4, 6]))  # Right + Z (HP)
    command = env.sent[-1]
    assert command.endswith("\n")
    assert len(command) == 21  # 10 bits P1 + 10 bits P2 + newline
    assert set(command[:20]) <= {"0", "1"}


def test_double_ko_is_flagged_and_not_scored_as_loss():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(0, 0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["double_ko"] is True
    assert info["win"] == 0          # a draw is not a win ...
    assert info["loss"] == 0         # ... but it is not a loss either


def test_timeout_truncation_is_flagged():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env._steps = config.MAX_STEPS_PER_ROUND - 1
    env.queue([make_payload(100, 100)])
    _, _, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is False
    assert truncated is True
    assert info["timeout"] is True
    assert info["win"] == 0


def test_hp_sentinel_frame_is_flagged_and_does_not_terminate():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    # 0xFF on both players: a round-transition sentinel, not a double KO.
    env.queue([make_payload(255, 255)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert info["hp_sentinel"] is True
    assert terminated is False


def test_episode_steps_are_reported_on_termination():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(176, 170), make_payload(176, 0)])
    env.step(np.array([0, 0]))
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["episode_steps"] == 2


def test_reward_components_sum_to_total():
    from envs.reward import RewardConfig, RewardState, compute_reward

    cfg = RewardConfig()
    state = RewardState(prev_my_hp=176.0, prev_enemy_hp=176.0,
                        prev_rel_dist=80.0, combo_counter=0,
                        frames_since_last_hit=0)
    reward, new_state, parts = compute_reward(
        state, my_hp=170.0, enemy_hp=150.0, rel_dist=60.0,
        terminated=False, cfg=cfg,
    )
    assert reward == pytest.approx(sum(parts.values()))
    assert parts["damage"] == pytest.approx(26.0)
    assert parts["taken"] == pytest.approx(-0.77 * 6.0)
    assert new_state.prev_my_hp == 170.0
    assert new_state.prev_enemy_hp == 150.0


def test_potential_based_shaping_telescopes_to_zero_on_a_round_trip():
    """PBRS guarantee (Ng, Harada & Russell 1999): shaping over a closed loop
    of states contributes (gamma^n - 1) * Phi, i.e. exactly 0 at gamma = 1.
    This is what makes the 50x scale-up safe."""
    from envs.reward import RewardConfig, RewardState, compute_reward

    cfg = RewardConfig(gamma=1.0)
    state = RewardState(176.0, 176.0, 80.0, 0, 0)
    total_shaping = 0.0
    for dist in (60.0, 40.0, 60.0, 80.0):
        _, state, parts = compute_reward(state, 176.0, 176.0, dist, False, cfg)
        total_shaping += parts["shaping"]
    assert total_shaping == pytest.approx(0.0, abs=1e-9)


def test_spacing_potential_has_no_dead_zone_across_the_measured_range():
    """The regression this task exists to fix: the old potential was
    identically zero for every d >= 80, which telemetry measured at 52.2%
    of all training steps. Every adjacent pair must now differ."""
    from envs.reward import RewardConfig, spacing_potential

    cfg = RewardConfig()
    values = [spacing_potential(float(d), cfg) for d in range(0, 188)]
    deltas = [b - a for a, b in zip(values, values[1:])]
    assert all(abs(d) > 1e-6 for d in deltas), "flat region found in the potential"


def test_spacing_potential_peaks_at_poke_range():
    from envs.reward import RewardConfig, spacing_potential

    cfg = RewardConfig()
    values = [spacing_potential(float(d), cfg) for d in range(0, 188)]
    assert values.index(max(values)) == int(cfg.peak_dist) == 70
    # Decays on BOTH sides -- a monotone "closer is better" potential would
    # teach pure rushdown, which is the wrong game for Ryu.
    assert spacing_potential(20.0, cfg) < spacing_potential(70.0, cfg)
    assert spacing_potential(150.0, cfg) < spacing_potential(70.0, cfg)


def test_closing_from_max_range_to_poke_range_is_worth_about_one_light_hit():
    """Previously this whole traverse was worth +0.05 against a -8.6 time
    penalty. It must now be on the same order as landing a hit."""
    from envs.reward import RewardConfig, RewardState, compute_reward

    cfg = RewardConfig(gamma=1.0)
    state = RewardState(176.0, 176.0, 187.0, 0, 0)
    shaping = 0.0
    for dist in range(186, 69, -1):
        _, state, parts = compute_reward(state, 176.0, 176.0, float(dist), False, cfg)
        shaping += parts["shaping"]
    assert 1.5 < shaping < 4.0


def test_time_penalty_over_a_full_round_no_longer_rivals_a_hit():
    """Measured mean episode length is ~570 steps."""
    from envs.reward import RewardConfig

    cfg = RewardConfig()
    assert cfg.time_penalty * 570 < 2.0


def test_combo_counter_extends_within_window_and_resets_outside():
    from envs.reward import RewardConfig, RewardState, compute_reward

    cfg = RewardConfig()
    state = RewardState(176.0, 176.0, 80.0, 0, 5)
    # Hit inside the combo window -> counter increments.
    _, state, parts = compute_reward(state, 176.0, 166.0, 80.0, False, cfg)
    assert state.combo_counter == 1
    assert parts["combo"] == pytest.approx(0.5)

    state.frames_since_last_hit = 99  # far outside the window
    _, state, parts = compute_reward(state, 176.0, 156.0, 80.0, False, cfg)
    assert state.combo_counter == 1
    assert parts["combo"] == pytest.approx(0.5)


def test_terminal_bonus_is_applied_only_when_terminated():
    from envs.reward import RewardConfig, RewardState, compute_reward

    cfg = RewardConfig()
    state = RewardState(176.0, 20.0, 80.0, 0, 0)
    _, _, parts = compute_reward(state, 176.0, 0.0, 80.0, True, cfg)
    assert parts["terminal"] == pytest.approx(65.0)

    state = RewardState(20.0, 176.0, 80.0, 0, 0)
    _, _, parts = compute_reward(state, 0.0, 176.0, 80.0, True, cfg)
    assert parts["terminal"] == pytest.approx(-50.0)


def test_v3_parses_the_expanded_24_field_payload_identically():
    """The wider payload must not break v1/v2/v3 or any saved model."""
    legacy = FakeBizHawkEnv([make_payload(176, 176, rel_dist=90)])
    obs_legacy, _ = legacy.reset()

    wide = FakeBizHawkEnv([make_payload(176, 176, rel_dist=90, extended=True)])
    obs_wide, _ = wide.reset()

    assert np.array_equal(obs_legacy, obs_wide)
    assert legacy.extra_ram == {}
    assert wide.extra_ram["p2_btn"] == 0


def test_expanded_payload_exposes_the_recovered_fields():
    env = FakeBizHawkEnv([make_payload(176, 176, extended=True,
                                       p1_act_lo=5, p2_btn=8, p2_air=1)])
    env.reset()
    assert env.extra_ram["p1_act_lo"] == 5
    assert env.extra_ram["p2_btn"] == 8
    assert env.extra_ram["p2_air"] == 1
