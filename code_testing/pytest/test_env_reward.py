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

from fakes.fake_bizhawk import (KO_HP, KO_HP_DEEP, FakeBizHawkEnv,
                                FakeBizHawkEnvV4, FakeLeagueEnv, make_payload)
import core.config as config
from envs.sf2_v4 import V4_FRAME_DIM


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
    env.queue([make_payload(120, KO_HP)])
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
    env.queue([make_payload(KO_HP, KO_HP)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["double_ko"] is True
    assert info["draw"] is True      # current key; double_ko is its alias
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


def test_single_sided_hp_sentinel_does_not_fabricate_a_loss():
    """P1 HP reading the 0xFF sentinel while P2 reads a real value is a
    menu/transition frame on one side only, not a KO. Before the fix this
    zeroed only raw[0] for the ko/reward check (hp_sentinel required BOTH
    sides to read sentinel), so my_hp=0 < enemy_hp produced a fabricated
    info["loss"] worth -127 reward (terminal -50 + taken -77) from a menu
    frame.
    """
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(255, 120)])  # P1 sentinel, P2 real HP
    _, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is False
    assert truncated is False
    assert info.get("loss", 0) == 0
    assert reward == pytest.approx(0.0)


def test_p1_and_p2_sentinel_flags_track_each_side_independently():
    """hp_sentinel stays the 'any sentinel this frame' flag every existing
    consumer reads; p1_sentinel/p2_sentinel are the new per-side detail."""
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(255, 120)])
    env.step(np.array([0, 0]))
    assert env.p1_sentinel is True
    assert env.p2_sentinel is False
    assert env.hp_sentinel is True


def test_both_sided_hp_sentinel_yields_zero_reward_and_preserves_reward_state():
    """A round-transition frame where both players read the sentinel used to
    feed compute_reward a fabricated (0, 0) HP pair -- roughly +23.5 of pure
    noise from a menu frame (damage_dealt=100, damage_taken=100). It must
    contribute exactly 0.0 reward and leave reward_state untouched, so the
    next real frame diffs against the last real HP rather than zero.
    """
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(255, 255)])
    _, reward, terminated, _, _ = env.step(np.array([0, 0]))
    assert terminated is False
    assert reward == pytest.approx(0.0)

    # The next real frame at the same HP must show zero damage in both
    # directions -- if reward_state had been corrupted to (0, 0) by the
    # sentinel frame, this would instead read damage_dealt=176 (clamped to
    # 100) and damage_taken=176 (clamped to 100).
    env.queue([make_payload(176, 176)])
    _, _, _, _, info2 = env.step(np.array([0, 0]))
    assert info2["reward_parts"]["damage"] == pytest.approx(0.0)
    assert info2["reward_parts"]["taken"] == pytest.approx(0.0)


def test_genuine_single_sided_loss_reports_loss_flag():
    """No test previously asserted directly on info['loss']. A real KO --
    my_hp at 0, enemy_hp alive, no sentinel on either side -- must terminate
    and set info['loss'] = 1."""
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(KO_HP, 120)])
    _, reward, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["loss"] == 1
    assert info["win"] == 0


def test_episode_steps_are_reported_on_termination():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(176, 170), make_payload(176, KO_HP)])
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
    """Measured mean episode length is ~570 steps. This used to assert only
    on the RewardConfig constant and never called compute_reward, so it
    would keep passing even if the time penalty were wired up wrong (e.g.
    applied on hit steps too, or scaled incorrectly) -- exactly the kind of
    silent error C3 shows the per-step reward budget is sensitive to.
    Actually accumulate it over a simulated no-damage round instead.
    """
    from envs.reward import RewardConfig, RewardState, compute_reward

    cfg = RewardConfig()
    state = RewardState(176.0, 176.0, 80.0, 0, 0)
    total_time_penalty = 0.0
    for _ in range(570):
        _, state, parts = compute_reward(state, 176.0, 176.0, 80.0, False, cfg)
        total_time_penalty += parts["time"]

    assert total_time_penalty == pytest.approx(-cfg.time_penalty * 570)
    assert abs(total_time_penalty) < 2.0


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


def test_v4_socket_death_returns_a_correctly_shaped_obs_and_full_info():
    """The RuntimeError recovery path in step() must return an observation
    matching the env's ACTUAL observation_space shape (92 for v4, not the
    v2/v3-sized 2216 it used to hardcode), and an info dict carrying the same
    hp_sentinel/reward_parts keys every other return path includes -- a shape
    or key mismatch here surfaces as an unrelated SB3 error deep into a run.
    """
    # Deliberately skip reset(): self.frames is empty (its natural state
    # before the very first successful frame), which is exactly the branch
    # of the RuntimeError handler that hardcoded a v2/v3-sized zero array.
    env = FakeBizHawkEnvV4([])
    assert len(env.frames) == 0

    def _dead_socket(command):
        raise RuntimeError("socket closed")

    env.send_command = _dead_socket
    obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))

    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    assert reward == 0.0
    assert terminated is True
    assert truncated is False
    assert info["socket_death"] is True
    assert "hp_sentinel" in info
    assert "reward_parts" in info


def test_v4_corrupt_payload_after_a_good_frame_does_not_crash():
    """StreetFighterBaseEnv's corrupt-payload failsafe repeats the last good
    frame verbatim (self.frames[-1][:TOTAL_OBS_DIM]). For v4 that frame is
    already the compact 23-dim layout, not the 554-dim v2/v3 one-hot layout.
    Re-running the one-hot argmax extraction on a 23-element array reads an
    empty slice for the P2 action one-hot (full[266:522]) and raises
    ValueError -- which propagates out of step() (parsing happens outside its
    try/except) and kills a SubprocVecEnv worker. StreetFighterEnvV4 must
    detect the failsafe frame and pass it through unchanged instead.
    """
    env = FakeBizHawkEnvV4([make_payload(176, 176, extended=True,
                                         p1_act_lo=5, p2_btn=8, p2_air=1)])
    good_obs, _ = env.reset()
    good_frame = good_obs[-V4_FRAME_DIM:]

    env.queue(["0 1,2,3"])  # deliberately corrupt: not 13 or 24 fields
    obs, reward, terminated, truncated, info = env.step(np.array([0, 0]))

    assert obs.shape == (V4_FRAME_DIM * config.NUM_FRAMES,)
    latest_frame = obs[-V4_FRAME_DIM:]
    assert latest_frame.shape == (V4_FRAME_DIM,)
    assert np.array_equal(latest_frame, good_frame)


# --------------------------------------------------------------------------
# Anti-jump gate (Run B): ground_gate_shaping redefines the potential over
# the extended state (dist, airborne) -- Phi = spacing_potential(dist) while
# grounded, 0 while airborne -- so jump-approach stops collecting shaping.
# Still pure PBRS over the extended state, hence policy-invariant.
# --------------------------------------------------------------------------

def test_ground_gate_zeroes_phi_on_airborne_frames():
    from envs.reward import RewardConfig, RewardState, compute_reward, spacing_potential

    cfg = RewardConfig(gamma=1.0, ground_gate_shaping=True)
    state = RewardState(176.0, 176.0, 100.0, 0, 0)

    # Ground -> air while closing distance: the jump forfeits the whole
    # accumulated potential instead of cashing in Phi(70) - Phi(100).
    _, state, parts = compute_reward(state, 176.0, 176.0, 70.0, False, cfg,
                                     airborne=True, prev_airborne=False)
    assert parts["shaping"] == pytest.approx(-spacing_potential(100.0, cfg))
    assert state.prev_airborne is True

    # Air -> air: Phi is 0 on both ends, no shaping regardless of distance.
    _, state, parts = compute_reward(state, 176.0, 176.0, 60.0, False, cfg,
                                     airborne=True, prev_airborne=state.prev_airborne)
    assert parts["shaping"] == pytest.approx(0.0)

    # Air -> ground: landing re-earns the potential at the landing distance.
    _, state, parts = compute_reward(state, 176.0, 176.0, 70.0, False, cfg,
                                     airborne=False, prev_airborne=state.prev_airborne)
    assert parts["shaping"] == pytest.approx(spacing_potential(70.0, cfg))
    assert state.prev_airborne is False


def test_ground_gate_shaping_telescopes_across_air_ground_transitions():
    """PBRS over the EXTENDED state: a closed loop through mixed air/ground
    states must still sum to (gamma^n - 1) * Phi = 0 at gamma = 1. This is
    exactly the property that makes the gate safe to bolt on."""
    from envs.reward import RewardConfig, RewardState, compute_reward

    cfg = RewardConfig(gamma=1.0, ground_gate_shaping=True)
    state = RewardState(176.0, 176.0, 80.0, 0, 0)  # start: (80, grounded)
    total_shaping = 0.0
    # Two jump arcs and a walk, returning to the exact starting (80, ground).
    for dist, air in ((70.0, True), (60.0, True), (70.0, False),
                      (90.0, True), (80.0, False)):
        _, state, parts = compute_reward(state, 176.0, 176.0, dist, False, cfg,
                                         airborne=air,
                                         prev_airborne=state.prev_airborne)
        total_shaping += parts["shaping"]
    assert total_shaping == pytest.approx(0.0, abs=1e-9)


def test_ground_gate_default_off_is_bit_identical_to_the_old_outputs():
    """With ground_gate_shaping at its False default the airborne arguments
    must be COMPLETELY inert: same float ops, bit-identical totals and parts,
    for every existing caller that never heard of the gate."""
    from envs.reward import RewardConfig, RewardState, compute_reward, spacing_potential

    cfg = RewardConfig()  # gate off by default
    transitions = [(80.0, 70.0), (70.0, 100.0), (187.0, 0.0), (60.0, 60.0)]
    for prev_d, d in transitions:
        old_style = compute_reward(
            RewardState(176.0, 170.0, prev_d, 0, 0),
            176.0, 150.0, d, False, cfg,
        )
        new_style = compute_reward(
            RewardState(176.0, 170.0, prev_d, 0, 0),
            176.0, 150.0, d, False, cfg,
            airborne=True, prev_airborne=True,  # must be ignored when gated off
        )
        assert new_style[0] == old_style[0]          # bit-identical total
        assert new_style[2] == old_style[2]          # bit-identical parts
        # And both equal the pre-gate formula exactly.
        assert old_style[2]["shaping"] == (
            cfg.gamma * spacing_potential(d, cfg) - spacing_potential(prev_d, cfg)
        )


def test_ground_gate_env_wiring_gates_shaping_in_step():
    """The constructor kwarg must reach compute_reward: an airborne approach
    step pays -Phi(prev) under the gate instead of collecting the ungated
    gamma*Phi(70) - Phi(100) > 0."""
    from envs.reward import spacing_potential

    gated = FakeBizHawkEnv([make_payload(176, 176, rel_dist=100, extended=True)],
                           ground_gate=True)
    gated.reset()
    gated.queue([make_payload(176, 176, rel_dist=70, extended=True, p1_air=1)])
    _, _, _, _, info = gated.step(np.array([0, 0]))
    cfg = gated.reward_cfg
    assert info["reward_parts"]["shaping"] == pytest.approx(
        -spacing_potential(100.0, cfg))

    plain = FakeBizHawkEnv([make_payload(176, 176, rel_dist=100, extended=True)])
    plain.reset()
    plain.queue([make_payload(176, 176, rel_dist=70, extended=True, p1_air=1)])
    _, _, _, _, info = plain.step(np.array([0, 0]))
    cfg = plain.reward_cfg
    assert info["reward_parts"]["shaping"] == pytest.approx(
        cfg.gamma * spacing_potential(70.0, cfg) - spacing_potential(100.0, cfg))


def test_reset_initializes_prev_airborne_from_the_post_load_frame():
    airborne_start = FakeBizHawkEnv([make_payload(176, 176, extended=True, p1_air=1)])
    airborne_start.reset()
    assert airborne_start.reward_state.prev_airborne is True

    # Legacy 13-field client: extra_ram is empty, airborne must read False.
    legacy = FakeBizHawkEnv([make_payload(176, 176)])
    legacy.reset()
    assert legacy.reward_state.prev_airborne is False


def test_ep_air_frac_counts_airborne_non_sentinel_steps():
    env = FakeBizHawkEnv([make_payload(176, 176, extended=True)])
    env.reset()
    env.queue([
        make_payload(176, 176, extended=True, p1_air=1),   # air
        make_payload(176, 176, extended=True, p1_air=0),   # ground
        make_payload(255, 255, extended=True, p1_air=1),   # sentinel: excluded
        make_payload(176, KO_HP, extended=True, p1_air=0),  # terminal KO, ground
    ])
    for _ in range(3):
        env.step(np.array([0, 0]))
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    # 3 non-sentinel steps, 1 airborne; the sentinel frame is in NEITHER the
    # numerator nor the denominator (same exclusion as the rel_dist samples).
    assert info["ep_air_frac"] == pytest.approx(1.0 / 3.0)


def test_ep_air_frac_resets_between_episodes():
    env = FakeBizHawkEnv([make_payload(176, 176, extended=True)])
    env.reset()
    env.queue([make_payload(176, 176, extended=True, p1_air=1),
               make_payload(176, KO_HP, extended=True, p1_air=1)])
    env.step(np.array([0, 0]))
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["ep_air_frac"] == pytest.approx(1.0)

    # Second reset consumes TWO payloads (stale in-flight + post-load).
    env.queue([make_payload(176, 176, extended=True),
               make_payload(176, 176, extended=True)])
    env.reset()
    env.queue([make_payload(176, KO_HP, extended=True, p1_air=0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["ep_air_frac"] == pytest.approx(0.0)


# ==========================================================================
# KO SEMANTICS (Run A post-mortem, 2026-08-25)
#
# Run A reported double_ko_rate rising 0.36 -> 0.55 while win_rate fell to
# 0.086, and that was read as the policy learning to farm draws. It was not.
# Two live measurement passes on the headless stable-retro core (40,000 and
# 4x30,000 emulator frames, 29 KOs, Champion.Level1.RyuVsGuile) found:
#
#   * The HP words at 0xFF8042 / 0xFF82C2 are SIGNED. Their entire observed
#     range is 0..176 and -1. Nothing has ever been read in 177..65525, so
#     every value the code called an "HP sentinel" was -1 seen through the
#     wrong type -- the KO signal itself, inverted into "refuse to terminate".
#   * HP == 0 does NOT mean dead. Both words sit at exactly 0 for hundreds of
#     consecutive frames during round transitions with nobody KO'd (49.4% of
#     frames in one run), and a live fighter was measured at 0 for 437 frames
#     while still dealing damage.
#   * The HP < 0 window is >= 33 emulator frames wide (median 33, max 449) and
#     the WINNER's HP is frozen and intact throughout it, so a 4-frame sampler
#     catches it with probability 1. The [0, 0] reset frame the old code
#     waited for is exactly 1 frame wide -- caught 1 time in 4.
#   * ZERO simultaneous deaths in 29 KOs. Every "double KO" was an artifact.
#
# Each test below fails on the pre-fix code.
# ==========================================================================

def test_a_kod_fighters_hp_word_decodes_to_negative_not_to_65535():
    """The one-line root cause. The Lua client sends read_u16_be and the retro
    integration typed the field ">u2", so -1 arrived as 65535 -- larger than
    every threshold in the codebase, hence "unreadable" instead of "dead"."""
    from envs.reward import hp_to_signed

    assert hp_to_signed(65535) == -1
    assert hp_to_signed(65526) == -10
    assert hp_to_signed(-1) == -1          # already signed (data.json ">i2")
    # Live values pass through untouched, including the two that matter.
    for live in (0, 1, 88, 176, 200):
        assert hp_to_signed(live) == live


def test_hp_zero_is_a_live_reading_and_must_not_terminate_the_episode():
    """MEASURED: a fighter sat at exactly 0 HP for 437 frames while dealing
    damage, and both words hold 0 through every round transition. The old
    `hp <= 0` test called all of that death -- which is how a round-transition
    frame became a terminal DOUBLE_KO."""
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(0, 120)])
    _, _, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is False
    assert truncated is False
    assert info.get("loss", 0) == 0
    assert info.get("double_ko", False) is False


def test_both_hp_words_at_zero_is_a_round_transition_not_a_draw():
    """THE artifact behind Run A's double_ko_rate = 0.55. The ROM blanks both
    HP words for one frame between rounds; the old code terminated there and
    labelled it DOUBLE_KO, because by then the real KO's -1 had already been
    thrown away as a sentinel."""
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(0, 0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is False
    assert info.get("double_ko", False) is False


def test_a_real_ko_terminates_on_the_negative_hp_frame_and_names_the_winner():
    """The KO frame used to be the one frame the env REFUSED to act on: the
    loser's -1 tripped hp_sentinel, which blocked termination for the whole
    ~110-step KO animation. It must terminate immediately, with the winner
    identified from the side that is still alive."""
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(35, KO_HP)])
    _, reward, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["win"] == 1
    assert info["loss"] == 0
    assert info["draw"] is False
    # The terminal payoff is actually PAID on this frame -- the old code
    # zeroed reward and emptied reward_parts on every sentinel frame, so the
    # KO that ended the round earned exactly nothing.
    assert info["reward_parts"]["terminal"] == pytest.approx(65.0)
    assert reward > 0.0


def test_the_winner_is_not_charged_its_remaining_hp_as_damage_taken():
    """P0-2. The old code terminated on the [0, 0] reset frame, where
    `damage_taken = prev_my_hp - 0` billed the WINNER its entire remaining
    health and `damage_dealt = prev_enemy_hp - 0` paid the LOSER the winner's.
    Measured consequences: a win with 61 HP left scored -21.48, a loss with the
    opponent on 170 HP scored +114.72 -- the sign of the round result inverted.
    Terminating on the KO frame instead leaves both HP reads truthful.
    """
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(170, 20), make_payload(170, KO_HP)])
    env.step(np.array([0, 0]))
    _, reward, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True and info["win"] == 1
    parts = info["reward_parts"]
    assert parts["taken"] == pytest.approx(0.0)   # winner lost no HP this step
    assert parts["terminal"] == pytest.approx(65.0)
    assert reward > 0.0

    # ... and the mirror image: losing is negative, not positive.
    loser = FakeBizHawkEnv([make_payload(176, 176)])
    loser.reset()
    loser.queue([make_payload(20, 170), make_payload(KO_HP, 170)])
    loser.step(np.array([0, 0]))
    _, reward, terminated, _, info = loser.step(np.array([0, 0]))
    assert terminated is True and info["loss"] == 1
    assert info["reward_parts"]["damage"] == pytest.approx(0.0)
    assert reward < 0.0


def test_winning_pays_strictly_more_than_losing_on_identical_hp_trades():
    """The gradient the outcome channel is supposed to carry. Same damage
    exchange, opposite winner: the win must dominate. Under the old code this
    comparison came out BACKWARDS on real data (E[reward | won the round] =
    -5.35 vs E[reward | lost] = +23.75)."""
    def play(final):
        env = FakeBizHawkEnv([make_payload(176, 176)])
        env.reset()
        env.queue([make_payload(100, 100), final])
        env.step(np.array([0, 0]))
        return env.step(np.array([0, 0]))[1]

    assert play(make_payload(100, KO_HP)) > play(make_payload(KO_HP, 100))


def test_a_draw_is_never_worth_more_than_losing_cleanly():
    """REQUIREMENT 1. The old terminal block ran two independent `if`s, so a
    double KO collected win_bonus AND loss_penalty: +65 - 50 = +15, NET
    POSITIVE and +65 better than a clean loss. Break-even against playing for
    the win sat at p = 65/115 = 0.565, and Run A's measured win share among
    decisive rounds was 0.566 -- the channel carried no gradient at all."""
    from envs.reward import RewardConfig, RewardState, compute_reward

    cfg = RewardConfig()
    def terminal(my_ko, enemy_ko):
        _, _, parts = compute_reward(
            RewardState(10.0, 10.0, 80.0, 0, 0), 0.0, 0.0, 80.0, True, cfg,
            my_ko=my_ko, enemy_ko=enemy_ko)
        return parts["terminal"]

    win, loss, draw = terminal(False, True), terminal(True, False), terminal(True, True)
    assert win == pytest.approx(65.0)
    assert loss == pytest.approx(-50.0)
    assert draw == pytest.approx(-50.0)
    assert draw <= 0.0, "a draw must never be a net bonus"
    assert draw <= loss < win
    # No probability of winning makes seeking a draw the better bet:
    # p*win + (1-p)*loss >= draw for every p in [0, 1].
    for i in range(11):
        p = i / 10.0
        assert p * win + (1 - p) * loss >= draw - 1e-9


def test_draw_payoff_is_one_explicit_branch_not_a_sum_of_two_bonuses():
    """REQUIREMENT 1, structurally. If the draw payoff were still
    win_bonus - loss_penalty, moving EITHER of those would move it. It must
    depend only on draw_penalty."""
    from envs.reward import RewardConfig, RewardState, compute_reward

    def draw_terminal(**kw):
        cfg = RewardConfig(**kw)
        _, _, parts = compute_reward(
            RewardState(10.0, 10.0, 80.0, 0, 0), 0.0, 0.0, 80.0, True, cfg,
            my_ko=True, enemy_ko=True)
        return parts["terminal"]

    baseline = draw_terminal()
    assert draw_terminal(win_bonus=1000.0) == baseline
    assert draw_terminal(loss_penalty=1000.0) == baseline
    assert draw_terminal(draw_penalty=7.0) == pytest.approx(-7.0)


def test_explicit_ko_flags_override_the_hp_value_in_compute_reward():
    """The env reads HP through a floor of 0 (V4_SINGLE_LOW pins it there), so
    the sign is gone by the time compute_reward sees it. The death flags carry
    it instead; without them a KO'd fighter at obs-HP 0 is indistinguishable
    from a live one at 0."""
    from envs.reward import RewardConfig, RewardState, compute_reward

    cfg = RewardConfig()
    # my_hp reads 0 but the flags say the ENEMY died -> this is a win.
    _, _, parts = compute_reward(RewardState(10.0, 10.0, 80.0, 0, 0),
                                 0.0, 0.0, 80.0, True, cfg,
                                 my_ko=False, enemy_ko=True)
    assert parts["terminal"] == pytest.approx(65.0)


def test_compute_reward_without_ko_flags_is_bit_identical_to_the_old_behaviour():
    """Back-compat: every caller predating the fix passes no flags and must
    keep getting the historical `hp <= 0` classification, bit for bit."""
    from envs.reward import RewardConfig, RewardState, compute_reward

    cfg = RewardConfig()
    for my_hp, enemy_hp in ((0.0, 176.0), (176.0, 0.0), (120.0, 90.0)):
        legacy = compute_reward(RewardState(176.0, 176.0, 80.0, 0, 0),
                                my_hp, enemy_hp, 70.0, True, cfg)
        explicit = compute_reward(RewardState(176.0, 176.0, 80.0, 0, 0),
                                  my_hp, enemy_hp, 70.0, True, cfg,
                                  my_ko=my_hp <= 0, enemy_ko=enemy_hp <= 0)
        assert legacy[0] == explicit[0]
        assert legacy[2] == explicit[2]


def test_outcome_flags_partition_every_terminal_episode():
    """REQUIREMENT 4: win / loss / draw / timeout must be mutually exclusive
    and exhaustive, so the four TensorBoard rates sum to 1.0. Run A's rates
    did not partition anything -- loss_rate was not even logged, it had to be
    reconstructed as 1 - win - double_ko - timeout."""
    cases = {
        "win":     make_payload(35, KO_HP),
        "loss":    make_payload(KO_HP, 35),
        "draw":    make_payload(KO_HP, KO_HP),
    }
    for name, final in cases.items():
        env = FakeBizHawkEnv([make_payload(176, 176)])
        env.reset()
        env.queue([final])
        _, _, terminated, truncated, info = env.step(np.array([0, 0]))
        assert terminated is True and truncated is False
        flags = {"win": bool(info["win"]), "loss": bool(info["loss"]),
                 "draw": bool(info["draw"]), "timeout": bool(info["timeout"])}
        assert sum(flags.values()) == 1, (name, flags)
        assert flags[name] is True

    # Timeout is the fourth cell of the partition.
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env._steps = config.MAX_STEPS_PER_ROUND - 1
    env.queue([make_payload(100, 100)])
    _, _, terminated, truncated, info = env.step(np.array([0, 0]))
    assert (terminated, truncated) == (False, True)
    assert sum([bool(info["win"]), bool(info["loss"]),
                bool(info["draw"]), bool(info["timeout"])]) == 1


def test_ko_is_attributed_to_the_acting_player_under_the_p2_perspective():
    """League/self-play runs the same payload from P2's side. The death flags
    have to flip with the HP words, or the winner and loser swap."""
    env = FakeBizHawkEnv([make_payload(176, 176)], player=2)
    env.reset()
    # Raw payload order is (p1, p2): P1 is dead, so the P2-perspective agent WON.
    env.queue([make_payload(KO_HP, 35)])
    _, reward, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["win"] == 1 and info["loss"] == 0
    assert info["reward_parts"]["terminal"] == pytest.approx(65.0)


def test_a_ko_frame_still_counts_toward_the_spacing_aggregates():
    """The KO frame is the most informative frame of the round -- both
    fighters on screen at real positions -- so it belongs in the rel_dist
    samples even though its HP word is a sentinel. Only a sentinel with NO
    death (a menu / round-transition frame) is excluded."""
    env = FakeBizHawkEnv([make_payload(176, 176, extended=True)])
    env.reset()
    env.queue([
        make_payload(176, 176, rel_dist=100, extended=True),
        make_payload(255, 255, rel_dist=187, extended=True),  # menu: excluded
        make_payload(176, KO_HP, rel_dist=60, extended=True),  # KO: counted
    ])
    for _ in range(2):
        env.step(np.array([0, 0]))
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["ep_rel_dist_mean"] == pytest.approx(np.mean([100, 60]))


# --------------------------------------------------------------------------
# Retro backend: the same semantics, one layer lower.
# --------------------------------------------------------------------------

def test_retro_integration_types_hp_signed_and_exposes_the_winner_counters():
    """The one-letter root cause in the custom integration: ">u2" made every
    KO read 65535. The integration stable-retro ships has always said ">i2".
    matches_won / enemy_matches_won are the independent cross-check -- measured
    ticking +1 emulator frame after the loser's HP goes negative on 29/29 KOs.
    """
    import json

    path = (Path(__file__).resolve().parents[2] / "retro_integration"
            / "StreetFighterIISpecialChampionEdition-Genesis-v0" / "data.json")
    info = json.loads(path.read_text())["info"]
    assert info["p1_hp"]["type"] == ">i2"
    assert info["p2_hp"]["type"] == ">i2"
    assert info["p1_hp"]["address"] == 16744514      # 0xFF8042
    assert info["p2_hp"]["address"] == 16745154      # 0xFF82C2
    assert info["matches_won"] == {"address": 16744922, "type": "|u1"}
    assert info["enemy_matches_won"] == {"address": 16745562, "type": "|u1"}


def test_retro_frame_assembly_treats_a_negative_hp_word_as_dead():
    """With data.json fixed to ">i2" the KO arrives as -1. Under the old
    `raw > 200` test that is NOT a sentinel, so the observation would carry
    hp = -1 -- below V4_SINGLE_LOW's floor of 0 and outside the declared
    observation space."""
    from envs.retro_env import RamTrack, assemble_v4_frame, V4_SINGLE_LOW

    ram = dict(
        p1_hp=176, p2_hp=-1, p1_x=100, p2_x=200, p1_y=40, p2_y=40,
        p1_state_word=0, p2_state_word=0, p1_proj_x=0, p2_proj_x=0,
        p1_char=0, p2_char=1, rel_dist=100, p1_btn=0, p2_btn=0,
        p1_air_raw=0, p2_air_raw=14, rel_y_dist=0,
        p1_chest=192, p1_head=192, p2_chest=192, p2_head=192,
    )
    frame, _, p1_sent, p2_sent = assemble_v4_frame(ram, RamTrack(), is_reset=True)
    assert p2_sent and not p1_sent
    assert (frame[0], frame[1]) == (176.0, 0.0)
    assert frame[1] >= V4_SINGLE_LOW[1], "HP escaped the observation space floor"

    # The legacy unsigned encoding must decode to exactly the same thing, so
    # the death test does not depend on which side typed the RAM read.
    ram_u = dict(ram, p2_hp=65535)
    frame_u, _, _, p2_sent_u = assemble_v4_frame(ram_u, RamTrack(), is_reset=True)
    assert p2_sent_u is p2_sent
    assert np.array_equal(frame_u, frame)


# --------------------------------------------------------------------------
# TIME OVER: the second way a SF2 round ends.
#
# Found while validating the fix on the live core (40-episode run, episode 28):
# the round clock expired with HP at (7, 16), the ROM awarded the round to the
# opponent and ticked enemy_matches_won -- and NEITHER HP word ever went
# negative. The HP sign is blind to this by construction, so the env played 469
# further steps into the next round and then logged a TIMEOUT worth 0 instead
# of the loss it was. The winner counters are the only signal that sees it.
#
# Driven through the real RetroSF2Env.step with a scripted RAM stream, so no
# emulator, no ROM and no stable-retro are needed.
# --------------------------------------------------------------------------

def _base_ram(**overrides):
    ram = dict(
        p1_hp=176, p2_hp=176, p1_x=100, p2_x=200, p1_y=40, p2_y=40,
        p1_state_word=0, p2_state_word=0, p1_proj_x=0, p2_proj_x=0,
        p1_char=0, p2_char=1, rel_dist=100, p1_btn=0, p2_btn=0,
        p1_air_raw=0, p2_air_raw=14, rel_y_dist=0,
        p1_chest=192, p1_head=192, p2_chest=192, p2_head=192,
        matches_won=0, enemy_matches_won=0,
        # Round clock, BCD. 0x99 = "the round is still running", which is what
        # every frame of a live round looks like.
        round_timer=0x99,
    )
    ram.update(overrides)
    return ram


def _make_retro_env(scripted_rams):
    """A RetroSF2Env whose libretro core is a list of RAM dicts."""
    from collections import deque
    from envs import retro_env as R
    from envs.reward import RewardConfig, RewardState, RoundTracker

    env = object.__new__(R.RetroSF2Env)

    class _Core:
        """One scripted RAM dict per AGENT step, not per emulator frame:
        RetroSF2Env.step calls this FRAME_SKIP times and reads only the last
        result, exactly as the Lua client samples once per command cycle."""
        def __init__(self, rams):
            self._rams = list(rams)
            self._n = 0
        def step(self, action):
            self._n += 1
            idx = min((self._n - 1) // R.FRAME_SKIP, len(self._rams) - 1)
            return None, 0.0, False, False, self._rams[idx]

    env._env = _Core(scripted_rams)
    env._buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT",
                    "C", "Y", "X", "Z"]
    env._action_cache = {}
    env._track = R.RamTrack()
    env.frames = deque(maxlen=R.NUM_FRAMES)
    env.sticky_enabled, env.sticky_counter, env.sticky_direction = True, 0, None
    env.trainable, env.player = True, 1
    env.reward_cfg = RewardConfig()
    env.reward_state = RewardState(176.0, 176.0, 100.0, 0, 0)
    env._steps, env.footsie_steps = 0, 0
    env._ep_rel_dists, env._ep_air_steps = [], 0
    env.hp_sentinel = env.p1_sentinel = env.p2_sentinel = False
    env.p1_ko = env.p2_ko = False
    env.matches_won = env.enemy_matches_won = 0
    env.round_timer = 0x99
    env._round = RoundTracker()
    env._round.reset(timer=0x99)
    env.current_state_file = "FAKE.State"
    env.prev_my_hp = env.prev_enemy_hp = 176.0
    env.prev_rel_dist = 100.0
    env.combo_counter = env.frames_since_last_hit = 0
    return env


def test_retro_env_terminates_on_a_ko_and_names_the_winner():
    env = _make_retro_env([_base_ram(p1_hp=35, p2_hp=-1)])
    _, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is True and truncated is False
    assert info["win"] == 1 and info["loss"] == 0 and info["draw"] is False
    assert info["time_over"] is False
    assert info["reward_parts"]["terminal"] == pytest.approx(65.0)
    assert reward > 0.0


def test_retro_env_terminates_on_a_time_over_awarded_by_the_counters():
    """No HP is negative -- the round was decided on the clock. Without the
    counter trigger this episode runs on into the next round and ends as a
    TIMEOUT paying 0, which is what happened on the live core."""
    env = _make_retro_env([_base_ram(p1_hp=7, p2_hp=16, enemy_matches_won=1)])
    _, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is True and truncated is False
    assert info["loss"] == 1 and info["win"] == 0
    assert info["time_over"] is True           # decided on the clock, not a KO
    assert info["timeout"] is False            # NOT the 1500-step truncation
    assert info["reward_parts"]["terminal"] == pytest.approx(-50.0)
    assert reward < 0.0

    # ... and the winning side of the same mechanism.
    env = _make_retro_env([_base_ram(p1_hp=16, p2_hp=7, matches_won=1)])
    _, reward, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True and info["win"] == 1
    assert info["time_over"] is True
    assert info["reward_parts"]["terminal"] == pytest.approx(65.0)


def test_retro_env_prefers_the_hp_sign_over_the_lagging_counters():
    """The counters tick one emulator frame AFTER the HP goes negative, so on
    the death frame itself HP is authoritative and the counters read stale.
    Measured on 13 of 40 episodes -- a counter-triggered design would have
    misread every one of them."""
    env = _make_retro_env([_base_ram(p1_hp=40, p2_hp=-1,
                                     matches_won=0, enemy_matches_won=0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["win"] == 1                     # from the HP sign alone
    assert info["matches_won_delta"] == 0       # counter had not ticked yet


def test_retro_env_does_not_terminate_while_both_fighters_are_alive():
    """HP == 0 on both sides is a round-transition frame, and the counters have
    not moved: nothing has been decided, so the episode must continue."""
    env = _make_retro_env([_base_ram(p1_hp=0, p2_hp=0)])
    _, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is False and truncated is False
    assert "win" not in info
    assert reward == pytest.approx(0.0)


def test_retro_env_counter_reset_at_match_end_cannot_fabricate_a_result():
    """The counters reset to 0 when a match ends, which makes the delta
    NEGATIVE. Only a strictly positive delta may award a round."""
    env = _make_retro_env([_base_ram(p1_hp=90, p2_hp=90,
                                     matches_won=0, enemy_matches_won=0)])
    env._round.reset(matches_won=1, enemy_matches_won=1, timer=0x99)
    _, _, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is False and truncated is False


def test_retro_env_episode_covers_exactly_one_round():
    """P0-4: the shipped env refused to terminate for the whole ~110-step KO
    animation, by which point the ROM had refilled both bars -- so episodes ran
    ~1.9 rounds and accumulated 300 HP of 'damage dealt' against a 176 HP bar.
    The KO frame must end the episode immediately."""
    rams = ([_base_ram(p1_hp=100, p2_hp=50 - i) for i in range(3)]
            + [_base_ram(p1_hp=100, p2_hp=-1)]          # the KO
            + [_base_ram(p1_hp=176, p2_hp=176)] * 5)    # round 2 refill
    env = _make_retro_env(rams)
    steps = 0
    while True:
        _, _, terminated, truncated, info = env.step(np.array([0, 0]))
        steps += 1
        if terminated or truncated:
            break
        assert steps < 9, "episode ran past the KO into the next round"
    assert terminated is True
    assert steps == 4 and info["episode_steps"] == 4
    assert info["win"] == 1


# --------------------------------------------------------------------------
# The 26-field payload: the BizHawk rig's half of the time-over fix.
#
# The Lua client is production hardware owned by another track and cannot be
# tested from here, so the Python side is made ready FIRST and the remaining
# work is a two-line additive Lua edit (documented verbatim in
# agent/stage0-runbook.md). Until that edit ships, the 24-field payload keeps
# working and time-over detection is simply inactive on that rig.
# --------------------------------------------------------------------------

def test_resolve_round_result_prefers_a_ko_over_the_lagging_counters():
    from envs.reward import resolve_round_result

    # KO wins even when the counters point the other way (they lag by a frame).
    assert resolve_round_result(True, False, my_award_delta=1) == (True, False)
    # No KO, opponent's counter ticked -> I lost the round on the clock.
    assert resolve_round_result(False, False, enemy_award_delta=1) == (True, False)
    # No KO, my counter ticked -> I won it on the clock.
    assert resolve_round_result(False, False, my_award_delta=1) == (False, True)
    # Nothing happened.
    assert resolve_round_result(False, False) == (False, False)
    # A counter RESET at match end makes the delta negative: never a result.
    assert resolve_round_result(False, False, my_award_delta=-1,
                                enemy_award_delta=-1) == (False, False)
    # Both ticking at once is a draw.
    assert resolve_round_result(False, False, my_award_delta=1,
                                enemy_award_delta=1) == (True, True)


def test_bizhawk_accepts_the_26_field_payload_and_reads_the_counters():
    env = FakeBizHawkEnv([make_payload(176, 176, counters=True)])
    env.reset()
    env.queue([make_payload(90, 120, counters=True,
                            matches_won=0, enemy_matches_won=1)])
    _, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    # Nobody's HP went negative -- this round was decided on the clock.
    assert terminated is True and truncated is False
    assert info["loss"] == 1 and info["win"] == 0
    assert info["reward_parts"]["terminal"] == pytest.approx(-50.0)
    assert reward < 0.0


def test_the_24_field_payload_still_parses_and_ignores_time_over():
    """Back-compat with the Lua client actually deployed on the rig today."""
    env = FakeBizHawkEnv([make_payload(176, 176, extended=True)])
    env.reset()
    env.queue([make_payload(90, 120, extended=True)])
    _, _, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is False and truncated is False
    assert env.matches_won == 0 and env.enemy_matches_won == 0


def test_26_and_24_field_payloads_produce_identical_observations():
    """Fields 25-26 are purely additive: the observation must not move."""
    wide = FakeBizHawkEnv([make_payload(176, 150, rel_dist=90, counters=True,
                                        matches_won=1, enemy_matches_won=2)])
    obs_wide, _ = wide.reset()
    narrow = FakeBizHawkEnv([make_payload(176, 150, rel_dist=90, extended=True)])
    obs_narrow, _ = narrow.reset()
    assert np.array_equal(obs_wide, obs_narrow)
    assert wide.matches_won == 1 and wide.enemy_matches_won == 2


def test_counters_flip_with_the_p2_perspective():
    env = FakeBizHawkEnv([make_payload(176, 176, counters=True)], player=2)
    env.reset()
    # Raw order is (p1, p2): P1's counter ticked, so the P2-perspective agent
    # LOST the round on the clock.
    env.queue([make_payload(120, 90, counters=True,
                            matches_won=1, enemy_matches_won=0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["loss"] == 1 and info["win"] == 0


def test_counter_baseline_is_taken_per_episode_not_absolute():
    """A savestate loaded mid-match already has non-zero counters. Only the
    delta since reset may end a round, or every episode would end on step 1."""
    env = FakeBizHawkEnv([make_payload(176, 176, counters=True,
                                       matches_won=1, enemy_matches_won=1)])
    env.reset()
    assert env.matches_won == 1
    env.queue([make_payload(120, 120, counters=True,
                            matches_won=1, enemy_matches_won=1)])
    _, _, terminated, truncated, _ = env.step(np.array([0, 0]))
    assert terminated is False and truncated is False


# ==========================================================================
# REGRESSION TESTS for the defects two adversarial validators found in the
# first cut of the round-semantics fix. Every one of these fails against the
# code as it stood before this block was written.
# ==========================================================================

_WIDE_EXTRAS = dict(p1_act_lo=7, p2_act_lo=9, p1_btn=33, p2_btn=0,
                    p1_air=1, p2_air=1, rel_y_dist=55,
                    p1_chest=101, p1_head=103, p2_chest=104, p2_head=105)


def test_every_payload_width_at_or_above_24_fills_extra_ram():
    """P0: `if len(raw) == 24:` sent the 26-field payload -- the one this fix
    exists to accept -- straight to `extra_ram = {}`.

    That silently zeroed 8 of the 23 v4 observation dims and pinned `airborne`
    permanently False, disabling the anti-jump ground gate. It could only have
    detonated on the BizHawk rig, on the day the Lua edit landed, with nothing
    on this machine able to reproduce it. Every optional block is gated `>=`
    now; this pins that for EVERY accepted width, with values that differ from
    the `.get()` fallbacks (the old test used all-default extras, whose values
    coincided exactly with the fallbacks, so it could not have failed).
    """
    from envs.base_env import ACCEPTED_PAYLOAD_WIDTHS

    reference = None
    for width in ACCEPTED_PAYLOAD_WIDTHS:
        if width < 24:
            continue
        kwargs = dict(_WIDE_EXTRAS, extended=True)
        if width >= 26:
            kwargs.update(counters=True, matches_won=1, enemy_matches_won=2)
        if width >= 27:
            kwargs.update(clock=True, round_timer=0x88)
        payload = make_payload(176, 150, rel_dist=90, **kwargs)
        assert len(payload.split(" ")[-1].split(",")) == width

        env = FakeBizHawkEnvV4([payload])
        obs, _ = env.reset()
        assert env.extra_ram, f"{width}-field payload dropped extra_ram"
        for key, value in _WIDE_EXTRAS.items():
            assert env.extra_ram[key] == value, (width, key)
        # The airborne read the ground gate depends on must survive too.
        assert bool(env.extra_ram.get("p1_air", 0)) is True
        assert env.reward_state.prev_airborne is True

        if reference is None:
            reference = obs
        else:
            assert np.array_equal(obs, reference), (
                f"{width}-field payload moved the observation")


def test_a_wider_payload_never_changes_the_v4_observation():
    """The 24 -> 26 comparison the first cut shipped used FakeBizHawkEnv (v3,
    554-dim), the ONE observation layout that never reads extra_ram, with
    all-default extra fields. It was structurally incapable of failing. On the
    v4 layout with non-default fields it fails immediately."""
    wide = FakeBizHawkEnvV4([make_payload(176, 150, rel_dist=90, counters=True,
                                          matches_won=1, enemy_matches_won=2,
                                          **_WIDE_EXTRAS)])
    obs_wide, _ = wide.reset()
    narrow = FakeBizHawkEnvV4([make_payload(176, 150, rel_dist=90,
                                            extended=True, **_WIDE_EXTRAS)])
    obs_narrow, _ = narrow.reset()
    assert np.array_equal(obs_wide, obs_narrow)
    assert wide.matches_won == 1 and wide.enemy_matches_won == 2
    assert narrow.matches_won == 0 and narrow.enemy_matches_won == 0


def test_the_ground_gate_still_gates_on_a_counter_bearing_payload():
    """The concrete downstream damage of the width bug: with extra_ram gone,
    `airborne` is False forever and --ground_gate silently stops gating."""
    def shaping(**payload_kwargs):
        env = FakeBizHawkEnvV4([make_payload(176, 176, rel_dist=120,
                                             **payload_kwargs)],
                               ground_gate=True)
        env.reset()
        env.queue([make_payload(176, 176, rel_dist=70, **payload_kwargs)])
        _, _, _, _, info = env.step(np.array([0, 0]))
        return info["reward_parts"].get("shaping", 0.0)

    airborne = dict(_WIDE_EXTRAS, p1_air=1)
    grounded = dict(_WIDE_EXTRAS, p1_air=0)
    # Airborne on both frames -> the gate zeroes Phi on both sides -> no
    # shaping at all. Grounded -> the approach is rewarded.
    assert shaping(extended=True, **airborne) == pytest.approx(0.0)
    assert shaping(counters=True, **airborne) == pytest.approx(0.0)
    assert shaping(counters=True, **grounded) > 0.0


# --------------------------------------------------------------------------
# The once-per-round latch. `terminated = ko if self.trainable else False`
# suppresses TERMINATION on an eval env but still handed `terminated=ko` to
# compute_reward, and a round result is a STATE hundreds of frames wide, not
# an event. Measured on the live core before the fix: 1,773 terminal payments
# in 2,500 steps (episode return -22,290), and via the counter path -- whose
# delta never returns to zero -- one time over paid a terminal on every
# remaining step of the run, forever.
# --------------------------------------------------------------------------

def test_a_ko_window_pays_its_terminal_exactly_once_on_a_non_trainable_env():
    # NOTE: with trainable=False, reset() reads exactly ONE payload (the
    # trainable path drains a stale one first), so it consumes BOOT_STALE and
    # every frame below is a step frame.
    env = FakeBizHawkEnvV4([], trainable=False)
    env.reset()
    # A KO window is 33-457 emulator frames: many consecutive agent steps see
    # the same negative HP word. Then the ROM refills the bars.
    window = [make_payload(120, KO_HP, extended=True)] * 10
    recovery = [make_payload(176, 176, extended=True)] * 5
    env.queue(window + recovery)

    payments = []
    for _ in range(len(window) + len(recovery)):
        _, _, terminated, truncated, info = env.step(np.array([0, 0]))
        assert terminated is False and truncated is False, "trainable=False"
        bonus = info["reward_parts"].get("terminal", 0.0)
        if bonus:
            payments.append(bonus)
    assert payments == [pytest.approx(65.0)], payments


def test_a_second_round_still_pays_after_the_latch_re_arms():
    """The latch must suppress a REPEAT, not every later round."""
    env = FakeBizHawkEnvV4([], trainable=False)
    env.reset()
    env.queue([make_payload(120, KO_HP, extended=True)] * 4
              + [make_payload(176, 176, extended=True)] * 3
              + [make_payload(KO_HP, 130, extended=True)] * 4)
    payments = []
    for _ in range(11):
        _, _, _, _, info = env.step(np.array([0, 0]))
        bonus = info["reward_parts"].get("terminal", 0.0)
        if bonus:
            payments.append(bonus)
    assert payments == [pytest.approx(65.0), pytest.approx(-50.0)], payments


def test_a_counter_time_over_cannot_pay_forever_on_a_non_trainable_env():
    """The counter delta LATCHES -- it never returns to 0 inside an episode --
    so before the fix one time over paid its terminal on every subsequent step
    of the run. The tracker consumes the delta when it reports it."""
    env = FakeBizHawkEnv([], trainable=False)
    env.reset()
    env.queue([make_payload(90, 120, counters=True, enemy_matches_won=1)] * 30)
    payments = [info["reward_parts"].get("terminal", 0.0)
                for info in (env.step(np.array([0, 0]))[4] for _ in range(30))]
    assert [p for p in payments if p] == [pytest.approx(-50.0)]


def test_the_trainable_path_is_unchanged_by_the_latch():
    """A trainable env terminates on the firing frame, so the latch must be
    invisible there: same step, same payoff, same flags."""
    env = FakeBizHawkEnvV4([make_payload(176, 176, extended=True)])
    env.reset()
    env.queue([make_payload(120, KO_HP, extended=True)])
    _, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is True and truncated is False
    assert info["win"] == 1 and info["reward_parts"]["terminal"] == pytest.approx(65.0)


def test_a_savestate_captured_mid_ko_does_not_terminate_on_step_1():
    """reset() used to "clear" p1_ko/p2_ko and claim that protected step 1. It
    did not -- step() re-derives them from the same still-negative HP word on
    the very next payload, so the flags were clear for exactly zero steps. The
    latch is what actually protects it, and it re-arms once the bars refill."""
    env = FakeBizHawkEnvV4([make_payload(140, KO_HP, extended=True)])
    env.reset()
    env.queue([make_payload(140, KO_HP, extended=True)] * 3
              + [make_payload(176, 176, extended=True)] * 2
              + [make_payload(150, KO_HP, extended=True)])
    for i in range(5):
        _, _, terminated, _, info = env.step(np.array([0, 0]))
        assert terminated is False, f"terminated on the stale KO at step {i+1}"
    # A genuinely NEW KO after the refill still terminates.
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True and info["win"] == 1


def test_the_counter_baseline_follows_a_match_boundary_down():
    """The counters reset to 0 when a MATCH ends. With a baseline that could
    only be set at reset(), a match boundary inside one episode left the delta
    permanently negative and silently disabled counter detection for the rest
    of the episode -- a miss, never a false positive, but a silent one."""
    env = FakeBizHawkEnv([make_payload(176, 176, counters=True,
                                       matches_won=2, enemy_matches_won=1)])
    env.reset()
    assert (env.matches_won, env.enemy_matches_won) == (2, 1)
    # The match ends: both counters reset to 0. Under a baseline that could
    # only be set at reset() the delta is now -2 and stays negative forever.
    env.queue([make_payload(150, 150, counters=True,
                            matches_won=0, enemy_matches_won=0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is False and "win" not in info
    # The first round of the NEW match must still be detectable.
    env.queue([make_payload(150, 90, counters=True,
                            matches_won=1, enemy_matches_won=0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True and info["win"] == 1
    assert info["reward_parts"]["terminal"] == pytest.approx(65.0)


# --------------------------------------------------------------------------
# The THIRD way a round ends: TIME OVER with EQUAL HP ("DRAW GAME"). The ROM
# ends the round and ticks NEITHER counter, so an env watching only HP and the
# counters does not terminate at all -- it plays a whole extra round and
# truncates as a TIMEOUT worth 0, the exact Run A pathology.
#
# Measured on the live core (both HP pinned to 120, clock left to run out):
# the round clock at 0xFF972A goes 0x99 -> 0x00 in BCD, reads 0 for 91-131
# agent steps at every time over -- ~10 agent steps BEFORE the winner's
# counter moves -- and is the only marker present on a draw game.
# --------------------------------------------------------------------------

def test_an_equal_hp_time_over_is_a_draw_not_a_silent_timeout():
    env = _make_retro_env([_base_ram(p1_hp=120, p2_hp=120, round_timer=0)])
    _, reward, terminated, truncated, info = env.step(np.array([0, 0]))
    assert terminated is True and truncated is False
    assert info["draw"] is True and info["double_ko"] is True
    assert info["win"] == 0 and info["loss"] == 0 and info["timeout"] is False
    assert info["time_over"] is True
    assert info["matches_won_delta"] == 0 and info["enemy_matches_won_delta"] == 0
    assert info["reward_parts"]["terminal"] == pytest.approx(-50.0)


def test_the_clock_decides_a_decisive_time_over_before_the_counters_do():
    """At the buzzer the ROM awards the round to whoever has more health, and
    the clock says so ~10 agent steps before the counter ticks. Both fighters
    cap at 176, so "more health" and "higher percentage" are one comparison."""
    env = _make_retro_env([_base_ram(p1_hp=94, p2_hp=176, round_timer=0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True and info["loss"] == 1 and info["time_over"] is True
    assert info["enemy_matches_won_delta"] == 0     # counter had not moved yet

    env = _make_retro_env([_base_ram(p1_hp=176, p2_hp=94, round_timer=0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True and info["win"] == 1 and info["time_over"] is True


def test_a_ko_still_outranks_the_clock():
    env = _make_retro_env([_base_ram(p1_hp=-1, p2_hp=176, round_timer=0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True and info["loss"] == 1
    assert info["time_over"] is False, "a KO is not a time over"


def test_the_clock_cannot_decide_a_round_it_never_saw_running():
    """The clock also reads 0 on the inter-match / continue screens. Measured:
    23 of 23 such windows in a 40,000-step run, all with HP blanked to [0, 0]
    -- so the readability guard alone blocks them, and the arming rule blocks a
    savestate loaded straight into one."""
    # Never armed: the very first frame of the episode already reads 0.
    env = _make_retro_env([_base_ram(p1_hp=120, p2_hp=90, round_timer=0)])
    env._round.reset(timer=0)
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is False

    # Armed, but the frame is the [0, 0] blank the ROM paints between rounds.
    env = _make_retro_env([_base_ram(p1_hp=0, p2_hp=0, round_timer=0)])
    _, reward, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is False and reward == pytest.approx(0.0)


def test_a_transport_without_a_clock_falls_back_to_the_counters():
    """The 24/26-field BizHawk payload carries no clock. `timer=None` must
    switch the clock rule off rather than misfire it as a permanent 0."""
    from envs.reward import RoundTracker

    tracker = RoundTracker()
    tracker.reset(timer=None)
    assert tracker.resolve(False, False, my_hp=90, enemy_hp=120,
                           hp_readable=True, timer=None) == (False, False)
    assert tracker.time_over is False
    assert tracker.resolve(False, False, my_hp=90, enemy_hp=120,
                           hp_readable=True, timer=None,
                           enemy_matches_won=1) == (True, False)


def test_resolve_round_result_ranks_ko_over_clock_over_counters():
    from envs.reward import resolve_round_result

    # Clock beats a counter that points the other way.
    assert resolve_round_result(False, False, my_award_delta=1,
                                time_over=True, my_hp=10,
                                enemy_hp=100) == (True, False)
    # Equal HP on the buzzer is a draw, not "nothing happened".
    assert resolve_round_result(False, False, time_over=True,
                                my_hp=80, enemy_hp=80) == (True, True)
    # time_over without HP to compare cannot decide anything.
    assert resolve_round_result(False, False, time_over=True) == (False, False)


# --------------------------------------------------------------------------
# Self-play. league_env carries its own copy of step(), and has now silently
# kept the OLD semantics twice. FakeLeagueEnv exists so that cannot recur.
# --------------------------------------------------------------------------

def test_league_env_terminates_on_a_ko_and_names_the_winner():
    """Before the fix `ko = (my<=0 or enemy<=0) and not hp_sentinel` was False
    on the KO frame (the negative HP word sets the sentinel), so info["win"]
    was constant 0 -- and train_league.py fed that straight into
    pool_manager.record_outcome, recording every league match as a loss and
    turning the whole Elo ranking into noise."""
    env = FakeLeagueEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(120, KO_HP)])
    _, reward, terminated, truncated, info = env.step(np.zeros(10, dtype=int))
    assert terminated is True and truncated is False
    assert info["win"] == 1 and info["loss"] == 0 and info["draw"] is False
    assert info["reward_parts"]["terminal"] == pytest.approx(65.0)

    env = FakeLeagueEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(KO_HP, 120)])
    _, _, terminated, _, info = env.step(np.zeros(10, dtype=int))
    assert terminated is True and info["loss"] == 1 and info["win"] == 0
    assert info["reward_parts"]["terminal"] == pytest.approx(-50.0)


def test_league_env_does_not_score_the_round_transition_blank_as_a_draw():
    """The [0, 0] blank is league's single most common terminal. Under the old
    two-independent-ifs it paid +15; the first cut of this fix silently
    repriced it to -50 by making draw its own branch. It is neither: it is a
    round-transition frame that must not terminate at all."""
    env = FakeLeagueEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(0, 0)])
    _, reward, terminated, truncated, info = env.step(np.zeros(10, dtype=int))
    assert terminated is False and truncated is False
    assert reward == pytest.approx(0.0) and info["reward_parts"] == {}


def test_league_env_uses_raw_p1_p2_flags_not_the_perspective_flipped_ones():
    """_PerspectiveParser drives ONE shared env through _parse_payload twice,
    P1 then P2, so every perspective-flipped attribute (my_ko / enemy_ko /
    matches_won) ends the step holding P2's view while the reward is computed
    from P1's observation. Wiring league to my_ko/enemy_ko -- which is what the
    runbook's prescription said to do -- inverts every win and loss."""
    env = FakeLeagueEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(120, KO_HP)])
    _, _, _, _, info = env.step(np.zeros(10, dtype=int))
    # P2's HP went negative -> P1 (the learner) won.
    assert (env.p1_ko, env.p2_ko) == (False, True)
    # The perspective-flipped pair is P2's, i.e. the exact inverse...
    assert (env.my_ko, env.enemy_ko) == (True, False)
    # ...and the reported outcome follows the RAW pair.
    assert info["win"] == 1


def test_league_env_emits_the_same_terminal_info_contract_as_the_others():
    env = FakeLeagueEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(120, KO_HP)])
    _, _, _, _, info = env.step(np.zeros(10, dtype=int))
    for key in ("win", "loss", "draw", "double_ko", "timeout", "episode_steps",
                "matches_won_delta", "enemy_matches_won_delta", "time_over"):
        assert key in info, key


# --------------------------------------------------------------------------
# Cross-backend contract and the load-bearing HP invariant.
# --------------------------------------------------------------------------

def test_both_backends_emit_the_same_terminal_info_keys():
    """The rig is the side about to receive a blind, untestable Lua change; it
    is the LAST place that should be missing the audit trail for the exact
    signal being added. These keys existed only on retro_env in the first cut."""
    bizhawk = FakeBizHawkEnvV4([make_payload(176, 176, extended=True)])
    bizhawk.reset()
    bizhawk.queue([make_payload(120, KO_HP, extended=True)])
    _, _, _, _, bh_info = bizhawk.step(np.array([0, 0]))

    retro = _make_retro_env([_base_ram(p1_hp=120, p2_hp=-1)])
    _, _, _, _, rt_info = retro.step(np.array([0, 0]))

    shared = {"win", "loss", "draw", "double_ko", "timeout", "episode_steps",
              "matches_won_delta", "enemy_matches_won_delta", "time_over",
              "round_timer"}
    assert shared <= set(bh_info), shared - set(bh_info)
    assert shared <= set(rt_info), shared - set(rt_info)
    # round_timer is the one key whose VALUE is transport-specific: None on a
    # payload that cannot carry the clock (every width the rig sends today),
    # a BCD byte on retro. Its presence is the contract; its value is the
    # honest report of what this backend can see.
    for key in shared - {"round_timer"}:
        assert bh_info[key] == rt_info[key], key
    assert bh_info["round_timer"] is None
    assert rt_info["round_timer"] == 0x99


def test_a_deep_overkill_ko_is_a_death_not_an_unreadable_frame():
    """The docstrings used to claim the HP words "only ever hold 0..176 and
    -1", with "not a single reading in 177..65525". Measured over 160,000
    emulator frames here the negative set was {-1, -10}; two independent
    250,000+ frame re-measurements saw down to -27. Read unsigned, -27 is
    65509 -- INSIDE the interval the comment called empty. Nothing may key off
    -1/65535, and a deep overkill must still be a clean, attributed KO."""
    from fakes.fake_bizhawk import KO_HP_DEEP
    from envs.reward import hp_to_signed

    assert hp_to_signed(KO_HP_DEEP) == -27
    assert 177 <= KO_HP_DEEP <= 65525          # the "empty" interval, occupied
    assert KO_HP_DEEP > FakeBizHawkEnvV4.HP_SENTINEL_THRESHOLD

    env = FakeBizHawkEnvV4([make_payload(176, 176, extended=True)])
    env.reset()
    env.queue([make_payload(120, KO_HP_DEEP, extended=True)])
    _, reward, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True and info["win"] == 1
    assert info["reward_parts"]["terminal"] == pytest.approx(65.0)


def test_a_real_double_ko_is_scored_as_a_draw():
    """The runbook's measurement table said "KOs simultaneos reales: 0". They
    do happen: 96 frames with both HP words negative in a 160,000-frame random
    run here, and 1 in 150 episodes in an independent re-measurement. The
    payoff must be the draw branch, never win_bonus + loss_penalty."""
    env = FakeBizHawkEnvV4([make_payload(176, 176, extended=True)])
    env.reset()
    env.queue([make_payload(KO_HP, KO_HP_DEEP, extended=True)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["draw"] is True and info["win"] == 0 and info["loss"] == 0
    assert info["reward_parts"]["terminal"] == pytest.approx(-50.0)


def test_the_retro_integration_decodes_a_ko_and_exposes_the_round_signals():
    """Behavioural replacement for a change-detector that asserted data.json
    equals what had just been written into it. This drives the real numpy
    decode the libretro bridge performs, and pins the ADDRESSES against the
    integration stable-retro ships (which has always typed HP signed) rather
    than against our own file."""
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    ours = json.loads((root / "retro_integration"
                       / "StreetFighterIISpecialChampionEdition-Genesis-v0"
                       / "data.json").read_text())["info"]

    # The decode the >i2 typing actually performs on a KO'd fighter's word.
    raw = np.array([0xFF, 0xFF], dtype=np.uint8).tobytes()
    assert np.frombuffer(raw, dtype=ours["p1_hp"]["type"])[0] == -1
    raw = np.array([0xFF, 0xE5], dtype=np.uint8).tobytes()      # -27
    assert np.frombuffer(raw, dtype=ours["p1_hp"]["type"])[0] == -27
    # ...versus the >u2 typing it replaced, which is why the sentinel existed.
    assert np.frombuffer(raw, dtype=">u2")[0] == 65509

    # Addresses, cross-checked against the shipped stable-retro integration
    # rather than against ourselves.
    try:
        import stable_retro as _retro
    except ImportError:                                    # pragma: no cover
        pytest.skip("stable-retro not installed")
    shipped_path = (Path(_retro.data.path()) / "stable"
                    / "StreetFighterIISpecialChampionEdition-Genesis-v0"
                    / "data.json")
    if not shipped_path.exists():                          # pragma: no cover
        pytest.skip("shipped integration not available")
    shipped = json.loads(shipped_path.read_text())["info"]
    assert ours["p1_hp"]["address"] == shipped["health"]["address"]
    assert ours["p2_hp"]["address"] == shipped["enemy_health"]["address"]
    assert ours["p1_hp"]["type"] == shipped["health"]["type"] == ">i2"
    assert ours["matches_won"]["address"] == shipped["matches_won"]["address"]
    # The shipped file reads the enemy counter as a >u4 whose LAST byte is the
    # |u1 we read; that is the same counter, one byte wide.
    assert (ours["enemy_matches_won"]["address"]
            == shipped["enemy_matches_won"]["address"] + 3)
    # The round clock, measured on the live core: BCD 0x99 -> 0x00.
    assert ours["round_timer"] == {"address": 16750378, "type": "|u1"}


def test_the_clock_result_flips_with_the_p2_perspective():
    """The clock decides by comparing the two HP words, so it inherits the
    perspective flip. Getting this backwards inverts every league/P2 outcome,
    and the counter version of the test could not have caught it."""
    for raw_hp, expected in (((176, 94), "loss"), ((94, 176), "win")):
        env = FakeBizHawkEnvV4([make_payload(176, 176, clock=True,
                                             counters=True)], player=2)
        env.reset()
        env.queue([make_payload(*raw_hp, clock=True, counters=True,
                                round_timer=0)])
        _, _, terminated, _, info = env.step(np.array([0, 0]))
        assert terminated is True and info["time_over"] is True
        assert info[expected] == 1, (raw_hp, expected, info)
        assert info["win"] + info["loss"] == 1
