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

from fakes.fake_bizhawk import FakeBizHawkEnv, FakeBizHawkEnvV4, make_payload
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
    env.queue([make_payload(0, 120)])
    _, reward, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["loss"] == 1
    assert info["win"] == 0


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
        make_payload(176, 0, extended=True, p1_air=0),     # terminal KO, ground
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
               make_payload(176, 0, extended=True, p1_air=1)])
    env.step(np.array([0, 0]))
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["ep_air_frac"] == pytest.approx(1.0)

    # Second reset consumes TWO payloads (stale in-flight + post-load).
    env.queue([make_payload(176, 176, extended=True),
               make_payload(176, 176, extended=True)])
    env.reset()
    env.queue([make_payload(176, 0, extended=True, p1_air=0)])
    _, _, terminated, _, info = env.step(np.array([0, 0]))
    assert terminated is True
    assert info["ep_air_frac"] == pytest.approx(0.0)
