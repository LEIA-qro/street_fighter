# test_reset_protocol.py
#
# Guards the wire-protocol phase discipline in StreetFighterBaseEnv.reset():
# exactly one payload is always pending when reset() runs (Lua sends before it
# waits), so reset() must drain that stale payload, build its observation from
# the REAL post-savestate-load frame, and re-prime the one-message offset with
# a neutral command. Before this, reset() returned the previous episode's
# final frame (or the ROM boot screen) copied 4x into the frame stack.

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import numpy as np
import pytest

import core.config as config
from fakes.fake_bizhawk import FakeBizHawkEnv, make_payload


def test_reset_returns_post_load_frame_not_stale():
    # Queue order mirrors the wire: the boot payload is auto-prepended by the
    # fake (stale), then the post-load frame with a distinctive rel_dist.
    env = FakeBizHawkEnv([make_payload(176, 176, rel_dist=42)])
    obs, _ = env.reset()
    # Frame layout: rel_dist is index 9 of each 554-float frame.
    assert float(obs[9]) == 42.0


def test_reset_sends_reset_then_neutral_prime():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    assert env.sent[0].startswith("RESET ")
    assert env.sent[1] == "0" * (2 * config.ACTION_DIM) + "\n"


def test_second_reset_consumes_stale_then_fresh():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(176, 100)])   # step payload
    env.step(env.action_space.sample())
    # Second reset: the in-flight payload (previous episode's last frame)
    # plays the stale role, then the post-load frame follows.
    env.queue([make_payload(176, 90, rel_dist=150),   # stale: mid-fight frame
               make_payload(176, 176, rel_dist=55)])  # fresh: loaded state
    obs, _ = env.reset()
    assert float(obs[1]) == 176.0
    assert float(obs[9]) == 55.0


def test_frame_stack_primed_with_fresh_frame():
    env = FakeBizHawkEnv([make_payload(176, 176, rel_dist=42)])
    obs, _ = env.reset()
    frames = obs.reshape(config.NUM_FRAMES, -1)
    assert np.allclose(frames, frames[0])
    assert float(frames[0][9]) == 42.0


def test_episode_spacing_aggregates_reach_terminal_info():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([
        make_payload(176, 176, rel_dist=100),
        make_payload(176, 176, rel_dist=60),
        make_payload(176, 0, rel_dist=60),   # KO ends the episode
    ])
    for _ in range(2):
        _, _, terminated, _, info = env.step(env.action_space.sample())
        assert not terminated
    _, _, terminated, _, info = env.step(env.action_space.sample())
    assert terminated
    assert info["ep_rel_dist_mean"] == pytest.approx(np.mean([100, 60, 60]), rel=1e-6)
    assert info["ep_rel_dist_median"] == 60.0
    # One of three samples (rel_dist=100) is at or past the 80 boundary.
    assert info["ep_rel_dist_frac_far"] == pytest.approx(1 / 3, rel=1e-6)


def test_spacing_samples_reset_between_episodes():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([make_payload(176, 0, rel_dist=150)])  # instant KO at far range
    _, _, terminated, _, info = env.step(env.action_space.sample())
    assert terminated and info["ep_rel_dist_frac_far"] == 1.0

    env.queue([make_payload(176, 176), make_payload(176, 176)])  # stale + fresh
    env.reset()
    env.queue([make_payload(176, 0, rel_dist=10)])  # instant KO at close range
    _, _, terminated, _, info = env.step(env.action_space.sample())
    assert terminated and info["ep_rel_dist_frac_far"] == 0.0


def test_sentinel_frames_excluded_from_spacing_stats():
    env = FakeBizHawkEnv([make_payload(176, 176)])
    env.reset()
    env.queue([
        make_payload(255, 255, rel_dist=187),  # sentinel frame: excluded
        make_payload(176, 0, rel_dist=20),
    ])
    env.step(env.action_space.sample())
    _, _, terminated, _, info = env.step(env.action_space.sample())
    assert terminated
    assert info["ep_rel_dist_mean"] == 20.0
    assert info["ep_rel_dist_frac_far"] == 0.0
