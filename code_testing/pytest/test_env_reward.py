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
