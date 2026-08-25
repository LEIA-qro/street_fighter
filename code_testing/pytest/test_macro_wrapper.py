# test_macro_wrapper.py

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pytest

from envs.action_macros import N_PRIMITIVES, N_ACTIONS, MACRO_NAMES
from envs.macro_wrapper import MacroActionWrapper

FRAME = 554
STACK = 4


class StubEnv(gym.Env):
    """Records every (direction, button) it receives and returns a fixed obs."""

    def __init__(self, rel_x=100.0, terminate_after=None):
        self.action_space = spaces.MultiDiscrete([9, 7])
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(FRAME * STACK,), dtype=np.float32
        )
        self.received = []
        self.rel_x = rel_x
        self.terminate_after = terminate_after
        self.sticky_enabled = True
        self.n = 0

    def _obs(self):
        obs = np.zeros(FRAME * STACK, dtype=np.float32)
        for f in range(STACK):
            obs[f * FRAME + 2] = self.rel_x   # rel_x lives at index 2 of each frame
        return obs

    def reset(self, seed=None, options=None):
        self.received.clear()
        self.n = 0
        return self._obs(), {}

    def step(self, action):
        self.received.append((int(action[0]), int(action[1])))
        self.n += 1
        terminated = self.terminate_after is not None and self.n >= self.terminate_after
        return self._obs(), 1.0, terminated, False, {"n": self.n}


def test_wrapper_exposes_a_flat_discrete_action_space():
    env = MacroActionWrapper(StubEnv())
    assert env.action_space == spaces.Discrete(N_ACTIONS)


def test_wrapper_disables_sticky_direction_on_reset():
    """The sticky hack force-holds L/R for two extra agent steps, which would
    overwrite the middle of every motion input."""
    inner = StubEnv()
    env = MacroActionWrapper(inner)
    env.reset()
    assert inner.sticky_enabled is False


def test_primitive_action_consumes_exactly_one_inner_step():
    inner = StubEnv()
    env = MacroActionWrapper(inner)
    env.reset()
    env.step(4 * 7 + 6)  # direction 4 (Right), button 6 (HP)
    assert inner.received == [(4, 6)]


def test_macro_action_consumes_all_of_its_inner_steps():
    inner = StubEnv()
    env = MacroActionWrapper(inner)
    env.reset()
    idx = N_PRIMITIVES + MACRO_NAMES.index("hadouken_lp")
    env.step(idx)
    assert inner.received == [(2, 0), (8, 0), (4, 4)]


def test_macro_reward_is_the_sum_over_its_inner_steps():
    inner = StubEnv()
    env = MacroActionWrapper(inner)
    env.reset()
    idx = N_PRIMITIVES + MACRO_NAMES.index("hadouken_lp")
    _, reward, _, _, _ = env.step(idx)
    assert reward == pytest.approx(3.0)  # 3 inner steps at 1.0 each


def test_macro_is_mirrored_when_the_opponent_is_on_the_left():
    inner = StubEnv(rel_x=-100.0)   # opponent to the agent's left
    env = MacroActionWrapper(inner)
    env.reset()
    idx = N_PRIMITIVES + MACRO_NAMES.index("hadouken_lp")
    env.step(idx)
    assert inner.received == [(2, 0), (7, 0), (3, 4)]


def test_macro_aborts_early_on_termination():
    inner = StubEnv(terminate_after=2)
    env = MacroActionWrapper(inner)
    env.reset()
    idx = N_PRIMITIVES + MACRO_NAMES.index("hadouken_lp")
    _, reward, terminated, _, info = env.step(idx)
    assert terminated is True
    assert len(inner.received) == 2       # third step never issued
    assert reward == pytest.approx(2.0)
    assert info["macro_steps"] == 2
