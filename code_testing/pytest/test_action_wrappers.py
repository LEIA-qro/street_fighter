# test_action_wrappers.py

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

from agents.common.action_wrappers import FlattenDiscreteActionWrapper


class _Stub(gym.Env):
    def __init__(self, space):
        self.action_space = space
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,),
                                            dtype=np.float32)
        self.received = None

    def reset(self, seed=None, options=None):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        self.received = action
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}


def test_multidiscrete_is_flattened_to_the_product():
    env = FlattenDiscreteActionWrapper(_Stub(spaces.MultiDiscrete([9, 7])))
    assert env.action_space == spaces.Discrete(63)


def test_multibinary_is_flattened_to_two_to_the_n():
    env = FlattenDiscreteActionWrapper(_Stub(spaces.MultiBinary(10)))
    assert env.action_space == spaces.Discrete(1024)


def test_multidiscrete_decode_is_a_bijection():
    inner = _Stub(spaces.MultiDiscrete([9, 7]))
    env = FlattenDiscreteActionWrapper(inner)
    env.reset()
    seen = set()
    for a in range(63):
        env.step(a)
        seen.add(tuple(int(x) for x in inner.received))
    assert len(seen) == 63


def test_multibinary_decode_round_trips_the_bit_string():
    inner = _Stub(spaces.MultiBinary(10))
    env = FlattenDiscreteActionWrapper(inner)
    env.reset()
    env.step(0b1010101010)
    assert list(inner.received) == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]


def test_unsupported_space_is_rejected():
    with pytest.raises(TypeError):
        FlattenDiscreteActionWrapper(
            _Stub(spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32))
        )


def test_sac_refuses_to_train_on_a_discrete_action_space():
    from agents.sac.agent import SACAgent

    with pytest.raises(NotImplementedError, match="SAC-Discrete"):
        SACAgent().train(env_fn=None, save_dir=None, steps=0)


def test_sac_refuses_to_tune_on_a_discrete_action_space():
    from agents.sac.agent import SACAgent

    # env_fn=None: if the guard were placed after any use of env_fn (e.g. to
    # spin up a SubprocVecEnv for the Optuna study), this would blow up with
    # a TypeError/AttributeError instead of the expected NotImplementedError.
    with pytest.raises(NotImplementedError, match="SAC-Discrete"):
        SACAgent().tune(env_fn=None, n_trials=0)
