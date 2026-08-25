# macro_wrapper.py
#
# Flattens the MultiDiscrete([9, 7]) action space into Discrete(N_ACTIONS) and
# executes macro actions as multi-step options against the wrapped environment.
#
# Reward over a macro is the undiscounted sum of its inner steps. This is the
# standard semi-MDP treatment (Sutton, Precup & Singh 1999); with FRAME_SKIP=4
# and macros of length <= 3, the intra-option discounting error at gamma=0.995
# is under 1.5% and is not worth the extra bookkeeping.

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from envs.action_macros import N_ACTIONS, decode


class MacroActionWrapper(gym.Wrapper):
    """Discrete(63 + n_macros) action space with temporally-extended macros.

    Args:
        env: an environment whose action_space is MultiDiscrete([9, 7]).
        obs_rel_x_index: index of rel_x within a single observation frame.
            rel_x = p2_x - p1_x, so rel_x >= 0 means the opponent is to the
            agent's right and macros are used unmirrored.
        frame_size: length of a single observation frame. 554 for v2/v3,
            14 for v4 (see src/envs/sf2_v4.py).
    """

    def __init__(self, env, obs_rel_x_index: int = 2, frame_size: int = 554):
        super().__init__(env)
        self.action_space = spaces.Discrete(N_ACTIONS)
        self.obs_rel_x_index = obs_rel_x_index
        self.frame_size = frame_size
        self._facing_right = True

    def _update_facing(self, obs: np.ndarray) -> None:
        """Reads rel_x out of the most recent frame of the stacked observation."""
        n_frames = max(1, len(obs) // self.frame_size)
        latest = (n_frames - 1) * self.frame_size
        self._facing_right = float(obs[latest + self.obs_rel_x_index]) >= 0.0

    def reset(self, **kwargs):
        # Macros are exact input sequences; the base env's sticky-direction hack
        # force-holds L/R for two extra agent steps and would overwrite them.
        if hasattr(self.env.unwrapped, "sticky_enabled"):
            self.env.unwrapped.sticky_enabled = False
        elif hasattr(self.env, "sticky_enabled"):
            self.env.sticky_enabled = False

        obs, info = self.env.reset(**kwargs)
        self._update_facing(obs)
        return obs, info

    def step(self, action):
        steps = decode(int(action), self._facing_right)

        total_reward = 0.0
        obs, terminated, truncated, info = None, False, False, {}
        executed = 0

        for direction, button in steps:
            obs, reward, terminated, truncated, info = self.env.step(
                np.array([direction, button], dtype=np.int64)
            )
            total_reward += float(reward)
            executed += 1
            if terminated or truncated:
                break

        info = dict(info)
        info["macro_steps"] = executed
        info["macro_action"] = int(action)

        self._update_facing(obs)
        return obs, total_reward, terminated, truncated, info
