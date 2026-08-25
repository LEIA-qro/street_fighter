# action_wrappers.py
#
# Canonical MultiBinary / MultiDiscrete -> Discrete flattening wrapper.
# Previously duplicated verbatim in agents/dqn/agent.py and
# agents/dqn/optuna_study.py, where the two copies could drift apart.

import numpy as np
from gymnasium import ActionWrapper, spaces


class FlattenDiscreteActionWrapper(ActionWrapper):
    """Presents a flat Discrete action space to value-based algorithms.

    * MultiBinary(n)     -> Discrete(2**n),        decoded as a binary string
    * MultiDiscrete(nvec) -> Discrete(prod(nvec)), decoded by successive divmod
    """

    def __init__(self, env):
        super().__init__(env)
        raw = env.action_space
        if isinstance(raw, spaces.MultiBinary):
            self._mode = "multibinary"
            self._n_buttons = int(raw.n)
            self.action_space = spaces.Discrete(2 ** self._n_buttons)
        elif isinstance(raw, spaces.MultiDiscrete):
            self._mode = "multidiscrete"
            self._nvec = raw.nvec.copy()
            self.action_space = spaces.Discrete(int(np.prod(self._nvec)))
        else:
            raise TypeError(
                f"FlattenDiscreteActionWrapper: unsupported action space "
                f"{type(raw).__name__}. Expected MultiBinary or MultiDiscrete."
            )

    def action(self, action):
        if self._mode == "multibinary":
            bits = format(int(action), f"0{self._n_buttons}b")
            return np.array([int(b) for b in bits], dtype=np.int8)

        decoded = []
        remaining = int(action)
        for n in reversed(self._nvec):
            decoded.append(remaining % int(n))
            remaining //= int(n)
        return np.array(list(reversed(decoded)), dtype=np.int64)
