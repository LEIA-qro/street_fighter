# env_sf2_v3.py
import numpy as np
from gymnasium import spaces
from envs.sf2_v2 import StreetFighterEnvV2

# Correct Genesis SF2 6-button layout
# Bits: [Up, Down, Left, Right, A(LK), B(MK), C(HK), X(LP), Y(MP), Z(HP)]
DIRECTION_MAP = {
    0: [0, 0, 0, 0], 1: [1, 0, 0, 0], 2: [0, 1, 0, 0],
    3: [0, 0, 1, 0], 4: [0, 0, 0, 1], 5: [1, 0, 1, 0],
    6: [1, 0, 0, 1], 7: [0, 1, 1, 0], 8: [0, 1, 0, 1],
}
BUTTON_MAP = {
    0: [0, 0, 0, 0, 0, 0], 1: [1, 0, 0, 0, 0, 0], 2: [0, 1, 0, 0, 0, 0],
    3: [0, 0, 1, 0, 0, 0], 4: [0, 0, 0, 1, 0, 0], 5: [0, 0, 0, 0, 1, 0],
    6: [0, 0, 0, 0, 0, 1],
}


def discrete_to_binary(action: np.ndarray) -> str:
    dir_bits = DIRECTION_MAP[int(action[0])]
    btn_bits = BUTTON_MAP[int(action[1])]
    return "".join(str(b) for b in dir_bits + btn_bits)


class StreetFighterEnvV3(StreetFighterEnvV2):
    """V3: MultiDiscrete([9, 7]) action space.

    Reduces valid combinations from 1,024 to 63.
    Preserves 10-bit Lua protocol and all v2 observation semantics.
    Corrects button mapping for Ryu's full moveset (punches + kicks preserved).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override ONLY the action space — observation space unchanged
        self.action_space = spaces.MultiDiscrete([9, 7])

    def _action_to_string(self, action: np.ndarray) -> str:
        """Convert MultiDiscrete array to 10-bit binary string command using corrective mapping."""
        return discrete_to_binary(action)
