# env_sf2_v2.py
import numpy as np
from gymnasium import spaces

import core.config as config
from envs.base_env import StreetFighterBaseEnv, TOTAL_OBS_DIM, ONE_HOT_ACT_DIM, ONE_HOT_CHAR_DIM


class StreetFighterEnvV2(StreetFighterBaseEnv):
    """Street Fighter II RL Environment with One-Hot Encoded Action IDs."""

    def __init__(self, rank=0, lua_path=config.TRAINING_ENV_CLIENT_LUA_PATH, trainable=True, debug_mode=True, player=1, verbose=True):
        super().__init__(
            rank=rank,
            lua_path=lua_path,
            trainable=trainable,
            debug_mode=debug_mode,
            player=player,
            verbose=verbose
        )

        self.action_space = spaces.MultiBinary(config.ACTION_DIM)

        # --- THE NEW HYBRID SPACE ---
        # Continuous bounds: P1_HP, P2_HP, RelX, RelY, P1_WallDist, P1_ProjX, P2_ProjX, P1_VelX, P2_VelX, RelDist
        cont_low  = [0., 0., -500., -200., 0., -1., -1., -100., -100., 0.]
        cont_high = [176., 176., 500., 200., 250., 500., 500., 100., 100., 187.]

        # One-Hot bounds: 544 zeros and ones
        act_low = [0.] * ONE_HOT_ACT_DIM
        act_high = [1.] * ONE_HOT_ACT_DIM
        char_low = [0.] * ONE_HOT_CHAR_DIM
        char_high = [1.] * ONE_HOT_CHAR_DIM

        single_frame_low = cont_low + act_low + char_low
        single_frame_high = cont_high + act_high + char_high

        # Change dtype throughout
        low  = np.array(single_frame_low  * config.NUM_FRAMES, dtype=np.float32)
        high = np.array(single_frame_high * config.NUM_FRAMES, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def _action_to_string(self, action) -> str:
        """Convert MultiBinary array to 10-bit binary string command."""
        return "".join(str(int(b)) for b in action)