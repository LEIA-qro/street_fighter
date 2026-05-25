import numpy as np
from gymnasium import spaces

import core.config as config
from envs.base_env import StreetFighterBaseEnv

class StreetFighterEnv(StreetFighterBaseEnv):
    """Street Fighter II RL Environment V1.
    Lightweight version producing a 40-dimensional float32 observation.
    """
    
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
        
        # Observation space: 10 continuous dims per frame * NUM_FRAMES
        # [P1_HP, P2_HP, RelX, RelY, P1_WallDist, P1_ProjX, P2_ProjX, P1_VelX, P2_VelX, RelDist]
        cont_low  = [0., 0., -500., -200., 0., -1., -1., -100., -100., 0.]
        cont_high = [176., 176., 500., 200., 250., 500., 500., 100., 100., 187.]
        
        low = np.array(cont_low * config.NUM_FRAMES, dtype=np.float32)
        high = np.array(cont_high * config.NUM_FRAMES, dtype=np.float32) 
        
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        
    def _action_to_string(self, action) -> str:
        """Convert MultiBinary array to 10-bit binary string command."""
        return "".join(str(int(b)) for b in action)
        
    def _parse_payload(self, data, is_reset=False):
        """Builds a lightweight 10-dimensional float32 observation per frame.
        Re-uses base_env logic but slices off the one-hot arrays.
        """
        obs = super()._parse_payload(data, is_reset)
        # obs is length 554 (10 continuous + 544 one-hot)
        return obs[:10]