import os
import random
import numpy as np
from gymnasium import spaces
import config
from bizhawk_base import BizHawkBaseEnv

class StreetFighterEnv(BizHawkBaseEnv):
    """Street Fighter II RL Environment."""
    
    def __init__(self):
        # Initialize the parent TCP bridge with config variables
        super().__init__(
            bizhawk_path=config.BIZHAWK_PATH,
            rom_path=config.ROM_PATH,
            lua_path=config.ENV_CLIENT_LUA_SCRIPT_PATH,
            host=config.HOST,
            port=config.PORT,
            reset_lua_path=config.RESET_CONFIG_LUA_SCRIPT_PATH
        )
        
        # Define Spaces
        self.action_space = spaces.MultiBinary(config.ACTION_DIM)
        
        # Observation space: [P1_HP, P2_HP, P1_X, P2_X, P1_Y, P2_Y, P1_Action_ID, P2_Action_ID, P1_Projectile_X, P2_Projectile_X]
        low = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, -1], dtype=np.int32)
        high = np.array([176, 176, 500, 500, 200, 200, 255, 255, 500, 500], dtype=np.int32) 
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        
        self.prev_p1_hp = 176
        self.prev_p2_hp = 176

    def step(self, action):
        # 1. Send Action via Parent Method
        action_string = "".join(str(int(b)) for b in action) + "\n"
        self.send_command(action_string)
        
        # 2. Receive State via Parent Method
        data = self.receive_payload()
        observation = self._parse_payload(data)
        
        current_p1_hp = observation[0]
        current_p2_hp = observation[1]
        
        # 3. Calculate Reward
        damage_dealt = max(0, self.prev_p2_hp - current_p2_hp)
        damage_taken = max(0, self.prev_p1_hp - current_p1_hp)
        reward = float(damage_dealt - damage_taken)
        
        self.prev_p1_hp = current_p1_hp
        self.prev_p2_hp = current_p2_hp
        
        terminated = bool(current_p1_hp <= 0 or current_p2_hp <= 0)
        
        return observation, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random Domain Selection
        chosen_state_file = random.choice(config.AVAILABLE_STATES)
        full_state_path = os.path.join(config.STATES_DIR, chosen_state_file)
        
        # Send Reset via Parent Method
        self.send_command(f"RESET {full_state_path}\n")
        
        # Wait for resulting state
        data = self.receive_payload()
        observation = self._parse_payload(data)
        
        self.prev_p1_hp = observation[0]
        self.prev_p2_hp = observation[1]
        
        return observation, {}
        
    def _parse_payload(self, data):
        """Converts raw string to strictly formatted numpy array."""
        parts = data.split(" ", 1)
        if len(parts) == 2:
            ram_values = [int(x) for x in parts[1].split(',')]
            ram_values[0] = 0 if ram_values[0] >= 65535 else ram_values[0]
            ram_values[1] = 0 if ram_values[1] >= 65535 else ram_values[1]
            return np.array(ram_values, dtype=np.int32)
        return np.zeros(config.ACTION_DIM, dtype=np.int32)