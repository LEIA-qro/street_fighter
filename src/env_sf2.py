import os
import random
import numpy as np
from gymnasium import spaces
import config
from bizhawk_base import BizHawkBaseEnv

class StreetFighterEnv(BizHawkBaseEnv):
    """Street Fighter II RL Environment."""
    
    def __init__(self, rank=0, lua_path=config.TRAINING_ENV_CLIENT_LUA_PATH, trainable=True, player=1):
        # Dinamic port assignment
        assigned_port = config.PORT + rank

        # Initialize the parent TCP bridge with config variables
        super().__init__(
            bizhawk_path=config.BIZHAWK_PATH,
            rom_path=config.ROM_PATH,
            lua_path=lua_path,
            host=config.HOST,
            port=assigned_port,
            reset_lua_path=config.RESET_CONFIG_LUA_SCRIPT_PATH,
            trainable=trainable
        )
        
        self.player = player  # Track which player this environment controls (1 or 2)
        # Define Spaces
        self.action_space = spaces.MultiBinary(config.ACTION_DIM)
        
        # Observation space: [P1_HP, P2_HP, P1_X, P2_X, P1_Y, P2_Y, P1_Action_ID, P2_Action_ID, P1_Projectile_X, P2_Projectile_X]
        low = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, -1], dtype=np.int32)
        high = np.array([176, 176, 500, 500, 200, 200, 255, 255, 500, 500], dtype=np.int32) 
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        
        self.prev_my_hp = 176
        self.prev_enemy_hp = 176

    def step(self, action):
        # 1. Send Action via Parent Method
        action_string = "".join(str(int(b)) for b in action)

        if self.player == 1:
            full_command = action_string + "0000000000\n" # P1 acts, P2 does nothing
        else:
            full_command = "0000000000" + action_string + "\n" # P1 does nothing, P2 acts

        self.send_command(full_command)
        
        # 2. Receive State via Parent Method
        data = self.receive_payload()
        observation = self._parse_payload(data)
        
        current_my_hp = observation[0]
        current_enemy_hp = observation[1]
        
        # 3. Calculate Reward
        # We define reward purely based on damage dealt to the opponent. To avoid the "Coward's Local Optimum".
        damage_dealt = max(0, self.prev_enemy_hp - current_enemy_hp)


        # Clamp phantom damage spikes from memory glitches
        if damage_dealt > 70: damage_dealt = 0
        # if damage_taken > 70: damage_taken = 0

        # reward = float(damage_dealt - damage_taken)
        reward = float(damage_dealt)

        self.prev_my_hp = current_my_hp
        self.prev_enemy_hp = current_enemy_hp
        
        if self.trainable:
            terminated = bool(current_my_hp <= 0 or current_enemy_hp <= 0)
        else:
            terminated = False
        
        return observation, reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random Domain Selection
        chosen_state_file = random.choice(config.RYU_ONLY_STATES)
        full_state_path = os.path.join(config.STATES_DIR, chosen_state_file)
        
        # Send Reset via Parent Method
        if self.reset_lua_path and self.trainable:
            self.send_command(f"RESET {full_state_path}\n")
        
        # Wait for resulting state
        data = self.receive_payload()
        observation = self._parse_payload(data)
        
        self.prev_my_hp = observation[0]
        self.prev_enemy_hp = observation[1]
        
        return observation, {}
        
    def _parse_payload(self, data):
        """Converts raw string to strictly formatted numpy array."""
        parts = data.split(" ", 1)
        if len(parts) == 2:
            ram_values = [int(x) for x in parts[1].split(',')]
            ram_values[0] = 0 if ram_values[0] > 200 else ram_values[0]
            ram_values[1] = 0 if ram_values[1] > 200 else ram_values[1]
            raw_obs = np.array(ram_values, dtype=np.int32)
            
            # THE PERSPECTIVE FLIP
            if self.player == 2:
                # We swap all P1 indices with P2 indices so the agent always sees itself as index 0, 2, 4, etc.
                flipped_obs = np.array([
                    raw_obs[1], raw_obs[0], # HP (P2, P1)
                    raw_obs[3], raw_obs[2], # X
                    raw_obs[5], raw_obs[4], # Y
                    raw_obs[7], raw_obs[6], # Action ID
                    raw_obs[9], raw_obs[8]  # Projectile X
                ], dtype=np.int32)
                return flipped_obs
            else:
                return raw_obs
                
        return np.zeros(config.ACTION_DIM, dtype=np.int32)