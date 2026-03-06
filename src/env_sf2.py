import os
import random
import numpy as np
from gymnasium import spaces
from collections import deque
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
        # --- THE NEW FRAME STACKING SPACE ---
        # We define a single frame's bounds as a standard Python list
        single_low = [0, 0, 0, 0, 0, 0, 0, 0, -1, -1]
        single_high = [176, 176, 500, 500, 200, 200, 255, 255, 500, 500]
        
        # Multiplying a Python list by NUM_FRAMES (4) duplicates it, making it length 40!
        low = np.array(single_low * config.NUM_FRAMES, dtype=np.int32)
        high = np.array(single_high * config.NUM_FRAMES, dtype=np.int32) 
        
        # SB3 now knows to expect a 40-dimensional array
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        # ------------------------------------
        
        self.prev_my_hp = 176
        self.prev_enemy_hp = 176

        self.frames = deque(maxlen=config.NUM_FRAMES)  

    def _get_obs(self):
        """Flattens the conveyor belt into a single 40-dimensional array."""
        return np.concatenate(self.frames)

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

        self.frames.append(observation)
        
        current_my_hp = observation[0]
        current_enemy_hp = observation[1]
        
        # 3. Calculate Reward
        # We define reward purely based on damage dealt to the opponent. To avoid the "Coward's Local Optimum".
        damage_dealt = max(0, self.prev_enemy_hp - current_enemy_hp)
        damage_taken = max(0, self.prev_my_hp - current_my_hp)


        # Clamp phantom damage spikes from memory glitches
        if damage_dealt > 70: damage_dealt = 0
        if damage_taken > 70: damage_taken = 0

        # reward = float(damage_dealt)
        # THE GRANDMASTER REWARD FUNCTION
        # +1.0x for Damage Dealt (Aggression)
        # -0.25x for Damage Taken (Self-Preservation / Blocking)
        # -0.01 for Time Passed  (Urgency / Anti-Turtle)
        # THE COMBO-ENGINE REWARD LOGIC
        if damage_dealt > 0:
            # Hit landed: Reward Damage + 1 Flat Bonus for Hit Count (combo incentive)
            reward = float(damage_dealt) + 1.0 - (0.25 * float(damage_taken))
        else:
            # Empty frame: Apply pain penalty and exact -0.01 bleed
            reward = -(0.25 * float(damage_taken)) - 0.01

        self.prev_my_hp = current_my_hp
        self.prev_enemy_hp = current_enemy_hp
        
        if self.trainable:
            terminated = bool(current_my_hp <= 0 or current_enemy_hp <= 0)
        else:
            terminated = False
        
        return self._get_obs(), reward, terminated, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Random Domain Selection
        # Phase selection
        chosen_state_file = random.choice(config.RYU_ONLY_STATES_PHASE_1)
        full_state_path = os.path.join(config.STATES_DIR, chosen_state_file)
        
        # Send Reset via Parent Method
        if self.reset_lua_path and self.trainable:
            self.send_command(f"RESET {full_state_path}\n")
        
        # Wait for resulting state
        data = self.receive_payload()
        observation = self._parse_payload(data)
        
        self.prev_my_hp = observation[0]
        self.prev_enemy_hp = observation[1]

        self.frames.clear()
        for _ in range(config.NUM_FRAMES):
            self.frames.append(observation)
        
        return self._get_obs(), {}
        
    def _parse_payload(self, data):
        """Converts raw string to strictly formatted numpy array."""
        parts = data.split(" ", 1)
        if len(parts) == 2:
            try:
                # Attempt to parse the 10 integers
                ram_values = [int(x) for x in parts[1].split(',')]
                ram_values[0] = 0 if ram_values[0] > 200 else ram_values[0]
                ram_values[1] = 0 if ram_values[1] > 200 else ram_values[1]
                raw_obs = np.array(ram_values, dtype=np.int32)
                
                # THE PERSPECTIVE FLIP
                if self.player == 2:
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
                    
            except ValueError:
                # If the string is corrupted in any way, gracefully ignore it and return zeros
                pass 
                
        return np.zeros(config.ACTION_DIM, dtype=np.int32)