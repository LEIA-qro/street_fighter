import os
import random
import numpy as np
from gymnasium import spaces
from collections import deque
import config
from bizhawk_base import BizHawkBaseEnv

# The new dimensions per frame
CONTINUOUS_DIM = 8  # HP(2), X(2), Y(2), Proj_X(2)
ACT_CATEGORIES = 256    # 0 to 255 possible Action IDs
CHAR_CATEGORIES = 16 # 16 possible characters (for future conditioning)
ONE_HOT_ACT_DIM = ACT_CATEGORIES * 2    # P1 + P2 Action Arrays
ONE_HOT_CHAR_DIM = CHAR_CATEGORIES * 2  # P1 + P2 Char Arrays
# 8 + 512 + 32 = 552 features per frame
TOTAL_OBS_DIM = CONTINUOUS_DIM + ONE_HOT_ACT_DIM + ONE_HOT_CHAR_DIM

class StreetFighterEnvV2(BizHawkBaseEnv):
    """Street Fighter II RL Environment with One-Hot Encoded Action IDs."""
    
    def __init__(self, rank=0, lua_path=config.TRAINING_ENV_CLIENT_LUA_PATH, trainable=True, player=1):
        assigned_port = config.PORT + rank

        super().__init__(
            bizhawk_path=config.BIZHAWK_PATH,
            rom_path=config.ROM_PATH,
            lua_path=lua_path,
            host=config.HOST,
            port=assigned_port,
            reset_lua_path=config.RESET_CONFIG_LUA_SCRIPT_PATH,
            trainable=trainable
        )
        
        self.player = player  
        self.action_space = spaces.MultiBinary(config.ACTION_DIM)
        
        # --- THE NEW HYBRID SPACE ---
        # Continuous bounds: P1_HP, P2_HP, P1_X, P2_X, P1_Y, P2_Y, P1_ProjX, P2_ProjX
        cont_low = [0, 0, 0, 0, 0, 0, -1, -1]
        cont_high = [176, 176, 500, 500, 200, 200, 500, 500]
        
        # One-Hot bounds: 552 zeros and ones
        act_low = [0] * ONE_HOT_ACT_DIM
        act_high = [1] * ONE_HOT_ACT_DIM
        
        char_low = [0] * ONE_HOT_CHAR_DIM
        char_high = [1] * ONE_HOT_CHAR_DIM
        
        single_frame_low = cont_low + act_low + char_low
        single_frame_high = cont_high + act_high + char_high
        
        low = np.array(single_frame_low * config.NUM_FRAMES, dtype=np.int32)
        high = np.array(single_frame_high * config.NUM_FRAMES, dtype=np.int32) 
        
        # The new space is exactly 2080 dimensions (520 * 4 frames)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)
        # ------------------------------------
        
        self.prev_my_hp = 176
        self.prev_enemy_hp = 176
        self.frames = deque(maxlen=config.NUM_FRAMES) 

    def _get_obs(self):
        return np.concatenate(self.frames)

    def _one_hot(self, val, num_classes):
        """Universal One-Hot Encoder"""
        arr = np.zeros(num_classes, dtype=np.int32)
        safe_val = max(0, min(int(val), num_classes - 1))
        arr[safe_val] = 1
        return arr

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
        chosen_state_file = random.choice(config.TRAINING_STATES)
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
        """Builds the 552-dimensional hybrid array."""
        # Grab strictly the CSV string, stripping off the leading zero
        csv_string = data.strip().split(" ")[-1]
        parts = csv_string.split(",")
        
        if len(parts) == 12:
            try:
                raw = [int(x) for x in parts]
                
                raw[0] = 0 if raw[0] > 200 else raw[0]
                raw[1] = 0 if raw[1] > 200 else raw[1]
                
                # PERSPECTIVE FLIP
                if self.player == 2:
                    p1_hp, p2_hp = raw[1], raw[0]
                    p1_x, p2_x   = raw[3], raw[2]
                    p1_y, p2_y   = raw[5], raw[4]
                    p1_act, p2_act = raw[7], raw[6]
                    p1_proj, p2_proj = raw[9], raw[8]
                    p1_char, p2_char = raw[11], raw[10]
                else:
                    p1_hp, p2_hp = raw[0], raw[1]
                    p1_x, p2_x   = raw[2], raw[3]
                    p1_y, p2_y   = raw[4], raw[5]
                    p1_act, p2_act = raw[6], raw[7]
                    p1_proj, p2_proj = raw[8], raw[9]
                    p1_char, p2_char = raw[10], raw[11]

                # 1. Continuous Features
                cont_obs = np.array([p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, p1_proj, p2_proj], dtype=np.int32)
                
                # 2. Categorical Features (One-Hot Encoded)
                p1_act_oh = self._one_hot(p1_act, ACT_CATEGORIES)
                p2_act_oh = self._one_hot(p2_act, ACT_CATEGORIES)
                p1_char_oh = self._one_hot(p1_char, CHAR_CATEGORIES)
                p2_char_oh = self._one_hot(p2_char, CHAR_CATEGORIES)
                
                # 3. Smash them all together
                return np.concatenate((cont_obs, p1_act_oh, p2_act_oh, p1_char_oh, p2_char_oh))

            except ValueError:
                pass
                
        # Failsafe: Return 520 zeros if the string is corrupted
        return np.zeros(TOTAL_OBS_DIM, dtype=np.int32)