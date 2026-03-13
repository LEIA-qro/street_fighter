# env_sf2_v2.py
import os
import random
import numpy as np
from gymnasium import spaces
from collections import deque
import config
from bizhawk_base import BizHawkBaseEnv

CONTINUOUS_DIM = 10  # HP(2), X(2), Y(2), Proj_X(2), Vel_X(2) ← was 8
ACT_CATEGORIES = 256
CHAR_CATEGORIES = 16
ONE_HOT_ACT_DIM = ACT_CATEGORIES * 2
ONE_HOT_CHAR_DIM = CHAR_CATEGORIES * 2
TOTAL_OBS_DIM = CONTINUOUS_DIM + ONE_HOT_ACT_DIM + ONE_HOT_CHAR_DIM  # 10+512+32 = 554

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
        # Velocity range: max observable delta per 4-frame skip is ~60px
        cont_low  = [0., 0., 0., 0., 0., 0., -1., -1., -60., -60.]
        cont_high = [176., 176., 500., 500., 200., 200., 500., 500., 60., 60.]
        
        # One-Hot bounds: 552 zeros and ones
        act_low = [0.] * ONE_HOT_ACT_DIM
        act_high = [1.] * ONE_HOT_ACT_DIM
        char_low = [0.] * ONE_HOT_CHAR_DIM
        char_high = [1.] * ONE_HOT_CHAR_DIM
        
        single_frame_low = cont_low + act_low + char_low
        single_frame_high = cont_high + act_high + char_high

        # Change dtype throughout
        low  = np.array(single_frame_low  * config.NUM_FRAMES, dtype=np.float32)  # was int32
        high = np.array(single_frame_high * config.NUM_FRAMES, dtype=np.float32)  # was int32
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32) # was int32
    
        # ------------------------------------
        
        self.prev_my_hp    = 176
        self.prev_enemy_hp = 176
        self.prev_p1_x     = 0
        self.prev_p2_x     = 0
        self.frames        = deque(maxlen=config.NUM_FRAMES)

    def _get_obs(self): return np.concatenate(self.frames)

    def _one_hot(self, val, num_classes):
        """Universal One-Hot Encoder"""
        arr = np.zeros(num_classes, dtype=np.float32)  # float32 to match obs dtype
        safe_val = max(0, min(int(val), num_classes - 1))
        arr[safe_val] = 1.0
        return arr

    def step(self, action):
        # 1. Send Action via Parent Method
        action_string = "".join(str(int(b)) for b in action)

        full_command = (action_string + "0000000000\n") if self.player == 1 else ("0000000000" + action_string + "\n")
        self.send_command(full_command)
        
        # 2. Receive State via Parent Method
        data = self.receive_payload()
        observation = self._parse_payload(data, is_reset=False)

        self.frames.append(observation)
        
        current_my_hp, current_enemy_hp = observation[0], observation[1]
        
        # 3. Calculate Reward
        # We define reward purely based on damage dealt to the opponent. To avoid the "Coward's Local Optimum".
        damage_dealt = max(0, self.prev_enemy_hp - current_enemy_hp)
        damage_taken = max(0, self.prev_my_hp - current_my_hp)

        damage_clamp = 100

        # Clamp phantom damage spikes from memory glitches
        if damage_dealt > damage_clamp: damage_dealt = 0
        if damage_taken > damage_clamp: damage_taken = 0

        # Footsie Spacing Reward (Distance is typically observation[2] and [3])
        # P1_X is usually at index 2, P2_X at index 3 in your cont_obs
        rel_dist = abs(observation[2] - observation[3])
        dist_reward = 0.05 if 70 <= rel_dist <= 150 else 0.0

        # reward = float(damage_dealt)
        # THE GRANDMASTER REWARD FUNCTION
        # +1.0x for Damage Dealt (Aggression)
        # -0.25x for Damage Taken (Self-Preservation / Blocking)
        # -0.01 for Time Passed  (Urgency / Anti-Turtle)
        # THE COMBO-ENGINE REWARD LOGIC
        if damage_dealt > 0:
            # Hit landed: Reward Damage + 1.5 Flat Bonus for Hit Count (combo incentive)
            reward = float(damage_dealt) + 1.5 - (0.35 * float(damage_taken)) + dist_reward
            
        else:
            # Empty frame: Apply pain penalty and exact -0.01 bleed
            reward = -(0.35 * float(damage_taken)) - 0.02 + dist_reward

        if current_enemy_hp <= 0: reward += 50.0
        elif current_my_hp <= 0: reward -= 50.0

        self.prev_my_hp, self.prev_enemy_hp = current_my_hp, current_enemy_hp
        
        terminated = bool(current_my_hp <= 0 or current_enemy_hp <= 0) if self.trainable else False
        
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
        observation = self._parse_payload(data, is_reset=True)
        
        self.prev_my_hp, self.prev_enemy_hp = observation[0], observation[1]

        self.frames.clear()
        for _ in range(config.NUM_FRAMES): self.frames.append(observation)
        
        return self._get_obs(), {}

    def _parse_payload(self, data, is_reset=False):
        """
        Builds a 554-dimensional float32 observation.
 
        Layout per frame:
          [0-9]    Continuous: HP(2), X(2), Y(2), ProjX(2), VelX(2)
          [10-265] P1 action one-hot (256)
          [266-521] P2 action one-hot (256)
          [522-537] P1 char one-hot (16)
          [538-553] P2 char one-hot (16)
        """
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
                    p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y = raw[1], raw[0], raw[3], raw[2], raw[5], raw[4]
                    p1_act, p2_act, p1_proj, p2_proj, p1_char, p2_char = raw[7], raw[6], raw[9], raw[8], raw[11], raw[10]
                else:
                    p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y = raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]
                    p1_act, p2_act, p1_proj, p2_proj, p1_char, p2_char = raw[6], raw[7], raw[8], raw[9], raw[10], raw[11]

                p1_vel_x = 0 if is_reset else int(np.clip(p1_x - self.prev_p1_x, -60, 60))
                p2_vel_x = 0 if is_reset else int(np.clip(p2_x - self.prev_p2_x, -60, 60))
                self.prev_p1_x, self.prev_p2_x = p1_x, p2_x
                # 1. Continuous Features
                cont_obs = np.array([p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y, p1_proj, p2_proj, p1_vel_x, p2_vel_x], dtype=np.float32)
                
                # 2. Categorical Features (One-Hot Encoded)
                # One-hot dims can stay int32 in the raw array, but the concat must be float32
                p1_act_oh  = self._one_hot(p1_act,  ACT_CATEGORIES)
                p2_act_oh  = self._one_hot(p2_act,  ACT_CATEGORIES)
                p1_char_oh = self._one_hot(p1_char, CHAR_CATEGORIES)
                p2_char_oh = self._one_hot(p2_char, CHAR_CATEGORIES)
                
                # 3. Smash them all together
                return np.concatenate((cont_obs, p1_act_oh, p2_act_oh, p1_char_oh, p2_char_oh))

            except ValueError: pass

        # Failsafe: Return 554 zeros if the string is corrupted
        return np.zeros(TOTAL_OBS_DIM, dtype=np.float32)