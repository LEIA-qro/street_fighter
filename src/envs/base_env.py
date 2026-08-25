# base_env.py
import os
import random
import time
import numpy as np
from gymnasium import spaces
from collections import deque

import core.config as config
from core.bizhawk_base import BizHawkBaseEnv

CONTINUOUS_DIM = config.OBS_DIM  # HP(2), RelX(1), RelY(1), WallDist(1), Proj_X(2), Vel_X(2), RelDist(1) = 10
ACT_CATEGORIES = 256
CHAR_CATEGORIES = 16
ONE_HOT_ACT_DIM = ACT_CATEGORIES * 2
ONE_HOT_CHAR_DIM = CHAR_CATEGORIES * 2
TOTAL_OBS_DIM = CONTINUOUS_DIM + ONE_HOT_ACT_DIM + ONE_HOT_CHAR_DIM  # 10+512+32 = 554


class StreetFighterBaseEnv(BizHawkBaseEnv):
    """Abstract Base Gym Environment for Street Fighter II.

    Handles connection, telemetry bridge, frame stacking, state parsing,
    reward calculation, and socket failure auto-recovery.
    Concrete subclasses override the action spaces and parsing helpers.
    """

    # HP reads above this are round-transition sentinels (0xFF etc.), not health.
    HP_SENTINEL_THRESHOLD = 200

    def __init__(self, rank=0, lua_path=config.TRAINING_ENV_CLIENT_LUA_PATH, trainable=True, debug_mode=True, player=1, verbose=True):
        assigned_port = config.PORT + rank

        super().__init__(
            bizhawk_path=config.BIZHAWK_PATH,
            rom_path=config.ROM_PATH,
            lua_path=lua_path,
            host=config.HOST,
            port=assigned_port,
            trainable=trainable,
            debug_mode=debug_mode,
            verbose=verbose
        )

        self.player = player
        self.active_training_states = config.TRAINING_STATES

        self.prev_my_hp    = 176
        self.prev_enemy_hp = 176
        self.prev_p1_x     = 0
        self.prev_p2_x     = 0
        self.frames        = deque(maxlen=config.NUM_FRAMES)

        # Counter systems
        self._steps = 0
        self.footsie_steps = 0
        self.prev_rel_dist = 80.0
        self.combo_counter = 0
        self.frames_since_last_hit = 0
        self.corrupt_payload_count = 0
        self.sticky_direction = None
        self.sticky_counter = 0
        self.hp_sentinel = False

    def set_training_states(self, new_states):
        """Receives broadcast from the Main Process and updates local memory."""
        self.active_training_states = new_states

    def _get_obs(self):
        return np.concatenate(self.frames)

    def _one_hot(self, val, num_classes):
        """Universal One-Hot Encoder"""
        arr = np.zeros(num_classes, dtype=np.float32)  # float32 to match obs dtype
        safe_val = max(0, min(int(val), num_classes - 1))
        arr[safe_val] = 1.0
        return arr

    def _action_to_string(self, action) -> str:
        """Converts step action object to standard 10-bit binary command string."""
        raise NotImplementedError("Subclasses must implement _action_to_string")

    def step(self, action):
        try:
            # 1. Convert action and Send Action via Parent Method
            action_string = self._action_to_string(action)

            # --- STICKY MOVEMENT LOGIC ---
            action_list = list(action_string)
            
            agent_left = action_list[2] == '1'
            agent_right = action_list[3] == '1'
            agent_crouch = action_list[1] == '1'
            
            # Cancel stickiness if the agent is crouching or inputs the opposite direction
            opposite_input = (self.sticky_direction == 'L' and agent_right) or (self.sticky_direction == 'R' and agent_left)
            
            if agent_crouch or opposite_input:
                self.sticky_counter = 0
                self.sticky_direction = None
                
            if self.sticky_counter > 0:
                if self.sticky_direction == 'L':
                    action_list[2] = '1'
                    action_list[3] = '0'  # Prevent conflicting Left+Right inputs
                elif self.sticky_direction == 'R':
                    action_list[3] = '1'
                    action_list[2] = '0'  # Prevent conflicting Left+Right inputs
                self.sticky_counter -= 1
            else:
                # Initiate stickiness on fresh directional inputs, provided we are not crouching
                if not agent_crouch:
                    if agent_left:
                        self.sticky_direction = 'L'
                        self.sticky_counter = 2
                    elif agent_right:
                        self.sticky_direction = 'R'
                        self.sticky_counter = 2
                        
            action_string = "".join(action_list)
            # -----------------------------

            full_command = (action_string + "0000000000\n") if self.player == 1 else ("0000000000" + action_string + "\n")
            self.send_command(full_command)
            # 2. Receive State via Parent Method
            data = self.receive_payload()

            self.debug_print(
                f"Command Sent: '{full_command}' | Raw Payload: '{data}'"
            )

        except RuntimeError as e:
            # Socket is dead. Return a terminal state so SB3 calls reset().
            # Do NOT let this propagate - it kills the SubprocVecEnv
            print(f"[Rank {self.port - config.PORT}] Socket error in step: {e}. Returning terminal obs.")
            obs = self._get_obs() if len(self.frames) > 0 else np.zeros(TOTAL_OBS_DIM * config.NUM_FRAMES, dtype=np.float32)
            return obs, 0.0, True, False, {"socket_death": True}

        observation = self._parse_payload(data, is_reset=False)
        self.frames.append(observation)

        # =====================================================
        # 3. Calculate Reward
        current_my_hp, current_enemy_hp = observation[0], observation[1]

        damage_clamp = 100
        damage_dealt = min(max(0, self.prev_enemy_hp - current_enemy_hp), damage_clamp)
        damage_taken = min(max(0, self.prev_my_hp    - current_my_hp),    damage_clamp)

        self._steps += 1

        COMBO_WINDOW = 6  # steps — a hit within 6 steps of the last extends the combo
        DAMAGE_TAKEN_PENALTY = 0.77
        FOOTSIE_RANGE_MAX    = 80     # 0x834C value — within effective fighting range
        FOOTSIE_BASE_REWARD  = 0.05

        # Read the engine-native distance from observation (index 9)
        rel_dist = float(observation[9])

        # Potential-Based Reward Shaping (PBRS) for footsie range
        def potential(d):
            return FOOTSIE_BASE_REWARD * max(0.0, 1.0 - d / FOOTSIE_RANGE_MAX)

        phi_curr = potential(rel_dist)
        phi_prev = potential(self.prev_rel_dist)
        # Using gamma = 0.99 consistent with PPO training discount factor
        dist_reward = 0.99 * phi_curr - phi_prev
        self.prev_rel_dist = rel_dist

        if rel_dist <= FOOTSIE_RANGE_MAX:
            self.footsie_steps += 1
        else:
            self.footsie_steps = 0

        if damage_dealt > 0:
            # Landing a hit resets the decay
            self.footsie_steps = 0

            if self.frames_since_last_hit == 0:
                pass # Continuous damage on consecutive frames, do not increment combo
            elif self.frames_since_last_hit <= COMBO_WINDOW:
                self.combo_counter += 1
            else:
                self.combo_counter = 1
            self.frames_since_last_hit = 0

            combo_bonus = min(self.combo_counter * 0.5, 4.0)  # caps at 8-hit combo

            reward = (float(damage_dealt)
                      + combo_bonus
                      - (DAMAGE_TAKEN_PENALTY * float(damage_taken))
                      + dist_reward)

        else:
            self.frames_since_last_hit += 1
            if self.frames_since_last_hit > COMBO_WINDOW:
                self.combo_counter = 0

            reward = -(DAMAGE_TAKEN_PENALTY * float(damage_taken)) - 0.015 + dist_reward

        if current_enemy_hp <= 0: reward += 65.0
        if current_my_hp <= 0: reward -= 50.0  # To avoid a tie

        self.prev_my_hp, self.prev_enemy_hp = current_my_hp, current_enemy_hp

        # A frame where BOTH HP reads are sentinels is a round transition, not a
        # KO. Terminating there fabricates episodes out of menu frames.
        ko = bool(current_my_hp <= 0 or current_enemy_hp <= 0) and not self.hp_sentinel
        terminated = ko if self.trainable else False
        truncated = (bool(self._steps >= config.MAX_STEPS_PER_ROUND) and not terminated) if self.trainable else False

        info = {
            "my_hp": current_my_hp,
            "enemy_hp": current_enemy_hp,
            "hp_sentinel": self.hp_sentinel,
        }
        if terminated or truncated:
            double_ko = bool(terminated and current_my_hp <= 0 and current_enemy_hp <= 0)
            info["double_ko"] = double_ko
            info["timeout"] = bool(truncated)
            info["episode_steps"] = self._steps
            info["win"] = 1 if (terminated and current_enemy_hp <= 0 and current_my_hp > 0) else 0
            info["loss"] = 1 if (terminated and current_my_hp <= 0 and current_enemy_hp > 0) else 0
            if hasattr(self, "current_state_file"):
                info["state_file"] = self.current_state_file

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        MAX_RETRIES = 2
        for attempt in range(MAX_RETRIES):
            try:
                # Random Domain Selection
                # Phase selection
                chosen_state_file = random.choice(self.active_training_states)
                self.current_state_file = chosen_state_file
                full_state_path = os.path.join(config.STATES_DIR, chosen_state_file)

                # Send Reset via Parent Method
                if self.trainable:
                    self.send_command(f"RESET {full_state_path}\n")

                # Wait for resulting state
                data = self.receive_payload()
                observation = self._parse_payload(data, is_reset=True)

                self.prev_my_hp    = float(observation[0]) if observation[0] > 0 else 176.0
                self.prev_enemy_hp = float(observation[1]) if observation[1] > 0 else 176.0

                # Internal absolute coordinates are still tracked for velocity calculation
                # (These are NOT in the final observation anymore, but we need them for deltas)
                csv_string = data.strip().split(" ")[-1]
                parts = csv_string.split(",")
                if len(parts) == 13:
                    raw = [int(x) for x in parts]
                    if self.player == 2:
                        self.prev_p1_x, self.prev_p2_x = raw[3], raw[2]
                    else:
                        self.prev_p1_x, self.prev_p2_x = raw[2], raw[3]

                # Reset reward tracking
                self._steps = 0
                self.footsie_steps = 0
                self.prev_rel_dist = float(observation[9])
                self.combo_counter = 0
                self.frames_since_last_hit = 0
                self.sticky_counter = 0
                self.sticky_direction = None
                self.hp_sentinel = False

                self.frames.clear()
                for _ in range(config.NUM_FRAMES): self.frames.append(observation)

                return self._get_obs(), {}

            except (RuntimeError, OSError) as e:
                rank = self.port - config.PORT
                if attempt < MAX_RETRIES - 1:
                    print(f"[Rank {rank}] RESET FAILED (Attempt {attempt+1}): {e}. Attempting self-healing respawn...")
                    try:
                        self.close()
                    except Exception: pass
                    time.sleep(5)
                    try:
                        self._start_emulator_bridge()
                    except Exception as bridge_err:
                        print(f"[Rank {rank}] Critical: Self-healing respawn failed: {bridge_err}")
                else:
                    raise RuntimeError(f"[Rank {rank}] BizHawk DEAD on reset after {MAX_RETRIES} attempts: {e}")

    def _parse_payload(self, data, is_reset=False):
        """Builds a 554-dimensional float32 observation.

        Layout per frame:
          [0-9]   Continuous: HP(2), RelX(1), RelY(1), WallDist(1), ProjX(2), VelX(2), RelDist(1)
          [10-265] P1 action one-hot (256)
          [266-521] P2 action one-hot (256)
          [522-537] P1 char one-hot (16)
          [538-553] P2 char one-hot (16)
        """
        # Grab strictly the CSV string, stripping off the leading zero
        csv_string = data.strip().split(" ")[-1]
        parts = csv_string.split(",")

        if len(parts) == 13:
            try:
                raw = [int(x) for x in parts]

                p1_sentinel = raw[0] > self.HP_SENTINEL_THRESHOLD
                p2_sentinel = raw[1] > self.HP_SENTINEL_THRESHOLD
                # Both players reading sentinel at once means a round transition,
                # not a simultaneous KO. Flag it so step() can refuse to terminate.
                self.hp_sentinel = p1_sentinel and p2_sentinel
                raw[0] = 0 if p1_sentinel else raw[0]
                raw[1] = 0 if p2_sentinel else raw[1]

                # PERSPECTIVE FLIP
                if self.player == 2:
                    p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y = raw[1], raw[0], raw[3], raw[2], raw[5], raw[4]
                    p1_act, p2_act, p1_proj, p2_proj, p1_char, p2_char = raw[7], raw[6], raw[9], raw[8], raw[11], raw[10]
                else:
                    p1_hp, p2_hp, p1_x, p2_x, p1_y, p2_y = raw[0], raw[1], raw[2], raw[3], raw[4], raw[5]
                    p1_act, p2_act, p1_proj, p2_proj, p1_char, p2_char = raw[6], raw[7], raw[8], raw[9], raw[10], raw[11]

                rel_dist = raw[12]

                # --- FIX C: TRANSLATION INVARIANCE ---
                rel_x = int(np.clip(p2_x - p1_x, -500, 500))
                rel_y = int(np.clip(p2_y - p1_y, -200, 200))
                p1_corner_dist = min(p1_x, 500 - p1_x) # Wall awareness

                p1_vel_x = 0 if is_reset else int(np.clip(p1_x - self.prev_p1_x, -100, 100))
                p2_vel_x = 0 if is_reset else int(np.clip(p2_x - self.prev_p2_x, -100, 100))

                # CRITICAL: Preserve internal tracking for velocity deltas
                self.prev_p1_x, self.prev_p2_x = p1_x, p2_x

                # 1. Continuous Features (10 dims)
                cont_obs = np.array([p1_hp, p2_hp, rel_x, rel_y, p1_corner_dist, p1_proj, p2_proj, p1_vel_x, p2_vel_x, rel_dist], dtype=np.float32)

                # 2. Categorical Features (One-Hot Encoded)
                p1_act_oh  = self._one_hot(p1_act,  ACT_CATEGORIES)
                p2_act_oh  = self._one_hot(p2_act,  ACT_CATEGORIES)
                p1_char_oh = self._one_hot(p1_char, CHAR_CATEGORIES)
                p2_char_oh = self._one_hot(p2_char, CHAR_CATEGORIES)

                # 3. Smash them all together
                return np.concatenate((cont_obs, p1_act_oh, p2_act_oh, p1_char_oh, p2_char_oh))

            except ValueError: pass

        # Failsafe: Return 554 zeros if the string is corrupted
        self.corrupt_payload_count += 1
        if self.corrupt_payload_count % 100 == 0:
            print(f"[WARNING] {self.corrupt_payload_count} corrupt payloads received. Check socket integrity.")
        # Return last known good observation instead of zeros:
        return self.frames[-1][:TOTAL_OBS_DIM] if len(self.frames) > 0 else np.zeros(TOTAL_OBS_DIM, dtype=np.float32)
