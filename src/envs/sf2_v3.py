# env_sf2_v3.py
import numpy as np
from gymnasium import spaces
from envs.sf2_v2 import StreetFighterEnvV2, TOTAL_OBS_DIM
import core.config as config

# Correct Genesis SF2 6-button layout
# Bits: [Up, Down, Left, Right, A(LK), B(MK), C(HK), X(LP), Y(MP), Z(HP)]
DIRECTION_MAP = {
    0: [0,0,0,0], 1: [1,0,0,0], 2: [0,1,0,0],
    3: [0,0,1,0], 4: [0,0,0,1], 5: [1,0,1,0],
    6: [1,0,0,1], 7: [0,1,1,0], 8: [0,1,0,1],
}
BUTTON_MAP = {
    0: [0,0,0,0,0,0], 1: [1,0,0,0,0,0], 2: [0,1,0,0,0,0],
    3: [0,0,1,0,0,0], 4: [0,0,0,1,0,0], 5: [0,0,0,0,1,0],
    6: [0,0,0,0,0,1],
}

def discrete_to_binary(action: np.ndarray) -> str:
    dir_bits = DIRECTION_MAP[int(action[0])]
    btn_bits = BUTTON_MAP[int(action[1])]
    return "".join(str(b) for b in dir_bits + btn_bits)


class StreetFighterEnvV3(StreetFighterEnvV2):
    """
    V3: MultiDiscrete([9, 7]) action space.
    Reduces valid combinations from 1,024 to 63.
    Preserves 10-bit Lua protocol and all v2 observation semantics.
    Corrects button mapping for Ryu's full moveset (punches + kicks preserved).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override ONLY the action space — observation space unchanged
        self.action_space = spaces.MultiDiscrete([9, 7])

    def step(self, action):
        try:
            action_string = discrete_to_binary(action)
            full_command = (
                (action_string + "0000000000\n") if self.player == 1
                else ("0000000000" + action_string + "\n")
            )
            self.send_command(full_command)
            data = self.receive_payload()
            self.debug_print(f"Command: '{full_command}' | Payload: '{data}'")

        except RuntimeError as e:
            print(f"[Rank {self.port - config.PORT}] Socket error in step: {e}.")
            obs = (self._get_obs() if len(self.frames) > 0
                   else np.zeros(TOTAL_OBS_DIM * config.NUM_FRAMES, dtype=np.float32))
            return obs, 0.0, True, False, {"socket_death": True}

        # All reward, observation, termination logic inherited from v2
        observation = self._parse_payload(data, is_reset=False)
        self.frames.append(observation)

        current_my_hp, current_enemy_hp = observation[0], observation[1]

        # --- Reward logic copy from v2 (DRY violation acceptable for isolation) ---
        damage_clamp = 100
        damage_dealt = min(max(0, self.prev_enemy_hp - current_enemy_hp), damage_clamp)
        damage_taken = min(max(0, self.prev_my_hp - current_my_hp), damage_clamp)

        COMBO_WINDOW = 6
        DAMAGE_TAKEN_PENALTY = 0.70
        FOOTSIE_RANGE_MAX = 80
        FOOTSIE_BASE_REWARD = 0.05
        FOOTSIE_DECAY_RATE = 0.05

        rel_dist = int(observation[9])

        if rel_dist <= FOOTSIE_RANGE_MAX:
            dist_reward = FOOTSIE_BASE_REWARD * np.exp(-FOOTSIE_DECAY_RATE * self.footsie_steps)
            self.footsie_steps += 1
        else:
            dist_reward = 0.0
            self.footsie_steps = 0

        if damage_dealt > 0:
            self.footsie_steps = 0
            if self.frames_since_last_hit <= COMBO_WINDOW:
                self.combo_counter += 1
            else:
                self.combo_counter = 1
            self.frames_since_last_hit = 0
            combo_bonus = min(self.combo_counter * 0.5, 4.0)
            reward = (float(damage_dealt) + combo_bonus
                      - (DAMAGE_TAKEN_PENALTY * float(damage_taken)) + dist_reward)
        else:
            self.frames_since_last_hit += 1
            if self.frames_since_last_hit > COMBO_WINDOW:
                self.combo_counter = 0
            reward = -(DAMAGE_TAKEN_PENALTY * float(damage_taken)) - 0.015 + dist_reward

        if current_enemy_hp <= 0: reward += 50.0
        if current_my_hp <= 0: reward -= 50.0

        self.prev_my_hp, self.prev_enemy_hp = current_my_hp, current_enemy_hp
        terminated = bool(current_my_hp <= 0 or current_enemy_hp <= 0) if self.trainable else False

        info = {}
        if terminated:
            info["win"] = 1 if current_enemy_hp <= 0 and current_my_hp > 0 else 0

        return self._get_obs(), reward, terminated, False, info
