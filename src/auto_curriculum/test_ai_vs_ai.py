import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from v1.env_sf2 import StreetFighterEnv

directories = config.get_directory()

# 1. The Mock Environment (Prevents Shadow Emulators)
class MockEnv(gym.Env):
    """A hollow shell environment purely used to trick VecNormalize into loading."""
    def __init__(self):
        super().__init__()
        self.action_space = spaces.MultiBinary(config.ACTION_DIM)
        low = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, -1], dtype=np.int32)
        high = np.array([176, 176, 500, 500, 200, 200, 255, 255, 500, 500], dtype=np.int32) 
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.int32)

def test_ai_vs_ai():
    print("Initializing AI vs AI Evaluation Mode...") 

    # 2. Boot ONE Master Environment (Handles the socket connection)
    print("Booting Master Socket...")
    master_env = StreetFighterEnv(lua_path=config.MATCH_TEST_ENV_CLIENT_LUA_PATH, trainable=False, rank=0)
    
    # 3. Reconstruct Normalization Math safely using MockEnv
    dummy_env = DummyVecEnv([lambda: MockEnv()])
    
    vec_norm_p1 = VecNormalize.load(os.path.join(directories["project_root"], config.TESTING_PKL_FILE_P1), dummy_env)
    vec_norm_p1.training = False
    vec_norm_p1.norm_reward = False

    vec_norm_p2 = VecNormalize.load(os.path.join(directories["project_root"], config.TESTING_PKL_FILE_P2), dummy_env)
    vec_norm_p2.training = False
    vec_norm_p2.norm_reward = False

    # 4. Load the Brains
    print("Loading Neural Networks...")
    model_p1 = PPO.load(os.path.join(directories["project_root"], config.TESTING_ZIP_FILE_P1), env=vec_norm_p1, device="cuda")
    model_p2 = PPO.load(os.path.join(directories["project_root"], config.TESTING_ZIP_FILE_P2), env=vec_norm_p2, device="cuda")

    print("\nFIGHT! (AI vs AI mode is running in the background)")

    # (Note: master_env.reset() has been DELETED to prevent deadlock)

    try:
        while True:
            # 1. Wait for BizHawk to send the raw 10-variable RAM string
            raw_payload = master_env.receive_payload()
            if not raw_payload:
                continue
                
            # 2. Parse the math for Player 1's perspective
            master_env.player = 1
            obs_p1_raw = master_env._parse_payload(raw_payload)
            
            # 3. Parse the math for Player 2's perspective (Flips the array!)
            master_env.player = 2
            obs_p2_raw = master_env._parse_payload(raw_payload)
            
            # 4. Apply standard deviation scaling
            obs_p1_norm = vec_norm_p1.normalize_obs(np.expand_dims(obs_p1_raw, axis=0))
            obs_p2_norm = vec_norm_p2.normalize_obs(np.expand_dims(obs_p2_raw, axis=0))
            
            # 5. Let the Brains think (Stochastic exploration allows organic fighting)
            act_p1, _states = model_p1.predict(obs_p1_norm, deterministic=False)
            act_p2, _states = model_p2.predict(obs_p2_norm, deterministic=False)
            
            # 6. Convert the binary arrays back into strings
            act_str_p1 = "".join(str(int(b)) for b in act_p1[0]) 
            act_str_p2 = "".join(str(int(b)) for b in act_p2[0])
            
            # 7. Glue them together and send the 20-Bit Master Command to BizHawk
            full_command = act_str_p1 + act_str_p2 + "\n"
            master_env.send_command(full_command)

    except KeyboardInterrupt:
        print("\nAI vs AI session ended by user.")
    finally:
        dummy_env.close()
        master_env.close()

if __name__ == "__main__":
    test_ai_vs_ai()