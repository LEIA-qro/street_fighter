import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

import config
from v1.env_sf2 import StreetFighterEnv
from bizhawk_base import BizHawkBaseEnv

directories = config.get_directory()

def test_agent():
    print("Initializing Street Fighter Evaluation Mode...")
    
    model_load_path = os.path.join(directories["project_root"], config.TESTING_ZIP_FILE_P2)
    vec_load_path = os.path.join(directories["project_root"], config.TESTING_PKL_FILE_P2)

    # 1. Boot a single emulator window
    env = StreetFighterEnv(lua_path = config.MATCH_TEST_ENV_CLIENT_LUA_PATH , trainable = False, rank=0, player=2) # Player 1 controls the agent, Player 2 is the dummy opponent
    # env.lua_path = config.MATCH_TEST_ENV_CLIENT_LUA_PATH  # Switch to the match test Lua script
    # env.trainable = False  # Disable randomization and resets for testing
    env = DummyVecEnv([lambda: env])
    
    # 2. Load the Normalization Math safely
    print(f"Loading normalization stats from {config.TESTING_PKL_FILE_P2}...")
    env = VecNormalize.load(vec_load_path, env)
    
    # CRITICAL: Lock the normalization math. Do NOT let it update during testing!
    env.training = False
    env.norm_reward = False 
    
    # 3. Load the Grandmaster Brain
    print(f"Loading neural network weights from {config.TESTING_ZIP_FILE_P2}...")
    model = PPO.load(model_load_path, env=env, device="cuda")
    
    print("\nFIGHT! (The AI engine is now running in the background)")
    obs = env.reset()
    
    try:
        while True:
            # deterministic=True forces the agent to pick its best calculated move
            action, _states = model.predict(obs, deterministic=False) # deterministic=True
            obs, reward, done, info = env.step(action)
            
    except KeyboardInterrupt:
        print("\nInteractive session ended by user.")
    finally:
        env.close()

if __name__ == "__main__":
    test_agent()