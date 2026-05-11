import os, sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

import config
from env_sf2_v2 import StreetFighterEnvV2
from selective_norm import SelectiveVecNormalize
# from bizhawk_base import BizHawkBaseEnv

directories = config.get_directory()

PLAYER = 1 # The agent's assigned player number (1 or 2)

def test_agent():
    print("Initializing Street Fighter Evaluation Mode...")
    
    model_load_path = os.path.join(directories["project_root"], config.TESTING_ZIP_FILE_P2)
    vec_load_path = os.path.join(directories["project_root"], config.TESTING_PKL_FILE_P2)

    # 1. Boot a single emulator window
    env = StreetFighterEnvV2(
        lua_path = config.MATCH_TEST_ENV_CLIENT_LUA_PATH , 
        trainable = False, 
        rank=-1, 
        player=PLAYER # AI controls the asigned player; the perspective parser will handle the rest. Use rank=-1 for a single env that won't be part of training.
        ) 
    
    env = DummyVecEnv([lambda: env])
    
    # 2. Load the Normalization Math safely
    print(f"Loading normalization stats from {config.TESTING_PKL_FILE_P2}...")
    env = SelectiveVecNormalize.load(vec_load_path, env)
    
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