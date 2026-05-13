import os, sys, argparse

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from core import config
from envs.sf2_v2 import StreetFighterEnvV2
from core.selective_norm import SelectiveVecNormalize

def test_agent():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_zip", type=str, required=True)
    parser.add_argument("--load_pkl", type=str, required=True)
    parser.add_argument("--player", type=int, default=1)
    args = parser.parse_args()

    print(f"Initializing Street Fighter Evaluation Mode for Player {args.player}...")
    
    model_load_path = os.path.join(config.PROJECT_ROOT, args.load_zip)
    vec_load_path = os.path.join(config.PROJECT_ROOT, args.load_pkl)

    # 1. Boot a single emulator window
    env = StreetFighterEnvV2(
        lua_path = config.MATCH_TEST_ENV_CLIENT_LUA_PATH , 
        trainable = False, 
        rank=-1, 
        player=args.player
        ) 
    
    env = DummyVecEnv([lambda: env])
    
    # 2. Load the Normalization Math safely
    print(f"Loading normalization stats from {args.load_pkl}...")
    env = SelectiveVecNormalize.load(vec_load_path, env)
    
    # CRITICAL: Lock the normalization math. Do NOT let it update during testing!
    env.training = False
    env.norm_reward = False 
    
    # 3. Load the Grandmaster Brain
    print(f"Loading neural network weights from {args.load_zip}...")
    model = PPO.load(model_load_path, env=env, device="cuda")
    
    print("\nFIGHT! (The AI engine is now running in the background)")
    obs = env.reset()
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=False) 
            obs, reward, done, info = env.step(action)
            
    except KeyboardInterrupt:
        print("\nInteractive session ended by user.")
    finally:
        env.close()

if __name__ == "__main__":
    test_agent()