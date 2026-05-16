import os, sys, argparse
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv

from core import config
from envs.sf2_v2 import StreetFighterEnvV2
from core.selective_norm import SelectiveVecNormalize

def get_model_class(algo_name):
    if algo_name.lower() == "sac":
        return SAC
    elif algo_name.lower() == "dqn":
        return DQN
    return PPO

def test_agent():
    config.generate_lua_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=str, default="ppo")
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
    
    def load_model_safely(algo, path):
        ModelClass = get_model_class(algo)
        custom_objs = {}
        if algo.lower() in ["dqn", "sac"]:
            custom_objs["buffer_size"] = 1  # Memory optimization
            
        try:
            model = ModelClass.load(
                path,
                device="cuda",
                custom_objects=custom_objs
            )
            print(f"Model ({algo.upper()}) loaded successfully.")
            return model
        except (AttributeError, TypeError, ValueError) as e:
            err_msg = str(e)
            # Detect common SB3 mismatch indicators
            is_mismatch = any(keyword in err_msg for keyword in [
                "unexpected keyword argument", 
                "object has no attribute",
                "missing 1 required positional argument"
            ])
            
            if is_mismatch:
                print(f"\n[CRITICAL ERROR] Algorithm '{algo.upper()}' is incompatible with the provided file: {path}")
                print(f"Details: {err_msg}")
            else:
                print(f"\n[ERROR] Unexpected error loading model: {err_msg}")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Unexpected error loading model: {e}")
            sys.exit(1)

    model = load_model_safely(args.algo, model_load_path)
    
    print(f"\nFIGHT! (The {args.algo.upper()} AI engine is now running in the background)")
    obs = env.reset()
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=False) 
            
            # Process action depending on algo
            if args.algo.lower() == "sac":
                bin_act = (action[0] > 0.0).astype(np.int8)
                processed_action = np.array([bin_act])
            elif args.algo.lower() == "dqn":
                val = action[0] if isinstance(action, np.ndarray) else action
                binary_str = format(val, f'0{config.ACTION_DIM}b')
                bin_act = np.array([int(b) for b in binary_str], dtype=np.int8)
                processed_action = np.array([bin_act])
            else:
                processed_action = action

            obs, reward, done, info = env.step(processed_action)
            
    except KeyboardInterrupt:
        print("\nInteractive session ended by user.")
    finally:
        env.close()

if __name__ == "__main__":
    test_agent()