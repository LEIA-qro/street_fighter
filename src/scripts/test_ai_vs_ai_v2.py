import os, sys, argparse
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

from core import config
from envs.sf2_v2 import StreetFighterEnvV2, TOTAL_OBS_DIM
from core.selective_norm import SelectiveVecNormalize

directories = config.get_directory()

class _MockV2Env(gym.Env):
    """
    Zero-cost shell with correct v2 obs/action spaces.
    Satisfies SelectiveVecNormalize.load()'s venv argument.
    No socket, no subprocess, no blocking.
    """
    def __init__(self):
        super().__init__()
        self.action_space = spaces.MultiBinary(config.ACTION_DIM)
        n = TOTAL_OBS_DIM * config.NUM_FRAMES  # 554 * 4 = 2216
        self.observation_space = spaces.Box(
            low=np.zeros(n,  dtype=np.float32),
            high=np.ones(n,  dtype=np.float32),
            dtype=np.float32
        )
    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

class _FrameBuffer:
    """
    Stateless per-agent frame stacker.
    Decouples stacking from env state so dual-perspective
    inference doesn't corrupt shared prev_x fields.
    """
    def __init__(self, n_frames: int, obs_dim: int):
        self.n_frames = n_frames
        self.obs_dim = obs_dim
        self.buffer = deque(maxlen=n_frames)

    def reset(self, first_obs: np.ndarray):
        self.buffer.clear()
        for _ in range(self.n_frames):
            self.buffer.append(first_obs.copy())

    def push(self, obs: np.ndarray) -> np.ndarray:
        self.buffer.append(obs.copy())
        return np.concatenate(list(self.buffer))  # shape: (obs_dim * n_frames,)


class _PerspectiveParser:
    """
    Wraps a single StreetFighterEnvV2 instance and provides
    isolated prev_x state per player perspective.
    Prevents the double-call side-effect bug.
    """
    def __init__(self, env: StreetFighterEnvV2, player: int):
        self.env = env
        self.player = player
        self.prev_p1_x = 0
        self.prev_p2_x = 0

    def parse(self, raw_payload: str, is_reset: bool = False) -> np.ndarray:
        # Temporarily inject isolated state into the env before parsing
        self.env.player     = self.player
        self.env.prev_p1_x  = self.prev_p1_x
        self.env.prev_p2_x  = self.prev_p2_x

        obs = self.env._parse_payload(raw_payload, is_reset=is_reset)

        # Save back the updated state from this perspective only
        self.prev_p1_x = self.env.prev_p1_x
        self.prev_p2_x = self.env.prev_p2_x

        return obs

def get_model_class(algo_name):
    if algo_name.lower() == "sac":
        return SAC
    elif algo_name.lower() == "dqn":
        return DQN
    return PPO

def process_action(act, algo):
    if algo.lower() == "sac":
        bin_act = (act[0] > 0.0).astype(np.int8)
        return "".join(str(b) for b in bin_act)
    elif algo.lower() == "dqn":
        val = act[0] if isinstance(act, np.ndarray) else act
        return format(val, f'0{config.ACTION_DIM}b')
    else:
        return "".join(str(int(b)) for b in act[0])

def test_ai_vs_ai():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo_p1", type=str, default="ppo")
    parser.add_argument("--load_zip_p1", type=str, required=True)
    parser.add_argument("--load_pkl_p1", type=str, required=True)
    parser.add_argument("--device_p1", type=str, default="auto")
    
    parser.add_argument("--algo_p2", type=str, default="ppo")
    parser.add_argument("--load_zip_p2", type=str, required=True)
    parser.add_argument("--load_pkl_p2", type=str, required=True)
    parser.add_argument("--device_p2", type=str, default="auto")
    args = parser.parse_args()

    # Auto-detect algorithm override for P1
    if args.load_zip_p1 != "None":
        lower_path_p1 = args.load_zip_p1.lower()
        if "dqn" in lower_path_p1 and args.algo_p1.lower() != "dqn":
            print(f"\n[WARN] Auto-detected 'dqn' in P1 path. Overriding algorithm '{args.algo_p1.upper()}' to 'DQN'.")
            args.algo_p1 = "dqn"
        elif "sac" in lower_path_p1 and args.algo_p1.lower() != "sac":
            print(f"\n[WARN] Auto-detected 'sac' in P1 path. Overriding algorithm '{args.algo_p1.upper()}' to 'SAC'.")
            args.algo_p1 = "sac"
        elif "ppo" in lower_path_p1 and args.algo_p1.lower() != "ppo":
            print(f"\n[WARN] Auto-detected 'ppo' in P1 path. Overriding algorithm '{args.algo_p1.upper()}' to 'PPO'.")
            args.algo_p1 = "ppo"
            
    # Auto-detect algorithm override for P2
    if args.load_zip_p2 != "None":
        lower_path_p2 = args.load_zip_p2.lower()
        if "dqn" in lower_path_p2 and args.algo_p2.lower() != "dqn":
            print(f"\n[WARN] Auto-detected 'dqn' in P2 path. Overriding algorithm '{args.algo_p2.upper()}' to 'DQN'.")
            args.algo_p2 = "dqn"
        elif "sac" in lower_path_p2 and args.algo_p2.lower() != "sac":
            print(f"\n[WARN] Auto-detected 'sac' in P2 path. Overriding algorithm '{args.algo_p2.upper()}' to 'SAC'.")
            args.algo_p2 = "sac"
        elif "ppo" in lower_path_p2 and args.algo_p2.lower() != "ppo":
            print(f"\n[WARN] Auto-detected 'ppo' in P2 path. Overriding algorithm '{args.algo_p2.upper()}' to 'PPO'.")
            args.algo_p2 = "ppo"

    # Set Model Labels for Lua UI
    p1_name = os.path.basename(args.load_zip_p1).replace(".zip", "")
    p2_name = os.path.basename(args.load_zip_p2).replace(".zip", "")
    config.P1_MODEL_NAME = f"AI: {p1_name}"
    config.P2_MODEL_NAME = f"AI: {p2_name}"

    config.generate_lua_config()

    # Device Auto-Logic
    device_p1 = args.device_p1
    if device_p1 == "auto":
        device_p1 = "cpu" if args.algo_p1.lower() == "ppo" else "cuda"
        
    device_p2 = args.device_p2
    if device_p2 == "auto":
        device_p2 = "cpu" if args.algo_p2.lower() == "ppo" else "cuda"
    
    # Suppress SB3 GPU Warnings for MlpPolicy
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="stable_baselines3")

    print("Initializing AI vs AI Evaluation Mode...") 
    print(f"P1 Device: {device_p1} | P2 Device: {device_p2}")

    print("Booting Master Socket...")
    # ── 1. Boot ONE master socket env ──
    master_env = StreetFighterEnvV2(
        lua_path=config.MATCH_TEST_ENV_CLIENT_LUA_PATH,
        trainable=False,
        rank=0,
        player=1  
    )
    
    # ── 2. Build perspective-isolated parsers ──
    parser_p1 = _PerspectiveParser(master_env, player=1)
    parser_p2 = _PerspectiveParser(master_env, player=2)

    # ── 3. Per-agent frame buffers ──
    buf_p1 = _FrameBuffer(n_frames=config.NUM_FRAMES, obs_dim=TOTAL_OBS_DIM)
    buf_p2 = _FrameBuffer(n_frames=config.NUM_FRAMES, obs_dim=TOTAL_OBS_DIM)

    # ── 4. Load normalizers ──
    dummy = DummyVecEnv([_MockV2Env])  
    vec_norm_p1 = SelectiveVecNormalize.load(
        os.path.join(config.PROJECT_ROOT, args.load_pkl_p1), dummy
    )
    vec_norm_p1.training = False

    vec_norm_p2 = SelectiveVecNormalize.load(
        os.path.join(config.PROJECT_ROOT, args.load_pkl_p2), dummy
    )
    vec_norm_p2.training = False

     # ── 5. Load models ──
    print(f"\nLoading Neural Networks...")
    
    def load_model_safely(algo, path, player_name, device):
        ModelClass = get_model_class(algo)
        custom_objs = {}
        if algo.lower() in ["dqn", "sac"]:
            custom_objs["buffer_size"] = 1  # Memory optimization: don't load the full replay buffer
            
        try:
            model = ModelClass.load(
                os.path.join(config.PROJECT_ROOT, path),
                device=device,
                custom_objects=custom_objs
            )
            print(f"{player_name} ({algo.upper()}) loaded from: {path}")
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
                print(f"\n[CRITICAL ERROR] Failed to load {player_name} model.")
                print(f"Algorithm '{algo.upper()}' is incompatible with the provided file: {path}")
                print(f"Details: {err_msg}")
            else:
                print(f"\n[ERROR] Unexpected error loading {player_name}: {err_msg}")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Unexpected error loading {player_name}: {e}")
            sys.exit(1)

    model_p1 = load_model_safely(args.algo_p1, args.load_zip_p1, "Player 1", device_p1)
    model_p2 = load_model_safely(args.algo_p2, args.load_zip_p2, "Player 2", device_p2)

    
    print(f"\n{('='*50)}")
    print("FIGHT! (AI vs AI V2 running...)")
    print(f"{('='*50)}")
    print("Press Ctrl + C to end the session and close the emulator.")

    # ── 6. Cold-start: prime frame buffers on first payload ──
    first_payload = master_env.receive_payload()
    obs_p1_raw = parser_p1.parse(first_payload, is_reset=True)
    obs_p2_raw = parser_p2.parse(first_payload, is_reset=True)
    buf_p1.reset(obs_p1_raw)
    buf_p2.reset(obs_p2_raw)

    try:
        while True:
            # Stack observations -> (2216,) each
            stacked_p1 = buf_p1.push(obs_p1_raw)  # Updated by previous iteration
            stacked_p2 = buf_p2.push(obs_p2_raw)

            # Normalize: SelectiveVecNormalize expects (n_envs, obs_dim)
            norm_p1 = vec_norm_p1.normalize_obs(stacked_p1[np.newaxis, :])
            norm_p2 = vec_norm_p2.normalize_obs(stacked_p2[np.newaxis, :])

            # Inference
            act_p1, _ = model_p1.predict(norm_p1, deterministic=False)
            act_p2, _ = model_p2.predict(norm_p2, deterministic=False)

            # Process action depending on algo
            cmd_p1 = process_action(act_p1, args.algo_p1)
            cmd_p2 = process_action(act_p2, args.algo_p2)

            # Build 20-bit command string
            cmd = cmd_p1 + cmd_p2 + "\n"
            master_env.send_command(cmd)

            # Receive next state and parse both perspectives
            raw = master_env.receive_payload()

            if not raw:
                continue

            obs_p1_raw = parser_p1.parse(raw, is_reset=False)
            obs_p2_raw = parser_p2.parse(raw, is_reset=False)

    except KeyboardInterrupt:
        print("\nAI vs AI session ended by user.")

    finally:
        master_env.close()

if __name__ == "__main__":
    test_ai_vs_ai()