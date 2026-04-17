import os, sys
from collections import deque

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium as gym
from gymnasium import spaces

import config
from env_sf2_v2 import StreetFighterEnvV2, TOTAL_OBS_DIM
from selective_norm import SelectiveVecNormalize

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
    
def test_ai_vs_ai():
    print("Initializing AI vs AI Evaluation Mode...") 

    print("Booting Master Socket...")
    # ── 1. Boot ONE master socket env (no DummyVecEnv — raw socket only) ──
    master_env = StreetFighterEnvV2(
        lua_path=config.MATCH_TEST_ENV_CLIENT_LUA_PATH,
        trainable=False,
        rank=0,
        player=1  # Default; overridden per-frame by _PerspectiveParser
    )
    
    # ── 2. Build perspective-isolated parsers ──
    parser_p1 = _PerspectiveParser(master_env, player=1)
    parser_p2 = _PerspectiveParser(master_env, player=2)

    # ── 3. Per-agent frame buffers (correct n_frames=4, dim=554) ──
    buf_p1 = _FrameBuffer(n_frames=config.NUM_FRAMES, obs_dim=TOTAL_OBS_DIM)
    buf_p2 = _FrameBuffer(n_frames=config.NUM_FRAMES, obs_dim=TOTAL_OBS_DIM)

    # ── 4. Load normalizers with the correct class ──
    # SelectiveVecNormalize.load requires a venv for obs_space; use a
    # DummyVecEnv wrapping a fresh env shell (immediately closed after load).
    dummy = DummyVecEnv([_MockV2Env])  # No BizHawk, no socket, instant
    vec_norm_p1 = SelectiveVecNormalize.load(
        os.path.join(directories["project_root"], config.TESTING_PKL_FILE_P1), dummy
    )
    vec_norm_p1.training = False

    vec_norm_p2 = SelectiveVecNormalize.load(
        os.path.join(directories["project_root"], config.TESTING_PKL_FILE_P2), dummy
    )
    vec_norm_p2.training = False

     # ── 5. Load models ──
    print(f"\nLoading Neural Networks...")
    model_p1 = PPO.load(
        os.path.join(directories["project_root"], config.TESTING_ZIP_FILE_P1),
        device="cuda"
    )
    print("Player 1 loaded from:", config.TESTING_ZIP_FILE_P1)
    model_p2 = PPO.load(
        os.path.join(directories["project_root"], config.TESTING_ZIP_FILE_P2),
        device="cuda"
    )
    print("Player 2 loaded from:", config.TESTING_ZIP_FILE_P2)

    
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
            # Stack observations → (2216,) each
            stacked_p1 = buf_p1.push(obs_p1_raw)  # Updated by previous iteration
            stacked_p2 = buf_p2.push(obs_p2_raw)

            # Normalize: SelectiveVecNormalize expects (n_envs, obs_dim)
            norm_p1 = vec_norm_p1.normalize_obs(stacked_p1[np.newaxis, :])
            norm_p2 = vec_norm_p2.normalize_obs(stacked_p2[np.newaxis, :])

            # Inference
            act_p1, _ = model_p1.predict(norm_p1, deterministic=False)
            act_p2, _ = model_p2.predict(norm_p2, deterministic=False)

            # Build 20-bit command string
            cmd = (
                "".join(str(int(b)) for b in act_p1[0]) +
                "".join(str(int(b)) for b in act_p2[0]) +
                "\n"
            )
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