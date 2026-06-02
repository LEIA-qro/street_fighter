# league_env.py
import os
import numpy as np
from gymnasium import spaces
import gymnasium as gym
from collections import deque
from typing import Optional, Tuple, Dict
from stable_baselines3.common.vec_env import DummyVecEnv

from core import config
from envs.base_env import TOTAL_OBS_DIM
from envs.sf2_v2 import StreetFighterEnvV2
from core.selective_norm import SelectiveVecNormalize

class _MockEnv(gym.Env):
    """Zero-cost Gym environment shell with correct obs/action spaces.
    
    Used to satisfy SelectiveVecNormalize.load()'s venv argument without spawning socket connections.
    """
    def __init__(self, version="v2"):
        super().__init__()
        if version == "v3":
            self.action_space = spaces.MultiDiscrete([9, 7])
        else:
            self.action_space = spaces.MultiBinary(config.ACTION_DIM)
            
        n = TOTAL_OBS_DIM * config.NUM_FRAMES  
        self.observation_space = spaces.Box(
            low=np.zeros(n, dtype=np.float32),
            high=np.ones(n, dtype=np.float32),
            dtype=np.float32
        )
    def reset(self, **kwargs):
        return np.zeros(self.observation_space.shape, dtype=np.float32), {}
    def step(self, action):
        return np.zeros(self.observation_space.shape, dtype=np.float32), 0.0, False, False, {}

class _FrameBuffer:
    """Stateless per-agent frame stacker to decouple P1 and P2 state tracking."""
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
        return np.concatenate(list(self.buffer))

class _PerspectiveParser:
    """Wraps a single environment instance to parse observations from P1 or P2 perspectives.
    
    Prevents coordinate and velocity delta pollution.
    """
    def __init__(self, env: StreetFighterEnvV2, player: int):
        self.env = env
        self.player = player
        self.prev_p1_x = 0
        self.prev_p2_x = 0

    def parse(self, raw_payload: str, is_reset: bool = False) -> np.ndarray:
        # Inject isolated coordinate history before parsing
        self.env.player = self.player
        self.env.prev_p1_x = self.prev_p1_x
        self.env.prev_p2_x = self.prev_p2_x

        obs = self.env._parse_payload(raw_payload, is_reset=is_reset)

        # Save back the updated coordinate history
        self.prev_p1_x = self.env.prev_p1_x
        self.prev_p2_x = self.env.prev_p2_x

        return obs

class StreetFighterLeagueEnv(StreetFighterEnvV2):
    """Gymnasium Environment for two-player League Matchmaking and training.
    
    Active P1 trains under standard PPO while P2 is controlled by cached pool opponents.
    Implements Dynamic Left/Right Action-Mirroring based on relative character position
    to ensure 100% translation-invariant fair matchups.
    """
    
    def __init__(self, rank=0, lua_path=config.TRAINING_ENV_CLIENT_LUA_PATH, 
                 trainable=True, debug_mode=True, player=1, verbose=True, version="v2"):
        
        self.version = version
        super().__init__(
            rank=rank,
            lua_path=lua_path,
            trainable=trainable,
            debug_mode=debug_mode,
            player=player,
            verbose=verbose
        )
        
        # Override action space if version is v3
        if self.version == "v3":
            self.action_space = spaces.MultiDiscrete([9, 7])
            
        # Decoupled perspective parsers
        self.parser_p1 = _PerspectiveParser(self, player=1)
        self.parser_p2 = _PerspectiveParser(self, player=2)
        
        # Stateless frame buffers
        self.buf_p1 = _FrameBuffer(n_frames=config.NUM_FRAMES, obs_dim=TOTAL_OBS_DIM)
        self.buf_p2 = _FrameBuffer(n_frames=config.NUM_FRAMES, obs_dim=TOTAL_OBS_DIM)
        
        # Opponent configuration
        self.opponent_id: str = "none"
        self.opponent_model = None
        self.opponent_vec_norm = None
        self.opponent_version = "v2"
        
        self.latest_p2_stacked: Optional[np.ndarray] = None
        self.latest_raw_payload: Optional[str] = None
        
    def set_opponent(self, opponent_id: str, model, vecnorm_path: str, opponent_version: str = "v2"):
        """Swaps the opponent policy and loads its corresponding frozen VecNormalize."""
        self.opponent_id = opponent_id
        self.opponent_model = model
        self.opponent_version = opponent_version
        
        # Load the frozen VecNormalize corresponding to the opponent's historical checkpoint
        try:
            dummy_venv = DummyVecEnv([lambda: _MockEnv(version=opponent_version)])
            self.opponent_vec_norm = SelectiveVecNormalize.load(vecnorm_path, dummy_venv)
            self.opponent_vec_norm.training = False
            if self.verbose:
                print(f"[LeagueEnv] Loaded opponent {opponent_id} with VecNormalize from: {os.path.basename(vecnorm_path)}")
        except Exception as e:
            self.opponent_vec_norm = None
            if self.verbose:
                print(f"[LeagueEnv][WARN] Could not load VecNormalize for opponent {opponent_id}: {e}. Disabling normalization for P2.")
                
    def set_opponent_paths(self, opponent_id: str, zip_path: str, vecnorm_path: str, opponent_version: str = "v2"):
        """Process-safe opponent loading: receives paths and loads/caches the model locally in the process."""
        self.opponent_id = opponent_id
        self.opponent_version = opponent_version
        
        if not hasattr(self, "_local_model_cache"):
            self._local_model_cache = {}
            
        if opponent_id in self._local_model_cache:
            self.opponent_model = self._local_model_cache[opponent_id]
        else:
            if self.verbose:
                print(f"[LeagueEnv][Rank {self.port - config.PORT}] Loading opponent model locally: {opponent_id}")
            try:
                from stable_baselines3 import PPO
                self.opponent_model = PPO.load(
                    zip_path,
                    device="cpu",
                    custom_objects={"buffer_size": 1}
                )
                self.opponent_model.policy.eval()
                for param in self.opponent_model.policy.parameters():
                    param.requires_grad = False
                self._local_model_cache[opponent_id] = self.opponent_model
            except Exception as e:
                print(f"[LeagueEnv][Rank {self.port - config.PORT}][ERROR] Failed to load PPO model locally from {zip_path}: {e}")
                self.opponent_model = None
                
        # Load the frozen VecNormalize
        try:
            dummy_venv = DummyVecEnv([lambda: _MockEnv(version=opponent_version)])
            self.opponent_vec_norm = SelectiveVecNormalize.load(vecnorm_path, dummy_venv)
            self.opponent_vec_norm.training = False
        except Exception as e:
            self.opponent_vec_norm = None
            if self.verbose:
                print(f"[LeagueEnv][WARN] Could not load VecNormalize for opponent {opponent_id}: {e}")
                
    def _mirror_left_right(self, action_string: str) -> str:
        """Flips absolute Left (index 2) and Right (index 3) commands to maintain relative controls."""
        chars = list(action_string)
        left = chars[2]
        right = chars[3]
        chars[2] = right
        chars[3] = left
        return "".join(chars)
        
    def _process_opponent_action(self, action) -> str:
        """Processes the opponent's action to its standard 10-bit binary string command."""
        from envs.sf2_v2 import StreetFighterEnvV2
        
        if self.opponent_version == "v3":
            from envs.sf2_v3 import discrete_to_binary
            val = action[0] if isinstance(action, np.ndarray) else action
            return discrete_to_binary(val)
        else:
            # MultiBinary 10-bit string
            act = action[0] if isinstance(action, np.ndarray) else action
            return "".join(str(int(b)) for b in act)

    def step(self, action):
        """Standard Gym step: runs active P1 step, executes P2 inference, and returns P1 observations."""
        try:
            # --- 1. PROCESS PLAYER 1 (MAIN AGENT) ACTION ---
            action_string_p1 = self._action_to_string(action)
            
            # --- 2. EXECUTE PLAYER 2 (OPPONENT POOL) POLICY ---
            if self.opponent_model is not None and self.latest_p2_stacked is not None:
                # Normalize observations using the opponent's frozen VecNormalize
                if self.opponent_vec_norm is not None:
                    norm_obs_p2 = self.opponent_vec_norm.normalize_obs(self.latest_p2_stacked[np.newaxis, :])
                else:
                    norm_obs_p2 = self.latest_p2_stacked[np.newaxis, :]
                    
                # Predict action
                act_p2, _ = self.opponent_model.predict(norm_obs_p2, deterministic=False)
                action_string_p2 = self._process_opponent_action(act_p2)
            else:
                action_string_p2 = "0000000000" # Idle fallback if no model is loaded

            # --- 3. APPLY DYNAMIC ACTION-MIRRORING FOR PVP FAIRNESS ---
            # Retrieve current positions from the environment parsed properties
            p1_x = self.parser_p1.prev_p1_x
            p2_x = self.parser_p1.prev_p2_x
            
            # If P1 is on the right side of P2, flip P1's absolute Left/Right controls
            if p1_x > p2_x:
                action_string_p1 = self._mirror_left_right(action_string_p1)
                
            # If P2 is on the right side of P1, flip P2's absolute Left/Right controls
            if p2_x > p1_x:
                action_string_p2 = self._mirror_left_right(action_string_p2)

            # Combine actions into 20-bit payload
            full_command = action_string_p1 + action_string_p2 + "\n"
            
            # Send action and receive new payload via socket
            self.send_command(full_command)
            data = self.receive_payload()
            self.latest_raw_payload = data

        except RuntimeError as e:
            print(f"[LeagueEnv][WARN] Socket error: {e}. Returning terminal state.")
            obs = self._get_obs() if len(self.frames) > 0 else np.zeros(TOTAL_OBS_DIM * config.NUM_FRAMES, dtype=np.float32)
            return obs, 0.0, True, False, {"socket_death": True}

        # --- 4. PARSE TELEMETRY PER PLAYER PERSPECTIVE ---
        obs_p1_raw = self.parser_p1.parse(data, is_reset=False)
        obs_p2_raw = self.parser_p2.parse(data, is_reset=False)
        
        # Update frame buffers
        self.frames.append(obs_p1_raw) # Main Gym observation stack
        self.latest_p2_stacked = self.buf_p2.push(obs_p2_raw) # Opponent stacked observation

        # --- 5. CALCULATE REWARD & OUTCOMES ---
        current_my_hp, current_enemy_hp = obs_p1_raw[0], obs_p1_raw[1]
        
        # Clamp RAM glitches to preserve training stability
        damage_clamp = 100
        damage_dealt = min(max(0, self.prev_enemy_hp - current_enemy_hp), damage_clamp)
        damage_taken = min(max(0, self.prev_my_hp - current_my_hp), damage_clamp)

        self._steps += 1
        
        # Footsie potential-based reward shaping
        COMBO_WINDOW = 6
        DAMAGE_TAKEN_PENALTY = 0.70
        FOOTSIE_RANGE_MAX = 80
        FOOTSIE_BASE_REWARD = 0.05
        
        rel_dist = float(obs_p1_raw[9])
        def potential(d):
            return FOOTSIE_BASE_REWARD * max(0.0, 1.0 - d / FOOTSIE_RANGE_MAX)
            
        phi_curr = potential(rel_dist)
        phi_prev = potential(self.prev_rel_dist)
        dist_reward = 0.99 * phi_curr - phi_prev
        self.prev_rel_dist = rel_dist

        if rel_dist <= FOOTSIE_RANGE_MAX:
            self.footsie_steps += 1
        else:
            self.footsie_steps = 0

        if damage_dealt > 0:
            self.footsie_steps = 0
            if self.frames_since_last_hit <= COMBO_WINDOW:
                self.combo_counter += 1
            else:
                self.combo_counter = 1
            self.frames_since_last_hit = 0
            combo_bonus = min(self.combo_counter * 0.5, 4.0)
            
            reward = float(damage_dealt) + combo_bonus - (DAMAGE_TAKEN_PENALTY * float(damage_taken)) + dist_reward
        else:
            self.frames_since_last_hit += 1
            if self.frames_since_last_hit > COMBO_WINDOW:
                self.combo_counter = 0
            reward = -(DAMAGE_TAKEN_PENALTY * float(damage_taken)) - 0.015 + dist_reward

        if current_enemy_hp <= 0: 
            reward += 50.0
        if current_my_hp <= 0: 
            reward -= 50.0

        self.prev_my_hp, self.prev_enemy_hp = current_my_hp, current_enemy_hp

        terminated = bool(current_my_hp <= 0 or current_enemy_hp <= 0) if self.trainable else False
        truncated = bool(self._steps >= config.MAX_STEPS_PER_ROUND) and not terminated

        info = {}
        if terminated or truncated:
            # Output win/loss explicitly for the League Pool Manager to record win rates
            win_outcome = 1 if current_enemy_hp <= 0 and current_my_hp > 0 else 0
            info["win"] = win_outcome
            info["opponent_id"] = self.opponent_id

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Standard Gym reset: resets emulator state and primes decoupled buffers on first telemetry payload."""
        obs_p1, info = super().reset(seed=seed, options=options)
        
        # Cold start telemetry
        data = self.latest_raw_payload if self.latest_raw_payload else self.receive_payload()
        
        # Prime parsers
        obs_p1_raw = self.parser_p1.parse(data, is_reset=True)
        obs_p2_raw = self.parser_p2.parse(data, is_reset=True)
        
        # Prime frame buffers
        self.buf_p1.reset(obs_p1_raw)
        self.buf_p2.reset(obs_p2_raw)
        
        self.latest_p2_stacked = self.buf_p2.push(obs_p2_raw)
        
        # Parse P1's initial variables
        self.prev_my_hp = float(obs_p1_raw[0]) if obs_p1_raw[0] > 0 else 176.0
        self.prev_enemy_hp = float(obs_p1_raw[1]) if obs_p1_raw[1] > 0 else 176.0
        self.prev_rel_dist = float(obs_p1_raw[9])
        
        self._steps = 0
        self.footsie_steps = 0
        self.combo_counter = 0
        self.frames_since_last_hit = 0
        
        # Fill standard SB3 frame stack
        self.frames.clear()
        for _ in range(config.NUM_FRAMES): 
            self.frames.append(obs_p1_raw)
            
        return self._get_obs(), {}
