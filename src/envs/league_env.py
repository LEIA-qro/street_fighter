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
from envs.reward import compute_reward
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
                    custom_objects={"buffer_size": 1, "learning_rate": 0.0, "clip_range": 0.0}
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
        # Same pure reward module and sentinel discipline as the single-player
        # path in base_env.step(). The league used to carry its own inline
        # copy of the OLD reward (dead-zone potential, hardcoded 0.99 discount,
        # -0.015/step, +/-50 terminals, no sentinel handling) -- self-play was
        # training against the exact bug the sf2-sota-rl-upgrade branch fixed.
        current_my_hp, current_enemy_hp = float(obs_p1_raw[0]), float(obs_p1_raw[1])
        rel_dist = float(obs_p1_raw[9])
        self._steps += 1

        if rel_dist <= self.reward_cfg.peak_dist:
            self.footsie_steps += 1
        else:
            self.footsie_steps = 0

        # SAME round rules as base_env.step and RetroSF2Env.step, via the same
        # shared objects -- self-play used to carry its own inline copy of the
        # OLD `hp <= 0 and not hp_sentinel` test, which is how it kept training
        # against a bug the single-player path had already fixed. Twice.
        #
        # The flags read here are the RAW p1/p2 ones, NOT self.my_ko /
        # self.enemy_ko: _PerspectiveParser drives one shared env through
        # _parse_payload twice, so every perspective-flipped attribute holds
        # whichever parse ran LAST (P2's). The reward below is P1's, so using
        # the flipped copies would invert every win and loss in the league.
        blanked = bool(current_my_hp == 0 and current_enemy_hp == 0)
        hp_readable = not (self.hp_sentinel or blanked)
        my_ko, enemy_ko = self._round.resolve(
            self.p1_ko, self.p2_ko,
            my_hp=current_my_hp, enemy_hp=current_enemy_hp,
            hp_readable=hp_readable,
            matches_won=self.p1_matches_won,
            enemy_matches_won=self.p2_matches_won,
            timer=self.round_timer)
        ko = bool(my_ko or enemy_ko)
        unreadable = bool(not hp_readable and not ko)

        if not unreadable:
            self._ep_rel_dists.append(rel_dist)

        if unreadable:
            # HP is not a health value this frame (round transition / menu /
            # the [0, 0] blank the ROM paints between rounds): skip reward and
            # refuse to terminate rather than diffing real HP against a
            # fabricated zero. Without this, a single-sided sentinel
            # fabricated a -50 "loss" out of a menu frame in every league
            # episode, and the [0, 0] blank scored as a DRAW.
            reward, reward_parts = 0.0, {}
        else:
            reward, self.reward_state, reward_parts = compute_reward(
                self.reward_state, current_my_hp, current_enemy_hp,
                rel_dist, ko, self.reward_cfg,
                my_ko=my_ko, enemy_ko=enemy_ko,
            )
            self.prev_my_hp = self.reward_state.prev_my_hp
            self.prev_enemy_hp = self.reward_state.prev_enemy_hp
            self.prev_rel_dist = self.reward_state.prev_rel_dist
            self.combo_counter = self.reward_state.combo_counter
            self.frames_since_last_hit = self.reward_state.frames_since_last_hit

        terminated = ko if self.trainable else False
        truncated = (bool(self._steps >= config.MAX_STEPS_PER_ROUND) and not terminated) if self.trainable else False

        info = {
            "my_hp": current_my_hp,
            "enemy_hp": current_enemy_hp,
            "hp_sentinel": self.hp_sentinel,
            "reward_parts": reward_parts,
        }
        if terminated or truncated:
            draw = bool(terminated and my_ko and enemy_ko)
            info["draw"] = draw
            info["double_ko"] = draw   # legacy alias, same event
            info["timeout"] = bool(truncated)
            info["episode_steps"] = self._steps
            info["win"] = 1 if (terminated and enemy_ko and not my_ko) else 0
            info["loss"] = 1 if (terminated and my_ko and not enemy_ko) else 0
            info["matches_won_delta"] = self._round.mw_delta
            info["enemy_matches_won_delta"] = self._round.emw_delta
            info["time_over"] = bool(terminated
                                     and not (self.p1_ko or self.p2_ko))
            info["opponent_id"] = self.opponent_id
            if hasattr(self, "current_state_file"):
                info["state_file"] = self.current_state_file
            self._attach_episode_spacing(info)

        return self._get_obs(), reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Standard Gym reset: resets emulator state and primes decoupled buffers on first telemetry payload."""
        obs_p1, info = super().reset(seed=seed, options=options)

        # The base reset already drained the stale in-flight payload and read
        # the real post-savestate-load frame; re-parse that same frame from
        # both perspectives. (The old code re-received on cold start -- which
        # stole the first step's payload -- and reused the PREVIOUS episode's
        # final payload on every later reset.)
        data = self._last_reset_payload
        self.latest_raw_payload = data

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
        # super().reset() already baselined the round tracker, but it did so
        # from whatever perspective self.player happened to hold; the two
        # re-parses above left it at P2. Re-baseline in RAW p1/p2 order, which
        # is the order step() resolves in.
        self._round.reset(matches_won=self.p1_matches_won,
                          enemy_matches_won=self.p2_matches_won,
                          timer=self.round_timer,
                          ko=bool(self.p1_ko or self.p2_ko))

        # Fill standard SB3 frame stack
        self.frames.clear()
        for _ in range(config.NUM_FRAMES): 
            self.frames.append(obs_p1_raw)
            
        return self._get_obs(), {}
