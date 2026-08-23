# selective_norm.py
import pickle
import numpy as np
import core.config as config
from stable_baselines3.common.vec_env import VecEnvWrapper

class SelectiveVecNormalize(VecEnvWrapper):
    """
    Normalizes only the continuous dimensions of a mixed continuous/one-hot
    observation vector. One-hot dimensions are passed through unchanged.
 
    Now includes Reward Normalization (Fix A) and proper state persistence.
    """
    def __init__(self, venv, n_continuous_dims=config.OBS_DIM, n_frames=config.NUM_FRAMES, 
                 clip=10.0, training=True, norm_reward=True, reward_clip=10.0):
        super().__init__(venv)
        self.n_cont = n_continuous_dims
        self.n_frames = n_frames
        
        obs_shape = venv.observation_space.shape[0]
        if obs_shape % n_frames != 0:
            raise ValueError(
                f"[SelectiveVecNormalize] Observation space dim ({obs_shape}) is not "
                f"divisible by n_frames ({n_frames}). Check your config.NUM_FRAMES!"
            )
            
        self.total_dim_per_frame = obs_shape // n_frames
        self.clip = clip
        self._training = training
        self._norm_reward = norm_reward
        self.reward_clip = reward_clip

        # Observation stats
        self.running_mean = np.zeros(n_continuous_dims, dtype=np.float64)
        self.running_var  = np.ones(n_continuous_dims, dtype=np.float64)
        self.count = 1e-4

        # Reward stats (Fix A)
        self.ret_rms_mean = 0.0
        self.ret_rms_var  = 1.0
        self.ret_rms_count = 1e-4
        self._returns = np.zeros(venv.num_envs, dtype=np.float64)

    def _update_stats(self, obs: np.ndarray):
        if not self._training:
            return

        # Only use the NEWEST frame to avoid 4x duplicate counting
        start_latest = (self.n_frames - 1) * self.total_dim_per_frame
        latest_cont = obs[:, start_latest : start_latest + self.n_cont].astype(np.float64)

        batch_mean = latest_cont.mean(axis=0)
        batch_var  = latest_cont.var(axis=0)
        n = latest_cont.shape[0]

        total = self.count + n
        delta = batch_mean - self.running_mean

        self.running_mean += delta * n / total
        self.running_var   = (
            self.running_var * self.count
            + batch_var * n
            + delta**2 * self.count * n / total
        ) / total
        self.count = total

    def normalize_obs(self, obs, update=True):
        if update:
            self._update_stats(obs)
        std = np.sqrt(self.running_var + 1e-8)
        norm_obs = obs.copy()
        
        for i in range(self.n_frames):
            start = i * self.total_dim_per_frame
            cont  = norm_obs[:, start : start + self.n_cont].astype(np.float64)
            normalized = (cont - self.running_mean) / std
            norm_obs[:, start : start + self.n_cont] = np.clip(
                normalized, -self.clip, self.clip
            ).astype(np.float32)
            
        return norm_obs

    def unnormalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Reverse observation normalization for the continuous features."""
        unnormed = obs.copy()
        std = np.sqrt(self.running_var + 1e-8)
        for i in range(self.n_frames):
            start = i * self.total_dim_per_frame
            cont  = unnormed[:, start : start + self.n_cont].astype(np.float64)
            unnormed_cont = cont * std + self.running_mean
            unnormed[:, start : start + self.n_cont] = unnormed_cont.astype(np.float32)
        return unnormed

    def _normalize_reward(self, rews: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """Discounted return running estimate (Welford online)."""
        # Note: Using a fixed 0.99 gamma for internal return estimation
        self._returns = self._returns * 0.99 + rews
        
        batch_mean = self._returns.mean()
        batch_var  = self._returns.var()
        n = len(self._returns)
        
        total = self.ret_rms_count + n
        delta = batch_mean - self.ret_rms_mean
        
        self.ret_rms_mean += delta * n / total
        self.ret_rms_var = (
            self.ret_rms_var * self.ret_rms_count
            + batch_var * n
            + delta**2 * self.ret_rms_count * n / total
        ) / total
        self.ret_rms_count = total
        
        # Reset returns on episode end
        self._returns[dones.astype(bool)] = 0.0

        std = np.sqrt(self.ret_rms_var + 1e-8)
        return np.clip(rews / std, -self.reward_clip, self.reward_clip).astype(np.float32)

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        obs = self.normalize_obs(obs)
        # Normalize terminal observations stored by VecEnv auto-reset.
        # Without this, PPO computes V(s_T) on unnormalized inputs at
        # episode boundaries, creating a discontinuity in value estimates.
        for i in range(len(dones)):
            if dones[i] and "terminal_observation" in infos[i]:
                terminal_obs = infos[i]["terminal_observation"]
                infos[i]["terminal_observation"] = self.normalize_obs(
                    terminal_obs.reshape(1, -1).astype(np.float32),
                    update=False
                )[0]
        if self._norm_reward and self._training:
            rews = self._normalize_reward(rews, dones)
        return obs, rews, dones, infos

    def reset(self):
        obs = self.venv.reset()
        # Reset internal returns tracker on environment reset
        self._returns = np.zeros(self.venv.num_envs, dtype=np.float64)
        return self.normalize_obs(obs)
    
    def save(self, path: str):
        stats = {
            "running_mean":   self.running_mean,
            "running_var":    self.running_var,
            "count":          self.count,
            "n_cont":         self.n_cont,
            "n_frames":       self.n_frames,
            "clip":           self.clip,
            # Reward normalization state (Fix A)
            "ret_rms_mean":   self.ret_rms_mean,
            "ret_rms_var":    self.ret_rms_var,
            "ret_rms_count":  self.ret_rms_count,
            "norm_reward":    self._norm_reward,
            "reward_clip":    self.reward_clip,
        }
        with open(path, "wb") as f:
            pickle.dump(stats, f)
        print(f"[SelectiveVecNormalize] Stats saved -> {path}")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    @classmethod
    def load(cls, path: str, venv):
        with open(path, "rb") as f:
            stats = pickle.load(f)
            
        obs_shape = venv.observation_space.shape[0]
        n_frames = stats["n_frames"]
        if obs_shape % n_frames != 0:
            raise ValueError(
                f"[SelectiveVecNormalize] Loaded n_frames ({n_frames}) is incompatible "
                f"with current environment observation space ({obs_shape})."
            )
            
        wrapper = cls(
            venv,
            n_continuous_dims=stats["n_cont"],
            n_frames=n_frames,
            clip=stats["clip"],
            norm_reward=stats.get("norm_reward", False), # Default for old pkls
            reward_clip=stats.get("reward_clip", 10.0),
        )
        wrapper.running_mean  = stats["running_mean"]
        wrapper.running_var   = stats["running_var"]
        wrapper.count         = stats["count"]
        # Restore reward stats
        wrapper.ret_rms_mean  = stats.get("ret_rms_mean", 0.0)
        wrapper.ret_rms_var   = stats.get("ret_rms_var", 1.0)
        wrapper.ret_rms_count = stats.get("ret_rms_count", 1e-4)
        print(f"[SelectiveVecNormalize] Stats loaded <- {path}")
        return wrapper
    
    @property
    def training(self):
        return self._training
 
    @training.setter
    def training(self, value):
        self._training = value
 
    @property
    def norm_reward(self):
        return self._norm_reward
 
    @norm_reward.setter
    def norm_reward(self, value: bool):
        self._norm_reward = value
