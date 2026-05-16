# selective_norm.py
import pickle
import numpy as np
import core.config as config
from stable_baselines3.common.vec_env import VecEnvWrapper

class SelectiveVecNormalize(VecEnvWrapper):
    """
    Normalizes only the continuous dimensions of a mixed continuous/one-hot
    observation vector. One-hot dimensions are passed through unchanged.
 
    Saves and loads via pickle (.pkl) to match SB3 VecNormalize conventions
    so resume scripts work without modification.
 
    Args:
        venv:              The wrapped vectorized environment.
        n_continuous_dims: Number of continuous features per frame (default 10).
        n_frames:          Number of stacked frames (default 4).
        clip:              Symmetric clip range after normalization (default 10.0).
    """
    def __init__(self, venv, n_continuous_dims=config.OBS_DIM, n_frames=config.NUM_FRAMES, clip=10.0, training=True):
        super().__init__(venv)
        self.n_cont = n_continuous_dims
        self.n_frames = n_frames
        
        # Bug 6 Fix: Strict dimension validation to prevent silent data corruption
        obs_shape = venv.observation_space.shape[0]
        if obs_shape % n_frames != 0:
            raise ValueError(
                f"[SelectiveVecNormalize] Observation space dim ({obs_shape}) is not "
                f"divisible by n_frames ({n_frames}). Check your config.NUM_FRAMES!"
            )
            
        self.total_dim_per_frame = obs_shape // n_frames
        self.clip = clip
        self._training = training
        self.running_mean = np.zeros(n_continuous_dims, dtype=np.float64)
        self.running_var  = np.ones(n_continuous_dims, dtype=np.float64)
        self.count = 1e-4

    def _update_stats(self, obs: np.ndarray):
        if not self._training:
            return

        # Only use the NEWEST frame (last in the stacked vector) to avoid 4x duplicate counting
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

    def normalize_obs(self, obs):
        self._update_stats(obs)
        
        # Pre-calculate normalization factors for efficiency
        # Variance floor of 1e-8 to prevent NaN/Inf
        std = np.sqrt(self.running_var + 1e-8)
        
        for i in range(self.n_frames):
            start = i * self.total_dim_per_frame
            cont  = obs[:, start : start + self.n_cont].astype(np.float64)
            
            # Vectorized normalization and clipping
            normalized = (cont - self.running_mean) / std
            obs[:, start : start + self.n_cont] = np.clip(
                normalized, -self.clip, self.clip
            ).astype(np.float32)
            
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        return self.normalize_obs(obs), rews, dones, infos

    def reset(self):
        obs = self.venv.reset()
        return self.normalize_obs(obs)
    
    # FIX BUG 3: Standardize Save/Load to .pkl and SB3 conventions
    def save(self, path: str):
        stats = {
            "running_mean": self.running_mean,
            "running_var":  self.running_var,
            "count":        self.count,
            "n_cont":       self.n_cont,
            "n_frames":     self.n_frames,
            "clip":         self.clip,
        }
        with open(path, "wb") as f:
            pickle.dump(stats, f)
        print(f"[SelectiveVecNormalize] Stats saved -> {path}")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Explicitly delegate env_method calls down to the vectorized environment."""
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    @classmethod
    def load(cls, path: str, venv):
        with open(path, "rb") as f:
            stats = pickle.load(f)
            
        # Bug 6 Fix: Verify compatibility between loaded stats and current environment
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
        )
        wrapper.running_mean = stats["running_mean"]
        wrapper.running_var  = stats["running_var"]
        wrapper.count        = stats["count"]
        print(f"[SelectiveVecNormalize] Stats loaded <- {path}")
        return wrapper
    

    # Expose training flag for API parity with VecNormalize
    # (SelectiveVecNormalize always updates stats during step — this flag
    # is a no-op kept for drop-in compatibility with resume script patterns.)
    @property
    def training(self):
        return self._training
 
    @training.setter
    def training(self, value):
        self._training = value   # Now actually respected
 
    # norm_reward parity — reward normalization not implemented here;
    # accepted silently to avoid AttributeError in resume scripts.
    @property
    def norm_reward(self):
        return False
 
    @norm_reward.setter
    def norm_reward(self, value):
        pass