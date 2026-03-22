# selective_norm.py
import pickle
import numpy as np
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
    def __init__(self, venv, n_continuous_dims=10, n_frames=4, clip=10.0, training=True):
        super().__init__(venv)
        self.n_cont = n_continuous_dims
        self.n_frames = n_frames
        self.total_dim_per_frame = venv.observation_space.shape[0] // n_frames # e.g. 2216 // 4 = 554
        self.clip = clip
        self._training = training  # Rename internal flag
        self.running_mean = np.zeros(n_continuous_dims, dtype=np.float64)
        self.running_var  = np.ones(n_continuous_dims, dtype=np.float64)
        self.count = 1e-4 # Small seed to avoid div-by-zero on first update

    def _update_stats(self, obs: np.ndarray):
        if not self._training:   # Guard: skip updates when frozen
            return
        # Extract only the continuous slices from all stacked frames
        cont_data = []
        for i in range(self.n_frames):
            start = i * self.total_dim_per_frame
            cont_data.append(obs[:, start : start + self.n_cont].astype(np.float64))
        
        cont_combined = np.concatenate(cont_data, axis=0)
        batch_mean = cont_combined.mean(axis=0)
        batch_var  = cont_combined.var(axis=0)
        n = cont_combined.shape[0]
        
        total = self.count + n
        delta = batch_mean - self.running_mean

        self.running_mean += delta * n / total
        self.running_var   = (self.running_var * self.count + batch_var * n + delta**2 * self.count * n / total) / total
        self.count = total

    # selective_norm.py — normalize_obs — obs is now float32, assignment works correctly
    def normalize_obs(self, obs):
        self._update_stats(obs)
        for i in range(self.n_frames):
            start = i * self.total_dim_per_frame
            cont  = obs[:, start : start + self.n_cont].astype(np.float64)
            cont  = (cont - self.running_mean) / np.sqrt(self.running_var + 1e-8)
            obs[:, start : start + self.n_cont] = np.clip(
                cont, -self.clip, self.clip
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
        print(f"[SelectiveVecNormalize] Stats saved → {path}")

    def env_method(self, method_name, *method_args, indices=None, **method_kwargs):
        """Explicitly delegate env_method calls down to the vectorized environment."""
        return self.venv.env_method(
            method_name, *method_args, indices=indices, **method_kwargs
        )

    @classmethod
    def load(cls, path: str, venv):
        with open(path, "rb") as f:
            stats = pickle.load(f)
        wrapper = cls(
            venv,
            n_continuous_dims=stats["n_cont"],
            n_frames=stats["n_frames"],
            clip=stats["clip"],
        )
        wrapper.running_mean = stats["running_mean"]
        wrapper.running_var  = stats["running_var"]
        wrapper.count        = stats["count"]
        print(f"[SelectiveVecNormalize] Stats loaded ← {path}")
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