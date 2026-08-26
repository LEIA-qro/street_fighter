# policy.py -- deterministic pure-numpy MLP policy for ES evaluation.
#
# Operates on the v4-style observation: 23 floats per frame (see
# envs/sf2_v4.py's frame layout) stacked NUM_FRAMES=4 deep = 92 floats.
# Architecture [92 -> 64 -> 64 -> 63], tanh hidden, argmax over the 63
# logits, decoded to the MultiDiscrete([9, 7]) action via divmod(a, 7).
#
# Torch-free on purpose: ES workers evaluate thousands of perturbed policies
# and must run on machines with nothing but numpy + stable-retro installed.
# ES needs no gradients, so a ~14k-parameter forward in numpy is both the
# simplest and the fastest option at this scale (BLAS matvec, no autograd
# bookkeeping, no per-call tensor allocation churn).

import numpy as np

OBS_FRAME_DIM = 23
NUM_FRAMES = 4
OBS_DIM = OBS_FRAME_DIM * NUM_FRAMES  # 92

HIDDEN_DIM = 64
N_MOVE = 9
N_ATTACK = 7
N_LOGITS = N_MOVE * N_ATTACK  # 63

# Per-frame input scale, 1/max(|low|, |high|) of the v4 Box bounds
# (envs/sf2_v4.py cont_low/cont_high + flags + ID highs, ACT_CATEGORIES=256,
# CHAR_CATEGORIES=16). Without this, raw channels reaching 500 saturate the
# first tanh layer for any reasonably-scaled weight vector and ES spends its
# whole budget crawling out of the flat region. Kept as a literal copy rather
# than importing envs.sf2_v4 so this module stays importable with numpy alone;
# test_es_core.py cross-checks nothing here (the env bounds are another
# track's contract), but the layout comment above pins the correspondence.
_FRAME_ABS_HIGH = np.array([
    176., 176., 500., 200., 250., 500., 500., 100., 100., 187.,  # continuous
    255., 192., 192.,                                            # rel_y/heads
    1., 1.,                                                      # airborne
    255., 255., 255., 255., 255., 255.,                          # act/btn IDs
    15., 15.,                                                    # char IDs
], dtype=np.float32)
OBS_SCALE = np.tile(1.0 / _FRAME_ABS_HIGH, NUM_FRAMES).astype(np.float32)

# Flat-pack layout: W1(64,92) b1(64) W2(64,64) b2(64) W3(63,64) b3(63).
_SHAPES = [
    (HIDDEN_DIM, OBS_DIM), (HIDDEN_DIM,),
    (HIDDEN_DIM, HIDDEN_DIM), (HIDDEN_DIM,),
    (N_LOGITS, HIDDEN_DIM), (N_LOGITS,),
]
NUM_PARAMS = sum(int(np.prod(s)) for s in _SHAPES)  # 14207


def init_flat(seed):
    """Fresh flat parameter vector: W ~ N(0, 1/fan_in), biases zero.

    Deterministic in the seed so every machine can reconstruct generation
    zero's mean from the master seed alone.
    """
    rng = np.random.default_rng(seed)
    parts = []
    for shape in _SHAPES:
        if len(shape) == 2:
            std = 1.0 / np.sqrt(shape[1])
            parts.append(rng.standard_normal(shape, dtype=np.float32).ravel() * std)
        else:
            parts.append(np.zeros(shape, dtype=np.float32))
    return np.concatenate(parts).astype(np.float32)


class MLPPolicy:
    """Deterministic MLP over the 92-float stacked v4 observation."""

    def __init__(self, flat=None):
        self._params = None  # unpacked views, rebuilt by set_flat
        self._flat = None
        self.set_flat(init_flat(0) if flat is None else flat)

    def get_flat(self):
        return self._flat.copy()

    def set_flat(self, flat):
        flat = np.asarray(flat, dtype=np.float32)
        if flat.shape != (NUM_PARAMS,):
            raise ValueError(f"expected flat vector of shape ({NUM_PARAMS},), got {flat.shape}")
        self._flat = flat.copy()
        params, offset = [], 0
        for shape in _SHAPES:
            size = int(np.prod(shape))
            params.append(self._flat[offset:offset + size].reshape(shape))
            offset += size
        self._params = params

    def act(self, obs):
        """obs (92,) float-like -> np.array([move 0-8, attack 0-6], int64)."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"expected obs of shape ({OBS_DIM},), got {obs.shape}")
        w1, b1, w2, b2, w3, b3 = self._params
        h = np.tanh(w1 @ (obs * OBS_SCALE) + b1)
        h = np.tanh(w2 @ h + b2)
        logits = w3 @ h + b3
        move, attack = divmod(int(np.argmax(logits)), N_ATTACK)
        return np.array([move, attack], dtype=np.int64)
