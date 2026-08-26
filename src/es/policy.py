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

def _shapes_for(in_dim):
    """Flat-pack layout: W1(64,in) b1(64) W2(64,64) b2(64) W3(63,64) b3(63)."""
    return [
        (HIDDEN_DIM, in_dim), (HIDDEN_DIM,),
        (HIDDEN_DIM, HIDDEN_DIM), (HIDDEN_DIM,),
        (N_LOGITS, HIDDEN_DIM), (N_LOGITS,),
    ]


class MLPPolicy:
    """Deterministic MLP over the 92-float stacked v4 observation.

    Subclasses change only the FEATURE map (what the first layer sees), never
    the wire observation: act() always takes the raw (92,) env obs, and
    IN_DIM is the post-feature width the flat parameter vector is sized for.
    """

    IN_DIM = OBS_DIM  # width after _features(); the first layer's fan-in

    @classmethod
    def shapes(cls):
        return _shapes_for(cls.IN_DIM)

    @classmethod
    def num_params(cls):
        return sum(int(np.prod(s)) for s in cls.shapes())

    @classmethod
    def init_flat(cls, seed):
        """Fresh flat parameter vector: W ~ N(0, 1/fan_in), biases zero.

        Deterministic in the seed so every machine can reconstruct generation
        zero's mean from the master seed alone.
        """
        rng = np.random.default_rng(seed)
        parts = []
        for shape in cls.shapes():
            if len(shape) == 2:
                std = 1.0 / np.sqrt(shape[1])
                parts.append(rng.standard_normal(shape, dtype=np.float32).ravel() * std)
            else:
                parts.append(np.zeros(shape, dtype=np.float32))
        return np.concatenate(parts).astype(np.float32)

    def __init__(self, flat=None):
        self._params = None  # unpacked views, rebuilt by set_flat
        self._flat = None
        self.set_flat(self.init_flat(0) if flat is None else flat)

    def get_flat(self):
        return self._flat.copy()

    def set_flat(self, flat):
        flat = np.asarray(flat, dtype=np.float32)
        expected = self.num_params()
        if flat.shape != (expected,):
            raise ValueError(f"expected flat vector of shape ({expected},), got {flat.shape}")
        self._flat = flat.copy()
        params, offset = [], 0
        for shape in self.shapes():
            size = int(np.prod(shape))
            params.append(self._flat[offset:offset + size].reshape(shape))
            offset += size
        self._params = params

    def _features(self, obs):
        """Raw (92,) env obs -> what the first layer sees. Base: scaled as-is."""
        return obs * OBS_SCALE

    def act(self, obs):
        """obs (92,) float-like -> np.array([move 0-8, attack 0-6], int64)."""
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (OBS_DIM,):
            raise ValueError(f"expected obs of shape ({OBS_DIM},), got {obs.shape}")
        w1, b1, w2, b2, w3, b3 = self._params
        h = np.tanh(w1 @ self._features(obs) + b1)
        h = np.tanh(w2 @ h + b2)
        logits = w3 @ h + b3
        move, attack = divmod(int(np.argmax(logits)), N_ATTACK)
        return np.array([move, attack], dtype=np.int64)


# --- char-conditioned variant ------------------------------------------------
# Evidence (bench 12 rivales, gens 64 vs 74): the scalar-char policy CYCLES
# matchup strategies instead of accumulating them -- it cracked Balrog/Ken/
# Sagat while losing Bison/Ryu/Vega, clean win rate pinned at 8/12 both times.
# The rival's character ID enters the v4 frame as ONE scalar (index 22, /15
# after OBS_SCALE), so a 64-unit tanh MLP cannot branch per matchup: rivals
# 0.07 apart on one input axis get nearly the same policy. The PPO baseline
# never had this problem -- its v3 layout one-hots both character IDs.

N_CHARS = 16
_CHAR_SLOTS = (21, 22)                      # p1_char, p2_char in the v4 frame
_CONT_DIM = OBS_FRAME_DIM - len(_CHAR_SLOTS)  # 21 non-char floats per frame
ONEHOT_FRAME_DIM = _CONT_DIM + len(_CHAR_SLOTS) * N_CHARS  # 53
ONEHOT_OBS_DIM = ONEHOT_FRAME_DIM * NUM_FRAMES             # 212
_CONT_SCALE = (1.0 / _FRAME_ABS_HIGH[:_CONT_DIM]).astype(np.float32)


def expand_char_onehot(obs):
    """(92,) v4 obs -> (212,): per frame, 21 scaled floats + two 16-way one-hots.

    Character IDs are already integers 0-15 in the frame; np.clip guards a
    corrupt read from indexing out of range rather than crashing an episode.
    """
    obs = np.asarray(obs, dtype=np.float32).reshape(NUM_FRAMES, OBS_FRAME_DIM)
    out = np.zeros((NUM_FRAMES, ONEHOT_FRAME_DIM), dtype=np.float32)
    out[:, :_CONT_DIM] = obs[:, :_CONT_DIM] * _CONT_SCALE
    ids = np.clip(obs[:, _CHAR_SLOTS].astype(np.int64), 0, N_CHARS - 1)
    rows = np.arange(NUM_FRAMES)
    out[rows, _CONT_DIM + ids[:, 0]] = 1.0
    out[rows, _CONT_DIM + N_CHARS + ids[:, 1]] = 1.0
    return out.reshape(-1)


class CharOneHotPolicy(MLPPolicy):
    """MLPPolicy whose first layer sees one-hot character IDs (matchup branching)."""

    IN_DIM = ONEHOT_OBS_DIM

    def _features(self, obs):
        return expand_char_onehot(obs)


# Wire registry: the name a coordinator pins in its checkpoint and serves in
# /theta ("policy" key); workers construct by name. Absent key = "v4", which
# is every run that predates the registry.
POLICIES = {"v4": MLPPolicy, "v4onehot": CharOneHotPolicy}
DEFAULT_POLICY = "v4"

# Backward-compatible module-level API (pre-registry callers and tests).
_SHAPES = MLPPolicy.shapes()
NUM_PARAMS = MLPPolicy.num_params()  # 14207


def init_flat(seed):
    return MLPPolicy.init_flat(seed)
