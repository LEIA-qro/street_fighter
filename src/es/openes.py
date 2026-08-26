# openes.py -- OpenAI-ES (Salimans et al. 2017, arXiv:1703.03864) core math.
#
# Everything here is a pure function of (state, seeds, fitnesses); the only
# RNG use is np.random.default_rng(<integer seed>), so a worker on another
# machine reconstructs a perturbation bit-identically from the seed alone --
# the wire never carries perturbation tensors, only integers (a 14k-float
# eps is ~56KB; a seed is 8 bytes; at pop 256 that is the difference between
# 14MB/generation of tensor traffic and nothing).
#
# Antithetic/mirrored sampling: population member i belongs to pair i//2 and
# contributes sign (+1 if i even else -1) times that pair's perturbation.
# Centered-rank shaping makes the update invariant to any monotone transform
# of raw fitness, which is what lets the fitness definition below evolve
# without retuning lr/sigma.

import glob
import json
import os
from dataclasses import dataclass, replace

import numpy as np

ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8

MAX_HP = 176.0  # full Genesis SF2 health bar; matches base_env.py's 176 sentinel


@dataclass
class ESState:
    """Everything needed to deterministically continue the optimisation.

    Two runs resumed from the same ESState and fed the same fitnesses
    produce bit-identical thetas: seeds derive from (master_seed,
    generation), Adam moments are carried explicitly, and no hidden global
    RNG is consulted anywhere.
    """
    theta: np.ndarray       # float32 (dim,) -- the ES mean
    sigma: float            # perturbation std
    lr: float               # Adam step size on the mean
    weight_decay: float     # L2 pull toward zero, applied to the gradient
    master_seed: int        # root of the whole seed tree
    generation: int         # next generation to be evaluated
    adam_m: np.ndarray      # float32 (dim,)
    adam_v: np.ndarray      # float32 (dim,)
    adam_t: int             # Adam step counter
    # Savestate rotation the run trains over (None = each worker's default
    # state, the pre-rotation behaviour). Part of the run's identity exactly
    # like sigma/lr: which opponents the fitnesses were measured against is
    # not a knob you can flip mid-run without invalidating the comparison.
    states: tuple = None
    # Which policy architecture theta parameterizes (key into policy.POLICIES).
    # Run identity like everything above: theta's very SHAPE depends on it.
    policy: str = "v4"


def normalize_states(states):
    """State list from any source (CLI, json.load) -> canonical tuple or None.

    One canonical form makes the resume identity check a plain equality:
    argparse hands lists, json hands lists, ESState carries tuples, and
    ['A'] != ('A',) would otherwise "differ" on every legitimate resume.
    """
    if not states:
        return None
    return tuple(str(s) for s in states)


def init_state(theta, sigma, lr, weight_decay, master_seed, states=None, policy="v4"):
    theta = np.asarray(theta, dtype=np.float32)
    zeros = np.zeros_like(theta)
    return ESState(theta=theta, sigma=float(sigma), lr=float(lr),
                   weight_decay=float(weight_decay), master_seed=int(master_seed),
                   generation=0, adam_m=zeros.copy(), adam_v=zeros.copy(), adam_t=0,
                   states=normalize_states(states), policy=str(policy))


def pair_seeds_for_generation(master_seed, generation, n_pairs):
    """Deterministic per-pair integer seeds for one generation.

    SeedSequence spawn keys give statistically independent streams per
    (master_seed, generation) without any state carried between generations,
    so coordinator and every worker agree on the seeds from two integers.
    """
    ss = np.random.SeedSequence(entropy=int(master_seed), spawn_key=(int(generation),))
    return [int(s) for s in ss.generate_state(n_pairs, dtype=np.uint64)]


def perturbation(dim, seed):
    """The pair's shared perturbation; float32 so worker/coordinator match."""
    return np.random.default_rng(int(seed)).standard_normal(int(dim), dtype=np.float32)


def members_for_generation(state, pop_size):
    """[(member_idx, pair_seed, sign), ...] for the generation state points at."""
    if pop_size % 2 != 0:
        raise ValueError("pop_size must be even (antithetic pairs)")
    seeds = pair_seeds_for_generation(state.master_seed, state.generation, pop_size // 2)
    return [(i, seeds[i // 2], 1 if i % 2 == 0 else -1) for i in range(pop_size)]


# Child-stream key for the episode->state draw. Any fixed value works; what
# matters is that it is NOT the bare pair seed: perturbation() consumes
# default_rng(pair_seed)'s stream directly, and drawing state indices from
# that same stream start would correlate WHICH opponent an episode gets with
# the perturbation's leading components -- exactly the confound this function
# exists to remove.
_STATE_STREAM_KEY = 0x57A7E


def states_for_member(pair_seed, episodes, n_states):
    """Which state index each of a member's episodes plays: [i0, i1, ...].

    Derived from the PAIR seed -- never the member index or sign -- so both
    halves of an antithetic pair (+eps and -eps) face the identical opponent
    sequence by construction. The pair's fitness DIFFERENCE then isolates the
    perturbation's effect: the opponent draw cancels out of the gradient
    term (shaped[+] - shaped[-]) instead of confounding it (classic common
    random numbers variance reduction). Pure function shared by the worker
    (to pick the states) and the tests (to assert what it picked).
    """
    if int(n_states) <= 0:
        raise ValueError("n_states must be positive")
    ss = np.random.SeedSequence(entropy=int(pair_seed),
                                spawn_key=(_STATE_STREAM_KEY,))
    rng = np.random.default_rng(ss)
    return [int(i) for i in rng.integers(0, int(n_states), size=int(episodes))]


def member_theta(state, seed, sign):
    """The perturbed parameter vector a worker evaluates for one member."""
    eps = perturbation(state.theta.shape[0], seed)
    return (state.theta + np.float32(sign) * np.float32(state.sigma) * eps).astype(np.float32)


def centered_ranks(fitnesses):
    """Map fitnesses to ranks scaled into [-0.5, 0.5], mean ~0.

    argsort-of-argsort assigns each entry its rank (ties broken by position,
    which is deterministic for a fixed input ordering); dividing by n-1 and
    centering yields the shaping of Salimans et al. sec. 2.1.
    """
    fitnesses = np.asarray(fitnesses, dtype=np.float64)
    n = fitnesses.shape[0]
    if n < 2:
        return np.zeros(n, dtype=np.float32)
    ranks = np.empty(n, dtype=np.float64)
    ranks[np.argsort(fitnesses, kind="stable")] = np.arange(n)
    return (ranks / (n - 1) - 0.5).astype(np.float32)


def es_update(state, fitnesses):
    """One ES step: fitnesses (aligned to member idx order) -> new ESState.

    Seeds are re-derived from (master_seed, generation) rather than taken as
    an argument, so an update can never be computed against seeds other than
    the ones the population was actually evaluated with.
    """
    fitnesses = np.asarray(fitnesses, dtype=np.float64)
    pop_size = fitnesses.shape[0]
    shaped = centered_ranks(fitnesses)

    dim = state.theta.shape[0]
    seeds = pair_seeds_for_generation(state.master_seed, state.generation, pop_size // 2)
    grad = np.zeros(dim, dtype=np.float32)
    for p, seed in enumerate(seeds):
        # members 2p (+eps) and 2p+1 (-eps) share one perturbation draw
        grad += (shaped[2 * p] - shaped[2 * p + 1]) * perturbation(dim, seed)
    grad /= np.float32(pop_size * state.sigma)
    grad -= np.float32(state.weight_decay) * state.theta  # ascent + L2 on the mean

    t = state.adam_t + 1
    m = ADAM_BETA1 * state.adam_m + (1 - ADAM_BETA1) * grad
    v = ADAM_BETA2 * state.adam_v + (1 - ADAM_BETA2) * grad * grad
    m_hat = m / (1 - ADAM_BETA1 ** t)
    v_hat = v / (1 - ADAM_BETA2 ** t)
    theta = state.theta + np.float32(state.lr) * m_hat / (np.sqrt(v_hat) + ADAM_EPS)

    return replace(state, theta=theta.astype(np.float32), generation=state.generation + 1,
                   adam_m=m.astype(np.float32), adam_v=v.astype(np.float32), adam_t=t)


def fitness_from_episode(info, steps):
    """Single episode -> scalar fitness. THE definition, shared by all workers.

    win dominates (1.0), hp margin breaks ties among wins/losses (max 0.5),
    and a tiny step cost prefers faster wins / slower losses without ever
    outweighing either (0.0001 * ~5000-step timeout episode = 0.5 max).
    """
    win = float(info.get("win", 0))
    margin = (float(info.get("my_hp", 0.0)) - float(info.get("enemy_hp", 0.0))) / MAX_HP
    # The step cost is capped at 0.5 so no episode length -- not even the
    # worker's 20k-step hard failsafe, which only fires if the env's own
    # truncation fails -- can outweigh a win (1.0) or invert win/loss order.
    return win * 1.0 + margin * 0.5 - min(0.0001 * float(steps), 0.5)


# ---------------------------------------------------------------------------
# Checkpointing: one .npz (arrays) + one .json sidecar (scalars). The json is
# human-readable at 2am over ssh; the npz keeps float32 bit-exact.
# ---------------------------------------------------------------------------

def save_checkpoint(state, path_base):
    """Write {path_base}.npz + {path_base}.json atomically-ish (npz first)."""
    np.savez_compressed(path_base + ".npz", theta=state.theta,
                        adam_m=state.adam_m, adam_v=state.adam_v)
    meta = {"sigma": state.sigma, "lr": state.lr, "weight_decay": state.weight_decay,
            "master_seed": state.master_seed, "generation": state.generation,
            "adam_t": state.adam_t, "dim": int(state.theta.shape[0]),
            # run identity like sigma/lr: which savestates the fitnesses were
            # measured against (null = workers' default state)
            "states": None if state.states is None else list(state.states),
            "policy": state.policy}
    tmp = path_base + ".json.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    # the .json is the marker of a complete checkpoint: rename last, so a
    # crash mid-save never leaves a json pointing at a half-written npz
    os.replace(tmp, path_base + ".json")


def load_checkpoint(path_base):
    with open(path_base + ".json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    arrays = np.load(path_base + ".npz")
    state = ESState(theta=arrays["theta"].astype(np.float32), sigma=float(meta["sigma"]),
                    lr=float(meta["lr"]), weight_decay=float(meta["weight_decay"]),
                    master_seed=int(meta["master_seed"]), generation=int(meta["generation"]),
                    adam_m=arrays["adam_m"].astype(np.float32),
                    adam_v=arrays["adam_v"].astype(np.float32), adam_t=int(meta["adam_t"]),
                    # .get(): checkpoints written before state rotation existed
                    # have no key at all and resume as single-state runs
                    states=normalize_states(meta.get("states")),
                    # same aging rule: pre-registry checkpoints are all v4
                    policy=str(meta.get("policy", "v4")))
    if state.theta.shape[0] != int(meta["dim"]):
        raise ValueError(f"checkpoint dim mismatch: json says {meta['dim']}, "
                         f"npz has {state.theta.shape[0]}")
    return state


def latest_checkpoint(checkpoint_dir):
    """Path base of the newest complete checkpoint in dir, or None."""
    candidates = []
    for j in glob.glob(os.path.join(checkpoint_dir, "gen_*.json")):
        base = j[:-len(".json")]
        if os.path.exists(base + ".npz"):
            candidates.append(base)
    return max(candidates) if candidates else None  # gen_%06d sorts lexically
