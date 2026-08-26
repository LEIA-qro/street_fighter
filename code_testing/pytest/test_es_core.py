# test_es_core.py
#
# Offline unit tests for the distributed ES harness (src/es/). No sockets,
# no emulator, no ROM: the coordinator/worker split is designed so that all
# the correctness-critical math (seed reconstruction, rank shaping, Adam,
# checkpoint resume) and the fault-tolerance logic (chunk leasing) are pure
# functions/classes testable exactly like this.

import os
import sys
from pathlib import Path

PROJECT_ROOT = str(Path(__file__).resolve().parents[2])
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

import numpy as np
import pytest

from es import openes, protocol
from es.openes import (
    ESState, centered_ranks, es_update, fitness_from_episode, init_state,
    load_checkpoint, member_theta, members_for_generation,
    pair_seeds_for_generation, perturbation, save_checkpoint,
)
from es.policy import MLPPolicy, NUM_PARAMS, N_MOVE, N_ATTACK, OBS_DIM, init_flat
from es.protocol import ChunkQueue, decode_theta, encode_theta, make_chunks


def _toy_state(dim=16, **overrides):
    kwargs = dict(sigma=0.1, lr=0.05, weight_decay=0.0, master_seed=1234)
    kwargs.update(overrides)
    return init_state(np.zeros(dim, dtype=np.float32), **kwargs)


# --------------------------------------------------------------------------
# Mirrored sampling: the wire carries integer seeds only, so a worker on a
# different machine (fresh interpreter, fresh rng) MUST rebuild the exact
# float32 perturbation the coordinator will use in its update.
# --------------------------------------------------------------------------

def test_two_machines_reconstruct_bitwise_identical_perturbations():
    seeds = pair_seeds_for_generation(master_seed=42, generation=7, n_pairs=4)
    # simulate two machines: two completely independent reconstructions
    machine_a = [perturbation(NUM_PARAMS, s) for s in seeds]
    machine_b = [perturbation(NUM_PARAMS, s) for s in seeds]
    for eps_a, eps_b in zip(machine_a, machine_b):
        assert eps_a.dtype == np.float32
        assert np.array_equal(eps_a, eps_b)  # bitwise, not approx


def test_antithetic_members_share_a_seed_with_opposite_signs():
    state = _toy_state()
    members = members_for_generation(state, pop_size=8)
    assert [i for i, _s, _g in members] == list(range(8))
    for p in range(4):
        idx_pos, seed_pos, sign_pos = members[2 * p]
        idx_neg, seed_neg, sign_neg = members[2 * p + 1]
        assert seed_pos == seed_neg
        assert (sign_pos, sign_neg) == (1, -1)
    th_pos = member_theta(state, members[0][1], members[0][2])
    th_neg = member_theta(state, members[1][1], members[1][2])
    assert np.array_equal(th_pos, -th_neg)  # theta is zeros, so exact mirror


def test_generations_and_master_seeds_get_distinct_seed_sets():
    a = pair_seeds_for_generation(42, 0, 8)
    b = pair_seeds_for_generation(42, 1, 8)
    c = pair_seeds_for_generation(43, 0, 8)
    assert set(a) != set(b)
    assert set(a) != set(c)


def test_odd_pop_size_is_rejected():
    with pytest.raises(ValueError):
        members_for_generation(_toy_state(), pop_size=7)


# --------------------------------------------------------------------------
# Centered-rank shaping (Salimans 2017 sec 2.1)
# --------------------------------------------------------------------------

def test_centered_ranks_are_bounded_zero_mean_and_order_preserving():
    fits = np.array([3.0, -1.0, 100.0, 0.5])
    shaped = centered_ranks(fits)
    assert shaped.dtype == np.float32
    assert shaped.min() == pytest.approx(-0.5)
    assert shaped.max() == pytest.approx(0.5)
    assert shaped.sum() == pytest.approx(0.0, abs=1e-6)
    assert np.array_equal(np.argsort(shaped), np.argsort(fits, kind="stable"))


def test_centered_ranks_ignore_monotone_transforms_of_fitness():
    fits = np.array([0.2, -3.0, 7.0, 1.1, 0.0])
    # the shaping must make the update invariant to e.g. reward rescaling
    assert np.array_equal(centered_ranks(fits), centered_ranks(fits * 1000.0 + 5.0))
    assert np.array_equal(centered_ranks(fits), centered_ranks(np.tanh(fits)))


# --------------------------------------------------------------------------
# Update determinism + checkpoint resume: same seeds and fitnesses must give
# the same theta whether computed twice in one process or across a
# save->load boundary (this is what makes a coordinator crash recoverable).
# --------------------------------------------------------------------------

def test_es_update_is_deterministic():
    fits = np.random.default_rng(0).normal(size=32)
    s1 = es_update(_toy_state(), fits)
    s2 = es_update(_toy_state(), fits)
    assert np.array_equal(s1.theta, s2.theta)
    assert np.array_equal(s1.adam_m, s2.adam_m)
    assert np.array_equal(s1.adam_v, s2.adam_v)
    assert s1.generation == s2.generation == 1
    assert s1.adam_t == s2.adam_t == 1


def test_update_moves_theta_and_advances_generation():
    state = _toy_state()
    fits = np.random.default_rng(1).normal(size=32)
    new = es_update(state, fits)
    assert not np.array_equal(new.theta, state.theta)
    assert np.array_equal(state.theta, np.zeros(16, dtype=np.float32))  # input untouched


def test_checkpoint_roundtrip_gives_identical_next_update(tmp_path):
    state = _toy_state(dim=32)
    fits0 = np.random.default_rng(2).normal(size=16)
    state = es_update(state, fits0)  # non-trivial Adam moments before saving

    base = str(tmp_path / "gen_000001")
    save_checkpoint(state, base)
    restored = load_checkpoint(base)
    assert np.array_equal(restored.theta, state.theta)
    assert restored.generation == state.generation
    assert restored.adam_t == state.adam_t

    fits1 = np.random.default_rng(3).normal(size=16)
    assert np.array_equal(es_update(state, fits1).theta, es_update(restored, fits1).theta)


def test_latest_checkpoint_picks_the_highest_generation(tmp_path):
    for g in (1, 2, 10):
        save_checkpoint(_toy_state(), str(tmp_path / f"gen_{g:06d}"))
    assert openes.latest_checkpoint(str(tmp_path)).endswith("gen_000010")


def test_json_without_npz_is_not_a_checkpoint(tmp_path):
    # crash-mid-save leaves at most an orphan file; resume must skip it
    save_checkpoint(_toy_state(), str(tmp_path / "gen_000001"))
    (tmp_path / "gen_000002.json").write_text("{}")
    assert openes.latest_checkpoint(str(tmp_path)).endswith("gen_000001")


# --------------------------------------------------------------------------
# Fitness definition (shared by every worker via openes.fitness_from_episode)
# --------------------------------------------------------------------------

def test_fitness_rewards_wins_hp_margin_and_speed():
    win_fast = fitness_from_episode({"win": 1, "my_hp": 176.0, "enemy_hp": 0.0}, steps=500)
    win_slow = fitness_from_episode({"win": 1, "my_hp": 176.0, "enemy_hp": 0.0}, steps=3000)
    close_loss = fitness_from_episode({"win": 0, "my_hp": 0.0, "enemy_hp": 10.0}, steps=500)
    assert win_fast == pytest.approx(1.0 + 0.5 - 0.05)
    assert win_fast > win_slow > close_loss
    # a win always beats a loss regardless of margins/steps within one round
    worst_win = fitness_from_episode({"win": 1, "my_hp": 1.0, "enemy_hp": 0.0}, steps=5000)
    best_loss = fitness_from_episode({"win": 0, "my_hp": 175.0, "enemy_hp": 176.0}, steps=1)
    assert worst_win > best_loss


# --------------------------------------------------------------------------
# Chunk leasing: the fault-tolerance core, factored into a pure class with an
# injectable clock precisely so it can be tested without sockets or sleeps.
# --------------------------------------------------------------------------

def _queue(lease_seconds=10.0):
    chunks = {"g000000-c000": [[0, 111, 1]], "g000000-c001": [[1, 111, -1]]}
    return ChunkQueue(chunks, lease_seconds, clock=lambda: 0.0)


def test_lease_hands_out_each_chunk_once():
    q = _queue()
    assert q.lease(now=0.0) is not None
    assert q.lease(now=0.0) is not None
    assert q.lease(now=0.0) is None  # both leased, none expired


def test_expired_lease_is_requeued_and_stealable():
    q = _queue(lease_seconds=10.0)
    cid, _payload, deadline = q.lease(now=0.0)
    assert deadline == pytest.approx(10.0)
    assert q.lease(now=9.9)[0] != cid          # not expired yet: other chunk
    stolen = q.lease(now=20.0)                 # both leases now expired
    assert stolen is not None
    assert q.done is False


def test_first_result_wins_and_duplicates_are_dropped():
    q = _queue(lease_seconds=10.0)
    cid, _p, _d = q.lease(now=0.0)
    q.lease(now=20.0)  # expiry requeues cid; a thief may re-lease it
    assert q.complete(cid, {"0": 1.0}) is True     # original worker reports late
    assert q.complete(cid, {"0": 99.0}) is False   # thief's duplicate dropped
    assert q.results[cid] == {"0": 1.0}
    # the requeued copy of cid must not be leasable after completion
    remaining = q.lease(now=50.0)
    assert remaining is None or remaining[0] != cid


def test_queue_done_when_all_chunks_completed():
    q = _queue()
    leases = [q.lease(now=0.0) for _ in range(2)]
    for cid, members, _deadline in leases:
        # complete() only accepts results covering EXACTLY the chunk's member
        # set (malformed submissions from stale fleet workers are rejected).
        q.complete(cid, {int(m[0]): 0.0 for m in members})
    assert q.done
    assert q.lease(now=100.0) is None


def test_queue_rejects_result_not_covering_the_chunks_members():
    q = _queue()
    cid, members, _deadline = q.lease(now=0.0)
    full = {int(m[0]): 0.0 for m in members}
    truncated = dict(list(full.items())[:-1])
    assert q.complete(cid, truncated) is False        # missing a member
    assert q.complete(cid, {**full, 9999: 1.0}) is False  # foreign index
    assert not q.done
    assert q.complete(cid, full) is True              # exact cover accepted


def test_unknown_chunk_id_is_refused():
    assert _queue().complete("g999999-c999", {}) is False


def test_make_chunks_partitions_members_and_embeds_generation():
    members = [(i, 1000 + i // 2, 1 if i % 2 == 0 else -1) for i in range(10)]
    chunks = make_chunks(members, chunk_size=4, generation=12)
    assert sorted(chunks) == ["g000012-c000", "g000012-c001", "g000012-c002"]
    flat = [m for cid in sorted(chunks) for m in chunks[cid]]
    assert [m[0] for m in flat] == list(range(10))
    assert len(chunks["g000012-c002"]) == 2  # remainder chunk


# --------------------------------------------------------------------------
# Theta wire encoding
# --------------------------------------------------------------------------

def test_theta_survives_the_wire_encoding_bitwise():
    theta = init_flat(5)
    version, decoded = decode_theta(encode_theta(theta, 3))
    assert version == 3
    assert decoded.dtype == np.float32
    assert np.array_equal(decoded, theta)


# --------------------------------------------------------------------------
# Policy: shapes, dtypes, ranges, determinism
# --------------------------------------------------------------------------

def test_policy_act_shape_dtype_and_range():
    pol = MLPPolicy(init_flat(7))
    rng = np.random.default_rng(0)
    for _ in range(50):
        obs = rng.uniform(-200, 500, size=OBS_DIM).astype(np.float32)
        action = pol.act(obs)
        assert action.shape == (2,)
        assert action.dtype == np.int64
        assert 0 <= action[0] < N_MOVE
        assert 0 <= action[1] < N_ATTACK


def test_policy_is_deterministic_and_flat_roundtrips():
    flat = init_flat(11)
    assert flat.shape == (NUM_PARAMS,)
    assert flat.dtype == np.float32
    a, b = MLPPolicy(flat), MLPPolicy(flat)
    assert np.array_equal(a.get_flat(), flat)
    obs = np.random.default_rng(1).uniform(-1, 1, size=OBS_DIM).astype(np.float32)
    assert np.array_equal(a.act(obs), b.act(obs))
    a.set_flat(np.zeros(NUM_PARAMS, dtype=np.float32))
    assert np.array_equal(b.get_flat(), flat)  # instances do not share storage


def test_policy_rejects_wrong_sizes():
    pol = MLPPolicy(init_flat(0))
    with pytest.raises(ValueError):
        pol.set_flat(np.zeros(10, dtype=np.float32))
    with pytest.raises(ValueError):
        pol.act(np.zeros(23, dtype=np.float32))


# --------------------------------------------------------------------------
# End-to-end (in memory, no HTTP): a full ES loop on a toy quadratic must
# actually optimise. This is the test that catches sign errors anywhere in
# the sampling -> shaping -> gradient -> Adam chain, which the unit tests
# above cannot see in isolation.
# --------------------------------------------------------------------------

def test_full_generation_loop_improves_a_quadratic_objective():
    dim, pop = 12, 64
    target = np.linspace(-1.0, 1.0, dim).astype(np.float32)

    def objective(theta):
        return -float(np.sum((theta - target) ** 2))

    state = _toy_state(dim=dim, sigma=0.1, lr=0.1, master_seed=99)
    start = objective(state.theta)
    for _ in range(50):
        members = members_for_generation(state, pop)
        fits = [objective(member_theta(state, seed, sign)) for _i, seed, sign in members]
        state = es_update(state, fits)
    end = objective(state.theta)
    assert end > start
    assert end > 0.25 * start  # at least 75% of the initial squared error gone
