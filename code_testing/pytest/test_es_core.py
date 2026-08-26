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


# --------------------------------------------------------------------------
# S3 restore. The madre is disposable: `terraform apply` on changed user_data
# REPLACES the instance and the local checkpoint dir dies with it. S3 used to
# be write-only, so a replaced madre silently restarted from generation 0 --
# the exact failure the uploads exist to prevent.
# --------------------------------------------------------------------------

class _FakeS3Client:
    def __init__(self, keys):
        self._keys = keys
        self.downloaded = []

    def get_paginator(self, _op):
        keys = self._keys

        class _P:
            def paginate(self, **kwargs):
                prefix = kwargs.get("Prefix", "")
                yield {"Contents": [{"Key": k} for k in keys if k.startswith(prefix)]}
        return _P()

    def download_file(self, bucket, key, dest):
        self.downloaded.append((key, dest))
        with open(dest, "w") as f:
            f.write("x")


def _install_fake_boto3(monkeypatch, client):
    import types
    fake = types.ModuleType("boto3")
    fake.client = lambda *a, **k: client
    monkeypatch.setitem(sys.modules, "boto3", fake)


def test_restore_from_s3_pulls_the_newest_generation(monkeypatch, tmp_path):
    from es import coordinator
    client = _FakeS3Client(["es/gen_000001.npz", "es/gen_000001.json",
                            "es/gen_000012.npz", "es/gen_000012.json",
                            "es/gen_000009.npz", "es/gen_000009.json"])
    _install_fake_boto3(monkeypatch, client)
    coordinator.restore_from_s3("bucket", str(tmp_path))
    # Fixed-width digits make a plain max() the newest generation, not gen_9.
    assert sorted(k for k, _ in client.downloaded) == [
        "es/gen_000012.json", "es/gen_000012.npz"]
    assert (tmp_path / "gen_000012.npz").exists()
    assert (tmp_path / "gen_000012.json").exists()


def test_restore_from_s3_is_a_noop_without_a_bucket(tmp_path):
    from es import coordinator
    coordinator.restore_from_s3(None, str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_restore_from_s3_survives_an_empty_bucket(monkeypatch, tmp_path):
    from es import coordinator
    _install_fake_boto3(monkeypatch, _FakeS3Client([]))
    coordinator.restore_from_s3("bucket", str(tmp_path))
    assert list(tmp_path.iterdir()) == []


def test_restore_from_s3_never_raises(monkeypatch, tmp_path):
    """Any S3 failure must degrade to a fresh start, never kill the madre."""
    from es import coordinator

    class _Boom:
        def get_paginator(self, _op):
            raise RuntimeError("credentials expired")
    _install_fake_boto3(monkeypatch, _Boom())
    coordinator.restore_from_s3("bucket", str(tmp_path))  # must not raise


# --------------------------------------------------------------------------
# Speculative re-leasing. A generation is a BARRIER: it ends only when the
# last member has a fitness, so on a 4-machine fleet with a 10x spread in
# throughput the slowest box holding the final chunk idles everyone else, and
# plain lease expiry only reacts after the full lease_seconds (300s in prod).
# At the tail, an idle worker may take a SECOND live lease and race it.
# --------------------------------------------------------------------------

def _spec_queue(n_chunks=1, lease_seconds=1000.0, **kwargs):
    """Queue of 2-member chunks with the tail race armed by default."""
    chunks = {f"g000000-c{i:03d}": [[2 * i, 111 + i, 1], [2 * i + 1, 111 + i, -1]]
              for i in range(n_chunks)}
    kwargs.setdefault("speculative_after", 5.0)
    kwargs.setdefault("speculative_when_remaining_below", 2)
    return ChunkQueue(chunks, lease_seconds, clock=lambda: 0.0, **kwargs)


def test_speculation_is_off_unless_asked_for():
    q = _queue(lease_seconds=10.0)  # constructed exactly like production used to
    q.lease(now=0.0)
    q.lease(now=0.0)
    assert q.lease(now=9.9) is None  # stale, tail reached -- but no speculation
    assert q.speculative_leases == 0


def test_speculative_release_hands_a_straggler_to_a_second_worker():
    q = _spec_queue(n_chunks=2, speculative_after=5.0)
    first = q.lease(now=0.0)[0]
    q.lease(now=1.0)
    assert q.lease(now=4.9) is None            # nothing outstanding long enough
    cid, _payload, deadline = q.lease(now=5.0)
    assert cid == first                        # the stalest chunk is the one raced
    assert deadline == pytest.approx(1005.0)
    assert q.speculative_leases == 1
    # a race, not an expiry: the original lease is untouched and still live
    assert q.requeue_expired(now=6.0) == []
    assert q.pending_count == 0


def test_speculation_waits_until_the_generation_is_nearly_drained():
    q = _spec_queue(n_chunks=5, speculative_after=5.0,
                    speculative_when_remaining_below=2)
    leases = [q.lease(now=0.0) for _ in range(5)]
    assert all(l is not None for l in leases)
    # every chunk is stale by now, but 5 are still unfinished: duplicating work
    # while the fleet has plenty ahead of it slows the generation down
    assert q.lease(now=100.0) is None
    for cid, members, _deadline in leases[:3]:
        assert q.complete(cid, {int(m[0]): 0.0 for m in members}) is True
    assert q.remaining_count == 2
    raced = q.lease(now=101.0)
    assert raced is not None and raced[0] == leases[3][0]


def test_speculation_never_preempts_fresh_pending_work():
    q = _spec_queue(n_chunks=3, speculative_after=5.0,
                    speculative_when_remaining_below=3)
    first = q.lease(now=0.0)[0]
    assert q.lease(now=100.0)[0] != first  # un-evaluated members come first,
    assert q.lease(now=100.0)[0] != first  # however stale the straggler is
    assert q.speculative_leases == 0
    assert q.lease(now=100.0)[0] == first  # only now, with nothing else to hand out
    assert q.speculative_leases == 1


def test_speculative_leases_are_capped():
    q = _spec_queue(n_chunks=1, speculative_after=5.0, max_concurrent_leases=2)
    cid = q.lease(now=0.0)[0]
    assert q.lease(now=10.0)[0] == cid   # one racer allowed
    assert q.lease(now=100.0) is None    # a third live lease would be wasted work
    assert q.lease(now=200.0) is None
    assert q.speculative_leases == 1


def test_expiry_still_requeues_a_chunk_whose_every_lease_died():
    q = _spec_queue(n_chunks=1, lease_seconds=10.0, speculative_after=5.0)
    cid = q.lease(now=0.0)[0]              # deadline 10
    assert q.lease(now=5.0)[0] == cid      # racer, deadline 15
    assert q.requeue_expired(now=12.0) == []  # racer still holds a live lease
    assert q.pending_count == 0
    assert q.requeue_expired(now=15.0) == [cid]  # both dead: genuinely orphaned
    assert q.pending_count == 1
    # the cap counts LIVE leases only -- if it counted lifetime ones, a chunk
    # whose workers keep dying would become unleasable and hang the barrier
    fresh = q.lease(now=16.0)
    assert fresh is not None and fresh[0] == cid


def test_first_racer_wins_and_the_loser_is_refused():
    q = _spec_queue(n_chunks=1, speculative_after=5.0)
    cid, members, _deadline = q.lease(now=0.0)
    assert q.lease(now=10.0)[0] == cid
    winner = {int(m[0]): 7.0 for m in members}
    assert q.complete(cid, winner) is True
    assert q.complete(cid, {int(m[0]): -1.0 for m in members}) is False
    assert q.results[cid] == winner
    assert q.done and q.pending_count == 0 and q.remaining_count == 0
    assert q.lease(now=500.0) is None  # nothing leasable, nothing corrupted


def test_a_racers_result_still_needs_the_exact_member_set():
    q = _spec_queue(n_chunks=1, speculative_after=5.0)
    cid, members, _deadline = q.lease(now=0.0)
    q.lease(now=10.0)
    full = {int(m[0]): 1.0 for m in members}
    assert q.complete(cid, dict(list(full.items())[:-1])) is False  # truncated
    assert q.complete(cid, {**full, 9999: 1.0}) is False            # foreign index
    assert not q.done
    assert q.complete(cid, full) is True


# --------------------------------------------------------------------------
# Coordinator fleet accounting. Constructed in-process: no socket is bound,
# no server thread runs -- submit_result/status/fleet_report are the whole
# surface the HTTP handler calls into anyway.
# --------------------------------------------------------------------------

def _coordinator(pop_size=4, chunk_size=2, lease_seconds=10.0, **kwargs):
    from es.coordinator import Coordinator
    coord = Coordinator(_toy_state(dim=8), pop_size, chunk_size, 1, lease_seconds,
                        **kwargs)
    coord.start_generation()
    return coord


def _submit(coord, work, worker, fitnesses=None, **extra):
    idx = [m[0] for m in work["members"]]
    body = {"chunk_id": work["chunk_id"], "generation": work["generation"],
            "worker": worker, "member_idx": idx,
            "fitnesses": [1.0] * len(idx) if fitnesses is None else fitnesses}
    body.update(extra)
    return coord.submit_result(body)


def test_status_worker_entry_shape_without_any_stats():
    coord = _coordinator()
    assert _submit(coord, coord.lease_work("m4"), "m4") is True
    entry = coord.status()["workers"]["m4"]
    assert set(entry) == {"age", "members_done", "members_total", "steps_per_s", "procs"}
    assert entry["members_done"] == 2 and entry["members_total"] == 2
    assert entry["steps_per_s"] is None and entry["procs"] is None  # never reported
    assert entry["age"] >= 0.0


def test_status_reports_per_worker_throughput_when_stats_are_sent():
    coord = _coordinator()
    _submit(coord, coord.lease_work("m4"), "m4",
            stats={"procs": 6, "steps_per_s": 3700.5, "episodes_per_s": 1.2,
                   "host": "m4.local"})
    entry = coord.status()["workers"]["m4"]
    assert entry["procs"] == 6
    assert entry["steps_per_s"] == pytest.approx(3700.5)


def test_last_known_throughput_survives_a_post_without_stats():
    coord = _coordinator(pop_size=8, chunk_size=2)
    _submit(coord, coord.lease_work("m4"), "m4", stats={"procs": 6, "steps_per_s": 3700.0})
    _submit(coord, coord.lease_work("m4"), "m4")  # older worker build, no stats
    entry = coord.status()["workers"]["m4"]
    assert entry["steps_per_s"] == pytest.approx(3700.0) and entry["procs"] == 6
    assert entry["members_done"] == 4


def test_garbage_worker_stats_never_crash_the_coordinator():
    coord = _coordinator(pop_size=8, chunk_size=2)
    accepted = _submit(coord, coord.lease_work("junk"), "junk",
                       stats={"procs": "eight", "steps_per_s": "fast"})
    assert accepted is True  # the fitnesses were fine; only the telemetry was junk
    entry = coord.status()["workers"]["junk"]
    assert entry["members_done"] == 2
    assert entry["steps_per_s"] is None and entry["procs"] is None

    for stats in ({}, None, "not-a-dict", [1, 2], {"steps_per_s": float("nan")},
                  {"steps_per_s": float("inf"), "procs": None}, {"procs": [3]},
                  {"steps_per_s": {"a": 1}}):
        assert coord.submit_result({"chunk_id": "g000000-c000", "generation": -1,
                                    "worker": "junk", "member_idx": [],
                                    "fitnesses": [], "stats": stats}) is False
    assert coord.status()["workers"]["junk"]["steps_per_s"] is None
    assert coord.status()["workers"]["junk"]["members_done"] == 2


def test_a_nonstring_worker_name_does_not_kill_the_handler_thread():
    coord = _coordinator()
    assert coord.submit_result({"chunk_id": "g000000-c000", "generation": -1,
                                "worker": {"name": "weird"}, "member_idx": [],
                                "fitnesses": []}) is False
    assert any("weird" in name for name in coord.status()["workers"])


def test_per_generation_members_reset_while_totals_accumulate():
    coord = _coordinator(pop_size=4, chunk_size=2)
    for _ in range(2):
        _submit(coord, coord.lease_work("m4"), "m4")
    fits = coord.wait_for_generation()  # returns immediately: the queue is done
    assert fits.shape == (4,)
    coord.apply_update(fits)
    coord.start_generation()
    entry = coord.status()["workers"]["m4"]
    assert entry["members_done"] == 0
    assert entry["members_total"] == 4


def test_duplicate_submission_neither_credits_the_loser_nor_changes_fitness():
    coord = _coordinator(pop_size=4, chunk_size=2)
    work = coord.lease_work("fast")
    idx = [m[0] for m in work["members"]]
    assert _submit(coord, work, "fast", fitnesses=[5.0, 6.0]) is True
    assert _submit(coord, work, "slow", fitnesses=[-1.0, -2.0]) is False
    workers = coord.status()["workers"]
    assert workers["fast"]["members_done"] == 2
    assert workers["slow"]["members_done"] == 0
    # and wait_for_generation's setdefault keeps the winner's numbers
    _submit(coord, coord.lease_work("fast"), "fast", fitnesses=[0.0, 0.0])
    fits = coord.wait_for_generation()
    assert fits[idx[0]] == pytest.approx(5.0)
    assert fits[idx[1]] == pytest.approx(6.0)


def test_fleet_report_and_wandb_metrics_split_the_work_per_machine():
    from es import coordinator
    coord = _coordinator(pop_size=8, chunk_size=2)
    for worker, procs, steps in (("desktop", 20, 9000.0), ("m4", 6, 3700.0)):
        for _ in range(2):
            _submit(coord, coord.lease_work(worker), worker,
                    stats={"procs": procs, "steps_per_s": steps})
    coord.lease_work("idle-laptop")  # polls for work, finishes nothing

    report = coord.fleet_report(seconds=10.0)
    assert report["workers_active"] == 2  # a poller that finished nothing is not one
    assert report["members"] == 8
    assert report["members_per_s"] == pytest.approx(0.8)
    assert report["total_steps_per_s"] == pytest.approx(12700.0)

    metrics = coordinator.fleet_metrics(report)
    assert metrics["fleet/total_steps_per_s"] == pytest.approx(12700.0)
    assert metrics["fleet/workers_active"] == 2
    assert metrics["worker/desktop/members"] == 4
    assert metrics["worker/m4/steps_per_s"] == pytest.approx(3700.0)
    assert "worker/idle-laptop/members" not in metrics

    line = coordinator.fleet_summary_line(report, generation=7)
    assert "gen 7" in line and "0.80 members/s" in line
    assert "desktop:4" in line and "m4:4" in line


def test_fleet_metrics_sanitise_names_and_omit_unknown_throughput():
    from es import coordinator
    report = {"total_steps_per_s": 0.0, "workers_active": 1, "members_per_s": 1.0,
              "speculative_leases": 0, "seconds": 8.0,
              "workers": {"wsl/laptop 2": {"members": 8, "steps_per_s": None,
                                           "procs": 12}}}
    metrics = coordinator.fleet_metrics(report)
    # '/' would fabricate a nesting level in the W&B metric tree
    assert metrics["worker/wsl_laptop_2/members"] == 8
    # no reading at all is a gap in the chart, not a fabricated zero
    assert not any(k.endswith("/steps_per_s") and k.startswith("worker/") for k in metrics)
    # the journalctl line is for humans: it keeps the machine's real name
    assert "wsl/laptop 2:8" in coordinator.fleet_summary_line(report, 0)


def test_status_keeps_its_documented_top_level_shape():
    coord = _coordinator()
    assert set(coord.status()) == {
        "generation", "pop_size", "members_done", "chunks_pending",
        "best_fitness_gen", "best_fitness_ever", "theta_version",
        "speculative_leases", "workers"}


def test_coordinator_does_not_speculate_by_default():
    coord = _coordinator(pop_size=4, chunk_size=2, lease_seconds=1000.0)
    coord.lease_work("a")
    coord.lease_work("b")
    assert coord.lease_work("idle") is None
    assert coord.status()["speculative_leases"] == 0


def test_coordinator_wires_speculative_releasing_into_its_queue():
    # lease_work() uses the real monotonic clock, so a threshold of ~0 (not a
    # sleep) is what makes the tail race observable in a unit test
    coord = _coordinator(pop_size=4, chunk_size=2, lease_seconds=1000.0,
                         speculative_after=1e-9, speculative_when_remaining_below=2)
    first = coord.lease_work("slow")["chunk_id"]
    assert coord.lease_work("also-slow")["chunk_id"] != first
    raced = coord.lease_work("idle")
    assert raced is not None and raced["chunk_id"] == first
    assert coord.status()["speculative_leases"] == 1
