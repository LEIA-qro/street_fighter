# protocol.py -- the JSON-over-HTTP contract between coordinator and workers,
# plus the pure chunk-leasing queue the coordinator drives it with.
#
# Endpoints (all bodies JSON, all responses JSON):
#
#   GET  /work?worker=<name>
#       -> {"work": null, "retry_in": <sec>}                     nothing leasable
#       -> {"work": {"generation": g, "theta_version": g,
#                    "chunk_id": "g000012-c003", "sigma": s,
#                    "episodes": e, "lease_seconds": L,
#                    "members": [[member_idx, pair_seed, sign], ...]}}
#
#   POST /result
#       <- {"chunk_id": ..., "generation": g, "worker": <name>,
#           "member_idx": [...], "fitnesses": [...]}
#       -> {"accepted": true|false}     false = duplicate or stale generation;
#                                       the worker just moves on either way
#
#   GET  /theta?version=<g>
#       -> {"version": <current g>, "npz_b64": <base64 of np.savez bytes>}
#          Always serves the CURRENT theta; the worker compares versions and
#          re-fetches if a /work lease names a newer one.
#
#   POST /result  (optional telemetry field, ignored by older coordinators)
#       <- {..., "stats": {"procs": n, "steps_per_s": f,
#                          "episodes_per_s": f, "host": "..."}}
#
#   GET  /status
#       -> {"generation", "pop_size", "members_done", "chunks_pending",
#           "best_fitness_gen", "best_fitness_ever", "theta_version",
#           "speculative_leases",
#           "workers": {name: {"age", "members_done", "members_total",
#                              "steps_per_s", "procs"}}}
#          NOTE: "workers" used to be {name: seconds_since_last_seen}; the
#          old scalar is now the "age" field of the per-worker object.
#
# Fault model: workers are stateless and expendable. A chunk not completed
# within lease_seconds silently re-enters the queue (work stealing), so a
# dead worker, a power loss, or a mid-episode SIGKILL never blocks the
# generation; if the original worker later reports anyway, first result wins
# and the duplicate is dropped. At the tail of a generation the same
# first-result-wins rule also powers speculative re-leasing (see ChunkQueue).

import base64
import io
import time

import numpy as np


class ChunkQueue:
    """Pure leasing queue for one generation. No I/O, injectable clock.

    Chunks are {chunk_id: payload}. Lifecycle per chunk:
    pending -> leased (deadline) -> completed, with leased -> pending again
    on expiry. complete() is first-result-wins and accepts results for
    already-expired leases (the work was still done; only duplicates of an
    already-completed chunk are refused).

    Speculative re-leasing (opt-in: off unless speculative_after is set).
    A generation is a BARRIER -- it ends only when the LAST member has a
    fitness -- so on a heterogeneous fleet the slowest machine holding the
    final chunk keeps every other machine idle, and plain expiry only reacts
    after the whole lease_seconds (300s in production). Once the generation
    is down to its last few incomplete chunks and one of them has been
    outstanding for speculative_after, an idle worker may take a SECOND live
    lease on it and race the straggler. First result wins; the loser is
    refused by complete() exactly like any other duplicate. The wasted work
    is bounded by construction: at most
    speculative_when_remaining_below * (max_concurrent_leases - 1) duplicate
    chunks, and only at the tail of a generation where the alternative is an
    idle fleet.
    """

    def __init__(self, chunks, lease_seconds, clock=time.monotonic,
                 speculative_after=None, speculative_when_remaining_below=2,
                 max_concurrent_leases=2):
        self._payloads = dict(chunks)
        self._pending = list(self._payloads.keys())  # FIFO
        self._leased = {}       # chunk_id -> [deadline, ...] one per live lease
        self._leased_at = {}    # chunk_id -> clock of its most recent hand-out
        self._results = {}      # chunk_id -> result payload
        self._lease_seconds = float(lease_seconds)
        self._clock = clock
        # None/0 disables speculation, which keeps the queue byte-for-byte the
        # old work-stealing-only queue for anyone constructing it positionally.
        self._speculative_after = (float(speculative_after)
                                   if speculative_after else None)
        self._speculative_below = int(speculative_when_remaining_below)
        self._max_leases = max(1, int(max_concurrent_leases))
        self._speculative_leases = 0

    def requeue_expired(self, now=None):
        """Move fully expired chunks back to pending; returns the requeued ids.

        A chunk returns to pending only when EVERY live lease on it has
        expired: while a speculative racer still holds a live lease the work
        is not actually orphaned.
        """
        now = self._clock() if now is None else now
        expired = []
        for cid, deadlines in list(self._leased.items()):
            live = [d for d in deadlines if now < d]
            if live:
                self._leased[cid] = live
            else:
                del self._leased[cid]
                self._leased_at.pop(cid, None)
                self._pending.append(cid)
                expired.append(cid)
        return expired

    def lease(self, now=None):
        """(chunk_id, payload, deadline) or None if nothing is leasable.

        Fresh pending work always wins over a speculative duplicate: burning a
        worker on a race while un-evaluated members are queued would slow the
        generation down, not speed it up.
        """
        now = self._clock() if now is None else now
        self.requeue_expired(now)
        if self._pending:
            return self._grant(self._pending.pop(0), now)
        cid = self._speculative_pick(now)
        if cid is None:
            return None
        self._speculative_leases += 1
        return self._grant(cid, now)

    def _grant(self, cid, now):
        deadline = now + self._lease_seconds
        self._leased.setdefault(cid, []).append(deadline)
        self._leased_at[cid] = now
        return cid, self._payloads[cid], deadline

    def _speculative_pick(self, now):
        """The stalest outstanding chunk worth racing, or None."""
        if self._speculative_after is None:
            return None
        if self.remaining_count > self._speculative_below:
            return None  # not the tail yet: the fleet still has real work ahead
        # The staleness clock restarts on every hand-out (including a
        # speculative one), so a chunk is never duplicated twice in a burst and
        # a chunk freshly re-leased after an expiry is treated as fresh work.
        candidates = [cid for cid, deadlines in self._leased.items()
                      if len(deadlines) < self._max_leases
                      and now - self._leased_at[cid] >= self._speculative_after]
        if not candidates:
            return None
        return min(candidates, key=lambda cid: self._leased_at[cid])

    def complete(self, chunk_id, result):
        """Record a result; False for duplicates/unknown ids (caller ignores)."""
        if chunk_id not in self._payloads or chunk_id in self._results:
            return False
        # Fleet workers are expendable and possibly running stale code: a
        # result must cover EXACTLY this chunk's member set, or it is rejected
        # and the chunk stays leased until expiry requeues it. A truncated or
        # foreign-index submission used to mark the chunk complete and crash
        # wait_for_generation with a KeyError one poll later.
        expected = {int(m[0]) for m in self._payloads[chunk_id]}
        if {int(k) for k in result} != expected:
            return False
        self._results[chunk_id] = result
        # Drop every trace of the chunk: the live leases of any speculative
        # racer still running it, and the copy expiry may have requeued into
        # pending. Neither may hand this chunk out again.
        self._leased.pop(chunk_id, None)
        self._leased_at.pop(chunk_id, None)
        if chunk_id in self._pending:
            self._pending.remove(chunk_id)
        return True

    @property
    def done(self):
        return len(self._results) == len(self._payloads)

    @property
    def pending_count(self):
        return len(self._pending)

    @property
    def remaining_count(self):
        """Chunks with no result yet -- pending plus leased. The tail gate."""
        return len(self._payloads) - len(self._results)

    @property
    def speculative_leases(self):
        """Duplicate leases handed out this generation (straggler races)."""
        return self._speculative_leases

    @property
    def results(self):
        return dict(self._results)


def encode_theta(theta, version):
    """theta -> the /theta response body. npz keeps float32 bit-exact."""
    buf = io.BytesIO()
    np.savez_compressed(buf, theta=np.asarray(theta, dtype=np.float32))
    return {"version": int(version),
            "npz_b64": base64.b64encode(buf.getvalue()).decode("ascii")}


def decode_theta(payload):
    """the /theta response body -> (version, theta float32)."""
    raw = base64.b64decode(payload["npz_b64"])
    theta = np.load(io.BytesIO(raw))["theta"].astype(np.float32)
    return int(payload["version"]), theta


def make_chunks(members, chunk_size, generation):
    """Split a generation's member list into {chunk_id: work payload}.

    Chunk ids embed the generation so a stale worker's POST for last
    generation's chunk can never collide with a current id.
    """
    chunks = {}
    for start in range(0, len(members), chunk_size):
        part = members[start:start + chunk_size]
        cid = f"g{generation:06d}-c{start // chunk_size:03d}"
        chunks[cid] = [[int(i), int(seed), int(sign)] for i, seed, sign in part]
    return chunks
