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
#   GET  /status
#       -> {"generation", "pop_size", "members_done", "chunks_pending",
#           "best_fitness_gen", "best_fitness_ever", "theta_version",
#           "workers": {name: seconds_since_last_seen}}
#
# Fault model: workers are stateless and expendable. A chunk not completed
# within lease_seconds silently re-enters the queue (work stealing), so a
# dead worker, a power loss, or a mid-episode SIGKILL never blocks the
# generation; if the original worker later reports anyway, first result wins
# and the duplicate is dropped.

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
    """

    def __init__(self, chunks, lease_seconds, clock=time.monotonic):
        self._payloads = dict(chunks)
        self._pending = list(self._payloads.keys())  # FIFO
        self._leased = {}       # chunk_id -> deadline
        self._results = {}      # chunk_id -> result payload
        self._lease_seconds = float(lease_seconds)
        self._clock = clock

    def requeue_expired(self, now=None):
        """Move expired leases back to pending; returns the requeued ids."""
        now = self._clock() if now is None else now
        expired = [cid for cid, deadline in self._leased.items() if now >= deadline]
        for cid in expired:
            del self._leased[cid]
            self._pending.append(cid)
        return expired

    def lease(self, now=None):
        """(chunk_id, payload, deadline) or None if nothing is leasable."""
        now = self._clock() if now is None else now
        self.requeue_expired(now)
        if not self._pending:
            return None
        cid = self._pending.pop(0)
        deadline = now + self._lease_seconds
        self._leased[cid] = deadline
        return cid, self._payloads[cid], deadline

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
        self._leased.pop(chunk_id, None)
        # a chunk can be completed while a re-lease of it sits in pending
        # (expiry requeued it, then the original worker reported): drop it
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
