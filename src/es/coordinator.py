# coordinator.py -- the single stateful end of the ES harness.
#
# stdlib http.server only (ThreadingHTTPServer): this must start on any box
# with bare python + numpy -- no fastapi/flask/uvicorn to install or version
# over a hotel wifi at 2am. Throughput is not a concern: the wire carries a
# few KB of JSON per chunk and one ~60KB theta per worker per generation.
#
# Generation loop: publish theta -> lease chunks to whoever polls /work ->
# collect the population's fitnesses -> ES update -> checkpoint (+ optional
# S3 / wandb, both lazy imports that degrade to a warning) -> repeat.
# On start it resumes from the newest checkpoint in --checkpoint-dir, so the
# coordinator process itself is also expendable.
#
#   python src/es/coordinator.py --pop-size 256 --checkpoint-dir models/es
#
# Fleet visibility: /result may carry an optional "stats" block
# ({"procs", "steps_per_s", "episodes_per_s", "host"}); workers that never
# send it are accounted for exactly as before, minus the throughput numbers.
#
# WIRE CHANGE (2026-08-25): GET /status used to report
#   "workers": {name: seconds_since_last_seen}
# and now reports
#   "workers": {name: {"age", "members_done", "members_total",
#                      "steps_per_s", "procs"}}
# -- the old scalar is the "age" field. Anything curling /status and reading
# that map as a number (a dashboard, a shell one-liner) needs the .age suffix.
# /status also gained a top-level "speculative_leases" counter.

import argparse
import json
import math
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ""):  # `python src/es/coordinator.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from es import openes, policy, protocol

# W&B key components: a '/' inside a worker name would fabricate a nesting
# level in the run's metric tree, and spaces make the panel names unusable.
_WANDB_KEY = re.compile(r"[^0-9A-Za-z._-]+")


def _finite_number(value):
    """None unless `value` is a finite number.

    Worker telemetry is untrusted input: an older/newer worker may omit a key
    or send a string, and a division by zero on its side sends inf. Junk must
    degrade to "unknown", never to an exception inside the coordinator lock.
    """
    try:
        num = float(value)
    except (TypeError, ValueError):
        return None
    return num if math.isfinite(num) else None


def _new_worker_record():
    return {"last_seen": 0.0, "members_gen": 0, "members_total": 0,
            "steps_per_s": None, "procs": None}


class Coordinator:
    """Owns the ESState and the current generation's ChunkQueue.

    All mutable state is behind one lock; HTTP handler threads and the
    generation loop both go through the methods below. Coarse locking is
    fine: the critical sections are microseconds against multi-second
    episode evaluations.
    """

    def __init__(self, state, pop_size, chunk_size, episodes, lease_seconds,
                 speculative_after=None, speculative_when_remaining_below=2,
                 max_concurrent_leases=2):
        self.lock = threading.Lock()
        self.state = state
        self.pop_size = pop_size
        self.chunk_size = chunk_size
        self.episodes = episodes
        self.lease_seconds = lease_seconds
        self.speculative_after = speculative_after
        self.speculative_when_remaining_below = speculative_when_remaining_below
        self.max_concurrent_leases = max_concurrent_leases
        self.queue = None
        self.theta_payload = protocol.encode_theta(state.theta, state.generation)
        self.workers = {}            # name -> _new_worker_record()
        self.gen_started_at = time.time()
        self.best_ever = None
        self.best_gen = None

    def _touch(self, worker_name):
        """Mark a worker alive and return its record. Caller holds the lock."""
        # str(): a malformed POST could carry a list/dict here, and an
        # unhashable dict key would take the whole handler thread down.
        name = str(worker_name) if worker_name is not None else "?"
        record = self.workers.get(name)
        if record is None:
            record = self.workers[name] = _new_worker_record()
        record["last_seen"] = time.time()
        return record

    def start_generation(self):
        with self.lock:
            g = self.state.generation
            members = openes.members_for_generation(self.state, self.pop_size)
            self.queue = protocol.ChunkQueue(
                protocol.make_chunks(members, self.chunk_size, g), self.lease_seconds,
                speculative_after=self.speculative_after,
                speculative_when_remaining_below=self.speculative_when_remaining_below,
                max_concurrent_leases=self.max_concurrent_leases)
            self.theta_payload = protocol.encode_theta(self.state.theta, g)
            self.best_gen = None
            self.gen_started_at = time.time()
            for record in self.workers.values():
                record["members_gen"] = 0  # cumulative counts survive, per-gen resets
            print(f"[coord] generation {g}: {self.pop_size} members, "
                  f"{len(self.queue.results) + self.queue.pending_count} chunks", flush=True)

    def lease_work(self, worker_name):
        with self.lock:
            self._touch(worker_name)
            if self.queue is None:
                return None
            leased = self.queue.lease()
            if leased is None:
                return None
            cid, members, _deadline = leased
            return {"generation": self.state.generation,
                    "theta_version": self.state.generation,
                    "chunk_id": cid, "sigma": self.state.sigma,
                    "episodes": self.episodes, "lease_seconds": self.lease_seconds,
                    "members": members}

    @staticmethod
    def _absorb_stats(record, stats):
        """Fold an optional /result "stats" block into a worker's record.

        Every field is optional and every field is untrusted; a missing or
        unusable value leaves the last known reading in place rather than
        blanking it, so one malformed POST does not erase a worker from the
        throughput view.
        """
        if not isinstance(stats, dict):
            return
        steps = _finite_number(stats.get("steps_per_s"))
        if steps is not None:
            record["steps_per_s"] = steps
        procs = _finite_number(stats.get("procs"))
        if procs is not None:
            record["procs"] = int(procs)

    def submit_result(self, body):
        with self.lock:
            record = self._touch(body.get("worker", "?"))
            self._absorb_stats(record, body.get("stats"))
            if self.queue is None or body.get("generation") != self.state.generation:
                return False  # stale worker finishing last generation's chunk
            fits = dict(zip(body["member_idx"], body["fitnesses"]))
            accepted = self.queue.complete(body["chunk_id"], fits)
            if accepted:
                # only accepted results count: a duplicate (the loser of a
                # speculative race, or a late report) must not inflate a
                # machine's contribution with work someone else's result used
                record["members_gen"] += len(fits)
                record["members_total"] += len(fits)
                best = max(body["fitnesses"])
                self.best_gen = best if self.best_gen is None else max(self.best_gen, best)
                self.best_ever = best if self.best_ever is None else max(self.best_ever, best)
            return accepted

    def status(self):
        with self.lock:
            done = sum(len(f) for f in self.queue.results.values()) if self.queue else 0
            now = time.time()
            return {"generation": self.state.generation, "pop_size": self.pop_size,
                    "members_done": done,
                    "chunks_pending": self.queue.pending_count if self.queue else 0,
                    "best_fitness_gen": self.best_gen, "best_fitness_ever": self.best_ever,
                    "theta_version": self.theta_payload["version"],
                    "speculative_leases": self.queue.speculative_leases if self.queue else 0,
                    # Was {name: age_seconds}; the scalar is now the "age" key.
                    # Rounded/short so a raw `curl | python -m json.tool` (or
                    # plain eyeballing) stays readable in a terminal.
                    "workers": {n: {"age": round(now - w["last_seen"], 1),
                                    "members_done": w["members_gen"],
                                    "members_total": w["members_total"],
                                    "steps_per_s": (None if w["steps_per_s"] is None
                                                    else round(w["steps_per_s"], 1)),
                                    "procs": w["procs"]}
                                for n, w in self.workers.items()}}

    def fleet_report(self, seconds=None):
        """What each machine actually contributed to the generation just ended.

        Must be read BEFORE the next start_generation(), which resets the
        per-generation counters. Only workers that landed at least one member
        this generation are listed: an idle-but-polling laptop is not a
        contributor, and counting it would deflate every fleet average.
        """
        with self.lock:
            seconds = float(time.time() - self.gen_started_at
                            if seconds is None else seconds)
            workers = {n: {"members": w["members_gen"], "steps_per_s": w["steps_per_s"],
                           "procs": w["procs"]}
                       for n, w in self.workers.items() if w["members_gen"] > 0}
            members = sum(w["members"] for w in workers.values())
            steps = sum(w["steps_per_s"] for w in workers.values()
                        if w["steps_per_s"] is not None)
            return {"seconds": seconds, "members": members,
                    "members_per_s": members / seconds if seconds > 0 else 0.0,
                    "workers_active": len(workers), "total_steps_per_s": steps,
                    "speculative_leases": (self.queue.speculative_leases
                                           if self.queue else 0),
                    "workers": workers}

    def wait_for_generation(self):
        """Block until every chunk of the current generation has a result."""
        while True:
            with self.lock:
                self.queue.requeue_expired()  # keep stealing even if no one polls
                if self.queue.done:
                    fits = {}
                    for chunk_fits in self.queue.results.values():
                        for idx, f in chunk_fits.items():
                            fits.setdefault(int(idx), float(f))  # first result wins
                    return np.array([fits[i] for i in range(self.pop_size)])
            time.sleep(0.5)

    def apply_update(self, fitnesses):
        with self.lock:
            self.state = openes.es_update(self.state, fitnesses)


def fleet_metrics(report):
    """fleet_report() -> W&B metrics.

    The point of these panels is that "how much is this machine contributing"
    is answerable from the dashboard, without anyone ssh-ing into a box to run
    a benchmark by hand. Workers that never sent a stats block get a members
    series but no steps_per_s series -- a gap is honest, a fabricated 0 would
    read as a machine that stalled.
    """
    metrics = {"fleet/total_steps_per_s": report["total_steps_per_s"],
               "fleet/workers_active": report["workers_active"],
               "fleet/members_per_s": report["members_per_s"],
               "fleet/speculative_leases": report["speculative_leases"]}
    for name, w in report["workers"].items():
        key = _WANDB_KEY.sub("_", name) or "unnamed"
        metrics[f"worker/{key}/members"] = w["members"]
        if w["steps_per_s"] is not None:
            metrics[f"worker/{key}/steps_per_s"] = w["steps_per_s"]
    return metrics


def fleet_summary_line(report, generation):
    """One line per generation for journalctl on the madre: the member split
    across machines, which is what a heterogeneous fleet is judged on."""
    split = " ".join(f"{n}:{w['members']}" for n, w in
                     sorted(report["workers"].items(),
                            key=lambda kv: (-kv[1]["members"], kv[0])))
    line = (f"[coord] gen {generation} fleet: {report['seconds']:.1f}s, "
            f"{report['members_per_s']:.2f} members/s, "
            f"{report['workers_active']} workers, "
            f"{report['total_steps_per_s']:.0f} steps/s | {split or 'no results'}")
    if report["speculative_leases"]:
        line += f" | {report['speculative_leases']} speculative"
    return line


class _Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        coord = self.server.coordinator
        url = urlparse(self.path)
        if url.path == "/work":
            name = parse_qs(url.query).get("worker", ["?"])[0]
            work = coord.lease_work(name)
            self._send({"work": work, "retry_in": 2.0} if work is None else {"work": work})
        elif url.path == "/theta":
            with coord.lock:
                self._send(coord.theta_payload)
        elif url.path == "/status":
            self._send(coord.status())
        else:
            self._send({"error": "unknown path"}, code=404)

    def do_POST(self):
        coord = self.server.coordinator
        if urlparse(self.path).path != "/result":
            self._send({"error": "unknown path"}, code=404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send({"accepted": coord.submit_result(body)})
        except (ValueError, KeyError, TypeError) as e:
            self._send({"error": f"bad result body: {e}"}, code=400)

    def log_message(self, *args):
        pass  # /work polls every couple of seconds; default logging would drown stdout


# ---------------------------------------------------------------------------
# Optional sinks: both are best-effort. Training must never die because a
# laptop lost wifi to S3 or wandb rotated a key.
# ---------------------------------------------------------------------------

def make_s3_uploader(bucket):
    if not bucket:
        return None
    try:
        import boto3  # lazy: not a hard dep of the ES harness
        client = boto3.client("s3")
    except Exception as e:
        print(f"[coord] S3 disabled ({e})", flush=True)
        return None

    def upload(path):
        try:
            client.upload_file(path, bucket, "es/" + os.path.basename(path))
        except Exception as e:
            print(f"[coord] S3 upload of {path} failed ({e}); continuing", flush=True)
    return upload


def restore_from_s3(bucket, checkpoint_dir):
    """Pull the newest checkpoint out of S3 when the local dir has none.

    The madre is disposable infra: a terraform apply that touches user_data
    REPLACES the instance and its local disk goes with it. Without this, S3 was
    a write-only backup -- a replaced (or relocated) madre silently restarted
    from generation 0, which is precisely the failure the uploads exist to
    prevent. Best-effort like every other sink: any failure just means a fresh
    start, never a crash.
    """
    if not bucket:
        return
    try:
        import boto3
        client = boto3.client("s3")
        keys = []
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix="es/"):
            for obj in page.get("Contents", []):
                if obj["Key"].endswith(".npz"):
                    keys.append(obj["Key"])
        if not keys:
            print("[coord] S3 has no checkpoint yet; starting fresh", flush=True)
            return
        newest = max(keys)  # gen_000123.npz sorts correctly: fixed-width digits
        base = newest[:-len(".npz")]
        os.makedirs(checkpoint_dir, exist_ok=True)
        for ext in (".npz", ".json"):
            dest = os.path.join(checkpoint_dir, os.path.basename(base) + ext)
            client.download_file(bucket, base + ext, dest)
        print(f"[coord] restored {os.path.basename(base)} from s3://{bucket}",
              flush=True)
    except Exception as e:
        print(f"[coord] S3 restore skipped ({e}); starting fresh", flush=True)


def make_wandb_logger(project):
    if not project:
        return None
    try:
        import wandb  # lazy; WANDB_MODE=offline works without a key
        default_mode = "online" if os.environ.get("WANDB_API_KEY") else "offline"
        run = wandb.init(project=project, resume="allow",
                         mode=os.environ.get("WANDB_MODE", default_mode))
        return lambda metrics, step: run.log(metrics, step=step)
    except Exception as e:
        print(f"[coord] wandb disabled ({e})", flush=True)
        return None


def load_or_init_state(args):
    # A replaced instance has an empty local dir but a full S3 bucket.
    if openes.latest_checkpoint(args.checkpoint_dir) is None:
        restore_from_s3(args.s3_bucket, args.checkpoint_dir)
    base = openes.latest_checkpoint(args.checkpoint_dir)
    if base is not None:
        state = openes.load_checkpoint(base)
        print(f"[coord] resumed generation {state.generation} from {base}", flush=True)
        # Determinism rule: the checkpoint's hyperparameters win on resume.
        for name, cli_val in (("sigma", args.sigma), ("lr", args.lr),
                              ("weight_decay", args.weight_decay),
                              ("master_seed", args.master_seed)):
            ckpt_val = getattr(state, name)
            if cli_val != ckpt_val:
                print(f"[coord] WARNING: --{name.replace('_', '-')}={cli_val} ignored; "
                      f"checkpoint pins {name}={ckpt_val} (delete the checkpoint "
                      f"dir for a fresh run)", flush=True)
        return state
    theta = policy.init_flat(args.master_seed)
    print(f"[coord] fresh start: {theta.shape[0]} params, master_seed {args.master_seed}",
          flush=True)
    return openes.init_state(theta, args.sigma, args.lr, args.weight_decay, args.master_seed)


def main():
    ap = argparse.ArgumentParser(description="OpenAI-ES coordinator")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8823)
    ap.add_argument("--pop-size", type=int, default=256, help="even; antithetic pairs")
    ap.add_argument("--chunk-size", type=int, default=8, help="members per work lease")
    ap.add_argument("--sigma", type=float, default=0.02)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=0.005)
    ap.add_argument("--episodes-per-eval", type=int, default=1)
    ap.add_argument("--lease-seconds", type=float, default=300.0)
    # Straggler mitigation. A generation is a barrier, so at its tail the
    # whole fleet idles behind one slow machine; racing that chunk costs at
    # most --speculative-tail-chunks duplicate evaluations and only ever at
    # the tail. 0 restores pure expiry-based work stealing.
    ap.add_argument("--speculative-after", type=float, default=60.0,
                    help="seconds a tail chunk may sit before an idle worker races it "
                         "(0 disables speculative re-leasing)")
    ap.add_argument("--speculative-tail-chunks", type=int, default=2,
                    help="only race stragglers once this many chunks (or fewer) remain")
    ap.add_argument("--max-chunk-leases", type=int, default=2,
                    help="cap on simultaneous leases of one chunk (original + racers)")
    ap.add_argument("--checkpoint-dir", default=os.path.join("models", "es"))
    ap.add_argument("--generations", type=int, default=0, help="0 = run forever")
    ap.add_argument("--master-seed", type=int, default=20260825)
    ap.add_argument("--s3-bucket", default=None)
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    state = load_or_init_state(args)
    coord = Coordinator(state, args.pop_size, args.chunk_size,
                        args.episodes_per_eval, args.lease_seconds,
                        speculative_after=args.speculative_after,
                        speculative_when_remaining_below=args.speculative_tail_chunks,
                        max_concurrent_leases=args.max_chunk_leases)
    s3_upload = make_s3_uploader(args.s3_bucket)
    wandb_log = make_wandb_logger(args.wandb_project)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.coordinator = coord
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[coord] listening on {args.host}:{args.port}", flush=True)

    done = 0
    try:
        while args.generations <= 0 or done < args.generations:
            g = coord.state.generation
            coord.start_generation()
            t0 = time.time()
            fitnesses = coord.wait_for_generation()
            coord.apply_update(fitnesses)

            base = os.path.join(args.checkpoint_dir, f"gen_{coord.state.generation:06d}")
            openes.save_checkpoint(coord.state, base)
            if s3_upload:
                s3_upload(base + ".npz")
                s3_upload(base + ".json")

            dt = time.time() - t0
            # read before the next start_generation() clears the per-gen counts
            report = coord.fleet_report(seconds=dt)
            metrics = {"fitness/mean": float(fitnesses.mean()),
                       "fitness/best": float(fitnesses.max()),
                       "fitness/worst": float(fitnesses.min()),
                       "theta/norm": float(np.linalg.norm(coord.state.theta)),
                       "time/generation_seconds": dt}
            metrics.update(fleet_metrics(report))
            if wandb_log:
                wandb_log(metrics, step=g)
            print(fleet_summary_line(report, g), flush=True)
            print(f"[coord] gen {g} done in {dt:.1f}s: "
                  f"mean {metrics['fitness/mean']:.4f} best {metrics['fitness/best']:.4f}",
                  flush=True)
            done += 1
    except KeyboardInterrupt:
        print("[coord] interrupted; latest checkpoint is the resume point", flush=True)
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
