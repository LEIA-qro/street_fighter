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

import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

if __package__ in (None, ""):  # `python src/es/coordinator.py`
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from es import openes, policy, protocol


class Coordinator:
    """Owns the ESState and the current generation's ChunkQueue.

    All mutable state is behind one lock; HTTP handler threads and the
    generation loop both go through the methods below. Coarse locking is
    fine: the critical sections are microseconds against multi-second
    episode evaluations.
    """

    def __init__(self, state, pop_size, chunk_size, episodes, lease_seconds):
        self.lock = threading.Lock()
        self.state = state
        self.pop_size = pop_size
        self.chunk_size = chunk_size
        self.episodes = episodes
        self.lease_seconds = lease_seconds
        self.queue = None
        self.theta_payload = protocol.encode_theta(state.theta, state.generation)
        self.workers = {}            # name -> last-seen wall time
        self.best_ever = None
        self.best_gen = None

    def start_generation(self):
        with self.lock:
            g = self.state.generation
            members = openes.members_for_generation(self.state, self.pop_size)
            self.queue = protocol.ChunkQueue(
                protocol.make_chunks(members, self.chunk_size, g), self.lease_seconds)
            self.theta_payload = protocol.encode_theta(self.state.theta, g)
            self.best_gen = None
            print(f"[coord] generation {g}: {self.pop_size} members, "
                  f"{len(self.queue.results) + self.queue.pending_count} chunks", flush=True)

    def lease_work(self, worker_name):
        with self.lock:
            self.workers[worker_name] = time.time()
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

    def submit_result(self, body):
        with self.lock:
            self.workers[body.get("worker", "?")] = time.time()
            if self.queue is None or body.get("generation") != self.state.generation:
                return False  # stale worker finishing last generation's chunk
            fits = dict(zip(body["member_idx"], body["fitnesses"]))
            accepted = self.queue.complete(body["chunk_id"], fits)
            if accepted:
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
                    "workers": {n: round(now - t, 1) for n, t in self.workers.items()}}

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
    ap.add_argument("--checkpoint-dir", default=os.path.join("models", "es"))
    ap.add_argument("--generations", type=int, default=0, help="0 = run forever")
    ap.add_argument("--master-seed", type=int, default=20260825)
    ap.add_argument("--s3-bucket", default=None)
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    state = load_or_init_state(args)
    coord = Coordinator(state, args.pop_size, args.chunk_size,
                        args.episodes_per_eval, args.lease_seconds)
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
            metrics = {"fitness/mean": float(fitnesses.mean()),
                       "fitness/best": float(fitnesses.max()),
                       "fitness/worst": float(fitnesses.min()),
                       "theta/norm": float(np.linalg.norm(coord.state.theta)),
                       "time/generation_seconds": dt}
            if wandb_log:
                wandb_log(metrics, step=g)
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
