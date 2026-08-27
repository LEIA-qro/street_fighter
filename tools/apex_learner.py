# apex_learner.py -- el learner del Rainbow distribuido (Ape-X).
#
#   .venv/bin/python tools/apex_learner.py --port 8090 --device auto
#
# Corre en LA maquina con GPU (Legion). Los actores (tools/apex_actor.py) le
# POSTean transiciones y jalan pesos; la config del run viaja dentro de
# /weights, asi que los actores no llevan hiperparametros de aprendizaje.
# Checkpoints .pt identicos en formato a los de train_rainbow.py: el banco
# los examina con `bench_12rivals.py --arm rainbow --ckpt <pt>` sin cambios.
#
# Politica de flota: el learner ocupa "el slot de entrenamiento" de su
# maquina; los actores son el slot de las demas.

import argparse
import json
import os
import random
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np
import torch

from agents.apex import ApexLearner


def pick_device(flag):
    if flag != "auto":
        return flag
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class _Handler(BaseHTTPRequestHandler):
    def _send(self, obj, code=200):
        body = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        learner = self.server.learner
        path = urlparse(self.path).path
        if path == "/weights":
            self._send(learner.weights_payload())
        elif path == "/status":
            self._send(learner.status(time.time()))
        else:
            self._send({"error": "unknown path"}, code=404)

    def do_POST(self):
        learner = self.server.learner
        if urlparse(self.path).path != "/transitions":
            self._send({"error": "unknown path"}, code=404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            self._send(learner.ingest(body, time.time()))
        except (ValueError, KeyError, TypeError) as e:
            self._send({"error": f"bad transitions body: {e}"}, code=400)

    def log_message(self, *args):
        pass


def main():
    ap = argparse.ArgumentParser(description="Ape-X learner (Rainbow-QR)")
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--quantiles", type=int, default=51)
    ap.add_argument("--no-onehot", action="store_true")
    ap.add_argument("--macros", action="store_true",
                    help="accion Discrete(72): los 9 macros del equipo como "
                         "opciones atomicas (config del run: los actores la "
                         "adoptan solos)")
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--buffer", type=int, default=500_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--per-alpha", type=float, default=0.5)
    ap.add_argument("--per-beta0", type=float, default=0.4)
    ap.add_argument("--beta-anneal-grads", type=int, default=200_000)
    ap.add_argument("--learn-start", type=int, default=20_000,
                    help="transiciones en buffer antes del primer gradiente")
    ap.add_argument("--target-sync", type=int, default=2_500,
                    help="pasos de gradiente entre syncs de la target")
    ap.add_argument("--weights-every", type=int, default=100,
                    help="pasos de gradiente entre versiones de /weights")
    ap.add_argument("--replay-ratio", type=float, default=8.0,
                    help="muestras por transicion ingerida (tope: el learner "
                         "se frena si va demasiado adelante de los actores)")
    ap.add_argument("--total-grads", type=int, default=0, help="0 = infinito")
    ap.add_argument("--ckpt-every", type=int, default=10_000,
                    help="pasos de gradiente entre checkpoints")
    ap.add_argument("--out", default=str(REPO / "models" / "rainbow_apex"))
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    device = pick_device(args.device)
    if args.macros:
        from envs.action_macros import N_ACTIONS as n_actions
    else:
        n_actions = 63
    torch.manual_seed(args.seed)
    learner = ApexLearner(hidden=args.hidden, quantiles=args.quantiles,
                          onehot=not args.no_onehot, gamma=args.gamma,
                          n_step=args.n_step, buffer_capacity=args.buffer,
                          lr=args.lr, per_alpha=args.per_alpha, device=device,
                          weights_every_grads=args.weights_every,
                          n_actions=n_actions, macros=args.macros)
    print(f"[learner] device={device} in_dim={learner.in_dim} "
          f"acciones={learner.n_actions} "
          f"buffer={args.buffer} batch={args.batch} replay_ratio<="
          f"{args.replay_ratio}", flush=True)

    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    server.learner = learner
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"[learner] listening on {args.host}:{args.port}", flush=True)

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project, entity="leia-qro-rl", id="rainbow-apex",
                                   name="rainbow-apex", resume="allow",
                                   group="dqn", tags=["dqn"],
                                   settings=wandb.Settings(x_disable_stats=True))
        except Exception as e:
            print(f"[learner] wandb off ({e})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(args.seed)
    losses = []
    last_log_t, last_log_grads, last_log_trans = time.time(), 0, 0

    def save_ckpt(tag):
        path = os.path.join(args.out, f"apex_{tag}.pt")
        torch.save({"state_dict": learner.online.state_dict(),
                    "meta": {"in_dim": learner.in_dim,
                             "quantiles": args.quantiles, "hidden": args.hidden,
                             "onehot": not args.no_onehot,
                             "n_actions": learner.n_actions,
                             "macros": args.macros,
                             "grad_steps": learner.grad_steps,
                             "args": vars(args)}}, path)
        return path

    try:
        while args.total_grads <= 0 or learner.grad_steps < args.total_grads:
            with learner.lock:
                buffered = learner.buffer.size
                ingested = learner.transitions_in
            if buffered < args.learn_start:
                time.sleep(0.5)
                continue
            # tope de replay ratio: no re-masticar el buffer si los actores
            # van lentos (sobreajuste al replay = el mal clasico de DQN)
            if learner.grad_steps * args.batch > args.replay_ratio * max(ingested, 1):
                time.sleep(0.05)
                continue
            beta = min(1.0, args.per_beta0 + (1.0 - args.per_beta0)
                       * learner.grad_steps / args.beta_anneal_grads)
            out = learner.train_tick(args.batch, beta, rng)
            if out is None:
                time.sleep(0.2)
                continue
            losses.append(out[0])
            losses = losses[-500:]
            if learner.grad_steps % args.target_sync == 0:
                learner.sync_target()
            if learner.grad_steps % args.ckpt_every == 0:
                save_ckpt(f"grads_{learner.grad_steps:08d}")
            if time.time() - last_log_t >= 30:
                s = learner.status(time.time())
                grads_s = (learner.grad_steps - last_log_grads) / (time.time() - last_log_t)
                trans_s = (s["transitions_in"] - last_log_trans) / (time.time() - last_log_t)
                actor_line = " ".join(
                    f"{n}:{r.get('steps_per_s', '?')}st/s"
                    for n, r in s["actors"].items())
                print(f"[learner] grads {s['grad_steps']} ({grads_s:.0f}/s) | "
                      f"buffer {s['buffer']} | trans/s {trans_s:.0f} | "
                      f"wr acum {s['win_rate_cum']} reciente200 {s['win_rate_recent200']} ({s['episodes']} eps) | "
                      f"loss {np.mean(losses):.4f} | {actor_line}",
                      flush=True)
                if wandb_run:
                    wandb_run.log({"loss": float(np.mean(losses)),
                                   "buffer": s["buffer"], "beta": beta,
                                   "grads_per_s": grads_s,
                                   "transitions_per_s": trans_s,
                                   "win_rate_cum": s["win_rate_cum"],
                                   "win_rate_recent200": s["win_rate_recent200"],
                                   "episodes": s["episodes"]},
                                  step=learner.grad_steps)
                last_log_t = time.time()
                last_log_grads = learner.grad_steps
                last_log_trans = s["transitions_in"]
    except KeyboardInterrupt:
        pass
    finally:
        path = save_ckpt(f"final_{learner.grad_steps:08d}")
        print(f"[learner] checkpoint final: {path}", flush=True)
        server.shutdown()


if __name__ == "__main__":
    main()
