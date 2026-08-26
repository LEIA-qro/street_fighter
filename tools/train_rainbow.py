# train_rainbow.py -- entrenamiento Rainbow-lite (QR) sobre el backend retro.
#
#   .venv/bin/python tools/train_rainbow.py --total-steps 2000000 --envs 6
#
# Politica de flota (Felipe, 2026-08-26): UN entrenamiento a la vez por
# maquina. Antes de lanzar esto en una maquina que corre el worker ES,
# detener el worker (la madre re-lease sus chunks sola).
#
# Piezas: agents/rainbow.py (nucleo puro), envs/discrete_sf2.py (wrappers).
# Los N emuladores corren en procesos hijos (AsyncVectorEnv spawn: stable-
# retro tolera UN emulador por proceso); la red entrena en el device que
# haya (cuda > mps > cpu). Con la red chica el cuello es la emulacion, igual
# que siempre en este proyecto.
#
# Checkpoints: {out}/rainbow_step_XXXXXXXX.pt (state_dict online + meta).
# El banco los examina con: tools/bench_12rivals.py --arm rainbow --ckpt <pt>
# Entrenamiento con --desync-max 30 por default: la leccion de robustez de
# la run 1 del ES, horneada desde el dia uno (bajarlo a 0 para el purismo
# de comparar contra ES/PPO en condiciones identicas de entrenamiento).

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np
import torch

from agents.rainbow import (NStepAccumulator, PERBuffer, QRDuelingNet,
                            linear_epsilon, make_taus, train_step)
from es.coordinator import resolve_states
from es.policy import OBS_DIM, ONEHOT_OBS_DIM


def pick_device(flag):
    if flag != "auto":
        return torch.device(flag)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    ap = argparse.ArgumentParser(description="Rainbow-lite (QR) sobre SF2 retro")
    ap.add_argument("--total-steps", type=int, default=2_000_000,
                    help="agent steps totales (suma sobre envs)")
    ap.add_argument("--envs", type=int, default=6)
    ap.add_argument("--states", default="manifest")
    ap.add_argument("--difficulty", default="1")
    ap.add_argument("--desync-max", type=int, default=30,
                    help="0 = condiciones identicas a ES/PPO; >0 = robustez horneada")
    ap.add_argument("--no-onehot", action="store_true",
                    help="features crudas 92 en vez de char one-hot 212")
    ap.add_argument("--buffer", type=int, default=200_000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--n-step", type=int, default=3)
    ap.add_argument("--quantiles", type=int, default=51)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--per-alpha", type=float, default=0.5)
    ap.add_argument("--per-beta0", type=float, default=0.4)
    ap.add_argument("--eps-decay-steps", type=int, default=300_000)
    ap.add_argument("--eps-end", type=float, default=0.02)
    ap.add_argument("--learn-start", type=int, default=20_000)
    ap.add_argument("--train-every", type=int, default=4,
                    help="vector-steps entre pasos de gradiente (o sea: un paso "
                         "de gradiente cada train_every*envs agent steps globales)")
    ap.add_argument("--target-sync", type=int, default=8_000,
                    help="pasos de gradiente entre syncs de la red target")
    ap.add_argument("--ckpt-every", type=int, default=100_000)
    ap.add_argument("--out", default=str(REPO / "models" / "rainbow"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=20260826)
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    states = resolve_states(args.states, args.difficulty)
    device = pick_device(args.device)
    in_dim = OBS_DIM if args.no_onehot else ONEHOT_OBS_DIM
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"[rainbow] {len(states)} estados | envs={args.envs} device={device} "
          f"in_dim={in_dim} desync<={args.desync_max} quantiles={args.quantiles}",
          flush=True)

    import gymnasium as gym

    def env_fn(rank):
        def _make():
            from envs.discrete_sf2 import make_discrete_sf2
            return make_discrete_sf2(states, seed=args.seed * 1000 + rank,
                                     desync_max=args.desync_max,
                                     onehot=not args.no_onehot)
        return _make

    venv = gym.vector.AsyncVectorEnv([env_fn(r) for r in range(args.envs)],
                                     context="spawn")

    online = QRDuelingNet(in_dim, n_quantiles=args.quantiles,
                          hidden=args.hidden).to(device)
    target = QRDuelingNet(in_dim, n_quantiles=args.quantiles,
                          hidden=args.hidden).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=args.lr, eps=1.5e-4)
    taus = make_taus(args.quantiles, device)

    buffer = PERBuffer(args.buffer, alpha=args.per_alpha)
    accums = [NStepAccumulator(args.n_step, args.gamma) for _ in range(args.envs)]

    wandb_run = None
    if args.wandb_project:
        try:
            import wandb
            wandb_run = wandb.init(project=args.wandb_project,
                                   name=f"rainbow-qr-{args.seed}", resume="allow")
        except Exception as e:
            print(f"[rainbow] wandb off ({e})", flush=True)

    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    import random as pyrandom
    per_rng = pyrandom.Random(args.seed)  # PER reproducible bajo --seed
    obs, _ = venv.reset(seed=args.seed)
    ep_return = np.zeros(args.envs)
    ep_wins, ep_count, recent_returns = 0, 0, []
    grad_steps, global_step = 0, 0
    t0, last_log_step, last_log_t = time.time(), 0, time.time()
    # gymnasium >= 1.0, modo NEXT_STEP (el default): el step terminal devuelve
    # la obs terminal + info real (dict de arrays con mascaras _key), y el
    # step SIGUIENTE es el autoreset -- su accion se ignora y su transicion
    # es basura inter-episodio que NO debe entrar al buffer.
    just_reset = np.zeros(args.envs, dtype=bool)

    while global_step < args.total_steps:
        eps = linear_epsilon(global_step, end=args.eps_end,
                             decay_steps=args.eps_decay_steps)
        with torch.no_grad():
            q = online.q_values(torch.as_tensor(obs, dtype=torch.float32,
                                                device=device))
            greedy = q.argmax(dim=1).cpu().numpy()
        explore = rng.random(args.envs) < eps
        actions = np.where(explore, rng.integers(0, 63, size=args.envs), greedy)

        next_obs, rewards, terms, truncs, infos = venv.step(actions)
        win_arr = infos.get("win")
        win_mask = infos.get("_win")
        for i in range(args.envs):
            if just_reset[i]:
                # este step fue el autoreset de env i: la accion se ignoro y
                # next_obs[i] es el primer frame del episodio nuevo. Nada de
                # esto entra al buffer ni al retorno del episodio.
                just_reset[i] = False
                continue
            ep_return[i] += rewards[i]
            # en NEXT_STEP, next_obs[i] del step terminal ES la obs terminal:
            # bootstrap correcto sin final_observation
            if terms[i]:
                cooked = accums[i].push(obs[i], actions[i], rewards[i],
                                        next_obs[i], True)
            elif truncs[i]:
                cooked = accums[i].push(obs[i], actions[i], rewards[i],
                                        next_obs[i], False)
                cooked += accums[i].flush()
            else:
                cooked = accums[i].push(obs[i], actions[i], rewards[i],
                                        next_obs[i], False)
            for tr in cooked:
                buffer.push(tr)
            if terms[i] or truncs[i]:
                just_reset[i] = True
                if win_arr is not None and (win_mask is None or win_mask[i]):
                    ep_wins += int(win_arr[i])
                ep_count += 1
                recent_returns.append(float(ep_return[i]))
                recent_returns = recent_returns[-200:]
                ep_return[i] = 0.0
        obs = next_obs
        global_step += args.envs

        if buffer.size >= args.learn_start and \
                global_step % (args.train_every * args.envs) < args.envs:
            beta = min(1.0, args.per_beta0 +
                       (1.0 - args.per_beta0) * global_step / args.total_steps)
            idxs, batch, weights = buffer.sample(args.batch, beta, rng=per_rng)
            loss, td = train_step(online, target, batch, weights, taus,
                                  args.gamma, optimizer, device)
            buffer.update_priorities(idxs, td)
            grad_steps += 1
            if grad_steps % args.target_sync == 0:
                target.load_state_dict(online.state_dict())
            if wandb_run and grad_steps % 200 == 0:
                wandb_run.log({"loss": loss, "epsilon": eps, "beta": beta,
                               "ep_return_mean200": float(np.mean(recent_returns))
                               if recent_returns else 0.0,
                               "win_rate_cum": ep_wins / max(ep_count, 1),
                               "buffer": buffer.size}, step=global_step)

        if global_step - last_log_step >= 20_000:
            rate = (global_step - last_log_step) / (time.time() - last_log_t)
            print(f"[rainbow] step {global_step} | {rate:.0f} steps/s | eps {eps:.3f} "
                  f"| eps hechos {ep_count} wr acum {ep_wins / max(ep_count, 1):.2f} "
                  f"| ret200 {np.mean(recent_returns) if recent_returns else 0:.1f} "
                  f"| grad {grad_steps} | buffer {buffer.size}", flush=True)
            last_log_step, last_log_t = global_step, time.time()

        if global_step % args.ckpt_every < args.envs:
            path = os.path.join(args.out, f"rainbow_step_{global_step:08d}.pt")
            torch.save({"state_dict": online.state_dict(),
                        "meta": {"in_dim": in_dim, "quantiles": args.quantiles,
                                 "hidden": args.hidden, "onehot": not args.no_onehot,
                                 "step": global_step, "args": vars(args)}}, path)

    path = os.path.join(args.out, f"rainbow_final_{global_step:08d}.pt")
    torch.save({"state_dict": online.state_dict(),
                "meta": {"in_dim": in_dim, "quantiles": args.quantiles,
                         "hidden": args.hidden, "onehot": not args.no_onehot,
                         "step": global_step, "args": vars(args)}}, path)
    print(f"[rainbow] listo: {path} ({time.time() - t0:.0f}s)", flush=True)
    venv.close()


if __name__ == "__main__":
    main()
