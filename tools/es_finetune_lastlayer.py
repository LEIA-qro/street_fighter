# es_finetune_lastlayer.py -- "etapa 2" en miniatura, en una sola maquina.
#
# Toma la MEJOR politica PPO que este proyecto ha producido (el checkpoint v3
# de 39.7M steps que vive en models/latest/), congela sus 3.1M de parametros
# menos la ULTIMA capa (action_net: ~4k), y deja que OpenES optimice esa capa
# contra el objetivo real -- ganar el round -- evaluando en el backend retro
# headless de esta maquina. Es el pipeline PPO->ES del handoff (7.6: "gradient
# free fine-tuning pass on a frozen, already-trained policy, perturbing only
# the last layer") sin flota ni madre: pool local de procesos, mismo modulo
# openes.py que usa la flota, misma fitness_from_episode.
#
# El checkpoint es v3 (obs one-hot de 554x4) y RetroSF2Env habla v4 (23x4).
# El frame v4 CONTIENE todo lo que el layout v3 codifica, asi que el adaptador
# de abajo reconstruye el frame v3 exacto (mismos 10 continuos + one-hots de
# los mismos IDs) y normaliza con las stats congeladas del entrenamiento
# (el pkl de SelectiveVecNormalize, leido crudo).
#
#   .venv/bin/python tools/es_finetune_lastlayer.py --generations 30
#
# Caveat documentado: el checkpoint se entreno en BizHawk (obs con 1 paso de
# retraso) y bajo el reward viejo; aqui se EVALUA con la semantica corregida.
# La fitness no usa shaping -- win + margen de vida + costo de pasos -- asi
# que el reward viejo no contamina la medicion, solo el comportamiento base.

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import numpy as np

from es import openes, resources

DEFAULT_ZIP = str(REPO / "models/latest/v3/ppo"
                  "/ppo_v3_autocurrTest27_lvl4_plus6_WR83pct_ckpt_39681358steps.zip")
DEFAULT_PKL = DEFAULT_ZIP.replace(".zip", "_vecnorm.pkl")

ACT_CATEGORIES, CHAR_CATEGORIES = 256, 16
V3_FRAME = 10 + 2 * ACT_CATEGORIES + 2 * CHAR_CATEGORIES  # 554


def v4_frame_to_v3(frame):
    """23-float v4 frame -> 554-float v3 frame (same values, one-hot layout).

    v4 indices (sf2_v4.py): [0:10] continuos identicos a v3, [15] p1_act_hi,
    [16] p2_act_hi, [21] p1_char, [22] p2_char. v3 = cont + one-hots de esos
    mismos IDs, en ese orden (base_env._parse_payload).
    """
    out = np.zeros(V3_FRAME, dtype=np.float32)
    out[:10] = frame[:10]
    out[10 + min(int(frame[15]), 255)] = 1.0
    out[10 + 256 + min(int(frame[16]), 255)] = 1.0
    out[522 + min(int(frame[21]), 15)] = 1.0
    out[538 + min(int(frame[22]), 15)] = 1.0
    return out


class FrozenNorm:
    """Las stats congeladas del pkl de SelectiveVecNormalize, aplicadas igual
    que en entrenamiento: solo los 10 dims continuos de CADA frame apilado."""

    def __init__(self, pkl_path):
        with open(pkl_path, "rb") as f:
            stats = pickle.load(f)
        assert stats["n_cont"] == 10 and stats["n_frames"] == 4, stats.keys()
        self.mean = np.asarray(stats["running_mean"], dtype=np.float64)
        self.std = np.sqrt(np.asarray(stats["running_var"], dtype=np.float64) + 1e-8)
        self.clip = float(stats.get("clip", 10.0))

    def __call__(self, stacked_2216):
        obs = stacked_2216.reshape(4, V3_FRAME)
        cont = (obs[:, :10] - self.mean) / self.std
        obs = obs.copy()
        obs[:, :10] = np.clip(cont, -self.clip, self.clip)
        return obs.reshape(-1).astype(np.float32)


# --- estado por proceso worker (un emulador + una copia del modelo) ---------
_ENV = _MODEL = _NORM = None
_FRAMES = None


def _init_worker(zip_path, pkl_path, nice_delta):
    global _ENV, _MODEL, _NORM, _FRAMES
    if nice_delta:
        resources.apply_nice(nice_delta)
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from envs.retro_env import RetroSF2Env
    _MODEL = PPO.load(zip_path, device="cpu")
    _ENV = RetroSF2Env(trainable=True)
    _NORM = FrozenNorm(pkl_path)
    _FRAMES = deque(maxlen=4)


def _theta_slices(policy):
    w, b = policy.action_net.weight, policy.action_net.bias
    return w, b, w.numel()


def _set_last_layer(policy, theta):
    import torch
    w, b, nw = _theta_slices(policy)
    with torch.no_grad():
        w.copy_(torch.from_numpy(theta[:nw].reshape(w.shape)))
        b.copy_(torch.from_numpy(theta[nw:]))


def _run_episode(seed):
    import torch
    torch.manual_seed(seed)
    obs92, _ = _ENV.reset()
    _FRAMES.clear()
    for i in range(4):
        _FRAMES.append(v4_frame_to_v3(obs92[i * 23:(i + 1) * 23]))
    steps, info = 0, {}
    while True:
        stacked = _NORM(np.concatenate(_FRAMES))
        action, _ = _MODEL.predict(stacked, deterministic=False)
        obs92, _r, term, trunc, info = _ENV.step(action)
        _FRAMES.append(v4_frame_to_v3(obs92[-23:]))
        steps += 1
        if term or trunc:
            return openes.fitness_from_episode(info, steps), int(info.get("win", 0))


def _eval_member(task):
    idx, theta_bytes, episodes, base_seed = task
    theta = np.frombuffer(theta_bytes, dtype=np.float32).copy()
    _set_last_layer(_MODEL.policy, theta)
    fits, wins = [], 0
    for e in range(episodes):
        f, w = _run_episode(base_seed * 1000 + e)
        fits.append(f)
        wins += w
    return idx, float(np.mean(fits)), wins, episodes


def main():
    ap = argparse.ArgumentParser(description="ES last-layer fine-tune de un PPO congelado")
    ap.add_argument("--zip", default=DEFAULT_ZIP)
    ap.add_argument("--pkl", default=DEFAULT_PKL)
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--pop", type=int, default=64, help="miembros por generacion (par)")
    ap.add_argument("--episodes", type=int, default=2, help="episodios por miembro")
    ap.add_argument("--generations", type=int, default=30)
    ap.add_argument("--baseline-episodes", type=int, default=24)
    ap.add_argument("--sigma-rel", type=float, default=0.3,
                    help="sigma = este factor x std(theta0): perturbacion relativa a la escala real de la capa")
    ap.add_argument("--lr-rel", type=float, default=0.5, help="lr = este factor x sigma")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--out", default=str(REPO / "benchmarks/es_finetune_log.jsonl"))
    args = ap.parse_args()

    from stable_baselines3 import PPO  # el padre tambien carga uno, para extraer theta0
    import torch
    torch.set_num_threads(1)
    model = PPO.load(args.zip, device="cpu")
    w, b, nw = _theta_slices(model.policy)
    theta0 = np.concatenate([w.detach().numpy().ravel(),
                             b.detach().numpy().ravel()]).astype(np.float32)
    sigma = args.sigma_rel * float(theta0.std())
    lr = args.lr_rel * sigma
    print(f"[es-ft] theta0: {theta0.size} params (ultima capa de "
          f"{sum(p.numel() for p in model.policy.parameters())}) | "
          f"std={theta0.std():.4f} -> sigma={sigma:.4f} lr={lr:.4f}", flush=True)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(args.procs, initializer=_init_worker,
                    initargs=(args.zip, args.pkl, args.nice))

    def evaluate(theta, n_eps, tag, base_seed):
        theta_b = theta.astype(np.float32).tobytes()
        per = max(1, n_eps // args.procs)
        tasks = [(i, theta_b, per, base_seed + i) for i in range(args.procs)]
        outs = pool.map(_eval_member, tasks)
        fits = [f for _i, f, _w, _e in outs]
        wins = sum(w for _i, _f, w, _e in outs)
        eps = sum(e for _i, _f, _w, e in outs)
        print(f"[es-ft] {tag}: fitness={np.mean(fits):+.3f} "
              f"win_rate={wins}/{eps}={wins / eps:.2f}", flush=True)
        return float(np.mean(fits)), wins / eps

    t0 = time.time()
    base_fit, base_wr = evaluate(theta0, args.baseline_episodes, "BASELINE (PPO congelado)", 7_000_000)

    state = openes.init_state(theta0, sigma, lr, weight_decay=0.0, master_seed=4242)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best = (base_fit, theta0.copy(), -1)
    for g in range(args.generations):
        members = openes.members_for_generation(state, args.pop)
        theta_by_member = {i: openes.member_theta(state, s, sign)
                           for i, s, sign in members}
        tasks = [(i, theta_by_member[i].astype(np.float32).tobytes(),
                  args.episodes, state.master_seed * 100 + g * args.pop + i)
                 for i, _s, _sig in members]
        outs = pool.map(_eval_member, tasks)
        outs.sort(key=lambda o: o[0])
        fits = np.array([f for _i, f, _w, _e in outs])
        wins = sum(w for _i, _f, w, _e in outs)
        eps = sum(e for _i, _f, _w, e in outs)
        state = openes.es_update(state, fits)
        gen_best = float(fits.max())
        if gen_best > best[0]:
            best = (gen_best, theta_by_member[int(np.argmax(fits))].copy(), g)
        row = {"gen": g, "fit_mean": float(fits.mean()), "fit_max": gen_best,
               "win_rate": wins / eps, "sigma": sigma, "elapsed_s": round(time.time() - t0, 1)}
        print(f"[es-ft] gen {g:02d}: mean={row['fit_mean']:+.3f} max={gen_best:+.3f} "
              f"win_rate={row['win_rate']:.2f} ({row['elapsed_s']:.0f}s)", flush=True)
        with open(args.out, "a") as f:
            f.write(json.dumps(row) + "\n")

    final_fit, final_wr = evaluate(state.theta, args.baseline_episodes, "FINAL (media de ES)", 9_000_000)
    np.savez(str(REPO / "benchmarks/es_finetune_best.npz"),
             theta=best[1], theta_mean=state.theta, theta0=theta0)
    print(f"\n[es-ft] RESUMEN  baseline fit={base_fit:+.3f} wr={base_wr:.2f}  ->  "
          f"final fit={final_fit:+.3f} wr={final_wr:.2f}  "
          f"(mejor individuo: {best[0]:+.3f} en gen {best[2]})", flush=True)
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
