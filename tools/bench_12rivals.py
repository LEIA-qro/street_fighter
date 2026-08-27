# bench_12rivals.py -- evaluacion limpia sobre LA MISMA rotacion de 12 rivales
# lvl1 que entrena la flota ES (resolve_states("manifest", "1"), identica a la
# de la madre por construccion: mismo commit, misma funcion, mismo manifest).
#
# Dos brazos, cada modelo en su regimen nativo:
#   --arm es   theta actual de la run (GET /theta de la madre, sin ruido),
#              MLPPolicy argmax -- determinista, asi que bastan 2 eps/estado
#              (el segundo solo verifica que el emulador repite bit a bit).
#   --arm ppo  el campeon legacy PPO (models/latest, 39.7M steps) con el
#              mismo cargador del banco historico (v4->v3 + FrozenNorm +
#              predict(deterministic=False) seedeado) -- estocastico, asi que
#              --eps-per-state (default 8) episodios por estado.
#
# La cifra comparable entre brazos y con la curva de W&B es el MEAN de
# fitness_from_episode sobre la rotacion uniforme (la run muestrea estados
# uniformemente, asi que su esperanza es el promedio por-estado).
#
#   .venv/bin/python tools/bench_12rivals.py --arm es
#   .venv/bin/python tools/bench_12rivals.py --arm ppo
#
# Corre con nice 10 y pocos procesos para convivir con el worker de la flota.

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
import urllib.request
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)  # resolve_states usa la ruta relativa del manifest

import numpy as np

from es import openes, protocol, resources
from es.coordinator import resolve_states
from es.policy import DEFAULT_POLICY, POLICIES, wrap_env_for_policy

MAX_EPISODE_STEPS = 20000  # mismo failsafe que el worker

CHAMPION_ZIP = str(REPO / "models/latest/v3/ppo"
                   "/ppo_v3_autocurrTest27_lvl4_plus6_WR83pct_ckpt_39681358steps.zip")
CHAMPION_PKL = CHAMPION_ZIP.replace(".zip", "_vecnorm.pkl")

# --- adaptador v4->v3 + norm congelada: copia exacta de es_finetune_lastlayer
ACT_CATEGORIES, CHAR_CATEGORIES = 256, 16
V3_FRAME = 10 + 2 * ACT_CATEGORIES + 2 * CHAR_CATEGORIES  # 554


def v4_frame_to_v3(frame):
    out = np.zeros(V3_FRAME, dtype=np.float32)
    out[:10] = frame[:10]
    out[10 + min(int(frame[15]), 255)] = 1.0
    out[10 + 256 + min(int(frame[16]), 255)] = 1.0
    out[522 + min(int(frame[21]), 15)] = 1.0
    out[538 + min(int(frame[22]), 15)] = 1.0
    return out


class FrozenNorm:
    def __init__(self, pkl_path):
        import pickle
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


# --- estado por proceso ------------------------------------------------------
_ENV = None
_ES_POLICY = None
_MODEL = _NORM = _FRAMES = None
_RAINBOW = None
_RAINBOW_ONEHOT = True
_RAINBOW_N_ACTIONS = 63
_NOISE = 0.0    # prob. de reemplazar la accion de la politica por una aleatoria
_DESYNC = 0     # frames neutrales (0..K, sorteados) antes de soltar el control

NEUTRAL_ACTION = np.array([0, 0], dtype=np.int64)  # sin direccion, sin boton


def _perturb_rng(state_idx, ep):
    """RNG propio de (estado, episodio): reproducible y sin correlacion entre
    episodios. Salt fijo para no colisionar con ningun otro stream del repo."""
    ss = np.random.SeedSequence(entropy=780537,
                                spawn_key=(int(state_idx), int(ep)))
    return np.random.default_rng(ss)


def _random_action(rng):
    return np.array([rng.integers(0, 9), rng.integers(0, 7)], dtype=np.int64)


def _init_es(theta_bytes, policy_name, nice_delta, noise, desync):
    global _ENV, _ES_POLICY, _NOISE, _DESYNC
    resources.apply_nice(nice_delta)
    _NOISE, _DESYNC = float(noise), int(desync)
    from envs.retro_env import RetroSF2Env
    cls = POLICIES[policy_name]
    # misma envoltura que el worker de flota: macro -> MacroActionWrapper
    _ENV = wrap_env_for_policy(RetroSF2Env(), cls)
    _ES_POLICY = cls(np.frombuffer(theta_bytes, dtype=np.float32).copy())


def _init_ppo(zip_path, pkl_path, nice_delta, noise, desync):
    global _ENV, _MODEL, _NORM, _FRAMES, _NOISE, _DESYNC
    resources.apply_nice(nice_delta)
    _NOISE, _DESYNC = float(noise), int(desync)
    import torch
    torch.set_num_threads(1)
    from stable_baselines3 import PPO
    from envs.retro_env import RetroSF2Env
    _MODEL = PPO.load(zip_path, device="cpu")
    _ENV = RetroSF2Env(trainable=True)
    _NORM = FrozenNorm(pkl_path)
    _FRAMES = deque(maxlen=4)


def _episode_es(task):
    state_name, state_idx, ep = task
    rng = _perturb_rng(state_idx, ep)
    obs, _ = _ENV.reset(options={"state": state_name})
    steps, info = 0, {}
    # desfase de arranque: N frames neutrales le corren la "pelicula" al rival
    # antes de que la politica vea su primer frame util
    for _ in range(int(rng.integers(0, _DESYNC + 1)) if _DESYNC else 0):
        obs, _r, term, trunc, info = _ENV.step(_ES_POLICY.neutral_action())
        steps += 1
        if term or trunc:
            break
    while steps < MAX_EPISODE_STEPS:
        if _NOISE and rng.random() < _NOISE:
            action = _ES_POLICY.random_action(rng)
        else:
            action = _ES_POLICY.act(obs)
        obs, _r, term, trunc, info = _ENV.step(action)
        steps += 1
        if term or trunc:
            break
    return (state_name, ep, openes.fitness_from_episode(info, steps),
            int(info.get("win", 0)), steps)


def _init_rainbow(ckpt_path, nice_delta, noise, desync):
    global _ENV, _RAINBOW, _RAINBOW_ONEHOT, _RAINBOW_N_ACTIONS, _NOISE, _DESYNC
    resources.apply_nice(nice_delta)
    _NOISE, _DESYNC = float(noise), int(desync)
    import torch
    torch.set_num_threads(1)
    from agents.rainbow import QRDuelingNet
    from envs.retro_env import RetroSF2Env
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt["meta"]
    _RAINBOW_N_ACTIONS = int(meta.get("n_actions", 63))
    _RAINBOW = QRDuelingNet(meta["in_dim"], n_actions=_RAINBOW_N_ACTIONS,
                            n_quantiles=meta["quantiles"],
                            hidden=meta["hidden"])
    _RAINBOW.load_state_dict(ckpt["state_dict"])
    _RAINBOW.eval()
    _RAINBOW_ONEHOT = bool(meta.get("onehot", True))
    _ENV = RetroSF2Env()
    if meta.get("macros", False):
        # checkpoint entrenado con macros: mismo wrapper que en entrenamiento
        from envs.macro_wrapper import MacroActionWrapper
        from es.policy import OBS_FRAME_DIM
        _ENV = MacroActionWrapper(_ENV, obs_rel_x_index=2,
                                  frame_size=OBS_FRAME_DIM)


def _episode_rainbow(task):
    state_name, state_idx, ep = task
    import torch
    from es.policy import expand_char_onehot
    # env envuelto (macros) habla acciones PLANAS int; el pelon, MultiDiscrete
    flat = _RAINBOW_N_ACTIONS != 63 or hasattr(_ENV, "obs_rel_x_index")
    rng = _perturb_rng(state_idx, ep)
    obs, _ = _ENV.reset(options={"state": state_name})
    steps, info = 0, {}
    neutral = 0 if flat else NEUTRAL_ACTION
    for _ in range(int(rng.integers(0, _DESYNC + 1)) if _DESYNC else 0):
        obs, _r, term, trunc, info = _ENV.step(neutral)
        steps += 1
        if term or trunc:
            break
    while steps < MAX_EPISODE_STEPS:
        if _NOISE and rng.random() < _NOISE:
            action = (int(rng.integers(0, _RAINBOW_N_ACTIONS)) if flat
                      else _random_action(rng))
        else:
            feats = expand_char_onehot(obs) if _RAINBOW_ONEHOT else obs
            with torch.no_grad():
                q = _RAINBOW.q_values(torch.as_tensor(
                    feats, dtype=torch.float32).unsqueeze(0))
            a = int(q.argmax(dim=1).item())
            if flat:
                action = a
            else:
                move, attack = divmod(a, 7)
                action = np.array([move, attack], dtype=np.int64)
        obs, _r, term, trunc, info = _ENV.step(action)
        steps += 1
        if term or trunc:
            break
    return (state_name, ep, openes.fitness_from_episode(info, steps),
            int(info.get("win", 0)), steps)


def _episode_ppo(task):
    state_name, state_idx, ep = task
    import torch
    # seed estable por (indice de estado, episodio): reproducible entre runs
    # (hash() de str esta aleatorizado por proceso, jamas usarlo aqui)
    torch.manual_seed(7_000_000 + state_idx * 1000 + ep)
    rng = _perturb_rng(state_idx, ep)
    obs92, _ = _ENV.reset(options={"state": state_name})
    _FRAMES.clear()
    for i in range(4):
        _FRAMES.append(v4_frame_to_v3(obs92[i * 23:(i + 1) * 23]))
    steps, info = 0, {}
    for _ in range(int(rng.integers(0, _DESYNC + 1)) if _DESYNC else 0):
        obs92, _r, term, trunc, info = _ENV.step(NEUTRAL_ACTION)
        _FRAMES.append(v4_frame_to_v3(obs92[-23:]))
        steps += 1
        if term or trunc:
            break
    while steps < MAX_EPISODE_STEPS:
        if _NOISE and rng.random() < _NOISE:
            action = _random_action(rng)
        else:
            stacked = _NORM(np.concatenate(_FRAMES))
            action, _ = _MODEL.predict(stacked, deterministic=False)
        obs92, _r, term, trunc, info = _ENV.step(action)
        _FRAMES.append(v4_frame_to_v3(obs92[-23:]))
        steps += 1
        if term or trunc:
            break
    return (state_name, ep, openes.fitness_from_episode(info, steps),
            int(info.get("win", 0)), steps)


def fetch_theta(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    version, theta = protocol.decode_theta(payload)
    return version, theta, str(payload.get("policy", DEFAULT_POLICY))


def report(tag, states, rows, meta, out_path):
    by_state = {s: [] for s in states}
    for s, _ep, f, w, steps in rows:
        by_state[s].append((f, w, steps))
    print(f"\n=== {tag} ===")
    all_f, all_w, n_eps = [], 0, 0
    for s in states:
        runs = by_state[s]
        fs = [f for f, _w, _st in runs]
        ws = sum(w for _f, w, _st in runs)
        all_f += fs
        all_w += ws
        n_eps += len(runs)
        print(f"  {s:<40} fit={np.mean(fs):+.3f}  wins={ws}/{len(runs)}")
    mean_f = float(np.mean(all_f))
    wr = all_w / n_eps
    print(f"  {'TOTAL':<40} fit={mean_f:+.3f}  win_rate={all_w}/{n_eps}={wr:.3f}")
    row = dict(meta, tag=tag, mean_fitness=mean_f, win_rate=wr,
               episodes=n_eps, wins=all_w,
               per_state={s: {"fitness": float(np.mean([f for f, _w, _st in by_state[s]])),
                              "wins": sum(w for _f, w, _st in by_state[s]),
                              "episodes": len(by_state[s])} for s in states},
               states_fingerprint=protocol.states_fingerprint(states))
    with open(out_path, "a") as f:
        f.write(json.dumps(row) + "\n")
    return mean_f, wr


def main():
    ap = argparse.ArgumentParser(description="Banco limpio sobre los 12 rivales lvl1 de la run")
    ap.add_argument("--arm", choices=["es", "ppo", "rainbow"], required=True)
    ap.add_argument("--difficulty", default="1",
                    help="que rotacion del manifest examinar (ej. 2 o 2,3): "
                         "el examen deja de ser solo lvl1")
    ap.add_argument("--ckpt", default=None,
                    help="rainbow: ruta al .pt guardado por train_rainbow.py")
    ap.add_argument("--theta-url", default="http://madre:8080/theta")
    ap.add_argument("--theta-npz", default=None,
                    help="alternativa offline: .npz de checkpoint con clave theta")
    ap.add_argument("--policy", default=DEFAULT_POLICY, choices=sorted(POLICIES),
                    help="solo con --theta-npz (el /theta de la madre trae el suyo)")
    ap.add_argument("--zip", default=CHAMPION_ZIP)
    ap.add_argument("--pkl", default=CHAMPION_PKL)
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--eps-per-state", type=int, default=None,
                    help="default: 2 para es (determinismo x2), 8 para ppo")
    ap.add_argument("--nice", type=int, default=10)
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="prob. por paso de reemplazar la accion por una aleatoria "
                         "(prueba de robustez: 0.05 = 5%% de los pasos)")
    ap.add_argument("--desync-max", type=int, default=0,
                    help="hasta N frames neutrales (sorteados por episodio) antes "
                         "de soltar el control: rompe la coreografia del arranque")
    ap.add_argument("--out", default=str(REPO / "benchmarks/bench_12rivals.jsonl"))
    args = ap.parse_args()

    states = resolve_states("manifest", args.difficulty)
    print(f"[bench] rotacion: {len(states)} estados, fingerprint "
          f"{protocol.states_fingerprint(states)}")
    perturbed = args.action_noise > 0 or args.desync_max > 0
    if perturbed:
        print(f"[bench] PERTURBADO: action_noise={args.action_noise} "
              f"desync_max={args.desync_max}")

    # es y rainbow son deterministas (argmax): 2 eps solo verifican; ppo
    # muestrea y necesita mas. Con perturbaciones todos necesitan muestra.
    eps = args.eps_per_state or (2 if args.arm in ("es", "rainbow") and not perturbed
                                 else 8)
    tasks = [(s, i, e) for i, s in enumerate(states) for e in range(eps)]
    ctx = mp.get_context("spawn")
    t0 = time.time()

    if args.arm == "es":
        if args.theta_npz:
            theta = np.load(args.theta_npz)["theta"].astype(np.float32)
            version = f"npz:{os.path.basename(args.theta_npz)}"
            policy_name = args.policy
        else:
            version, theta, policy_name = fetch_theta(args.theta_url)
        if theta.shape[0] != POLICIES[policy_name].num_params():
            raise SystemExit(f"[bench] theta {theta.shape[0]} params no cuadra con "
                             f"policy '{policy_name}' ({POLICIES[policy_name].num_params()})")
        print(f"[bench] theta version/gen {version}, policy {policy_name}, "
              f"{theta.shape[0]} params, ||theta||={float(np.linalg.norm(theta)):.3f}")
        pool = ctx.Pool(args.procs, initializer=_init_es,
                        initargs=(theta.tobytes(), policy_name, args.nice,
                                  args.action_noise, args.desync_max))
        rows = pool.map(_episode_es, tasks)
        meta = {"model": "es_theta", "theta_version": str(version), "policy": policy_name}
        tag = f"ES theta (gen {version}, {policy_name}) argmax"
        # verificacion de determinismo: mismo estado -> episodios identicos.
        # Solo aplica SIN perturbaciones (con ellas, variar es el punto).
        if eps >= 2 and not perturbed:
            drift = [s for s in states
                     if len({(f, st) for s2, _e, f, _w, st in rows if s2 == s}) > 1]
            print("[bench] determinismo: " +
                  ("OK (todas las repeticiones identicas)" if not drift
                   else f"OJO, {len(drift)} estados variaron: {drift}"))
    elif args.arm == "rainbow":
        if not args.ckpt:
            raise SystemExit("[bench] --arm rainbow requiere --ckpt")
        pool = ctx.Pool(args.procs, initializer=_init_rainbow,
                        initargs=(args.ckpt, args.nice,
                                  args.action_noise, args.desync_max))
        rows = pool.map(_episode_rainbow, tasks)
        meta = {"model": "rainbow", "ckpt": os.path.basename(args.ckpt)}
        tag = f"Rainbow {os.path.basename(args.ckpt)} greedy"
        if eps >= 2 and not perturbed:  # greedy = determinista, igual que es
            drift = [s for s in states
                     if len({(f, st) for s2, _e, f, _w, st in rows if s2 == s}) > 1]
            print("[bench] determinismo: " +
                  ("OK (todas las repeticiones identicas)" if not drift
                   else f"OJO, {len(drift)} estados variaron: {drift}"))
    else:
        pool = ctx.Pool(args.procs, initializer=_init_ppo,
                        initargs=(args.zip, args.pkl, args.nice,
                                  args.action_noise, args.desync_max))
        rows = pool.map(_episode_ppo, tasks)
        meta = {"model": "ppo", "zip": os.path.basename(args.zip)}
        tag = f"PPO {os.path.basename(args.zip)} predict estocastico"
    meta["action_noise"] = args.action_noise
    meta["desync_max"] = args.desync_max
    if perturbed:
        tag += f" [noise={args.action_noise} desync<={args.desync_max}]"

    pool.close()
    pool.join()
    report(tag, states, rows, meta, args.out)
    print(f"[bench] {len(tasks)} episodios en {time.time() - t0:.0f}s -> {args.out}")


if __name__ == "__main__":
    main()
