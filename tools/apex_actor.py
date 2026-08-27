# apex_actor.py -- actor del Rainbow distribuido (Ape-X). Uno por maquina.
#
#   .venv/bin/python tools/apex_actor.py --learner http://legion-wsl:8090 --procs 8
#
# Cada PROCESO corre un emulador con su propio epsilon (la escalera de Ape-X
# via agents.apex.apex_epsilon), cocina n-step localmente y manda las
# transiciones al padre; el padre las POSTea en lotes comprimidos y refresca
# los pesos del learner cada pocos segundos (archivo compartido + version en
# memoria compartida: los hijos recargan al ver la version cambiar).
#
# La config de aprendizaje (gamma, n_step, onehot, arquitectura) viene DEL
# LEARNER dentro de /weights -- este script no lleva hiperparametros de
# aprendizaje, solo de maquina (procs, estados, desync). Un actor con codigo
# viejo truena ruidoso al cargar pesos incompatibles (load_state_dict
# estricto), jamas alimenta el buffer con features equivocadas.
#
# Mismo contrato de robustez que el worker ES: red caida = backoff, jamas
# crash; SIGTERM/SIGINT terminan limpio. Los emuladores van con nice.

import argparse
import json
import multiprocessing as mp
import os
import signal
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np

from agents.apex import apex_epsilon, encode_transitions
from es.coordinator import resolve_states

FLUSH_TRANSITIONS = 800     # transiciones por POST
WEIGHTS_REFRESH_S = 5.0     # cadencia con la que el padre jala /weights
_STOP = False


def _log(msg):
    print(f"[actor] {msg}", flush=True)


def _http_json(url, body=None, timeout=60):
    try:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"} if body else {})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError) as e:
        _log(f"{url}: {e}")
        return None


# ---------------------------------------------------------------------------
# Proceso hijo: un emulador, un epsilon, un acumulador n-step
# ---------------------------------------------------------------------------

def _child(rank, procs, states, desync_max, weights_path, version_value,
           out_queue, nice_delta, seed):
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    import torch
    torch.set_num_threads(1)
    from agents.rainbow import NStepAccumulator, QRDuelingNet
    from es import resources
    from es.policy import expand_char_onehot
    from envs.discrete_sf2 import make_discrete_sf2
    if nice_delta:
        resources.apply_nice(nice_delta)

    # espera la primera version de pesos (el padre la baja antes de arrancar)
    while version_value.value < 0:
        time.sleep(0.2)
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    net = QRDuelingNet(config["in_dim"],
                       n_actions=int(config.get("n_actions", 63)),
                       n_quantiles=config["quantiles"],
                       hidden=config["hidden"])
    net.load_state_dict(ckpt["state"])  # estricto: actor stale truena aqui
    net.eval()
    onehot = bool(config["onehot"])
    accum = NStepAccumulator(config["n_step"], config["gamma"])
    eps = apex_epsilon(rank, procs)
    rng = np.random.default_rng(np.random.SeedSequence(
        entropy=int(seed), spawn_key=(int(rank),)))
    # obs crudas (92) en el wire: onehot=False en la fabrica; la expansion
    # one-hot es solo para el forward local
    n_actions = int(config.get("n_actions", 63))
    env = make_discrete_sf2(states, seed=seed * 100 + rank,
                            desync_max=desync_max, onehot=False,
                            macros=bool(config.get("macros", False)))
    my_version = version_value.value
    reload_failures = 0

    obs, _ = env.reset()
    ep_return, wins, count = 0.0, 0, 0
    while True:
        if version_value.value != my_version:
            try:
                ckpt = torch.load(weights_path, map_location="cpu",
                                  weights_only=False)
                if ckpt["config"] != config:
                    # learner reiniciado con OTRO run (gamma/arch/features):
                    # cocinar transiciones con la config vieja seria drift
                    # silencioso del objetivo. Ruidoso, como el worker ES.
                    raise SystemExit(
                        f"[actor:{rank}] la config del learner cambio "
                        f"({ckpt['config']} != {config}) -- relanzar el actor")
                net.load_state_dict(ckpt["state"])
                my_version = version_value.value
                reload_failures = 0
            except SystemExit:
                raise
            except Exception:
                # el padre puede estar reescribiendo el archivo; un fallo
                # transitorio se reintenta, un fallo PERSISTENTE truena
                reload_failures += 1
                if reload_failures >= 5:
                    raise
        if rng.random() < eps:
            action = int(rng.integers(0, n_actions))
        else:
            feats = expand_char_onehot(obs) if onehot else obs
            with torch.no_grad():
                q = net.q_values(torch.as_tensor(
                    np.asarray(feats, dtype=np.float32)).unsqueeze(0))
            action = int(q.argmax(dim=1).item())
        next_obs, reward, term, trunc, info = env.step(action)
        cooked = accum.push(obs, action, reward, next_obs, bool(term))
        if trunc and not term:
            cooked += accum.flush()
        for t in cooked:
            out_queue.put(("t", t))
        if term or trunc:
            wins += int(info.get("win", 0) or 0)
            count += 1
            out_queue.put(("ep", {"wins": int(info.get("win", 0) or 0),
                                  "count": 1}))
            obs, _ = env.reset()
        else:
            obs = next_obs


# ---------------------------------------------------------------------------
# Padre: pesos frescos + lotes al learner
# ---------------------------------------------------------------------------

def fetch_weights(learner_url, weights_path, version_value):
    """Baja /weights; escribe atomico y publica la version. -> version o None."""
    import torch
    payload = _http_json(f"{learner_url}/weights")
    if payload is None:
        return None
    from agents.apex import decode_weights
    version, config, state = decode_weights(payload)
    tmp = weights_path + ".tmp"
    torch.save({"config": config, "state": state}, tmp)
    os.replace(tmp, weights_path)
    version_value.value = version
    return version


def main():
    ap = argparse.ArgumentParser(description="Ape-X actor")
    ap.add_argument("--learner", required=True, help="http://host:port")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--states", default="manifest")
    ap.add_argument("--difficulty", default="1")
    ap.add_argument("--desync-max", type=int, default=30)
    ap.add_argument("--nice", type=int, default=10 if hasattr(os, "nice") else 0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--seed", type=int, default=20260827)
    args = ap.parse_args()

    learner_url = args.learner.rstrip("/")
    name = args.name or f"{socket.gethostname()}-{os.getpid()}"
    states = resolve_states(args.states, args.difficulty)
    # seed UNICO por maquina/proceso: con el default compartido, dos actores
    # en maquinas distintas emitirian streams de experiencia byte-identicos
    # (mismo emulador determinista + mismos sorteos) -- M maquinas llenando
    # el buffer con duplicados M-plicados. El nombre entra al seed.
    import zlib
    seed = args.seed ^ zlib.crc32(name.encode("utf-8"))

    def _sigterm(_s, _f):
        global _STOP
        _STOP = True
    signal.signal(signal.SIGTERM, _sigterm)
    signal.signal(signal.SIGINT, _sigterm)

    ctx = mp.get_context("spawn")
    version_value = ctx.Value("i", -1)
    # 30k < SEM_VALUE_MAX de macOS (32767): un maxsize mayor revienta el
    # BoundedSemaphore del Queue con EINVAL en Mac (en Linux ni se nota)
    out_queue = ctx.Queue(maxsize=30_000)
    weights_path = os.path.join(tempfile.gettempdir(),
                                f"apex_weights_{os.getpid()}.pt")

    # primera bajada de pesos ANTES de parir emuladores: si el learner no
    # esta, backoff aqui y no 8 emuladores huerfanos
    while fetch_weights(learner_url, weights_path, version_value) is None:
        if _STOP:
            return
        _log("learner inalcanzable; reintento en 10s")
        time.sleep(10)

    eps_list = [round(apex_epsilon(r, args.procs), 4) for r in range(args.procs)]
    _log(f"{name} -> {learner_url} | procs={args.procs} | epsilons={eps_list} "
         f"| estados={len(states)} desync<={args.desync_max}")

    children = [ctx.Process(target=_child,
                            args=(r, args.procs, states, args.desync_max,
                                  weights_path, version_value, out_queue,
                                  args.nice, seed), daemon=True)
                for r in range(args.procs)]
    for c in children:
        c.start()

    batch, ep_wins, ep_count = [], 0, 0
    sent_total, t0, last_weights = 0, time.time(), time.time()
    last_rate_t, last_rate_n = time.time(), 0
    try:
        while not _STOP:
            try:
                kind, item = out_queue.get(timeout=1.0)
            except Exception:
                kind = None
            if kind == "t":
                batch.append(item)
            elif kind == "ep":
                ep_wins += item["wins"]
                ep_count += item["count"]
            if len(batch) >= FLUSH_TRANSITIONS:
                # POSTs SIEMPRE de tamano fijo: durante una caida del learner
                # la cola local sigue creciendo, y un lote gigante seria
                # rechazado por MAX_BATCH_B64_BYTES al volver -- perdida
                # permanente en loop. Se manda una rebanada y el resto espera.
                chunk = batch[:FLUSH_TRANSITIONS]
                dt = max(time.time() - last_rate_t, 1e-6)
                rate = (sent_total - last_rate_n + len(chunk)) / dt
                body = {"actor": name, "b64": encode_transitions(chunk),
                        "stats": {"procs": args.procs,
                                  "steps_per_s": round(rate, 1),
                                  "host": socket.gethostname()},
                        "episodes": {"wins": ep_wins, "count": ep_count}}
                resp = _http_json(f"{learner_url}/transitions", body=body)
                if resp is not None:
                    del batch[:FLUSH_TRANSITIONS]
                    sent_total += len(chunk)
                    ep_wins, ep_count = 0, 0
                    if sent_total % (FLUSH_TRANSITIONS * 10) == 0:
                        _log(f"{sent_total} transiciones | {rate:.0f} trans/s | "
                             f"buffer learner {resp.get('buffer')}")
                    last_rate_t, last_rate_n = time.time(), sent_total
                else:
                    time.sleep(5)  # learner caido: reintento con la MISMA rebanada
                    # cola local acotada: replay es prescindible, la RAM no.
                    # Se descarta lo MAS VIEJO (lo mas lejos de la politica actual)
                    cap = FLUSH_TRANSITIONS * 20
                    if len(batch) > cap:
                        dropped = len(batch) - cap
                        del batch[:dropped]
                        _log(f"learner caido: descartadas {dropped} transiciones viejas")
            if time.time() - last_weights >= WEIGHTS_REFRESH_S:
                fetch_weights(learner_url, weights_path, version_value)
                last_weights = time.time()
    finally:
        for c in children:
            c.terminate()
        for c in children:
            c.join(timeout=5)
        try:
            os.remove(weights_path)
        except OSError:
            pass
        _log(f"{name} detenido ({sent_total} transiciones enviadas)")


if __name__ == "__main__":
    main()
