# watch_es.py -- visor en vivo del theta del ES: abre la ventana del emulador
# y lo pone a pelear contra la rotacion (o un rival fijo) a velocidad real.
#
#   .venv/bin/python tools/watch_es.py                        # theta vivo de la madre
#   .venv/bin/python tools/watch_es.py --state RYU_KEN_R1_lvl1
#   .venv/bin/python tools/watch_es.py --theta-npz benchmarks/run2_final/theta_final.npz --policy v4onehot
#   .venv/bin/python tools/watch_es.py --desync-max 30        # verlo en modo perturbado
#
# Al final de cada episodio imprime el resultado + ep_air_frac (LA metrica de
# "¿camina o brinca?": PPO saltarin ~0.47, random ~0.33, caminar de verdad
# <0.15) y la distancia mediana. Ctrl+C para salir.

import argparse
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np

from es import protocol
from es.coordinator import resolve_states
from es.policy import DEFAULT_POLICY, POLICIES

NEUTRAL = np.array([0, 0], dtype=np.int64)


def fetch_theta(url):
    with urllib.request.urlopen(url, timeout=30) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    version, theta = protocol.decode_theta(payload)
    return version, theta, str(payload.get("policy", DEFAULT_POLICY))


def main():
    ap = argparse.ArgumentParser(description="Visor en vivo del theta ES")
    ap.add_argument("--theta-url", default="http://madre:8080/theta")
    ap.add_argument("--theta-npz", default=None)
    ap.add_argument("--policy", default=DEFAULT_POLICY, choices=sorted(POLICIES),
                    help="solo con --theta-npz")
    ap.add_argument("--state", default=None,
                    help="un rival fijo (p.ej. RYU_KEN_R1_lvl1); default: rota los 12")
    ap.add_argument("--difficulty", default="1")
    ap.add_argument("--desync-max", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="1.0 = tiempo real (~15 acciones/s); 2.0 = doble")
    args = ap.parse_args()

    if args.theta_npz:
        theta = np.load(args.theta_npz)["theta"].astype(np.float32)
        version, policy_name = f"npz:{os.path.basename(args.theta_npz)}", args.policy
    else:
        version, theta, policy_name = fetch_theta(args.theta_url)
    policy = POLICIES[policy_name](theta)
    states = [args.state] if args.state else resolve_states("manifest",
                                                            args.difficulty)
    print(f"[visor] theta gen {version} ({policy_name}) | rivales: "
          f"{', '.join(s.replace('RYU_', '').replace('_R1_lvl1', '') for s in states)}")
    print("[visor] Ctrl+C para salir\n")

    from envs.retro_env import RetroSF2Env
    env = RetroSF2Env(render_mode="human")
    rng = np.random.default_rng(0)
    step_dt = (4 / 60.0) / max(args.speed, 0.01)  # frameskip 4 sobre 60 fps
    wins, count = 0, 0
    try:
        while True:
            state = states[count % len(states)]
            rival = state.replace("RYU_", "").replace("_R1_lvl1", "")
            obs, _ = env.reset(options={"state": state})
            for _ in range(int(rng.integers(0, args.desync_max + 1))
                           if args.desync_max else 0):
                obs, _r, term, trunc, info = env.step(NEUTRAL)
                time.sleep(step_dt)
            steps, info = 0, {}
            t0 = time.time()
            while True:
                obs, _r, term, trunc, info = env.step(policy.act(obs))
                steps += 1
                # marcapasos a tiempo real (la ventana ya ata a vsync, esto
                # solo evita correr rapido si el vsync no aplica)
                lag = t0 + steps * step_dt - time.time()
                if lag > 0:
                    time.sleep(lag)
                if term or trunc:
                    break
            win = int(info.get("win", 0) or 0)
            wins += win
            count += 1
            air = info.get("ep_air_frac")
            dist = info.get("ep_rel_dist_median")
            air_txt = (f"air_frac {air:.2f}" + (" <- CAMINA" if air < 0.25 else "")
                       ) if air is not None else "air_frac ?"
            dist_txt = f"dist_med {dist:.0f}" if dist is not None else ""
            print(f"[{count:3d}] vs {rival:9} -> "
                  f"{'GANA' if win else 'pierde'}  "
                  f"hp {int(info.get('my_hp', 0)):3d}-{int(info.get('enemy_hp', 0)):3d}  "
                  f"{air_txt}  {dist_txt}  ({steps} pasos, "
                  f"{wins}/{count} global)", flush=True)
    except KeyboardInterrupt:
        print(f"\n[visor] {wins}/{count} victorias")
    finally:
        env.close()


if __name__ == "__main__":
    main()
