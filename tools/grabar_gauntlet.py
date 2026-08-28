# grabar_gauntlet.py -- graba UNA PELEA COMPLETA contra los 12 rivales.
#
#   .venv/bin/python tools/grabar_gauntlet.py --difficulty 8
#
# Peleas de verdad, al mejor de tres, no rounds sueltos. Hasta hoy TODO lo que
# habiamos medido y visto era el PRIMER ROUND: retro_env.py:512 hace
# `terminated = ko if self.trainable else False`, asi que con trainable=True el
# episodio se corta en el primer KO y nunca hay round 2. De ahi que en pantalla
# nunca se viera una victoria: solo un KO y un corte a otro escenario.
# Este grabador usa trainable=False -- el juego sigue solo a round 2 y 3 como
# el juego real -- y cuenta rounds de la RAM hasta que alguien llega a dos.
#
# El video sale a 60 fps reales gracias al gancho por frame de RetroSF2Env
# (muestrear una vez por paso de agente daria 15 fps y se ve a tirones), con
# escalado 2x de vecino mas cercano para que el pixel art no se enjuague, y
# comprimido a un tamano objetivo para que se pueda mandar por WhatsApp.

import argparse
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "src"))
os.chdir(REPO)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from envs.retro_env import RetroSF2Env  # noqa: E402
from envs.macro_wrapper import MacroActionWrapper  # noqa: E402
from es.policy import OBS_FRAME_DIM, expand_char_onehot  # noqa: E402
from agents.rainbow import QRDuelingNet  # noqa: E402

RIVALES = ("BALROG", "BLANKA", "CHUNLI", "DHALSIM", "EHONDA", "GUILE",
           "KEN", "MBISON", "RYU", "SAGAT", "VEGA", "ZANGIEF")
ANCHO, ALTO, FPS = 320, 224, 60
CAMPEON = os.path.join("benchmarks", "apex_milestones", "apex_v3291_media990.pt")


def wilson(k, n, z=1.96):
    """Intervalo de confianza al 95%. Un porcentaje sin su intervalo, con n
    chica, es una opinion con aires de medicion."""
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    centro = (p + z * z / (2 * n)) / d
    margen = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, centro - margen), 3), round(min(1.0, centro + margen), 3)]


def cargar(ckpt_path, device="cpu"):
    torch.set_num_threads(1)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = ck["meta"]
    net = QRDuelingNet(m["in_dim"], n_actions=m["n_actions"],
                       n_quantiles=m["quantiles"], hidden=m["hidden"])
    net.load_state_dict(ck["state_dict"])
    net.to(device).eval()
    onehot = bool(m.get("onehot", True))
    macros = bool(m.get("macros", False))

    def actuar(obs):
        feats = expand_char_onehot(obs) if onehot else obs
        with torch.no_grad():
            q = net.q_values(torch.as_tensor(feats, dtype=torch.float32,
                                             device=device).unsqueeze(0))
        return int(q.argmax(dim=1).item())
    return actuar, macros, m


def abrir_ffmpeg(destino, crf):
    return subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{ANCHO}x{ALTO}", "-r", str(FPS), "-i", "-",
         # vecino mas cercano: el pixel art se agranda, no se difumina
         "-vf", f"scale={ANCHO * 2}:{ALTO * 2}:flags=neighbor",
         "-c:v", "libx264", "-preset", "slow", "-crf", str(crf),
         "-pix_fmt", "yuv420p", "-movflags", "+faststart", destino],
        stdin=subprocess.PIPE)


def recomprimir(entrada, salida, objetivo_mb):
    """Segunda pasada solo si hizo falta: clava el tamano sin regalar calidad."""
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", entrada], capture_output=True, text=True).stdout.strip())
    kbps = int((objetivo_mb * 1024 * 1024 * 8 / dur) / 1000) - 8
    base = ["ffmpeg", "-y", "-v", "error", "-i", entrada, "-c:v", "libx264",
            "-preset", "slow", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p", "-an"]
    subprocess.run(base + ["-pass", "1", "-f", "mp4", os.devnull], check=True)
    subprocess.run(base + ["-pass", "2", "-movflags", "+faststart", salida], check=True)
    for resto in ("ffmpeg2pass-0.log", "ffmpeg2pass-0.log.mbtree"):
        if os.path.exists(resto):
            os.remove(resto)


def main():
    ap = argparse.ArgumentParser(description="Graba una pelea completa contra los 12 rivales")
    ap.add_argument("--ckpt", default=CAMPEON)
    ap.add_argument("--difficulty", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--target-mb", type=float, default=16.0,
                    help="tamano objetivo del video (0 = sin limite)")
    ap.add_argument("--rounds-to-win", type=int, default=2, help="2 = al mejor de tres")
    ap.add_argument("--max-steps", type=int, default=6000, help="tope por pelea")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--repeticiones", type=int, default=1,
                    help="peleas por rival. Con 1 la n es 12 y el intervalo de "
                         "confianza va del 55%% al 95%%: sirve para el video y "
                         "para saber CONTRA QUIEN pierde, no como porcentaje.")
    ap.add_argument("--desync-max", type=int, default=0,
                    help="frames neutrales sorteados antes de soltar el control. "
                         "SIN esto la pelea es determinista y repetir no aporta "
                         "informacion: con greedy y estado fijo sale identica.")
    ap.add_argument("--sin-video", action="store_true",
                    help="solo medir (rapido): no graba ni codifica")
    args = ap.parse_args()

    destino = args.out or os.path.join(
        "benchmarks", f"gauntlet_lvl{args.difficulty}.mp4")
    os.makedirs(os.path.dirname(destino) or ".", exist_ok=True)
    crudo = destino.replace(".mp4", "_crudo.mp4")

    actuar, macros, meta = cargar(args.ckpt)
    print(f"[gauntlet] campeon {os.path.basename(args.ckpt)} "
          f"({meta['n_actions']} acciones, macros={macros}) contra los 12 "
          f"rivales en lvl{args.difficulty}, al mejor de "
          f"{args.rounds_to_win * 2 - 1}", flush=True)

    if args.repeticiones > 1 and not args.desync_max:
        print("[gauntlet] OJO: repeticiones>1 SIN --desync-max no aporta nada; "
              "greedy sobre estado fijo da peleas identicas. Usa --desync-max 30.",
              flush=True)

    ffmpeg = None if args.sin_video else abrir_ffmpeg(crudo, args.crf)
    escribir = (lambda _f: None) if ffmpeg is None else ffmpeg.stdin.write

    # trainable=False es LA pieza: sin el, el env corta en el primer KO.
    base = RetroSF2Env(trainable=False, frame_hook=lambda f: escribir(f.tobytes()))
    env = MacroActionWrapper(base, obs_rel_x_index=2,
                             frame_size=OBS_FRAME_DIM) if macros else base

    marcador, t0 = [], time.time()
    rng = np.random.default_rng(20260828)
    try:
      for rep in range(args.repeticiones):
        for rival in RIVALES:
            estado = f"RYU_{rival}_R1_lvl{args.difficulty}"
            obs, _ = env.reset(options={"state": estado})
            # El desfase es lo unico que hace distintas dos peleas del mismo
            # rival: rompe la coreografia del arranque y da variedad real.
            for _ in range(int(rng.integers(0, args.desync_max + 1))
                           if args.desync_max else 0):
                obs, _r, _t, _tr, _i = env.step(0 if macros else np.array([0, 0]))
            pasos = 0
            while pasos < args.max_steps:
                obs, _r, _t, _tr, _i = env.step(actuar(obs))
                pasos += 1
                ram = base._env.data.lookup_all()
                nuestros = int(ram["matches_won"])
                suyos = int(ram["enemy_matches_won"])
                if max(nuestros, suyos) >= args.rounds_to_win:
                    break
            gano = nuestros >= args.rounds_to_win
            marcador.append((rival, nuestros, suyos, gano, pasos))
            etiqueta = rival if args.repeticiones == 1 else f"{rival}#{rep + 1}"
            print(f"[gauntlet] {etiqueta:<11} {nuestros}-{suyos}  "
                  f"{'GANA LA PELEA' if gano else 'pierde'}  ({pasos} pasos)",
                  flush=True)
    finally:
        env.close()
        if ffmpeg is not None:
            ffmpeg.stdin.close()
            ffmpeg.wait()

    ganadas = sum(1 for _r, _n, _s, g, _p in marcador if g)

    # El acta por escrito, para que la consola pueda mostrar PELEAS COMPLETAS
    # en vez del win rate de rounds -- que es el numero halagador y el que se
    # presta a leerse como "le gana al juego" sin serlo.
    acta = destino.replace(".mp4", ".json")
    with open(acta, "w", encoding="utf-8") as f:
        json.dump({
            "fecha": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "checkpoint": os.path.basename(args.ckpt),
            "dificultad": args.difficulty,
            "rounds_para_ganar": args.rounds_to_win,
            "peleas_ganadas": ganadas,
            "peleas_totales": len(marcador),
            "repeticiones": args.repeticiones,
            "desync_max": args.desync_max,
            "ic95": wilson(ganadas, len(marcador)),
            "video": os.path.basename(destino),
            "rivales": [{"rival": r, "rounds_propios": n, "rounds_rival": sv,
                         "gano": g, "pasos": ps}
                        for r, n, sv, g, ps in marcador],
        }, f, ensure_ascii=False, indent=2)
    print(f"[gauntlet] acta: {acta}")

    mb = os.path.getsize(crudo) / 1048576 if ffmpeg is not None else 0.0
    ic = wilson(ganadas, len(marcador))
    print(f"\n[gauntlet] PELEAS COMPLETAS ganadas: {ganadas}/{len(marcador)} "
          f"({ganadas / len(marcador) * 100:.1f}%) "
          f"IC95 [{ic[0] * 100:.0f}% - {ic[1] * 100:.0f}%]"
          + ("  <- n chica: sirve para saber contra QUIEN pierde, no como "
             "porcentaje" if len(marcador) < 30 else ""))
    print(f"[gauntlet] video crudo: {mb:.1f} MB en {time.time() - t0:.0f}s")

    if ffmpeg is None:
        print("[gauntlet] sin video (--sin-video)")
        return
    if args.target_mb and mb > args.target_mb:
        print(f"[gauntlet] recomprimiendo a {args.target_mb} MB...", flush=True)
        recomprimir(crudo, destino, args.target_mb)
        os.remove(crudo)
    else:
        os.replace(crudo, destino)
    print(f"[gauntlet] LISTO: {destino} "
          f"({os.path.getsize(destino) / 1048576:.2f} MB)")


if __name__ == "__main__":
    main()
