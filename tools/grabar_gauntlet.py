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
# El buffer del emulador mide 320 de ancho, pero SF2 en Genesis dibuja solo
# 256 (modo H32): las 64 columnas de la derecha son negro muerto y hasta hoy
# se estaban codificando como si fueran imagen. Y el modo 256 sale a 4:3 en
# pantalla, o sea que sus pixeles NO son cuadrados: escalar 1:1 achata el
# juego. Se recorta a 256, se escala x4 entero (nitido, sin difuminar el pixel
# art) y se declara la proporcion 4:3 en el contenedor.
ANCHO_BUFFER, ANCHO, ALTO, FPS = 320, 256, 224, 60
ESCALA = 4
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
         "-s", f"{ANCHO_BUFFER}x{ALTO}", "-r", str(FPS), "-i", "-",
         "-vf", (f"crop={ANCHO}:{ALTO}:0:0,"
                 f"scale={ANCHO * ESCALA}:{ALTO * ESCALA}:flags=neighbor"),
         "-aspect", "4:3",
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
            "-preset", "slow", "-b:v", f"{kbps}k", "-pix_fmt", "yuv420p",
            "-aspect", "4:3", "-an"]
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
    ap.add_argument("--target-mb", type=float, default=0.0,
                    help="tope de tamano en MB. 0 (default) = manda la CALIDAD: "
                         "se codifica por CRF y pesa lo que pese. Referencia: "
                         "WhatsApp admite ~100 MB, y las 12 peleas completas a "
                         "CRF 20 caben debajo. Apretar mucho SI se nota: 17 min "
                         "en 18 MB son 145 kbps.")
    ap.add_argument("--rounds-to-win", type=int, default=2, help="2 = al mejor de tres")
    ap.add_argument("--max-steps", type=int, default=6000, help="tope por pelea")
    ap.add_argument("--crf", type=int, default=17,
                    help="calidad (menor = mejor). 17 es casi transparente; el "
                         "pixel art comprime bien y no cuesta tanto")
    ap.add_argument("--repeticiones", type=int, default=1,
                    help="peleas por rival. Con 1 la n es 12 y el intervalo de "
                         "confianza va del 55%% al 95%%: sirve para el video y "
                         "para saber CONTRA QUIEN pierde, no como porcentaje.")
    ap.add_argument("--desync-max", type=int, default=30,
                    help="frames neutrales sorteados antes de soltar el control. "
                         "ACTIVO POR DEFECTO: sin esto la pelea es determinista "
                         "y repetir no aporta NADA -- greedy sobre estado fijo "
                         "da peleas identicas. Poner 0 solo para grabar video.")
    ap.add_argument("--action-noise", type=float, default=0.0,
                    help="probabilidad por paso de sustituir la accion por una "
                         "aleatoria. Segunda fuente de variedad, independiente "
                         "del desfase: mide si la estrategia sobrevive a que le "
                         "interrumpan la ejecucion.")
    ap.add_argument("--rivales", default=None,
                    help="subconjunto separado por comas (ej. BALROG,EHONDA,GUILE). "
                         "Las 12 peleas al mejor de 3 son ~17 min y caben enteras "
                         "en el limite de WhatsApp (~100 MB) a buena calidad; usa "
                         "esto solo si quieres un clip corto de rivales concretos.")
    ap.add_argument("--seed", type=int, default=None,
                    help="semilla del sorteo. Por defecto es el reloj, para que "
                         "dos corridas no se copien; fijala para reproducir.")
    ap.add_argument("--sin-video", action="store_true",
                    help="solo medir (rapido): no graba ni codifica")
    args = ap.parse_args()

    destino = args.out or os.path.join(
        "benchmarks", f"gauntlet_lvl{args.difficulty}.mp4")
    os.makedirs(os.path.dirname(destino) or ".", exist_ok=True)
    crudo = destino.replace(".mp4", "_crudo.mp4")

    actuar, macros, meta = cargar(args.ckpt)
    n_riv = len(RIVALES if not args.rivales else args.rivales.split(","))
    print(f"[gauntlet] campeon {os.path.basename(args.ckpt)} "
          f"({meta['n_actions']} acciones, macros={macros}) contra {n_riv} "
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

    rivales = RIVALES
    if args.rivales:
        pedidos = tuple(r.strip().upper() for r in args.rivales.split(",") if r.strip())
        desconocidos = [r for r in pedidos if r not in RIVALES]
        if desconocidos:
            raise SystemExit(f"[gauntlet] rival desconocido: {desconocidos} "
                             f"(validos: {', '.join(RIVALES)})")
        rivales = pedidos

    marcador, t0 = [], time.time()
    # Semilla del reloj por defecto: dos corridas seguidas deben ser muestras
    # distintas del mismo modelo, no la misma corrida repetida.
    semilla = args.seed if args.seed is not None else int(time.time() * 1000) % (2**32)
    rng = np.random.default_rng(semilla)
    print(f"[gauntlet] semilla {semilla} | desfase <={args.desync_max} | "
          f"ruido {args.action_noise}", flush=True)
    try:
      for rep in range(args.repeticiones):
        for rival in rivales:
            estado = f"RYU_{rival}_R1_lvl{args.difficulty}"
            obs, _ = env.reset(options={"state": estado})
            # El desfase es lo unico que hace distintas dos peleas del mismo
            # rival: rompe la coreografia del arranque y da variedad real.
            for _ in range(int(rng.integers(0, args.desync_max + 1))
                           if args.desync_max else 0):
                obs, _r, _t, _tr, _i = env.step(0 if macros else np.array([0, 0]))
            pasos = 0
            n_acc = meta.get("n_actions", 72)
            while pasos < args.max_steps:
                accion = (int(rng.integers(0, n_acc))
                          if args.action_noise and rng.random() < args.action_noise
                          else actuar(obs))
                obs, _r, _t, _tr, _i = env.step(accion)
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

    # El acta por escrito. SEMANTICA DE ARCHIVOS (aprendida por las malas: una
    # corrida de video con n=12 piso la medicion de n=360 porque compartian
    # nombre): la MEDICION DE REGISTRO (--sin-video) vive en
    # benchmarks/gauntlet_lvlN.json y es lo que la consola muestra; una corrida
    # CON video escribe su acta como sidecar del video (<video>.json) y no toca
    # el registro -- un video es una ilustracion, no una medicion.
    if args.sin_video:
        acta = os.path.join(os.path.dirname(destino) or ".",
                            f"gauntlet_lvl{args.difficulty}.json")
    else:
        acta = destino.replace(".mp4", ".video.json")
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
            "action_noise": args.action_noise,
            "semilla": semilla,
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
