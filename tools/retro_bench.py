# retro_bench.py -- benchmark portable del backend headless (stable-retro).
#
# UN comando, cualquier maquina:
#
#     python3 tools/retro_bench.py
#
# Hace todo solo: instala stable-retro si falta, importa el ROM de roms/ si el
# juego no esta integrado, corre el barrido de procesos con render_mode=None
# (OBLIGATORIO: sin el, retro.make abre una ventana pyglet y mides el vsync del
# monitor a 60fps, no el emulador), imprime la tabla y appendea el JSON con los
# datos de la maquina a benchmarks/retro_bench.jsonl. Mandense ese archivo (o
# la tabla) al grupo.
#
# Windows nativo NO puede correrlo (stable-retro no publica wheels win_amd64);
# el script lo detecta e imprime los pasos de WSL2. Dentro de WSL2 la emulacion
# CPU corre a velocidad practicamente nativa, asi que el numero si representa a
# la maquina.
#
# Contexto: BizHawk en la 13900K se estanca en ~1,160 agent steps/s agregados
# (16 o 24 emuladores, da igual). stable-retro en una MacBook M4 dio ~3,700
# fps/proceso y ~19,700 fps agregados con 8 procesos (~4,900 agent steps/s a
# frame-skip 4). La compuerta de EGGROLL es 25,000 agent steps/s ENTRE TODAS
# las maquinas; este script mide cuanto aporta cada una.

import argparse
import json
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import time

GAME = "StreetFighterIISpecialChampionEdition-Genesis-v0"
STATE = "Champion.Level1.RyuVsGuile"
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WSL_HELP = """
stable-retro no tiene wheels para Windows nativo. Corre esto dentro de WSL2:

    wsl --install -d Ubuntu        (solo la primera vez; reinicia si lo pide)
    wsl
    cd /mnt/<disco>/<ruta al repo>  p.ej. cd /mnt/d/GitHub/Street/street_fighter
    python3 tools/retro_bench.py

Dentro de WSL2 la emulacion corre a velocidad practicamente nativa.
Si Ubuntu no trae pip:  sudo apt update && sudo apt install -y python3-pip
"""


def ensure_stable_retro():
    try:
        import stable_retro  # noqa: F401
        return
    except ImportError:
        pass
    print("[setup] Instalando stable-retro...")
    base = [sys.executable, "-m", "pip", "install", "stable-retro"]
    for extra in ([], ["--user", "--break-system-packages"]):
        result = subprocess.run(base + extra, capture_output=True, text=True)
        if result.returncode == 0:
            return
    print(result.stdout[-2000:])
    print(result.stderr[-2000:])
    sys.exit("[setup] No pude instalar stable-retro; revisa el error de pip arriba.")


def ensure_rom():
    import stable_retro as retro
    try:
        retro.data.get_romfile_path(GAME)
        return
    except (FileNotFoundError, KeyError):
        pass
    roms_dir = os.path.join(REPO_ROOT, "roms")
    print(f"[setup] Importando el ROM desde {roms_dir}...")
    subprocess.run([sys.executable, "-m", "stable_retro.import", roms_dir], check=True)
    retro.data.get_romfile_path(GAME)  # raises if the import did not take


def cpu_model():
    system = platform.system()
    try:
        if system == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True).stdout.strip()
        if system == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def worker(steps, obs_kind, out):
    import stable_retro as retro
    kwargs = dict(game=GAME, state=STATE, render_mode=None)
    if obs_kind == "ram":
        kwargs["obs_type"] = retro.Observations.RAM
    env = retro.make(**kwargs)
    env.reset()
    actions = [env.action_space.sample() for _ in range(64)]
    for i in range(200):
        env.step(actions[i % 64])
    t0 = time.perf_counter()
    for i in range(steps):
        _, _, terminated, truncated, _ = env.step(actions[i % 64])
        if terminated or truncated:
            env.reset()
    fps = steps / (time.perf_counter() - t0)
    env.close()
    out.put(fps)


def run(n_procs, steps, obs_kind):
    out = mp.Queue()
    procs = [mp.Process(target=worker, args=(steps, obs_kind, out))
             for _ in range(n_procs)]
    for p in procs:
        p.start()
    fps = [out.get() for _ in procs]
    for p in procs:
        p.join()
    return fps


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark portable de stable-retro (SF2'SCE Genesis)")
    parser.add_argument("--procs", default=None,
                        help="lista de conteos de procesos, p.ej. 1,4,8,16")
    parser.add_argument("--steps", type=int, default=6000,
                        help="frames medidos por proceso")
    parser.add_argument("--obs", default="image", choices=["image", "ram"])
    args = parser.parse_args()

    if platform.system() == "Windows":
        sys.exit(WSL_HELP)

    ensure_stable_retro()
    ensure_rom()

    n_cpus = os.cpu_count() or 4
    if args.procs:
        counts = [int(x) for x in args.procs.split(",")]
    else:
        counts = sorted({1, 4, 8, 12, 16, n_cpus})
        counts = [c for c in counts if c <= n_cpus]

    machine = {
        "hostname": platform.node(),
        "cpu": cpu_model(),
        "logical_cpus": n_cpus,
        "system": f"{platform.system()} {platform.release()}",
        "wsl": "microsoft" in platform.release().lower(),
    }
    print(f"[bench] {machine['hostname']} -- {machine['cpu']} "
          f"({n_cpus} CPUs logicos, {machine['system']})")
    print(f"{'procs':>5} {'fps/proc (min-max)':>22} {'fps agregado':>13} "
          f"{'agent steps/s (fs4)':>20}")

    rows = []
    for n in counts:
        fps = run(n, args.steps, args.obs)
        agg = sum(fps)
        rows.append({**machine, "procs": n, "obs": args.obs,
                     "steps_per_proc": args.steps,
                     "fps_per_proc_min": min(fps), "fps_per_proc_max": max(fps),
                     "fps_aggregate": agg, "agent_steps_per_s_fs4": agg / 4,
                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")})
        print(f"{n:>5} {min(fps):>10.0f}-{max(fps):<11.0f} {agg:>13.0f} {agg / 4:>20.0f}")
        sys.stdout.flush()

    best = max(rows, key=lambda r: r["fps_aggregate"])
    print(f"\n[bench] PICO: {best['fps_aggregate']:.0f} fps agregados con "
          f"{best['procs']} procesos = {best['agent_steps_per_s_fs4']:.0f} agent steps/s "
          f"(la compuerta de EGGROLL es 25,000 entre toda la flota)")

    out_dir = os.path.join(REPO_ROOT, "benchmarks")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "retro_bench.jsonl")
    with open(out_path, "a") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    print(f"[bench] Resultados appendeados a {out_path} -- mandenlo al grupo.")


if __name__ == "__main__":
    mp.set_start_method("spawn")
    main()
