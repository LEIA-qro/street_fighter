# leia_hub.py -- los OJOS de la flota (v1 del plan de reconstruccion).
#
#   .venv/bin/python tools/leia_hub.py            # muestrea y alarma
#   .venv/bin/python tools/leia_hub.py --once     # una pasada (para probar)
#
# Sustituye a los scripts de scratchpad que vigilaban la flota y que morian
# con la sesion que los lanzo (paso el 2026-08-28: canario, selector y velador
# muertos ~1h sin que nadie se enterara -- "se cayo el observador, no lo
# observado"). Tres diferencias de fondo con aquellos:
#
#   1. EL CENSO MANDA. Compara /status contra fleet/fleet.json: alarma por
#      "menos maquinas frescas que las ESPERADAS", no por "ninguna fresca"
#      (la regla vieja era ciega por construccion: con 1 de 4 vivos callaba).
#      Y vigila aparte a las maquinas marcadas criticas -- hoy el canario, sin
#      el cual las ventanas de lvl1-3 son datos rancios, no datos.
#   2. NO SE SUICIDA. El velador viejo hacia SystemExit(1) en la primera
#      alerta, o sea que dejaba de vigilar justo cuando algo iba mal. Este
#      sigue muestreando, y una alarma se resuelve sola cuando la condicion
#      se va (con histeresis para no parpadear).
#   3. LA HISTORIA VIVE EN EL REPO, no en /tmp bajo un id de sesion.
#
# Productor UNICO de las tasas derivadas (grads/s, trans/s, replay ratio): se
# calculan aqui, de dos muestras consecutivas, y nadie mas las re-deriva.
#
# Stdlib pura a proposito: tiene que poder correr en cualquiera de los cuatro
# rigs sin instalar nada.

import argparse
import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLEET_JSON = os.path.join(REPO, "fleet", "fleet.json")
HIST_DIR = os.path.join(REPO, "fleet", "history")
SAMPLES = os.path.join(HIST_DIR, "hub_samples.jsonl")
ALARMS = os.path.join(HIST_DIR, "hub_alarms.jsonl")
WEB = os.path.join(REPO, "web")
DESIGN = os.path.join(REPO, "design")
CAMPEON = os.path.join(REPO, "benchmarks", "apex_milestones",
                       "apex_escalera_best.pt.json")
SELECTOR = os.path.join(HIST_DIR, "apex_selector_v3.jsonl")
BENCHMARKS = os.path.join(REPO, "benchmarks")

# Lo ultimo que se midio, en memoria. El servidor LEE de aqui y nunca vuelve a
# consultar al learner: un productor, muchos lectores. Asi la pantalla no puede
# multiplicar la carga sobre el learner por mas pestanas que se abran.
ESTADO = {"muestra": None, "arranque": time.time()}


def campeon_actual():
    """La tarjeta de identidad del campeon vigente, o None si no hay."""
    try:
        with open(CAMPEON, encoding="utf-8") as f:
            c = json.load(f)
    except (OSError, ValueError):
        return None
    c["archivo"] = os.path.basename(CAMPEON).replace(".json", "")
    return c


def coronaciones(n=12):
    """Historial de campeones: solo las filas que de verdad coronaron."""
    try:
        with open(SELECTOR, encoding="utf-8") as f:
            filas = [json.loads(ln) for ln in f if ln.strip()]
    except (OSError, ValueError):
        return []
    return [r for r in filas if r.get("nuevo_mejor")][-n:]


def gauntlets():
    """Las actas de peleas COMPLETAS, por dificultad, la mas reciente de cada.

    Es el numero que de verdad contesta "le gana al juego": el win rate de la
    escalera es de rounds de APERTURA, que corre por encima y se presta a
    leerse como si fueran peleas.
    """
    out = {}
    try:
        nombres = [n for n in os.listdir(BENCHMARKS)
                   if n.startswith("gauntlet_lvl") and n.endswith(".json")]
    except OSError:
        return out
    for n in nombres:
        try:
            with open(os.path.join(BENCHMARKS, n), encoding="utf-8") as f:
                a = json.load(f)
        except (OSError, ValueError):
            continue
        lvl = str(a.get("dificultad"))
        if lvl not in out or a.get("fecha", "") > out[lvl].get("fecha", ""):
            out[lvl] = a
    return out


def historia(maximo=180):
    """Las ultimas muestras, submuestreadas a `maximo` puntos.

    Una consola que solo ensena el instante no deja ver si algo se esta
    degradando: la tendencia es la mitad de la informacion. Se submuestrea
    aqui y no en el navegador para no mandar megabytes por la red.
    """
    try:
        with open(SAMPLES, encoding="utf-8") as f:
            filas = [json.loads(ln) for ln in f if ln.strip()]
    except (OSError, ValueError):
        return []
    filas = [f for f in filas if f.get("grads_per_s") is not None]
    if len(filas) > maximo:
        paso = len(filas) / maximo
        filas = [filas[int(i * paso)] for i in range(maximo)]
    return [{"ts": f["ts"], "grads_per_s": f.get("grads_per_s"),
             "trans_per_s": f.get("trans_per_s"),
             "replay_ratio": f.get("replay_ratio"),
             "vivas": f.get("vivas"), "esperadas": f.get("esperadas"),
             "wr": f.get("wr_recent200"), "por_nivel": f.get("por_nivel")}
            for f in filas]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass  # el hub ya imprime lo suyo; no queremos el ruido de acceso

    def _enviar(self, cuerpo, tipo="application/json", codigo=200):
        if isinstance(cuerpo, str):
            cuerpo = cuerpo.encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", tipo + "; charset=utf-8")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def _archivo(self, ruta, tipo):
        try:
            with open(ruta, "rb") as f:
                self._enviar(f.read(), tipo)
        except OSError:
            self._enviar("no encontrado", "text/plain", 404)

    def do_GET(self):
        ruta = self.path.split("?")[0]
        # La app compilada (React+shadcn, build de consola-app/) es la consola
        # principal; el HTML plano queda en /simple como respaldo sin build.
        if ruta in ("/", "/index.html") and os.path.isdir(os.path.join(WEB, "app")):
            self._archivo(os.path.join(WEB, "app", "index.html"), "text/html")
        elif ruta.startswith("/app/"):
            rel = os.path.normpath(ruta[len("/app/"):])
            if rel.startswith(".."):
                self._enviar("no", "text/plain", 400); return
            destino = os.path.join(WEB, "app", rel or "index.html")
            if os.path.isdir(destino):
                destino = os.path.join(destino, "index.html")
            tipo = {"js": "text/javascript", "css": "text/css", "html": "text/html",
                    "svg": "image/svg+xml", "png": "image/png",
                    "woff2": "font/woff2"}.get(destino.rsplit(".", 1)[-1], "application/octet-stream")
            self._archivo(destino, tipo)
        elif ruta == "/simple":
            self._archivo(os.path.join(WEB, "consola.html"), "text/html")
        elif ruta == "/champion-chrome.css":
            self._archivo(os.path.join(DESIGN, "champion-chrome.css"), "text/css")
        elif ruta == "/api/fleet":
            # El plano de control crudo, para la zona de configuracion de la
            # consola. Se sirve tal cual esta en disco: editar el plano ES
            # editar este archivo.
            self._archivo(FLEET_JSON, "application/json")
        elif ruta == "/api/state":
            plano = cargar_plano()
            self._enviar(json.dumps({
                "muestra": ESTADO["muestra"],
                "plano": plano,
                "campeon": campeon_actual(),
                "coronaciones": coronaciones(),
                "gauntlets": gauntlets(),
                "historia": historia(),
                "hub_desde": ESTADO["arranque"],
                "ahora": time.time(),
            }, ensure_ascii=False))
        else:
            self._enviar("no encontrado", "text/plain", 404)


    def do_PUT(self):
        # Escritura REAL del plano de control desde la consola. Validacion
        # minima pero suficiente: JSON parseable, con expected[] y umbrales,
        # y cada maquina con id y host. Escritura atomica (tmp + rename) y
        # respaldo de la version anterior -- el plano es el censo del que
        # dependen las alarmas: corromperlo apagaria los ojos.
        if self.path.split("?")[0] != "/api/fleet":
            self._enviar(json.dumps({"error": "ruta desconocida"}), codigo=404)
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            if n > 256 * 1024:
                raise ValueError("plano demasiado grande")
            nuevo = json.loads(self.rfile.read(n).decode("utf-8"))
            if not isinstance(nuevo.get("expected"), list) or not nuevo["expected"]:
                raise ValueError("expected[] vacio o ausente")
            for m in nuevo["expected"]:
                if not m.get("id") or not m.get("host"):
                    raise ValueError(f"maquina sin id/host: {m}")
            if not isinstance(nuevo.get("umbrales"), dict):
                raise ValueError("umbrales ausentes")
        except (ValueError, KeyError, TypeError) as e:
            self._enviar(json.dumps({"error": str(e)}), codigo=400)
            return
        try:
            import shutil
            shutil.copy2(FLEET_JSON, FLEET_JSON + ".bak")
            tmp = FLEET_JSON + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(nuevo, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(tmp, FLEET_JSON)
            self._enviar(json.dumps({"ok": True, "respaldo": "fleet.json.bak"}))
            print("[hub] fleet.json actualizado desde la consola", flush=True)
        except OSError as e:
            self._enviar(json.dumps({"error": str(e)}), codigo=500)


def servir(puerto):
    srv = ThreadingHTTPServer(("127.0.0.1", puerto), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    print(f"[hub] consola en http://127.0.0.1:{puerto}", flush=True)


def ahora():
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def cargar_plano():
    """El plano, normalizado. Acepta `run` (una sola, formato viejo) o `runs`
    (lista con una marcada `activa`): la consola no debe romperse el dia que
    haya dos entrenamientos, ni asumir que el unico que existe es este."""
    with open(FLEET_JSON, encoding="utf-8") as f:
        plano = json.load(f)
    if "runs" not in plano:
        plano["runs"] = [dict(plano.get("run", {}), id="actual", activa=True)]
    activa = next((r for r in plano["runs"] if r.get("activa")), None)
    plano["run"] = activa or (plano["runs"][0] if plano["runs"] else {})
    return plano


def leer_status(url, timeout=15):
    with urllib.request.urlopen(url.rstrip("/") + "/status", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def censo(status, plano):
    """Compara lo que se ESPERA (fleet.json) contra lo que REPORTA el learner.

    Devuelve una fila por maquina esperada, incluidas las mudas: una maquina
    ausente tiene que aparecer como fila fantasma, no desaparecer de la vista
    (el /status del learner solo lista lo que alguna vez escribio, y jamas
    poda: por eso hoy hay 12 entradas para 4 hosts).
    """
    frescura = int(plano["umbrales"]["actor_fresco_s"])
    actores = status.get("actors", {})
    filas = []
    for esperada in plano["expected"]:
        candidatos = [(k, v) for k, v in actores.items()
                      if v.get("host") == esperada["host"]]
        vivo = [(k, v) for k, v in candidatos if v.get("age", 9e9) < frescura]
        vivo.sort(key=lambda kv: kv[1].get("age", 9e9))
        if vivo:
            nombre, a = vivo[0]
            filas.append({
                "id": esperada["id"], "estado": "vivo", "actor": nombre,
                "procs": a.get("procs"), "steps_per_s": a.get("steps_per_s"),
                "age_s": round(a.get("age", -1), 1),
                "critico": bool(esperada.get("critico")),
            })
        else:
            ultimo = min((v.get("age", 9e9) for _k, v in candidatos), default=None)
            filas.append({
                "id": esperada["id"], "estado": "MUDA", "actor": None,
                "procs": None, "steps_per_s": 0.0,
                "age_s": None if ultimo is None else round(ultimo, 1),
                "critico": bool(esperada.get("critico")),
            })
    return filas


def tasas(prev, cur):
    """grads/s, trans/s y replay ratio INSTANTANEO de dos muestras.

    El ratio usa el batch que reporta el servidor si lo expone; si no, cae al
    default documentado del learner (256, apex_learner.py) y lo marca como
    supuesto -- un ratio calculado con un batch adivinado no puede presentarse
    como medicion.
    """
    if prev is None:
        return {}
    dt = cur["t"] - prev["t"]
    if dt <= 0:
        return {}
    dg = (cur["status"]["grad_steps"] - prev["status"]["grad_steps"]) / dt
    dx = (cur["status"]["transitions_in"] - prev["status"]["transitions_in"]) / dt
    batch = cur["status"].get("batch")
    supuesto = batch is None
    batch = int(batch or 256)
    return {
        "grads_per_s": round(dg, 2),
        "trans_per_s": round(dx, 1),
        "replay_ratio": round(dg * batch / dx, 2) if dx > 0 else None,
        "batch": batch,
        "batch_supuesto": supuesto,
        "ventana_s": round(dt, 1),
    }


def evaluar(filas, status, plano):
    """-> lista de (clave, mensaje) de las condiciones de alarma ACTIVAS."""
    u = plano["umbrales"]
    activas = []
    vivas = [f for f in filas if f["estado"] == "vivo"]
    if len(vivas) < int(u["minimo_maquinas"]):
        mudas = [f["id"] for f in filas if f["estado"] == "MUDA"]
        activas.append(("flota_incompleta",
                        f"{len(vivas)}/{len(filas)} maquinas vivas; mudas: {', '.join(mudas)}"))
    for f in filas:
        if f["critico"] and f["estado"] == "MUDA":
            activas.append((f"critica_{f['id']}",
                            f"la maquina CRITICA '{f['id']}' esta muda: "
                            "nadie alimenta lvl1-3, el detector de olvido quedo "
                            "ciego y esas ventanas del /status son rancias"))
    lvl = status.get("win_rate_recent_by_lvl") or {}
    bajos = [f"lvl{k}={v}" for k, v in lvl.items()
             if k in ("1", "2", "3") and isinstance(v, (int, float))
             and v < float(u["olvido_umbral_lvl1a3"])]
    if bajos and any(f["critico"] and f["estado"] == "vivo" for f in filas):
        # Solo tiene sentido si el canario ESTA vivo: si no, el dato es rancio.
        activas.append(("olvido", "posible olvido de niveles bajos: " + ", ".join(bajos)))
    return activas


def empujar(titulo, cuerpo):
    """Notificacion al operador. Local por ahora, A PROPOSITO.

    El plan pide una alarma que EMPUJE, y hoy no existe NINGUN canal en el
    repo (verificado: cero ntfy/telegram/slack/webhook/smtp en tools/). Un
    canal externo (ntfy, Telegram) manda datos de la flota a un tercero: eso
    es decision de Felipe, no de este script. Mientras tanto: notificacion de
    macOS -- suena y se ve aunque la terminal este tapada -- y la fila en el
    JSONL, que es lo que hace la alarma auditable.
    """
    try:
        subprocess.run(
            ["osascript", "-e",
             f'display notification {json.dumps(cuerpo)} with title {json.dumps(titulo)} sound name "Submarine"'],
            check=False, capture_output=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        pass
    print(f"\a[hub] ALARMA · {titulo} · {cuerpo}", flush=True)


def main():
    ap = argparse.ArgumentParser(description="Hub de observabilidad de la flota LEIA")
    ap.add_argument("--poll", type=float, default=60.0, help="segundos entre muestras")
    ap.add_argument("--once", action="store_true", help="una sola pasada y salir")
    ap.add_argument("--serve", type=int, default=0,
                    help="puerto de la consola web (0 = sin servidor)")
    args = ap.parse_args()

    plano = cargar_plano()
    url = plano["run"]["learner"]
    hist = int(plano["umbrales"]["histeresis_muestras"])
    os.makedirs(HIST_DIR, exist_ok=True)

    prev = None
    consecutivas = {}   # clave de alarma -> muestras seguidas activa
    abiertas = set()    # alarmas ya empujadas (no se repiten hasta resolverse)

    if args.serve:
        servir(args.serve)

    print(f"[hub] vigilando {url} | esperadas: "
          f"{', '.join(m['id'] for m in plano['expected'])} | poll {args.poll:.0f}s",
          flush=True)

    while True:
        try:
            status = leer_status(url)
            cur = {"t": time.time(), "status": status}
            filas = censo(status, plano)
            muestra = {
                "ts": ahora(),
                "grad_steps": status.get("grad_steps"),
                "buffer": status.get("buffer"),
                "wr_recent200": status.get("win_rate_recent200"),
                "por_nivel": status.get("win_rate_recent_by_lvl"),
                "maquinas": filas,
                "vivas": sum(1 for f in filas if f["estado"] == "vivo"),
                "esperadas": len(filas),
                **tasas(prev, cur),
            }
            with open(SAMPLES, "a", encoding="utf-8") as f:
                f.write(json.dumps(muestra, ensure_ascii=False) + "\n")
            ESTADO["muestra"] = muestra
            prev = cur

            activas = dict(evaluar(filas, status, plano))
            for clave, msg in activas.items():
                consecutivas[clave] = consecutivas.get(clave, 0) + 1
                if consecutivas[clave] >= hist and clave not in abiertas:
                    abiertas.add(clave)
                    empujar("Flota LEIA", msg)
                    with open(ALARMS, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"ts": ahora(), "evento": "abre",
                                            "clave": clave, "mensaje": msg},
                                           ensure_ascii=False) + "\n")
            for clave in list(abiertas):
                if clave not in activas:
                    abiertas.discard(clave)
                    consecutivas[clave] = 0
                    print(f"[hub] alarma RESUELTA: {clave}", flush=True)
                    with open(ALARMS, "a", encoding="utf-8") as f:
                        f.write(json.dumps({"ts": ahora(), "evento": "resuelve",
                                            "clave": clave}, ensure_ascii=False) + "\n")
            for clave in list(consecutivas):
                if clave not in activas:
                    consecutivas[clave] = 0

            vivas = muestra["vivas"]
            print(f"[hub] {muestra['ts']} | grads {muestra['grad_steps']:,} "
                  f"({muestra.get('grads_per_s', '?')}/s) | maquinas {vivas}/{muestra['esperadas']} "
                  f"| trans/s {muestra.get('trans_per_s', '?')} "
                  f"| ratio {muestra.get('replay_ratio', '?')} "
                  f"| lvls {muestra['por_nivel']}", flush=True)
        except Exception as e:  # noqa: BLE001 -- el hub JAMAS se muere solo
            print(f"[hub] muestra fallida ({type(e).__name__}: {e}); sigo", flush=True)
            with open(ALARMS, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": ahora(), "evento": "error",
                                    "detalle": str(e)[:300]}, ensure_ascii=False) + "\n")

        if args.once:
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    sys.exit(main())
