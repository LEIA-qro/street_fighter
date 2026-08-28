# 41 · PLAN DE SISTEMA Y EJECUCIÓN — consola LEIA

Síntesis del panel (7 propuestas `20`–`27`, 4 jueces `30`–`33`) resuelta contra
código y contra el sistema VIVO. Fecha de escritura: **2026-08-28 10:03 CST**,
rama `stage0-metrics-and-semantics`, HEAD `47643d78`.

## 0. Lo que medí yo mismo antes de escribir (no heredado)

Dos lecturas de `GET http://desktop-4090-ubuntu-wsl:8090/status` separadas 20.0 s,
desde esta Mac, 2026-08-28 10:03 CST:

| dato | valor |
|---|---|
| grad_steps | 1,937,866 (**27.70 grads/s**) |
| transitions_in | 207,224,800 (**1,437.3 trans/s**) |
| episodes | +8.13/s · weights_version 3,810 (+1 en 20 s) |
| buffer | 1,000,000 (saturado, señal muerta) |
| escalera viva | 1:1.00 · 2:0.995 · 3:0.985 · 4:0.94 · 5:0.94 · 6:0.92 · 7:0.89 · **8:0.745** |
| dict `actors` | **12 entradas, 4 hosts, UNA fresca** (`SSS-207877`, age 0.6 s) |
| resto | `mini-fzamorano` age 3,263 s · `LAPTOP-1VCCKQL9` 58,660 s · `PEREA` hasta 94,142 s |

Cuatro hechos que gobiernan todo el plan:

1. **La flota está en 1 de 4 y nadie se enteró.** Incluido el canario de la Mac
   (`--difficulty 1..8`, la única defensa contra el olvido de lvl1‑3), muerto hace
   ~54 min. `ps aux | grep -E "apex_|night_watch|selector"` en esta Mac → **cero
   procesos**: murieron el actor, el selector y el velador.
2. **El velador es ciego por construcción.** Última fila de
   `night_watch_dqn.jsonl` (t=1787929390, ~1 h de antigüedad, intervalo nominal
   10 min): `"actores_frescos": 2`. No alertó porque su regla es "ningún actor
   fresco" (`night_watch_dqn.py:81-83`) y con 1 de 4 nunca dispara. Y vive en
   `/private/tmp/claude-502/…` — una purga de /tmp borra la historia entera.
3. **El learner NO está hambriento.** `apex_learner.py:96` fija `--batch 256` y
   `:107` `--replay-ratio 8.0`; el tope que el bucle aplica (`:240-241`) es
   **acumulado desde el resume**, no instantáneo. Ratio instantáneo hoy =
   27.70×256/1437.3 = **4.93 < 8**. El learner está limitado por **cómputo**
   (~28 grads/s), que es exactamente el pendiente #3 de
   `agent/memory/08-cola-manana.md:40-62`. Queda **rechazado** el KPI
   "replay_ratio 9.29 = STARVED" de `21-option-flota-ux.md` (juez producto V19).
4. **`fleet.json` no existe** (`find . -name "fleet*.json"` → vacío) y **el plan de
   anoche tampoco se leyó** (`agent/dashboard/PLAN-reconstruccion.md`, 2026‑08‑28
   00:02, commit `781fe61f`). El mayor riesgo medido del proyecto no es
   arquitectónico: es **re-derivar el mismo plan cada noche sin ejecutar nada**.

Correcciones de hecho que este plan propaga (y que rompen cosas si se ignoran):

- `VENV_PYTHON` **NO** está clavado a Windows: `web_dashboard.py:24-28` es una
  cadena de fallback `.venv/Scripts/python.exe` → `.venv/bin/python` →
  `sys.executable`. El impedimento en esta Mac es BizHawk/ROM, no el intérprete.
- En gradio 6.25.0 `gr.Blocks.__init__` **no** acepta `theme` y `.launch()` **sí**.
  `web_dashboard.py:2320` está **correcto**; `11-discover-historia.md` y
  `13-discover-registry.md` están equivocados y seguirlos **rompe el arranque**.
- **Sí hay red de pruebas parcial** sobre el dashboard:
  `code_testing/pytest/test_telemetry_dashboard.py` y
  `code_testing/pytest/test_model_testing_config.py` importan `web_dashboard`.
  `get_stand_checkpoint_status` (`:349`) tiene test verde → se **desconecta**, no
  se borra. (`27-option-matar.md` afirma lo contrario; es falso.)

---

## 1. ARQUITECTURA HÍBRIDA DEFINITIVA

### 1.1 El diagrama en palabras

Tres planos, no dos. La separación no es "cliente/servidor" sino **quién posee el
hardware**:

**Plano de CÓMPUTO (local, uno por rig, ya existe y no se toca).**
En cada máquina viven el emulador, la ROM, los savestates, torch y los modelos.
Aquí corren `tools/apex_actor.py` (actores Ape‑X), el learner
(`tools/apex_learner.py`, hoy en desktop-4090:8090, `--host 0.0.0.0` en `:135`),
y en el rig BizHawk los subprocesos que lanza el Gradio actual
(`web_dashboard.py:424` `stream_logs`). **Ningún plano superior está en el camino
crítico de este**: si todo lo demás muere, la run sigue entrenando. Regla dura.

**Plano de AGENTE LOCAL (uno por máquina, nuevo, *un solo proceso*).**
`tools/leia_agent.py`, stdlib primero, con dos modos por bandera:
`--observe` (heartbeat + censo + servir la UI local) y `--supervise` (además,
convergencia hacia el estado deseado; apagado por defecto). Escucha en
`127.0.0.1:8770`. **Rechazo explícito de la duplicación** que proponían
`22-option-arquitectura.md` (`src/api/app_local.py`) y `23-option-fleet-agent.md`
(`tools/fleet_agent.py`) por separado: son dos programas para la misma máquina y
ninguno supervisaría al otro (juez arquitectura).

**Plano COMPARTIDO (uno para el equipo, nuevo).**
`tools/leia_hub.py`: sampler único del learner, censo `expected[]`, historia
durable **en el repo**, alarma que empuja, y —cuando llegue— la UI en modo
observador. Habla por la **tailnet `leia-qro.org.github`** (verificado por el
panel: 12 ms directo sin DERP, MagicDNS, HTTP 200 en ~30–80 ms por nombre corto,
cero puertos a Internet, cero costo, cero cuentas). **AWS prohibido**
(`00-DECISIONES.md:20`) y `madre` excluida por ser EC2.

### 1.2 Dónde vive el plano compartido — y por qué NO en la 4090

**Rechazado `desktop-4090-ubuntu-wsl:8091`** aunque sea la máquina "24/7":
- `ssh desktop-4090-ubuntu-wsl` → *Connection refused* (juez arquitectura). **No
  existe camino remoto de administración**: si el hub muere ahí, sólo Santiago,
  físicamente, lo revive.
- Es WSL2 bajo el Windows de un tercero; muere con cada reinicio del anfitrión.
- Su llave de tailnet caduca 2027‑02‑23; el único nodo con expiración desactivada
  es `madre`, excluida.
- Ya es learner **y** único actor vivo: el observador viviría dentro del observado
  y con correlación de fallo total.

**Decisión de este plan (v1): el hub corre en `mini-fzamorano` (esta Mac) bajo
`launchd`**, no lanzado a mano. La razón no es que la Mac sea más fiable —hoy se
le murieron tres procesos— sino que es la única máquina donde **existe un
supervisor de verdad** (launchd con `KeepAlive`) y cuyo dueño puede administrarla
en el momento. La mitigación M3 de `22` ("sampler espejo en la Mac") tal como
estaba redactada **ya falló en vivo antes de existir**: era otro proceso lanzado a
mano. `launchd` es la corrección.

**Requisito, no mitigación:** el hub es **relocalizable por hostname**. Su
dirección vive en `fleet/fleet.json` (`hub_url`), nunca hardcodeada. Emplazamiento
definitivo = decisión de Felipe (§7, D3).

### 1.3 Cómo habla el cliente con los dos planos

La página se sirve **desde el agente local** (`127.0.0.1:8770`), nunca desde el
hub. Eso elimina por diseño el campo minado de *Private Network Access* y
contenido mixto (una página https del hub haciendo `fetch` a `http://127.0.0.1`).
Consecuencia: el 90 % de las interacciones (lanzar, parar, log, jugable) es
**same-origin**; el hub sólo necesita CORS de lectura para `127.0.0.1:8770`.

Descubrimiento por `GET /capabilities` → `{plane, node_id, host, gpu, emulator,
hub_url, roles, supervise}`, derivado en el render, jamás por banderas de build.
Un rig sin GPU o sin emulador **no dibuja habilitado** lo que no puede hacer.

**Doble camino de datos (dead-man's switch).** La UI trata "sin muestra del hub
desde hace X" como **estado de primera clase**, no como página vacía; y el agente
local sabe leer `/status` directo, de modo que la consola degrada a lectura
directa cuando el hub calla. Sin esto, el fallo observado hoy se repite con más
código.

### 1.4 Transporte del estado en vivo

| flujo | transporte | por qué |
|---|---|---|
| estado de flota | **sampler único** en el hub, cadencia fija 10 s, escribe JSONL en el repo; los clientes leen del hub | grads/s y trans/s **sólo existen diferenciando dos lecturas**: si cada navegador poléa, cada operador ve una tasa distinta y cerrar la pestaña borra la historia |
| log de un job local | SSE desde el agente local, con `Last-Event-ID` = offset de byte de `runs/<job_id>/stdout.log` | `stream_logs` (`web_dashboard.py:424`) ya es un generador puro de strings → `StreamingResponse` sin reescribir su lógica; y arregla el defecto real de que recargar el navegador huerfana el proceso |
| comandos | POST | son raros (un clic cada minutos) |
| red de seguridad | polling con `ETag` cada 10 s | **camino obligatorio y probado, no opcional** |

**Corrección al argumento de `22` §3:** su justificación del sampler ("si cada
pestaña poléa se multiplica la carga sobre el proceso que entrena") es **falsa por
dos órdenes de magnitud** — el juez midió 12 clientes a 0.5 s = −1.6 % de
gradientes, ruido. La decisión sobrevive por las otras tres razones (derivadas,
consistencia, durabilidad). Importa corregirlo porque un argumento falso se
reutiliza luego para justificar cosas que no se siguen.

**SSE no es el camino de confianza.** `tools/apex_actor.py:190-195` documenta un
incidente real: en red ajena "el POST /transitions se queda en *timed out* con los
GET pasando bien". Un stream de larga vida es justo lo que un NAT de WSL2 corta en
silencio, y `EventSource` no distingue "conectado y callado" de "conectado y sin
novedad". Por eso: **latido explícito cada 15 s dentro del propio SSE**, y el
fallback con ETag se prueba en el arnés, no se documenta y se olvida.

**Cerrojo del lado del learner (ninguna propuesta lo contempló).** `status()` toma
el **mismo lock** que `sample()` (`src/agents/apex.py:245` vs `:229`). Añadir al
learner un snapshot cacheado ≤1 s servido **sin tomar el lock**, más límite de
tasa por IP. Son unas líneas, es aditivo, y hace irrelevante la disciplina del
cliente para siempre — que es lo correcto en un repo que tiene un
`gr.Timer(0.1, active=True)` repintando 10 veces por segundo con la pestaña oculta
(`web_dashboard.py:2133`).

**Higiene inmediata, independiente de todas las fases:** el learner escucha en
`0.0.0.0` sin autenticación (`tools/apex_learner.py:135`, `--host` default
`0.0.0.0` en `:85`) y `POST /transitions` acepta cualquier `actor` como string
(`src/agents/apex.py:181`). Cambiar el bind a la IP de la tailnet es **una línea**
y hay que hacerlo **antes** de repartir cualquier URL al equipo.

---

## 2. LA CAPA DE API SOBRE EL PYTHON EXISTENTE

Principio: **nada se reescribe; se promueve**. Las funciones de
`web_dashboard.py` que ya son lógica pura pasan a routers y el Gradio actual las
llama a ellas, de modo que `/legacy` funciona como **oráculo** de la extracción
(si sigue lanzando, la extracción es correcta).

Coste de dependencias **cero**, verificado por el panel en el `.venv` del repo:
`gradio 6.25.0` ya arrastra `fastapi 0.141.1`, `uvicorn 0.52.4`,
`starlette 1.6.0`, `pydantic 2.13.4`. Y `gradio.mount_gradio_app` existe con
`gradio.routes.App` como subclase de `fastapi.FastAPI` → el Blocks se monta
**intacto** en `/legacy`, un proceso, un puerto.

### 2.1 Catálogo de artefactos — `GET /api/artifacts`

| función promovida | línea | qué aporta |
|---|---|---|
| `get_model_files` | `:155` | `.zip` SB3 por carpeta (`production`/`tuning`/`latest`) |
| `get_stand_checkpoint_files` | `:260` | `.pt` Ape‑X bajo `benchmarks/apex_milestones` (`:125`) |
| `get_stand_default_checkpoint` | `:294` | heurística de "el mejor por defecto" |
| `_resolve_stand_checkpoint` | `:320` | ruta relativa canónica + meta |
| `_load_stand_checkpoint_meta` | `:186` | lee el sidecar y **hoy lo tira** |
| `_stand_sidecar_metrics` | `:246` | wr por tier del sidecar |
| `get_stand_checkpoint_status` | `:349` | in_dim / n_actions / quantiles / weights_version / escalera L1‑L8 |
| `get_all_state_files` | `:414` | savestates |
| `handle_model_upload` | `:1007` | ingreso de artefacto externo |

Contrato: `Artifact { id, kind: "sb3_zip"|"vecnorm_pkl"|"apex_pt"|"savestate",
path, size, mtime, meta: {...} | null, meta_schema: "v1"|"v2"|"v3"|null,
provenance: "sidecar"|"inferido"|"desconocido" }`.

**`provenance` es obligatorio y no cosmético.** 454–462 de 500 filas de
`benchmarks/bench_12rivals.jsonl` dicen `apex_eval_tmp.pt`, hay tres esquemas de
sidecar incompatibles y `.pt` huérfanos (verificado: `apex_best_desync.pt.json`
121 B vs `apex_v1952_latest.pt.json` 617 B, y `apex_final_286k_run_lvl1.pt`,
`apex_grads90k_PLENO12de12.pt`, `apex_grads_00090000.pt`,
`apex_rescate_congelado_v331.pt`, `apex_v3291_media990.pt` **sin sidecar**). La
regla R1 de `20-option-ia.md` ("algo/env se derivan del artefacto") es correcta
como diseño y **hoy imposible como dato**: el picker diría "desconocido" en la
mayoría de los casos. Por eso la API expone la ignorancia en vez de inventarla, y
el **normalizador de procedencia es trabajo de `tools/`, no de la UI**.

Se **rescata el CONTENIDO** de `get_stand_checkpoint_status` (`:349`) aunque su
pegamento Gradio muera: es la tarjeta de identidad del campeón, y la borró
`5a852ae2`. Tiene test verde (`test_telemetry_dashboard.py:436`) → **desconectar,
no borrar**.

### 2.2 Jobs — `POST /api/jobs`, `GET /api/jobs`, `GET /api/jobs/{id}/log`, `POST /api/jobs/{id}/stop`

Un solo recurso para los seis lanzadores de hoy:

| lanzador | línea | tipo de job |
|---|---|---|
| `run_training` | `:731` | `train` |
| `run_matchup` | `:775` | `matchup` |
| `run_stand` | `:869` | `stand` (el jugable) |
| `run_tuning` | `:689` | `tune` — *fuera de v1* |
| `run_pbt` | `:1048` | `pbt` — **MUERTO** (ver §5) |
| `run_league` / `run_exploiter` | `:1063` / `:1085` | `league` / `exploiter` — *fuera de v1* |

Contrato: `JobSpec { type, node_id, params: {...} }` →
`Job { id, type, node_id, state: queued|running|stopping|exited, started_at,
exit_code, log_offset, cmd[] }`.

Tres reglas que nacen de defectos medidos:
1. **`JobsContext` con `capacity: 1`, no singleton.** `GlobalState`
   (`:31-38`) y el chequeo `busy` (`:432-434`, mensaje en `:449`) son UN slot
   global. La API lo modela como capacidad configurable desde el día uno para no
   repetir el bloqueador estructural que se está reconstruyendo para eliminar.
2. **Todo `stop` nombra a su víctima.** Hoy cinco botones (`:1810`, `:1811`,
   `:1902`, `:1903`, `:1984`) llaman a tres funciones (`graceful_stop_process:536`,
   `force_kill_process:633`, `stop_active_process:659`) sobre el **mismo** proceso:
   "Graceful Stop League" mata un PPO lanzado desde otra pestaña sin avisar.
3. **`force_kill` se reimplementa, no se porta.** `taskkill` (`:616`, `:650`) no
   existe fuera de Windows, `capture_output=True` traga el error y la función
   devuelve éxito **incondicionalmente** (`:657`). Contrato nuevo: el endpoint
   verifica la muerte del PID y devuelve `{killed: bool, pid, verified_at}`. Un
   stop que miente envenena el diagnóstico del siguiente fallo.

La parada **suave** sí es contrato real y se conserva: el marcador `.stop_training`
lo leen `auto_curriculum_callback.py:528`, `manual_curriculum_callback.py:419`,
`stand_leia.py:676`, `test_agent_v2.py:220`, `test_ai_vs_ai_v2.py:343`.

### 2.3 El jugable — `stand_leia.py` como servicio

`stand_leia.py` **no se importa como librería**: es un motor CLI probado
(decisión #4 del dueño) y su contrato es su `argparse`
(`stand_leia.py:535-562`: `--ckpt`, `--opponent`, `--opponent-type`,
`--cpu-level`, `--p2-ckpt`, `--p2-device`, `--p2-algo`, `--p2-env`,
`--p2-model-zip`, `--p2-model-pkl`, `--rematch-delay`, `--infinite-match`,
`--device`). Lo que se promueve es la **validación** que ya existe en
`run_stand` (`:869-912`): tipo de rival ∈ {human, cpu, model, sb3}, rival ∈
STAND_OPPONENTS, `cpu_level` ∈ 1..8, P2 Ape‑X obliga RYU, P2 SB3 exige zip+pkl
existentes. Eso **es** el esquema pydantic de `StandSpec` — copiarlo, no
reinventarlo. Complementos: `toggle_agent_state:942` → `POST /api/match/pause`;
`stop_match_process:958` → el `stop` genérico.

### 2.4 Config del rig — `GET/PUT /api/config`

`update_config_var:663`, `save_all_config:968`, `update_config_list:990`.
Hoy editan `src/core/config.py` **por `re.sub`** y hacen `importlib.reload` en
caliente: el operador deja el repo sucio sin saberlo. La capacidad vive (las vars
tienen consumidores reales: `lua/v2.0/training_env_client.lua:22,25,27`,
`base_env.py:51`), pero el estado pasa a un JSON/TOML fuera del código, y la
superficie se etiqueta como **"rig BizHawk"**, no como flota.
`get_best_tuning_params:705` (script Python en f-string ejecutado con `python -c`,
inyección trivial desde un Textbox) **no se promueve**: muere con Optuna.

### 2.5 Estado derivado — `GET /api/fleet` (hub) y `GET /api/learner` (proxy)

Sirve el snapshot del sampler, no un passthrough. Contrato en §3.4. Regla dura:
**`GET /weights` (5,335,856 B, sin ETag ni HEAD) NUNCA se pide desde la UI**; la
versión se lee de `weights_version` en `/status`.

Las tres funciones que hoy devuelven **HTML** —`get_league_pool_status_html:1104`,
`get_auto_curriculum_status_html:1220`, `get_live_telemetry_html:1438`— se
promueven a **datos**, no a HTML. La de telemetría además **no tiene fuente**:
`write_telemetry` sólo lo llaman `test_agent_v2.py` y `test_ai_vs_ai_v2.py`,
`stand_leia.py` no la llama y no hay `.telemetry.json` en disco. Muere la
implementación (272 líneas + `gr.Timer(0.1)` en `:2133`), **sobrevive el
contrato**: el panel vuelve el día que el motor emita telemetría.

---

## 3. EL FLEET-AGENT

### 3.1 El hallazgo que obliga a rediseñar la llave

`actors[*].host` devuelve **`SSS`, `PEREA`, `LAPTOP-1VCCKQL9`, `mini-fzamorano`**
(verificado hoy) — los nombres de las máquinas **Windows** — mientras la tailnet
las llama `desktop-4090-ubuntu-wsl`, `legion-wsl`, `laptop-omen5080-ubuntu-wsl`.
Causa: `tools/apex_actor.py:199` usa `socket.gethostname()` y dentro de WSL2 eso
hereda el nombre del anfitrión. Sólo la Mac coincide, por no ser WSL.

Consecuencia dura: **`fleet.json` no puede llavear por hostname ni por nodo de
Tailscale.** Necesita un `node_id` opaco elegido por humano en el alta y
propagado por el flag **`--name`, que YA EXISTE** (`apex_actor.py:189`) →
**cambio de wire de cero líneas de código**. Es el hallazgo más valioso del panel
y se aplica **antes** que cualquier UI.

### 3.2 Contrato de `fleet/fleet.json` (versionado en el repo)

```jsonc
{
  "schema": 1,
  "generation": 7,                       // monótono; es el número que la UI muestra
  "hub_url": "http://mini-fzamorano:8091",   // REQUISITO: relocalizable, nunca hardcodeado
  "defaults": {
    "git_ref": "fleet/2026-08-28-01",    // TAG ANOTADO o SHA. JAMÁS una rama.
    "learner_url": "http://desktop-4090-ubuntu-wsl:8090",
    "requirements": ["requirements.txt", "requirements-dqn.txt"]
  },
  "profiles": {
    "grande":     { "procs": "auto", "difficulty": "4,5,6,7,8", "flush": 800 },
    "canario":    { "procs": 12,     "difficulty": "1,2,3,4,5,6,7,8", "flush": 800 },
    "red-hostil": { "flush": 100 }     // apex_actor.py:190-195: POST muere, GET pasa
  },
  "nodes": {
    "sss":    { "owner": "santiago", "roles": ["learner","actor"], "profiles": ["grande"] },
    "legion": { "owner": "diego",    "roles": ["actor"], "profiles": ["grande"] },
    "omen":   { "owner": "diego",    "roles": ["actor"], "profiles": ["grande"] },
    "mac":    { "owner": "felipe",   "roles": ["actor","hub"], "profiles": ["canario"] }
  },
  "expected": ["sss", "legion", "omen", "mac"],   // EL CENSO — campo obligatorio
  "rollout": { "strategy": "canary", "canary": "mac", "soak_minutes": 20,
               "max_unavailable": 1, "freeze": true },
  "previous_ref": "fleet/2026-08-27-03"
}
```

Ocho reglas duras, cada una anclada a una falla real:

| # | regla | anclaje |
|---|---|---|
| A1 | la llave es `node_id` opaco, jamás `socket.gethostname()` ni el nodo Tailscale | `apex_actor.py:199` en WSL2 |
| A2 | `expected[]` es obligatorio y produce **fila fantasma** por cada nodo sin heartbeat | `night_watch_dqn.py:81-83` calló con 1 de 4 vivas |
| A3 | `git_ref` es tag anotado o SHA, nunca rama; fetch con **`--tags --force`** | mover un tag anotado NO se propaga con un fetch normal → el deploy **reporta éxito sin desplegar** |
| A4 | el supervisor **nunca** arranca, para ni reinicia un **learner** | un reinicio destruye el buffer de 1 M y el dict `actors` en memoria (`apex.py:181`), que `previous_ref` no restaura |
| A5 | `.fleet/node.json` **local, no versionado**: `paused_by_owner`, `max_cpu_share`, `quiet_hours`, `allow_auto_update`. El plano de control **no lo pisa** | son laptops personales de compañeros (`08-cola-manana.md:66`) |
| A6 | `procs: "auto"` se resuelve con `src/es/resources.py` `plan_procs` (ya sabe de WSL, reserve_cores y batería) — no se reinventa | evitar un segundo planificador |
| A7 | `fleet.json` aporta **sólo el censo esperado y el estado deseado**. **Ningún valor de runtime mostrado en pantalla sale de él** | `--difficulty` se declara aquí y **no viaja en el wire** (`apex_actor.py:271-275` manda sólo `procs/steps_per_s/host`); relanzar a mano es la operación NORMAL (`RUN_OMEN_DESDE_CERO.md §7`) → la UI mentiría sobre el experimento |
| A8 | `freeze: true` por defecto durante una run marcada crítica | hay 1.9 M grad_steps vivos |

### 3.3 Ciclo del supervisor (`tools/leia_agent.py --supervise`)

Seis pasos, bucle de 10 s con jitter:

1. **Observa** — `git fetch --tags --force` (**sin tocar el working tree**) y lee
   `fleet.json` con `git show <ref>:fleet/fleet.json` (**sin checkout**).
2. **Compara** — `desired_hash = sha256(target_sha + spec canónica + hash(requirements))`.
3. **Converge** — SIGTERM al hijo → drain → checkout **sólo con el hijo muerto** →
   `pip` si cambiaron los requirements → preflight → lanzar.
4. **Verifica la señal de arranque sano** — actor: `estados=96`
   (`apex_actor.py:232-233`); learner: `listening on` (`apex_learner.py:146`).
5. **Heartbeat cada 15 s** al hub, con encolado local si el hub no está.
6. **Duerme**.

**Preflight (6 chequeos, cada uno un incidente que ya costó tiempo):** sha1 de la
ROM `a5aad1d1` · `Imported 1 games` · `len(resolve_states) == 96` · config de
`/weights` compatible con el ckpt · disco libre · `on_battery`. Se injertan además
como **health check permanente del hub**, no sólo como semáforo de alta.

### 3.4 Cambios al código existente — TODOS aditivos, ninguno reinicia la run

| # | cambio | archivo | coste |
|---|---|---|---|
| C1 | pasar `--name <node_id>` a los actores | ya soportado, `apex_actor.py:189` | **cero código** |
| C2 | añadir `node_id`, `generation`, `git_sha`, `difficulty`, `weights_version_en_uso`, `hijos_vivos` al dict `stats` del POST | `apex_actor.py:271-275` | ~5 líneas; seguro porque `apex.py:184-186` filtra `stats` por llaves conocidas → un actor viejo manda menos campos y no rompe nada |
| C3 | exponer en `/status`: `server_time`, `uptime`, `training_state`, `buffer_capacity`, `batch`, `loss`, `beta`, `grads_per_s`, `trans_per_s` | `apex_learner.py` — **ya se calculan** en `:258-277` y sólo van a stdout y wandb | ~10 líneas |
| C4 | **podar `actors` por edad** y servir snapshot cacheado ≤1 s sin lock | `src/agents/apex.py:181`, `:245` | ~8 líneas; hoy el dict sólo crece (12 entradas para 4 hosts) |
| C5 | bind del learner a la IP de tailnet | `apex_learner.py:85`/`:135` | **1 línea**, higiene de seguridad |

`training_state` ∈ `warmup | training | throttled | starved | frozen | unreachable`.
El desambiguador entre **throttled** y **frozen** es `transitions_in` avanzando
mientras `grad_steps` no — y hoy nadie lo mira. Es exactamente la ambigüedad del
incidente del 2026‑08‑27 (`08-cola-manana.md`: HTTP 200 con el loop congelado 1.5 h).
Nota semántica: `warmup` no mapea a ningún estado de color existente; se resuelve
como `idle` con etiqueta "calentando" **hasta que Felipe decida** (§7, D5).

### 3.5 Modos de fallo y mitigación (los 8 que importan)

| # | fallo | detección | respuesta |
|---|---|---|---|
| F1 | crash loop | backoff **por `desired_hash`**, cuarentena a la 5ª | contar globalmente dejaría un deploy **corregido** sin aplicar |
| F2 | red caída al learner | — | **NO HACER NADA**: `apex_actor.py:225-229`/`:287` ya hacen backoff y descartan lo más viejo. Relanzar aquí es el error clásico que convierte un corte de red en un ciclo de arranques |
| F3 | NAT de WSL2 atorado (GET pasa, POST muere) | POSTs con timeout y GETs sanos | perfil `red-hostil` (`--flush 100`), `apex_actor.py:190-195` |
| F4 | `git fetch` falla | — | **nunca mata al hijo**; reporta `stale` |
| F5 | versión incompatible | `SystemExit` determinista de `apex_actor.py:114-120` | **no relanzar en bucle** |
| F6 | máquina dormida | salto de reloj de pared contra `monotonic` | re-sincroniza, no alarma |
| F7 | hijos muertos dentro del actor | el padre lanza N y **nunca los revisa** (`apex_actor.py:235-241`) | reportar `hijos_vivos` (C2) |
| F8 | muerte del propio supervisor | pidfile + `launchd`/`systemd --user Restart=always` | **un supervisor muerto en una máquina que aún corre su actor es indistinguible, desde el hub, de una máquina caída** — hay que decirlo, no esconderlo |

### 3.6 Alta de una máquina nueva

Línea base medida por el panel: `tools/RUN_OMEN_DESDE_CERO.md`, 7 secciones,
~30–40 min, 14 bloques copy-paste, 5 con trampas documentadas. Su §7 dice
literalmente *"tras cualquier deploy de código: relanzar el actor"* — esa frase
**es** la justificación entera del fleet-agent.

Objetivo: **3 toques humanos + 1 clic de Felipe**.
- **H1** WSL + energía (irreducible: admin, BIOS, reinicio).
- **H2** Tailscale con login GitHub y grant a la org (OAuth interactivo por diseño).
- **H3** un comando: `bash tools/fleet_enroll.sh --hub … --token … --node-id omen`.
  **No `curl | sh`**: se clona el repo y se corre un script versionado y revisable.
  Sirve la ROM desde el hub por la tailnet **con verificación de sha1 antes de
  importar**, en vez del `http.server` temporal que alguien levanta y mata a mano.
- **H4** Felipe aprueba en la UI y asigna perfil.

**Límite declarado, no escondido:** WSL2 no regala el arranque automático.
`systemd=true` en `/etc/wsl.conf` exige un `wsl --shutdown` desde Windows, y WSL no
arranca solo al prender la máquina sin una tarea de Windows. El enroll imprime los
one-liners de PowerShell y el nodo queda visiblemente en **"arranque manual"**
hasta que se observa un reinicio con reconexión. Fingir que es automático es
exactamente lo que hoy tiene máquinas offline sin que nadie se entere.

Simetría que hoy no existe: **`drain` / `retire` / `revoke`**. Sin `retire`, una
máquina apagada a propósito dispara la alarma para siempre y entrena al equipo a
ignorarla.

### 3.7 Deploy y ROLLBACK — el rollback propuesto no era un rollback

"Canario + soak 20 min + `max_unavailable: 1`" es **higiene de despliegue**, no
rollback. Dos huecos que lo rompen:
(a) si el mal deploy rompe la red —clase de fallo con precedente **literal** en
este repo, `apex_actor.py:190-195`— ningún nodo alcanza el hub **ni git** para
volver atrás; (b) `previous_ref` vive en el hub, que es un punto único de fallo.

**Rollback real, en tres reglas:**
1. `git_ref` apunta a **tags inmutables** (`fleet/2026-08-28-01`) y el deploy
   **cambia el valor**, en vez de mover un tag.
2. Cada nodo **pinea en disco** su último `desired_hash` arrancado con éxito y
   **revierte solo tras N fallos, sin consultar al hub**.
3. `freeze: true` por defecto durante runs críticas; promover está **deshabilitado**
   hasta que el soak del canario pase, y el botón de volver atrás lleva escritos
   `previous_ref` y su SHA.

---

## 4. PLAN POR FASES

Regla que gobierna el orden: **cada fase deja el sistema utilizable y el Gradio
actual sigue vivo**. Nadie se queda sin herramienta a media run. El dashboard
corre hoy en `http://127.0.0.1:7861` (HTTP 200 verificado) y **no se le quita
ninguna capacidad en ninguna fase**.

| fase | qué entrega | valor / riesgo | seguridad |
|---|---|---|---|
| **F0** | `fleet/fleet.json` con `expected[]` + rescate al repo de la historia que hoy vive en `/private/tmp/claude-502/…` | altísimo / nulo | **SEGURA** — cero código ejecutable, cero procesos tocados |
| **F1** | `tools/leia_hub.py` (sampler stdlib, poll 10 s → JSONL en el repo) + **alarma que empuja** con la regla corregida | altísimo / bajo | **SEGURA** — sólo lee `GET /status` |
| **F2** | C1 (`--name`) en el relanzamiento de actores | alto / bajo | **SEGURA** — cero código, pero **exige relanzar actores** → ventana planeada |
| **F3** | C2–C5 (campos aditivos, poda, snapshot sin lock, bind) | alto / medio | **EXIGE PRUEBAS** — toca el learner vivo; probar en learner de juguete (puerto 8099, buffer chico) y desplegar en reinicio planeado, como `RUN_LARGA_SANTIAGO.md §5` |
| **F4** | UNA pantalla de sólo lectura de flota, servida por el hub, HTML estático sin toolchain | alto / bajo | **SEGURA** — no toca nada existente |
| **F5** | `app_local` monta el Gradio **intacto** en `/legacy` (`mount_gradio_app`) | medio / nulo | **SEGURA** — no cambia el Blocks, sólo quién lo sirve |
| **F6** | extraer las 25 funciones de lógica a routers + `JobManager` de N jobs | medio / **alto** | **EXIGE PRUEBAS** — **único punto de no retorno**; en rama, con `/legacy` como oráculo, fuera de ventana de run crítica |
| **F7** | SPA Vite + React + shadcn sobre la API, con `/legacy` como red de seguridad | medio / medio | **EXIGE PRUEBAS** |
| **F8** | `--supervise`: convergencia, deploy por tag, rollback local | medio / **muy alto** | **EXIGE PRUEBAS + conversación de confianza con el equipo** |
| **F9** | retiro de `/legacy` por **checklist de capacidades firmado por el dueño**, no por fecha | — | — |

Orden justificado: F0 y F1 entregan el 80 % del valor observable en ~1 día y con
riesgo cero; F6 (la reescritura) es la que más cuesta y la que menos entrega
mientras la flota siga siendo invisible. **Invertir ese orden es el error que este
plan existe para evitar** — y es el que ya se cometió una vez: el panel de anoche
(`agent/dashboard/PLAN-reconstruccion.md:14`, commit `781fe61f`) puntuó la
migración completa **3/10 en viabilidad** y hoy `fleet.json` sigue sin existir.

---

## 5. EL ALCANCE v1 RECOMENDADO

### 5.1 La recomendación

**v1 = OJOS.** Sólo lectura, fuera del dashboard, sin toolchain de build, medido
en días. Es F0 + F1 + F2 (+ F3 en el siguiente reinicio planeado) + F4.

**v1.0 — un día, cero UI, cero riesgo**
1. `fleet/fleet.json` con `expected: ["sss","legion","omen","mac"]`.
2. **Rescatar al repo** `night_watch_dqn.jsonl` (37 KB, 197 filas) y
   `apex_selector_v3.jsonl` desde `/private/tmp/claude-502/…`. Hoy toda la
   historia de la flota —incluidas las 12 coronaciones— vive en /tmp bajo un id
   de sesión de Claude, con la ruta hardcodeada. **Una purga la borra antes de que
   exista la UI que la dibujaría.** Es trabajo de minutos y no debe esperar a
   ninguna fase.
3. `tools/leia_hub.py`: sampler stdlib, poll 10 s, JSONL en el repo. **Productor
   único** de grads/s, trans/s y del replay ratio.
4. **Alarma que EMPUJA**, con la regla corregida: *"menos máquinas frescas que las
   esperadas"*, histéresis 2/2, `ack ≠ resolver`, y que **no se suicide**
   (`night_watch_dqn.py:46` muere con `SystemExit(1)` al primer alerta).
   Verificado: `grep -rn "ntfy|telegram|slack|webhook|smtp"` sobre `tools/` y
   `src/agents/apex.py` → **cero**. No existe ningún camino de notificación.
   Bajo `launchd` en la Mac, no lanzada a mano.

**v1.1 — ~20 líneas aditivas, en el siguiente reinicio planeado**
C1 (`--name`) · C2 (difficulty y weights_version en el wire) · C3
(`training_state`, `batch`, `buffer_capacity`, `server_time`) · C4 (poda +
snapshot sin lock) · C5 (bind a tailnet).

**v1.2 — 3–5 días, UNA pantalla de sólo lectura**
Barra: learner (`training_state` + grads/s) · flota **n/N** · replay ratio contra
su tope · escalera de 8 tiers. Debajo: **grid por MÁQUINA ESPERADA** con fila
fantasma para cada `expected` mudo, `age` visible en toda celda, y la tarjeta de
identidad del campeón (§2.1).

**Criterio de éxito falsable, uno solo:** *la flota cae de 4 a 2 a las 03:00,
alguien recibe aviso en <15 min, y al abrir una página ve cuáles dos, desde cuándo
y con qué `difficulty`.* Nada más. Eso no lo entrega ninguna de las siete
propuestas completas, y lo entrega v1.0 en un día.

### 5.2 Qué queda fuera de v1 y por qué (resolución de los conflictos del panel)

**El juez escéptico gana la SECUENCIA; los demás ganan la FORMA.** No es una
componenda: son preguntas distintas. `22`/`23`/`24`/`25`/`26` describen
correctamente **cómo debe ser** el sistema terminado; `33` describe correctamente
**qué se puede entregar esta semana**. La arquitectura de §1–§3 es el destino y se
mantiene entera; lo que se recorta es **cuánto de ella se construye ahora**.

Tres conflictos resueltos explícitamente:

1. **SPA Vite+React+shadcn en v1 — NO.** No existe proyecto de frontend (0
   `package.json`, sin `components.json`, sin `styles.css`, sin `components/ui/`),
   y `node`/`bun` sólo viven en esta Mac mientras los tres rigs son WSL2/Windows:
   un pipeline Vite ata cada corrección de UI a la portátil de Felipe. **v1.2 es
   un HTML estático con CSS vars planas**, servido por el hub. La SPA entra en F7,
   cuando la pantalla haya demostrado qué se mira de verdad. Los 26 hexes
   pinneados + los 15 tokens de estado + la regla de tinta son directamente
   portables a CSS vars; los `--chart-*`, `--sidebar-*` y las tres familias
   tipográficas esperan al frontend real.

2. **El colapso de los 35 dropdowns — CORRECTO como diagnóstico, PREMATURO como
   obra.** Verificado que 4 de los 5 lanzadores no pueden producir nada:
   `grep -rn ray requirements*.txt` → **exit 1** contra 7 imports en
   `pbt_orchestrator.py`; `sac/agent.py:113` y `:334` levantan
   `NotImplementedError` en la **primera línea** de `train` y `tune`;
   `models/production/league/` **vacío**; cero artefactos de tuning; y
   `find models -type f` → **3 ficheros** (un `.zip`, su `_vecnorm.pkl`, un json de
   curriculum), o sea **un solo modelo SB3** cuando la rama IA‑vs‑IA necesita dos.
   Reconstruir esa superficie es ~70 % del presupuesto por ~5 % del valor.

3. **Lo que SÍ se aplica hoy al Gradio ACTUAL, como parche de horas** (no requiere
   que la reconstrucción exista, y elimina trampas vivas):
   - Matar la opción **`"v1"`** de entorno: `env_tools.py:22-30` ramifica
     `v4`/`v3`/`else→v2` — **elegir v1 entrena en v2 en silencio**. Y añadir
     **`v4`**, que es el contrato vivo y **no está en ningún dropdown**. Vocabulario
     cerrado `{v2, v3, v4}`.
   - Sacar **`sac`** de los tres vocabularios de algoritmo → `{ppo, dqn, apex}`.
   - **5 botones de parada → 1** que nombra a su víctima; **3 consolas → 1**
     (las tres comparten `elem_id="terminal"` en `:1817`, `:1898`, `:2004`: HTML
     inválido, y el CSS `#terminal textarea` de `:2321` sólo aplica a una).
   - Apagar `gr.Timer(0.1, active=True)` (`:2133`).

Lo demás que se **mata** (no se pospone): PBT entero, `taskkill`/Force Kill
mentiroso, `get_best_tuning_params` (inyección por f-string), el botón TensorBoard
(salida a un `gr.Textbox(visible=False)` en `:2222`, y cero ficheros
`events.out.tfevents` en el repo), `_DASHBOARD_RELOAD_HEAD` (`:44-79`) y
`normalize_apex_p2_selection` (`:768`, condición siempre falsa tras `c365ae98`,
con docstring hoy literalmente falsa).

Lo que se **pospone con candado**: Optuna, league y exploiter. La etiqueta "fuera
de v1" **sólo es válida si trae criterio de reentrada o fecha escritos**, y la
pregunta al canal es requisito previo a congelar la lista, no cortesía — la
evidencia es de **este** checkout y Diego y Santiago tienen sus propios `models/`.

**Y las pantallas de Campeones y del banco de 12 rivales quedan bloqueadas aguas
arriba, no aplazadas por gusto**: 454+ de 500 filas de `bench_12rivals.jsonl` dicen
`apex_eval_tmp.pt`, hay tres esquemas de sidecar incompatibles (uno con
`wr_lvl1..4` cuya media **no** es comparable con la de 8 tiers) y `.pt` sin sidecar
alguno. Una UI sobre eso publica un ranking falso con aire de autoridad. **Se
arregla en `tools/` (normalizador de procedencia), no en la UI.**

---

## 6. RIESGOS Y CÓMO SE DETECTAN

| # | riesgo | señal que lo delata | detector concreto |
|---|---|---|---|
| R1 | **Este plan no se ejecuta y se re-deriva mañana** — ya pasó: `PLAN-reconstruccion.md` (781fe61f, 00:02) no lo citó ni una de las 7 propuestas de hoy | `find . -name "fleet*.json"` sigue vacío en 24 h | el propio F0. Si al final del día no existe `fleet/fleet.json`, el plan falló, independientemente de su calidad |
| R2 | **Colapso parcial de flota invisible** (ocurriendo AHORA: 1 de 4) | ninguna, y ése es el punto | alarma de F1 con `frescas < len(expected)`, histéresis 2/2 |
| R3 | **La UI miente sobre el experimento** porque `--difficulty` se declara en `fleet.json` y no viaja en el wire | un actor relanzado a mano con otro rango se ve "correcto" | regla A7 + C2: hasta que el actor lo reporte, la columna se pinta **"sin dato"**, jamás el valor declarado |
| R4 | **Número rancio con color vivo** — `PEREA-602` reporta `procs 40` y `46,479.4 steps/s` congelados hace 17.6 h | ninguna sin `age` | regla "rancio ≠ dato": toda celda derivada de un POST viejo se pinta *stale* usando `age` (`apex.py:258`); prohibido agregar `steps_per_s` crudo (es lo que hace `night_watch_dqn.py:70`) |
| R5 | **Learner congelado con HTTP 200** (precedente: 1.5 h el 2026‑08‑27) | `transitions_in` avanza y `grad_steps` no | `training_state` de C3. Y **cada número vivo lleva su edad de lectura**: HTTP 200 no es liveness |
| R6 | **Un mal deploy tumba la flota** (1.9 M grad_steps vivos) | soak del canario | F8 nace apagado; `freeze: true`; rollback local sin consultar al hub (§3.7) |
| R7 | **`git fetch` sin `--tags --force`**: el sistema entero reporta verde y al día mientras corre la versión vieja | ninguna — **falla en silencio reportando éxito** | comparar el SHA **efectivo del working tree** contra `git_ref`, no el ref pedido |
| R8 | **El hub cae y la consola parece vacía en vez de ciega** | — | dead-man's switch: "sin muestra desde hace X" es estado de primera clase; el agente local degrada a `/status` directo |
| R9 | **El supervisor muere y su máquina parece caída** (o al revés) | indistinguible desde el hub | pidfile + `launchd`/`systemd Restart=always`, y el heartbeat distingue "agente vivo, hijo muerto" de "silencio total" |
| R10 | **`/weights` (5.3 MB, sin ETag ni HEAD) descargado por un poller** | tráfico | regla dura §2.5; y el límite de tasa por IP de C4 lo hace inaplicable por accidente |
| R11 | **El learner sigue limitado por cómputo** (~28 grads/s con ratio instantáneo 4.93 < tope 8): sumar máquinas **no** es la palanca | el ratio nunca toca su tope | es el pendiente #3 de `08-cola-manana.md:40-62` (vectorizar `_featurize`, `PERBuffer.sample`, contención del lock). **Perfilar en learner de juguete, no en la run viva** |
| R12 | **Cero tests sobre la capa que se reescribe en F6** — hay 2 ficheros que tocan `web_dashboard.py`, y ejercitan justo lo que hay que preservar | una regresión sólo se ve a mano | `/legacy` como oráculo + esos 2 tests como puerta de F6; `get_stand_checkpoint_status` **no se borra** (test verde en `test_telemetry_dashboard.py:436`) |
| R13 | **El consentimiento local es una promesa, no un mecanismo**: el agente corre como el dueño y puede escribir el `.fleet/node.json` que dice respetar | — | no es un problema técnico: la conversación de confianza (`08-cola-manana.md:66`) es **precondición de F8**, no un trámite |
| R14 | **Dos librerías de primitivas** en el bundle de F7: `@shadcn/combobox` trae `@base-ui/react` mientras el resto usa `radix` | — | es lo que el registry manda hoy; se anota y se presupuesta probar juntos los dos modelos de portal y foco. No se "arregla" |

---

## 7. 🌅 Para Felipe — decisiones que sólo él puede tomar

**D1 · El lima `#C8EC40` como `--accent` es un BLOQUEADOR técnico, no estético.**
En shadcn, `accent` **es** el token de hover/focus/highlight: `dropdown-menu`
(×5 `focus:bg-accent`), `button` (×3), `command`, `select`, `toggle`, y el
`combobox` (`data-highlighted:bg-accent`). El día del primer `shadcn add`, cada
botón fantasma, cada item de menú enfocado y cada fila del picker bajo el cursor
se pinta de lima — y el presupuesto de "≤5 componentes, la firma se usa poco"
queda violado **por construcción**. Arreglo mínimo propuesto: `--accent` vuelve a
superficie neutra derivada de `--muted`, y el lima se declara como token propio
`--champion` con sus cinco usos cerrados. **Es cambio al contrato de color de
`00-DECISIONES.md §2`, así que la decisión es tuya, no del agente.**

**D2 · Los tokens derivados.** El bloque real necesita ~58 valores donde `§2`
pinneó 26 (16 de rampa de tier, 10 de `-subtle`, 6 de `-fg` para los 5 fallos de
AA medidos, 10 de chart). Los *hues* son derivados de los tuyos, pero los
**valores** son nuevos. ¿Autorizas la derivación bajo la regla "ningún hue nuevo"?

**D3 · Dónde vive el hub.** Este plan propone la Mac bajo `launchd` para v1, por
ser la única máquina con supervisión real y administración inmediata. Descartada
la 4090 (`ssh` → *connection refused*: nadie puede revivirla en remoto).
Alternativas: (a) Mac + espejo en la 4090; (b) una Raspberry / mini dedicada
—técnicamente correcta, sin hardware libre hoy—; (c) Cloudflare Tunnel + Access
como escape hatch **sólo** si aparece un compañero sin Tailscale.

**D4 · Cómo empuja la alarma.** Hoy no existe **ningún** canal (verificado: cero
hits de `ntfy|telegram|slack|webhook|smtp` en `tools/` y `apex.py`). Opciones:
ntfy.sh (trivial, pero es un servicio público externo), un bot de Telegram
(privado, requiere token) o correo. Es una decisión de privacidad tuya.

**D5 · `warmup` no cabe en los 5 estados de color.** No es *running* (no hay
gradientes) ni *degraded* (es correcto y esperado). O se pinta como `idle` con
etiqueta "calentando", o se abre un sexto estado —y entonces la lista deja de
estar cerrada, que es justo lo que el contrato de tokens existe para evitar.

**D6 · Cuándo se enciende `--supervise` (F8).** El 80 % del valor —**ver** que la
flota cayó— no necesita ni una escritura. Encender la convergencia automática
sobre laptops personales de compañeros y una run de 1.9 M grad_steps es la
decisión de mayor riesgo del paquete. Criterio propuesto: v1.0 con **dos semanas
sin falsos positivos** + la conversación de confianza. Tú pones la fecha.

**D7 · Techo de plan de la tailnet.** Hay exactamente 3 usuarios hoy. El 4º
compañero puede ser un problema de plan, no técnico. Confirmar en la consola de
admin **antes** de prometer "acceso para todo el equipo".

**D8 · El candado de "fuera de v1".** Optuna, league, exploiter, matchup SB3‑vs‑SB3
y la telemetría quedan pospuestos con evidencia sólida de desuso, pero la
evidencia es de **este** checkout. Sin fecha o criterio de reentrada escritos, en
tres meses son "muertos" sin que nadie lo haya decidido. ¿Preguntas en el canal
antes de congelar la lista?

**D9 · Prioridad relativa: `_featurize` vs consola.** El learner está limitado por
cómputo (27.70 grads/s, ratio 4.93 contra tope 8). Vectorizar `_featurize` promete
3–8× más gradientes **con el mismo hardware y la misma flota** (pendiente #3), y
compite por el mismo día de trabajo que F1. Los dos son baratos; el orden es tuyo.

---

### Anexo: hallazgos fuera de alcance (anotados, nunca `spawn_task`)

- `night_watch_dqn.py:46` muere con `SystemExit(1)` al primer alerta: el detector
  se suicida justo cuando empieza a servir.
- `apex_selector_v3.sh:44-45` corona con `torch.save` + `json.dump` **sin
  tmp+rename**: la coronación no es atómica y la UI puede mostrar un `.pt` y un
  `.json` que discrepan, sin forma de saberlo (sólo un sidecar de nueve trae sha256).
- `wr_media` del campeón es una **marca de agua alta** de un estimador de 48
  eps/tier que sólo se guarda cuando supera al anterior (v3291 dio 0.990 y el
  examen siguiente 0.956): leer la timeline como progreso monótono es leer ruido.
- El eje X de wandb (`grad_steps`) **retrocede** al reiniciar sin `--resume-ckpt`,
  y la cola llena tira filas en silencio (`apex_learner.py:279-282`): la consola
  debe distinguir "hueco en wandb" de "hueco en el entrenamiento".
- PPO/BizHawk **no reporta a wandb**: ese hueco se cierra en `train.py` (~15
  líneas, pendiente #7), no en la UI.
- Existen dos implementaciones distintas del mismo acto de subir un savestate
  (`handle_league_state_upload:1373` vs `handle_state_upload:2287`), y
  `toggle_league_matchup_mode:1365` / `toggle_exploiter_matchup_mode:1369` son
  byte-idénticas.
