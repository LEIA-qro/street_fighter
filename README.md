# Street Fighter II — Reinforcement Learning

Enseñarle a una IA a jugar *Street Fighter II' — Special Champion Edition* (Sega Genesis),
contra los doce rivales del juego y en los ocho niveles de dificultad.

**El juego está resuelto.** El campeón es un Ape-X DQN distribuido:

| lvl1 | lvl2 | lvl3 | lvl4 | lvl5 | lvl6 | lvl7 | lvl8 |
|---|---|---|---|---|---|---|---|
| 100% | 100% | 100% | 100% | 99.0% | 99.0% | 97.9% | 93.8% |

Win rate de rounds de apertura con desfase aleatorio de arranque, n=768. En **peleas
completas al mejor de tres en el nivel 8** hace **~90%** (dos mediciones independientes
de n=360, 91.7% [88–94] y 89.7% [86–92]). Pesa 4 MB y vive en
`benchmarks/apex_milestones/apex_v3291_media990.pt`.

> **[`HANDOFF.md`](HANDOFF.md) es el documento de entrada.** Estado completo, la cola de
> trabajo, las decisiones que esperan a un humano y las reglas de la casa. Este README es
> el mapa técnico. `agent/memory/` es la memoria viva del proyecto.

---

## Arranque rápido

```bash
git clone git@github.com:LEIA-qro/street_fighter.git
cd street_fighter
python3.12 -m venv .venv
.venv/bin/pip install torch        # aparte: depende de tu hardware (pytorch.org)
.venv/bin/pip install -r requirements.txt -r requirements-retro.txt
.venv/bin/python -m pytest code_testing/pytest -q     # 630 verdes
```

El ROM va en `roms/`, con sha1 `a5aad1d108046d9388e33247610dafb4c6516e0b` — **ningún otro
dump sirve** para stable-retro. Se importa con `python -m stable_retro.import roms/`
(es `stable_retro`, no `retro`).

En Windows nativo no se puede instalar stable-retro: no publica wheels para `win_amd64`,
así que las máquinas Windows del equipo lo corren dentro de WSL2. macOS y Linux instalan
nativo. La suite de tests corre en cualquier parte, sin emulador y sin BizHawk.

```bash
# Ver al campeón pelear, y grabarlo en video
.venv/bin/python tools/grabar_gauntlet.py --difficulty 8

# Medirlo contra los 12 rivales (el brazo del Ape-X se llama "rainbow")
.venv/bin/python tools/bench_12rivals.py --arm rainbow \
  --ckpt benchmarks/apex_milestones/apex_v3291_media990.pt \
  --difficulty 8 --desync-max 30

# La interfaz local (Gradio). Escucha solo en esta máquina; --auth para exponerla
.venv/bin/python src/scripts/web_dashboard.py
```

---

## Los dos backends

Es lo menos obvio del proyecto. Hay **dos emuladores** y **un solo contrato de
observación**; una política entrenada en uno funciona en el otro.

**stable-retro** (`src/envs/retro_env.py`) — Linux, macOS y WSL2. Headless, ~3,700 fps por
proceso, un emulador por proceso (límite de libretro). **Es donde se entrena.** No importa
`core/config.py` a propósito: no depende de nada de BizHawk.

**BizHawk** (`src/envs/base_env.py` + `lua/v2.0/`) — solo Windows. Un puente TCP en
lock-step contra el emulador. **Se queda** para evaluación visual, PvP y
humano-contra-IA; nadie lo va a eliminar. El ciclo es estrictamente 1-envío / 1-recepción:

```
┌────────────────────────────────┐                 ┌───────────────────────────────┐
│     BizHawk (Lua)              │                 │     Servidor Python           │
│  1. Lee WRAM big-endian        │                 │  1. Recibe la observación     │
│  2. Manda la cadena de vars    ├─[ TCP Socket ]─>│  2. Calcula la acción         │
│  3. Spinlock: bloquea          │                 │  3. Formatea y manda teclas   │
│  4. Lee el socket e inyecta    │<─[ TCP Socket ]─┤  4. Rollout / SGD             │
└────────────────────────────────┘                 └───────────────────────────────┘
```

El emulador bloquea hasta que Python responde, así que ni un frame se salta aunque el
backpropagation tenga un pico. **La diferencia que hay que recordar:** BizHawk entrega la
observación con un paso de retraso — es pipelining deliberado del protocolo, no un bug.

La paridad entre backends está validada bit a bit y hay tests que la sostienen. El
contrato vivo es **v4**: 23 floats por frame × 4 frames apilados.

---

## Los dos algoritmos

**Ape-X DQN** (`tools/apex_learner.py` + `tools/apex_actor.py`) — el que ganó. Un learner
con replay priorizado central en la máquina con GPU y N actores en el resto de la flota,
hablando HTTP por Tailscale. Red QR-DQN dueling, 72 acciones (63 primitivas + 9 macros).
Ganó con **~50 veces menos cómputo** que el mejor PPO del proyecto.

**OpenES en flota** (`src/es/`) — evolución, no gradientes. Una MLP numpy de 14k–22k
parámetros, pares antitéticos por semilla, centered ranks y Adam (Salimans 2017). Un
coordinador reparte trabajo y los workers son stateless. Llegó a ~75–82% honesto en el
banco de nivel 1. **Perdió la carrera**, con el mismo entorno y el mismo reward.

> **Por qué OpenES y no EGGROLL.** EGGROLL es OpenES con perturbaciones low-rank, pensado
> para redes enormes. A 22k parámetros el low-rank restringe la exploración sin ahorrar
> nada: el cuello de botella es el emulador, ~99.9% del tiempo, no el álgebra. La decisión
> está fechada en `agent/memory/02-decisiones.md`, el swap vive contenido en
> `src/es/openes.py` si la red crece, y el reconocimiento completo está en
> `agent/recon-2026-08-25-rumbo-a-eggroll.html`.

También hay **PPO sobre Stable-Baselines3** (`src/agents/ppo/`), que fue el campeón
durante meses y sigue siendo la política más robusta al ruido. Y andamio que **no está
vivo**, dicho aquí para que nadie lo descubra por las malas: `src/agents/sac/` lanza
`NotImplementedError` en la primera línea de sus dos entrypoints, y PBT
(`src/agents/pbt/`) necesita `ray[tune]`, que está deliberadamente fuera de
`requirements.txt`.

---

## El bug de seis meses

Del 2026-03-01 al 2026-08-26, **perder pagaba +55 de reward y ganar pagaba −12**.

Se decidía "muerto" con `hp <= 0` sobre un entero sin signo, y se calculaba el daño por
diferencia de HP en el frame terminal. La verdad de la ROM, medida sobre 300,000+ frames:
**HP va de −27 a 176**, `hp == 0` es una lectura VIVA que persiste cientos de frames, y el
marcador de muerte es **el signo** (65535 = −1).

El arreglo es una sola función de decodificación (`src/envs/reward.py`, `hp_to_signed`)
compartida por ambos backends, más un `RoundTracker` común. Medido sobre 150 episodios
después del fix: ganar +138, perder −96, sin traslape.

**Todo resultado anterior al 2026-08-26 se midió bajo ese régimen roto.** Los otros
diecinueve bugs cazados, cada uno con su test de regresión, están en
`agent/memory/03-bugs-cazados.md`. Las trampas que ya costaron horas, en
`agent/memory/07-gotchas.md`. Léelos antes de tocar el reward o el protocolo.

---

## Mapa del repositorio

```
street_fighter/
├── HANDOFF.md               # ← empieza aquí
├── agent/memory/            # la memoria del proyecto: arquitectura, decisiones,
│                            #   bugs, infra, bitácora de runs, cola, gotchas
├── src/
│   ├── core/                # config, puente TCP de BizHawk, extractor de WRAM,
│   │                        #   normalizador selectivo, telemetría, constantes de RL
│   ├── envs/                # retro_env.py (stable-retro) · sf2_v1..v4 (BizHawk)
│   │                        #   reward.py y action_macros.py: compartidos y puros
│   ├── agents/              # ppo · dqn · rainbow · apex · league · pbt · sac(muerto)
│   ├── es/                  # coordinador, worker, OpenES, política numpy, protocolo
│   └── scripts/             # train, tune, resume, matchups, stand_leia, web_dashboard
├── tools/                   # lo que se opera de verdad: apex_learner/actor,
│                            #   bench_12rivals, grabar_gauntlet, leia_hub,
│                            #   forge_states, y los runbooks RUN_*.md
├── benchmarks/              # modelos congelados (apex_milestones/) y actas crudas
├── retro_integration/       # integración custom: 25 vars de RAM y 96 savestates
│                            #   verificados (12 rivales × 8 niveles)
├── fleet/                   # fleet.json = censo de qué DEBERÍA correr; history/ = qué pasó
├── lua/v2.0/                # clientes lock-step de BizHawk
├── code_testing/pytest/     # 630 tests
├── doc/                     # justificaciones matemáticas, mapa de RAM, guía de CLI
├── infra/                   # terraform de la flota ES (DESMONTADO, ver abajo)
└── design/, web/            # paleta medida del juego y consola del hub de flota
```

---

## Operación

El entrenamiento vivo fue **distribuido**: cuatro máquinas del equipo coordinadas por
Tailscale sobre el tailnet de la organización, con el learner en la desktop con GPU y
actores en el resto. Métricas en Weights & Biases, equipo `leia-qro-rl`, dos proyectos
que no se mezclan: `leia-sf2-dqn` y `leia-sf2-es`.

Los runbooks por rol están en `tools/RUN_*.md` y son el detalle real de cómo se lanza cada
cosa. `tools/leia_hub.py` es la observabilidad: muestrea el `/status` del learner, lo
compara contra el censo de `fleet/fleet.json`, alarma cuando faltan máquinas y sirve una
consola web de una sola página.

**No hay nada corriendo hoy.** La run del curriculum se cerró a propósito el 2026-08-28 y
**la infraestructura de AWS se destruyó el 2026-09-11**: la instancia coordinadora llevaba
dieciséis días encendida sin trabajo. `infra/` conserva el terraform completo y su README
sigue siendo correcto de principio a fin para volver a levantarla. Lo que había en S3 y no
era reproducible se rescató al repo antes de destruir: ver `benchmarks/LEEME-runs-ES.md`.

---

## Las reglas de la casa

- **Identidad de git: `FelipeJackFox` / `felipaupz@gmail.com`, siempre.** El historial ya
  quedó partido en tres identidades una vez.
- **`git pull` antes de lanzar cualquier cosa**, en todas las máquinas. Los modelos se
  distribuyen por git, no copiándolos a mano.
- **Un entrenamiento por máquina.**
- **Tras desplegar código nuevo, relanza workers y actores** — la banca por fingerprint
  deja esperando al código viejo, no lo mata.
- **Lanza con `nohup` o `tmux`, nunca en la terminal del editor.** Cerrar el editor mató
  una run.
- **Los modelos guardados deben seguir cargando.** Cambiar la observación o el espacio de
  acciones es un entorno nuevo, nunca una modificación en sitio.
- **El campeón se selecciona, no se toma de los pesos vivos.** La política churea: entre
  90k y 192k gradientes el número limpio cayó de 100 a 83 mientras el honesto se mantenía.
- **El número honesto es el de desfase, no el limpio.** Un banco greedy con estado fijo da
  episodios idénticos: repetir no añade muestra. Toda acta lleva n e intervalo de Wilson.

---

## Documentación

| Documento | Qué contiene |
|---|---|
| [`HANDOFF.md`](HANDOFF.md) | El estado del proyecto, la cola de trabajo y lo que espera decisión |
| [`agent/memory/`](agent/memory/) | Memoria viva: arquitectura, decisiones fechadas, bugs, infra, runs, gotchas |
| [`doc/README.md`](doc/README.md) | Justificación del puente lock-step, mapa de WRAM del 68000, espacio de observación |
| [`doc/DEVELOPER_CLI_GUIDE.md`](doc/DEVELOPER_CLI_GUIDE.md) | Referencia de banderas y modos de cada script |
| [`tools/RUN_*.md`](tools/) | Runbooks por rol: learner, helpers, onboarding de máquina nueva, stand |
| [`infra/README.md`](infra/README.md) | Cómo levantar la flota ES en AWS desde cero |
| [`agent/dashboard/`](agent/dashboard/) | Auditoría del dashboard: qué controles mienten y qué ramas no se alcanzan |

El equipo: Felipe ([FelipeJackFox](https://github.com/FelipeJackFox)), Diego
([Perea094](https://github.com/Perea094)) y Santiago
([SantiagoSaldanaS](https://github.com/SantiagoSaldanaS)), con agentes de Claude Code.
Los tres son owners de la organización [LEIA-qro](https://github.com/LEIA-qro).

Licencia: ver [`LICENSE`](LICENSE).
