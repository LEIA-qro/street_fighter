# 08 — Cola de trabajo (escrita 2026-08-27 ~00:00, la noche de los tres algoritmos)

La cola VIVA post-noche-histórica. Sustituye a 06-pendientes para lo operativo
(06 queda de histórico). Orden = prioridad sugerida.

## Lo primero de mañana

1. **Fix numpy en el actor Ape-X** — los hijos actúan con torch forward por paso
   individual: ~0.5-1ms de overhead de despacho por consulta en una red de ~1M
   params. Convertir el ACTING a numpy puro (como es/policy.py; torch solo para
   cargar el state_dict) ≈ 2-3× throughput por core. Diagnóstico medido: hijos
   al 50-66% CPU (mi Mac) / desktop al 30% total. Mitigación aplicada anoche:
   sobre-suscribir procs (Mac 12, desktop 44, Legion 30). Ambas componen.
   Verificar adversarial antes de desplegar (toca el acting de una run viva).
2. ~~**Inferencia de savestates lvl5-8 (el atajo)**~~ — **HECHO (madrugada
   2026-08-27)**, con giro científico: el poke de solo `0xFE45` resultó
   COSMÉTICO (la escena copia la dificultad al cargar; validación conductual
   FAIL — el forjado jugaba idéntico a su donante). Un diff masivo de blobs
   sobre los 68 estados auténticos encontró el conjunto real: **`0x97B2`
   (copia en-pelea, = nivel−1), `0x96B8` (parámetro derivado no lineal) y
   `0xBA35/38/58` (derivado lineal triplicado)**. Pokeando los 6 bytes:
   validación PASS en 3 rivales × 2 direcciones (480 eps, tests de
   permutación) → **las 28 celdas faltantes forjadas; la malla 12×8 = 96
   COMPLETA**, verify_states 240/240, manifest documentado
   (`difficulty_ram_in_fight`). Herramientas: `tools/forge_states.py` +
   `tools/validate_forged_states.py` (+19 tests). **Extender la run a
   `--difficulty 1..8` = decisión de equipo; solo se relanzan actores.**
3. **Fleet-agent** ("dar de alta y olvidarse"): `fleet.json` en el repo como
   plano de control (qué corre cada máquina) + supervisor por máquina (loop:
   git pull rama `fleet-stable` → comparar manifiesto/HEAD → relanzar hijo →
   heartbeat; backoff anti-crash-loop). Deploy = mover el tag. Conversación de
   confianza con el equipo antes de encenderlo.

## ⚠️ INCIDENTE madrugada 2026-08-27: learner CONGELADO (grads 33103)

- ~00:40, tras llenarse el buffer (1M): `/status` vivo, ingesta corriendo,
  pero **cero gradientes** — el main thread quedó bloqueado a media
  iteración FUERA del lock (el `finally` nunca corrió: no fue crash).
  Sospechoso principal: `wandb.log` colgado (el mal conocido); alternativa:
  kernel CUDA atorado. **El traceback del Ctrl+C de Santiago es el
  diagnóstico definitivo — pedirle que lo copie.**
- **Fix ya commiteado**: wandb en hilo daemon con cola drop-on-full
  (`apex_learner.py`) — un sidecar muerto ya no puede frenar el
  entrenamiento. Reinicio documentado en `RUN_LARGA_SANTIAGO.md` §5
  (git pull + resume del último ckpt + `--weights-every 500`).
- La escalera ANTES del congelamiento (33k grads): lvl1 .96-.97 / lvl2
  .90-.93 / lvl3 .84-.90 / lvl4 .50-.56 — subiendo parejo, sin mínimo
  cobarde. La flota siguió generando toda la noche (experiencia a un
  learner paralizado: inofensivo, solo cómputo tirado).

## Vigilancia de la run del curriculum (corriendo AHORA)

- **Learner**: desktop-4090-ubuntu-wsl:8090, `--macros` (72 acciones, PRIMER
  run con macros de la historia), buffer 1M, wandb id `rainbow-apex-curriculum`.
  Actores: desktop 44p + Legion 30p + Mac 12p, todos `--difficulty 1,2,3,4`
  (curriculum UNIFORME: decisión de Felipe — bandas por máquina sesgarían el
  buffer hacia el tier de la CPU fuerte; el PER enfoca solo).
- **Selector v2 en la Mac** (scratchpad/apex_selector_v2.sh + .jsonl): cada 30
  min examina el θ vivo greedy con desfase EN CADA TIER (bench --difficulty N)
  y guarda el mejor por promedio en benchmarks/apex_milestones/
  apex_curriculum_best.pt(.json).
- **Riesgo a vigilar: "coward local minimum" de Diego** — si lvl4 muestra wr
  cayendo + episodios acortándose (aprender a perder rápido), quitar lvl4 de
  los ACTORES (--difficulty 1,2,3 + relanzar; el learner no se toca). La
  escalera temprana (15 min: lvl1 85/lvl2 74/lvl3 65/lvl4 36, todo subiendo)
  contradice el colapso por ahora.
- Dashboard DQN v2: https://wandb.ai/leia-qro-rl/leia-sf2-es?nw=zd7dgnfgz3s
  (win_rate_recent200 + escalera por lvl). Dashboard ES:
  ?nw=kjfkszyz11g. Criterio de paro: escalera plana ~2 evaluaciones nocturnas.
- Al próximo REINICIO del learner (no urge): subir `--weights-every` 100→500
  (versión de pesos cada ~10s en vez de ~2s: menos recargas de los hijos).

## Experimentos en cola (decisión de equipo)

- **PPO + macros en la desktop** (`train.py --macros`, red fresca) — la prueba
  A/B de la tesis del action space en la pista PPO. Bloqueado mientras la
  desktop hospede el learner (un entrenamiento por máquina).
- **ES run 4**: `--policy v4onehot_macro --sigma-final 0.012
  --sigma-decay-gens 600` + perturbaciones de run 3, cuando haya slot de
  máquina. La madre (EC2) está VIVA pero ociosa esperándola (unit con flags de
  run 3; cambiar prefijo S3 a es-run4-macro).
- ~~**Humano vs IA en BizHawk**~~ — **CONSTRUIDO (2026-08-27 tarde, modo
  STAND para promocionar LEIA)**: `lua/v2.0/stand_env_client.lua` (payload
  de 25 vars crudas de data.json) + `src/scripts/stand_leia.py` (obs v4
  idéntica al entrenamiento, macros calcados del wrapper, KO por signo,
  time-over, marcador, rematch, P2 = pad físico por passthrough) +
  `tools/RUN_STAND_LEIA.md`. Revisión adversarial de 16 agentes: 12
  hallazgos aplicados (el crítico: el prefijo `<len> ` que BizHawk
  antepone habría matado el parser al primer payload). 9 tests de paridad
  (gold test MacroPlayer≡MacroActionWrapper). **FALTA solo el end-to-end
  en Windows con BizHawk** (~5 min con el runbook). Lo viejo:
  su `test_ai_vs_ai_v2.py` +
  `match_test_env_client.lua` ya hacen VS con doble inyección; falta (1) dejar
  de inyectar P2 en el Lua (passthrough de teclado), (2) adaptador de θ ES/DQN
  a v4-BizHawk (lo puedo escribir; probar exige alguien en la desktop).
  Caveat: obs BizHawk con 1 paso de lag (transferencia inversa probada).
- **Backlog de sentidos** (priorizar con lo que duela en tiers altos): medidor
  de STUN (no está en obs), Y de proyectiles (solo X), fase/frames del move
  rival (hay ID, no timing). Cada canal nuevo = obs nueva = runs nuevas: en
  paquete.
- **Caminar**: evidencia cuádruple de que saltar es óptimo en lvl1 (air_frac
  PPO .47/ES .45-.48/DQN .43 — aunque el DQN es el primero que MODULA por
  matchup: .23 vs Balrog). Los anti-aéreos de lvl3-4 ya están EN la run del
  curriculum — medir air_frac del campeón nuevo; plan B: incentivo no
  invariante (con riesgo de hacking).

## Deuda técnica menor (aceptada, no urgente)

- ES worker theta_cache puede cruzar runs si colisionan números de gen en un
  swap de coordinator (fix real: nonce de run en el wire).
- Ape-X: duplicación por ack perdido (sin idempotency key) y carrera benigna
  de prioridades PER en wraparound — NOTEs de los verificadores, tolerables.
- wandb: 0.28.2 es la versión maldita (sidecar ahoga la historia — madre corre
  0.21.4); el 0.29 de Diego/Santiago funcionó bien. Regla: log con step
  explícito publica con 1 log de retraso (fila parcial); los reinicios
  frecuentes del proceso matan charts.
- metrics.jsonl de la madre (S3) es la observabilidad a prueba de wandb del ES.

## Recordatorios operativos

- ROM: sha1 `a5aad1d108046d9388e33247610dafb4c6516e0b`; import con ruta
  RELATIVA correcta; los .pt de torch son zips (renombrar si llegan .zip);
  Taildrop NO cruza usuarios (compartir por http.server en la tailnet).
- pkill SIEMPRE acotado al venv del repo (`street_fighter/.venv`) — dos
  incidentes de colateral anoche (orchestrator backend y el worker ES propio).
- Tras desplegar código: RELANZAR workers/actores (la banca deja esperando al
  código viejo, no lo mata).
- La desktop entrenó BizHawk 6 meses con OTRO dump del ROM (sin consecuencias
  para BizHawk; para retro solo vale el a5aa).
