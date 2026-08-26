# Stage 0 Runbook — instrumentar y medir en la máquina de entrenamiento

**Para:** quien opere la desktop (i9-13900K / RTX 4090 / 64 GB).
**Rama:** `stage0-metrics-and-semantics` (parte de `sf2-sota-rl-upgrade`).
**Objetivo:** producir los tres números que hoy no existen — throughput agregado,
veredicto del spinlock, y la validación del fix de movimiento — antes de tocar
cualquier optimización.

Todo lo de abajo corre desde la raíz del proyecto (dentro de la carpeta de
BizHawk, como siempre).

## 0. Preparación (una vez)

```
git fetch origin
git checkout stage0-metrics-and-semantics
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pytest code_testing\pytest -q
```

La suite debe dar **126 passed**. Si algo falla, deténganse y avisen.

## 1. Línea base de throughput (~30-40 min)

```
.venv\Scripts\python.exe src\scripts\benchmark_throughput.py --env v3 --n_envs 1,8,16,24
```

- Anota la tabla completa (se appendea sola a `logs\throughput_bench.jsonl`).
- El número que manda: **agent steps/s con n_envs=16**. Toda referencia previa
  (~165 steps/s con 1 env) era de laptop.
- Después, el experimento A/B que decide si los 16 procesos worker valen algo:

```
.venv\Scripts\python.exe src\scripts\benchmark_throughput.py --env v3 --vec dummy --n_envs 16
```

  Si `dummy` (todo en un proceso) iguala o supera a `subproc`, los 16 workers
  de Python son puro costo de pickle y hay una simplificación grande disponible.
- Repite la corrida de 16 con `--env v4` para medir cuánto paga la observación
  angosta.

## 2. Veredicto del spinlock (~2 min)

```
.venv\Scripts\python.exe -m pip install psutil
.venv\Scripts\python.exe src\scripts\measure_spinlock.py --hold 15
```

Imprime un VERDICT explícito. Si dice TIMED POLL, se tacha "16 cores
quemados" del roadmap; si dice HOT BUSY-WAIT, arreglar el spinlock sube al
tope de la lista de optimizaciones.

## 3. Validación Stage 1 — ¿el fix hace que Ryu camine? (~1 día de compute)

```
.venv\Scripts\python.exe src\scripts\train.py --algo ppo --env v3 --auto_curriculum --steps 1000000 --device cpu
```

y en otra terminal:

```
tensorboard --logdir logs
```

Qué mirar en TensorBoard (todo nuevo en esta rama):

| Métrica | Baseline (política random) | Señal de éxito |
|---|---|---|
| `spacing/frac_steps_far` | **0.522** | baja sostenida hacia <0.35 |
| `spacing/ep_rel_dist_median` | **83** | se acerca a la banda de 70 |
| `reward/shaping_per_step` | ~0 | mismo orden que `reward/damage_per_step` acumulado por round |
| `throughput/agent_steps_per_s` | (lo que dé el paso 1) | estable; caídas = investigar |
| `env/hp_sentinel_frac` | ~2-5% esperado | >10% sostenido = problema de estados |

Si tras ~1M steps `spacing/frac_steps_far` no se mueve del baseline, el
diagnóstico central de la rama está mal y NO se sigue con macros/v4/algoritmos:
se re-diagnostica.

## 4. Qué cambió en esta rama (resumen para el que revisa)

**A — instrumentación:**
- `src/agents/metrics_callback.py` — TensorBoard: `spacing/*`, `reward/*`,
  `macros/*`, `episodes/*`, `env/*`, `throughput/*`. Conectado a PPO y QR-DQN.
- El env agrega en el info terminal `ep_rel_dist_mean/median/frac_far`
  (muestras no-sentinel del episodio).
- `src/scripts/benchmark_throughput.py` y `src/scripts/measure_spinlock.py`.

**E — semántica (todo con tests):**
- **Reset del protocolo:** `reset()` drena el payload viejo en vuelo, devuelve
  el frame real post-carga del savestate (antes devolvía el último frame del
  episodio ANTERIOR ×4), y re-arma el offset de un mensaje con un comando
  neutro para conservar el pipelining emulación↔inferencia. El lag de un paso
  DURANTE el episodio se conserva y queda documentado en `base_env.step()` —
  quitarlo serializaría el loop.
- **`league_env` porteado a `envs/reward.py`:** el self-play entrenaba contra
  el reward viejo (zona muerta, 0.99 hardcodeada, ±50, sin sentinels — con el
  falso-KO de menú incluido). Ahora usa el mismo módulo puro que single-player
  y emite los mismos info keys (+`opponent_id`).
- **atexit solo en el proceso principal** (`env_tools.py`): un worker que moría
  ejecutaba el sniper de PowerShell y mataba los 16 emuladores.
- **`SelectiveVecNormalize(gamma=AGENT_GAMMA)`**: la normalización de reward
  usaba 0.99 fijo con el agente a 0.995. Los `.pkl` viejos cargan igual.
- **QR-DQN `gradient_steps` 1→8** (agent y estudio Optuna a la par): el replay
  ratio era 1/64.
- **PBT**: `n_epochs=10, target_kl=0.03` → `4, None` (el defecto de Task 4 que
  nunca llegó a ese path).
- `stream_buffer` se limpia al respawnear el bridge; docstring de
  `macro_wrapper` corregido (23, no 14); f-string de debug ya no se construye
  en cada paso (`debug_mode` ahora default False).

**Nota de compatibilidad:** modelos guardados cargan sin cambios (ni obs ni
action space cambiaron). El primer obs de cada episodio ahora es el frame real
del estado cargado — estrictamente mejor, pero es un cambio de distribución en
t=0 respecto a los checkpoints viejos.
