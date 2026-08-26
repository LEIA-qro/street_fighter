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

La suite debe dar **177 passed**. Si algo falla, deténganse y avisen.

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

## 5. Runs A y B — desatorar el optimizador y anti-salto

Diagnóstico de la corrida de 1M del 2026-08-24: **el optimizador estuvo
congelado toda la corrida**. `lr=2.108e-05` (artefacto malo de Optuna) anneleado
linealmente a cero ⇒ `train/entropy_loss` clavado en el máximo de
MultiDiscrete([9,7]) (4.1431→4.1405) y `clip_fraction 0.000` de principio a
fin. El "se acerca saltando" observado es simplemente lo que parece una
política uniforme-random: 3 de las 9 direcciones son saltos.

**Run A — lr sano (sin cambios de código, el flag ya existía):**

```
.venv\Scripts\python.exe src\scripts\train.py --algo ppo --env v3 --auto_curriculum --steps 1000000 --lr 3e-4
```

**Run B — A + gate anti-salto** (`--ground_gate`: Phi(d, air) = potencial de
spacing solo en el suelo, 0 en el aire — sigue siendo PBRS puro sobre el
estado extendido, así que es policy-invariant; saltar deja de cobrar shaping
por acercarse):

```
.venv\Scripts\python.exe src\scripts\train.py --algo ppo --env v3 --auto_curriculum --steps 1000000 --lr 3e-4 --ground_gate
```

Qué mirar en TensorBoard, en este orden (si el paso 1 no ocurre, los demás no
significan nada):

| # | Métrica | Corrida congelada (baseline) | Señal de éxito |
|---|---|---|---|
| 1 | `train/entropy_loss` | **clavado en 4.143** toda la corrida — así se supo que el optimizador estaba congelado | **CAE** de 4.143 de forma sostenida |
| 2 | `train/clip_fraction` | **0.000** todo el run | **> 0** desde los primeros updates |
| 3 | `spacing/frac_steps_far` | ~0.46 | baja sostenida |
| 4 | `spacing/ep_air_frac` | ~0.33 (uniforme-random: 3/9 direcciones son salto) | baja **muy por debajo de 0.15** = camina en vez de saltar |

Si A destraba la entropía pero `ep_air_frac` no baja, el gate de B es el
tratamiento; si ni A mueve la entropía, el problema es otro y se
re-diagnostica antes de tocar rewards.

## 6. Post-mortem del Run A y fix de semántica de rondas (2026-08-25)

**El diagnóstico del Run A estaba mal, y la conclusión correcta es peor.** El
resumen decía "reward hacking: el agente aprendió a buscar double KOs". No es
eso. El agente **no puede influir en esa etiqueta**: el env nunca terminaba las
rondas, y `win_rate`, `double_ko_rate`, `timeout_rate` y `len_mean` del Run A
**no miden habilidad, miden la fase del frame-skip contra un frame de 1 frame de
ancho**. Una política *random* en esta Mac reproduce las métricas de "fin de
entrenamiento" del Run A casi exactamente (`double_ko` 0.55 vs 0.5478).

### 6.1 Qué encontró la medición

Dos corridas independientes sobre el core headless (40,000 y 4×30,000 frames de
emulador, 29 KOs, `Champion.Level1.RyuVsGuile`):

| hecho medido | valor |
|---|---|
| rango de las palabras de HP (0xFF8042 / 0xFF82C2), **vivo** | **0..176**. Cero lecturas en 177..32767 |
| rango de las palabras de HP, **muerto** | **negativo pequeño**: se han visto `{−1, −4, −5, −6, −7, −8, −9, −10, −11, −13, −27}` |
| KOs con ganador claro | **29 / 29** (y 52/52 en la re-medición) |
| KOs simultáneos reales (ambos HP < 0 a la vez) | **ocurren**: 96 frames en 160,000; ~1 de cada 150 episodios |
| ancho de la ventana HP < 0 | **33–457 frames** (mediana 33; los cientos son sólo el KO que cierra el match) |
| HP del GANADOR durante esa ventana | intacto y congelado en un solo valor (52/52) |
| ancho de la ventana [0, 0] que el código viejo esperaba | **1 frame** → se atrapa 1 de cada 4 |
| latencia de `matches_won` / `enemy_matches_won` tras el KO | **+1 frame, 29/29** |
| **reloj de ronda** (0xFF972A, BCD `0x99`→`0x00`) | lee 0 durante **91–131 agent steps** en cada TIME OVER |
| latencia del contador respecto al reloj en un TIME OVER | **+10 agent steps** (el reloj llega primero) |

> **Corrección (2026-08-26).** Tres filas de esta tabla decían otra cosa en la
> primera versión y eran **falsas**; quedan aquí corregidas porque dos de ellas
> son load-bearing:
>
> * *"el HP sólo vale 0..176 y −1; ni una lectura en 177..65525"*. El conjunto
>   negativo es más ancho — el golpe final resta más allá de cero antes de que
>   la ROM congele la palabra. Leídos como u16, `−13` es 65523 y `−27` es 65509,
>   **dentro** del intervalo que la frase declaraba vacío. El test de SIGNO no
>   se ve afectado (todos son negativos, todos > 32767 sin signo), pero
>   **nunca** endurezcas la prueba de muerte a `== −1` / `== 65535`: perdería la
>   mayoría de los KOs. `HP_SENTINEL_THRESHOLD = 200` sobrevive sólo como clamp
>   de observación, no como test de muerte.
> * *"KOs simultáneos reales: 0"*. Sí ocurren (medido arriba). Se clasifican
>   correctamente como `draw` / −50, pero `draw_rate` **no** va a ser 0 y en
>   ellos la ROM no tica **ningún** contador.
> * *"el KO real quedaba bloqueado ~110 agent steps"*. La ventana medida es de
>   33 frames de emulador (8.25 agent steps) en una ronda normal; sólo el KO
>   que cierra el match llega a 419–457 frames. La cifra 110 no corresponde a
>   ninguna de las dos.

**El "HP sentinel" nunca fue un sentinel.** Era un HP negativo leído con el
tipo equivocado: la señal de KO, invertida en "este frame es ilegible, no
termines". El KO real quedaba bloqueado toda la ventana de KO, y para cuando el
flag se limpiaba la ROM ya había reseteado ambas barras y **la identidad del
ganador ya no existía**. Sobrevivía `0/0` (→ `DOUBLE_KO`) o `176/176` (→ el episodio seguía
a la ronda siguiente).

**Y `hp == 0` NO significa muerto.** Se midió un luchador vivo a 0 HP durante
437 frames mientras pegaba, y entre rondas **ambas** palabras se quedan en 0
durante cientos de frames sin que nadie haya muerto. `hp <= 0` nunca fue un test
de muerte válido.

**Descartar del Run A** todo lo que dependa de `win_rate`, `loss_rate`,
`double_ko_rate`, `timeout_rate`, `len_mean`, `terminal_per_step` y
`hp_sentinel_frac`. **Sobrevive** lo que no toca el frame terminal: `spacing/*`
(rel_dist 83→74, air_frac 0.39), `entropy_loss`, `clip_fraction`, `approx_kl`.
La conclusión "el optimizador se desatoró y el spacing mejoró" se sostiene.

### 6.2 Qué se cambió

**Regla única de muerte: el SIGNO de la palabra de HP.** `envs/reward.py`
expone `hp_to_signed()` — único lugar donde se decodifica, importado por los dos
backends. Un KO es `hp < 0` **estricto**. Sirve igual para BizHawk (Lua manda
`read_u16_be`, así que el KO llega como 65535 → −1) y para retro.

**TRES formas de terminar una ronda, no una ni dos**
(`envs/reward.py:resolve_round_result`):
1. **KO** — `hp < 0`. Ventana ≥ 33 frames: ningún muestreo la puede perder.
2. **TIME OVER decisivo** — se acaba el reloj, la ROM adjudica la ronda al que
   tenga más vida y tica su contador, **sin que ningún HP se vuelva negativo**.
   Esto no lo reportó ninguno de los tres investigadores; salió validando el fix
   (episodio 28 de 40: contador ticó en el step 1031 con HP (7, 16), y el env
   siguió 469 steps más hasta truncar y loguear un TIMEOUT que valía 0 en vez de
   la derrota que era).
3. **TIME OVER con HP EMPATADO — "DRAW GAME"** — se acaba el reloj con las dos
   barras exactamente iguales. La ROM termina la ronda y **no tica ningún
   contador**. Un env que sólo mira HP + contadores **no termina en absoluto**:
   juega una ronda entera de más y trunca a los 1500 steps como TIMEOUT que vale
   0 — exactamente la patología del Run A que este cambio existe para matar.
   Medido en vivo (ambos HP fijados en 120, reloj hasta el final): la ronda
   terminó, las barras se rellenaron ~95 agent steps después, y los dos
   contadores siguieron en 0.

**Por eso el disparador de time-over es el RELOJ, no los contadores.** El reloj
de ronda vive en `0xFF972A` (un byte BCD, `0x99` → `0x00`) y:

* lee 0 durante **91–131 agent steps** (364–524 frames) en cada time over, así
  que ninguna cadencia de muestreo lo puede perder;
* llega **~10 agent steps antes** que el contador del ganador;
* está presente en el **DRAW GAME**, donde ningún contador se mueve;
* con el reloj en 0, el resultado se decide comparando las dos palabras de HP
  vivas — los dos luchadores topan en 176, así que "más vida" y "mayor
  porcentaje" son la misma comparación — y HP igual **es** un empate.

Dos guardas, ambas medidas: el reloj también lee 0 en las pantallas de fin de
match / continue (23 de 23 ventanas así en una corrida de 40,000 steps, todas
con el HP en blanco `[0,0]`), por lo que un time-over sólo se declara (a) en un
frame con HP **legible** y (b) después de haber visto el reloj **corriendo** en
este episodio.

El KO gana cualquier desacuerdo: el HP es exacto en su frame, los contadores
llegan 1 frame tarde y el muestreo de 4 frames cae sobre el frame de muerte
~1 de cada 4 veces (medido 13 de 40). Los contadores quedan como **fallback**
para un transporte que tenga contadores pero no reloj, y como pista de auditoría
en `info`.

**Un resultado de ronda se cobra UNA sola vez** (`envs/reward.RoundTracker`).
Un resultado no es un evento, es un **estado** que se mantiene cientos de frames
(33–457 en un KO, 364–524 con el reloj en 0). En un env con `trainable=False`
(eval de modelos, AI-vs-AI, cualquier harness que quiera un stream continuo)
`terminated` se fuerza a `False`, así que **nada consumía el resultado**: el
payoff terminal se cobraba en **cada** step de la ventana. Medido en el core en
vivo antes de este latch: **1,773 cobros terminales en 2,500 steps**, retorno
del episodio **−22,290**; y por la vía de los contadores, cuyo delta nunca
vuelve a cero, **un solo time over cobraba terminal en todos los steps
restantes de la corrida, para siempre**. `RoundTracker` cobra una vez y se
re-arma cuando el resultado se limpia. En `trainable=True` es invisible (el
episodio termina en el frame que dispara). De regalo tapa el hueco del
savestate capturado a mitad de un KO: `reset(ko=True)` arranca enganchado, así
que el resultado viejo se traga en vez de terminar el episodio en el step 1.
(El código anterior "limpiaba" `p1_ko`/`p2_ko` en `reset()` y decía que eso
protegía el step 1. No protegía nada: `step()` los vuelve a derivar de la misma
palabra de HP todavía negativa en el siguiente payload.)

**Empate con payoff propio y explícito** (requisito 1). Antes eran dos `if`
independientes: `+65 − 50 = +15`, **neto positivo**. Ahora son tres ramas
mutuamente excluyentes y `RewardConfig.draw_penalty = 50.0`.

**Frame "en blanco".** Con ambos HP exactamente en 0 y nadie muerto, el env
ahora **no** calcula reward: diffear eso contra el último HP real inventaba
~+73 de daño de una pantalla en blanco.

**Integración retro** (`retro_integration/.../data.json`): `p1_hp`/`p2_hp`
`">u2"` → **`">i2"`** (la integración que trae stable-retro siempre dijo `>i2`;
la nuestra tenía el typo). Se agregaron `matches_won` (0xFF81DA),
`enemy_matches_won` (0xFF845A) y `round_timer` (**0xFF972A**), los tres `|u1`.

> Cómo se ubicó `round_timer`: barrido de las 65,536 posiciones de la work RAM
> buscando una columna estrictamente decreciente durante una ronda. Único
> candidato: arranca en `0x99`, baja hasta `0x00` con 99 decrementos, se
> recarga a `0x99` al empezar la ronda siguiente y se **congela** (no llega a 0)
> cuando la ronda termina por KO.
>
> **Ojo con la paridad de byte:** `env.get_ram()` de stable-retro entrega la RAM
> de Genesis con los bytes **intercambiados** respecto del espacio de
> direcciones del 68000 que usan `data.json` y `mainmemory` de BizHawk —
> `get_ram()[off] == data.json[off XOR 1]`. Verificado con `data.set_value()`:
> escribir `round_timer` cambia `get_ram()[0x972B]`, escribir `matches_won`
> cambia `get_ram()[0x81DB]`. Que `data.json` y el Lua comparten espacio de
> direcciones está confirmado por las lecturas de 1 byte que ya existen en los
> dos lados (`p1_char` = `0xFF81DB` en `data.json`, `read_u8(0x81DB)` en el Lua;
> igual `p2_char`/`0x845B` y `p1_btn`/`0x81E2`).

**Contrato de info**: `win` / `loss` / `double_ko` / `timeout` siguen igual.
Nuevas: `draw` (nombre actual; `double_ko` queda como alias para no romper
dashboards ni comparaciones con el Run A), `time_over`, `matches_won_delta`,
`enemy_matches_won_delta` y `round_timer`. **Los tres backends emiten las mismas
llaves** — base_env (rig), retro_env y league_env — y hay un test que las compara
llave por llave. La primera versión de este fix sólo las emitía en `retro_env`,
es decir el desglose por causa existía únicamente en el backend con el que nadie
entrena. `round_timer` es la única llave cuyo **valor** depende del transporte:
`None` en un payload que no puede cargar el reloj (todos los anchos que el rig
manda hoy), un byte BCD en retro.

**`league_env` (self-play) YA está portado.** Usa el mismo `RoundTracker`, el
mismo `resolve_round_result` y el mismo predicado `unreadable`, y hay un
`FakeLeagueEnv` en los fakes para que se pueda manejar sin socket — que es la
razón por la que se quedó atrás dos veces seguidas. **Detalle que invierte
resultados si se copia mal:** `_PerspectiveParser` pasa UN payload por
`_parse_payload` dos veces (P1 y luego P2), así que todos los atributos con
perspectiva (`my_ko`, `enemy_ko`, `matches_won`, ...) terminan el step con la
vista de **P2**, mientras que el reward se calcula con la observación de P1.
league **debe** resolver con los crudos `p1_ko`/`p2_ko` y `p1_matches_won`/
`p2_matches_won`. La receta que traía este runbook (`resolve_round_result(
self.my_ko, self.enemy_ko)`) habría invertido cada victoria y cada derrota de la
liga.

**Métricas**: `episodes/loss_rate`, `episodes/draw_rate` y
`episodes/time_over_rate` nuevas. Esta última corta los **mismos** episodios por
CAUSA (reloj vs KO) en vez de por desenlace, así que no toca la partición; su
utilidad es que un transporte que no ve el reloj la deja clavada en 0.0 mientras
los time overs se acumulan como timeouts. Las cuatro
tasas **particionan** los episodios: si no suman 1.0, hay un desenlace mal
clasificado. En el Run A `loss_rate` ni siquiera se logueaba — había que
reconstruirla como `1 − win − dko − timeout`, asumiendo la partición que
justamente estaba rota.

### 6.3 Tabla de payoffs

| Desenlace | condición en el frame terminal | `terminal` | antes |
|---|---|---|---|
| Victoria | `enemy_hp < 0` (o la ROM me adjudicó la ronda) | **+65.0** | +65 (casi nunca alcanzado) |
| Derrota | `my_hp < 0` (o la ROM se la adjudicó al rival) | **−50.0** | −50 (casi nunca alcanzado) |
| **Empate** | ambos muertos en el mismo frame | **−50.0** | **+15.0** ← el bug |
| Timeout (1500 steps) | `truncated` | 0.0 | 0.0 |
| Frame ilegible / en blanco | sentinel o `0/0`, sin resultado | 0.0, sin terminar | igual |

Comparaciones que importan: empate vs derrota **0.0** (antes +65). Punto de
indiferencia para forzar un empate: **no existe** — `p·65 + (1−p)·(−50) ≥ −50`
para todo `p`. Antes era `p < 0.565`, y la cuota real de victorias del Run A era
**0.566**: el canal de resultado aportaba gradiente **cero**.

### 6.4 Validación en vivo (40 episodios, política random, esta Mac)

| métrica | código viejo | **con el fix** |
|---|---|---|
| `double_ko_rate` / `draw_rate` | 0.5500 | **0.0000** |
| `timeout_rate` | 0.2800 | **0.0000** |
| `win_rate` | 0.0200 | **0.1750** |
| `loss_rate` | (no se logueaba) | 0.8250 |
| `len_mean` | 1141 | **616** (≈1 ronda, no ~1.9) |
| `hp_sentinel_frac` | 0.1474 | **0.0016** |
| etiquetas correctas vs. contadores de RAM | 0 de 33 decididos | **40 / 40** |
| reward medio \| gané la ronda | **−12.47** | **+124.87** |
| reward medio \| perdí la ronda | **+55.42** | **−87.39** |

El signo del gradiente de resultado estaba **invertido**: el agente cobraba
+55.4 por perder y −12.5 por ganar.

**Re-validación tras el hardening (2026-08-26, 80 episodios random, esta Mac):**

| métrica | valor |
|---|---|
| partición win/loss/draw/timeout | **exacta en 80/80** |
| desenlaces | win 22 / loss 58 / draw 0 / timeout 0 |
| causa del terminal | KO 77, **reloj 3** ← time overs bajo política random, ya atribuidos |
| `len_mean` | 584 (≈1 ronda) |
| retorno medio \| gané | **+162.5** (peor victoria +104.4) |
| retorno medio \| perdí | **−100.4** (mejor derrota −16.3) |
| solapamiento win/loss | **ninguno** |
| env `trainable=False`, 3,000 steps | **3** cobros terminales (antes 1,773 en 2,500) |
| turtling puro, 3 episodios | 3/3 `loss` por **TIME OVER** atribuido (antes: timeout que valía 0) |
| HP empatado + reloj a 0 | **`draw`, −50** (antes: no terminaba, ronda extra y timeout 0) |

Suite: **413 passed** (`test_env_reward.py` sola: **82**). Cada test de
regresión nuevo se verificó re-introduciendo el defecto que persigue: el ancho
`== 24` tira 3, quitar el latch tira 4, y devolver a league los flags con
perspectiva tira 2.

### 6.5 PENDIENTE — el rig BizHawk necesita 3 lecturas de Lua

**La detección de KO ya funciona en el rig sin tocar el Lua**: `hp_to_signed()`
decodifica el 65535 del lado de Python. (Esto corrige el punto 6 del reporte
empírico, que pedía cambiar `read_u16_be` → `read_s16_be`: **no hace falta**, y
cambiarlo sin coordinar rompería a cualquier consumidor que espere u16.)

**Lo que SÍ falta es el TIME OVER.** El lado de Python **ya está listo y
testeado**: `_parse_payload` acepta 13 / 24 / 26 / **27** campos y cada ancho
tiene test. Mientras el Lua no cambie, en el rig las tres señales de time-over
están ausentes, la regla degrada a **sólo KO**, y una ronda que se decide en el
reloj sigue truncando como TIMEOUT que vale 0. `info["time_over"]` clavado en
`False` y `info["round_timer"]` en `None` **son** ese diagnóstico.

Cambio exacto en `lua/v2.0/training_env_client.lua` — **3 lecturas**, no 2: el
reloj es la señal primaria (llega antes que los contadores y es la única que ve
el DRAW GAME), y los contadores quedan como fallback y auditoría.

```lua
-- (1) junto a las demás lecturas de RAM, cerca de la línea 128:
local matches_won       = mainmemory.read_u8(0x81DA)
local enemy_matches_won = mainmemory.read_u8(0x845A)
local round_timer       = mainmemory.read_u8(0x972A)

-- (2) en el string.format del payload: 24 -> 27 campos.
--     Agregar TRES "%d" al final del formato, antes del \n, y los tres
--     nombres al final de la lista de argumentos, EN ESTE ORDEN:
"0 %d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n",
    ... rel_y_dist, p1_chest, p1_head, p2_chest, p2_head,
    matches_won, enemy_matches_won, round_timer)
```

Direcciones: BizHawk direcciona la RAM de Genesis con offset `0xFF0000`, así que
`0xFF81DA` → `0x81DA` y `0xFF972A` → `0x972A` (igual que `0xFF8042` → `0x8042`
en las líneas 74-75). Es **aditivo**: los campos 1-24 no se mueven y v1/v2/v3
siguen parseando igual. Los campos 25-26 son los que ya estaban documentados,
así que un cliente que sólo aplique la mitad del cambio (26 campos) también
funciona: se pierde el reloj, no el resto.

**Auto-verificación de 30 segundos antes de dar el cambio por bueno** (es la
única lectura nueva cuya paridad de byte no se pudo comprobar desde esta Mac):
al empezar una ronda, `read_u8(0x972A)` debe valer **`0x99` (153)** y bajar
`0x98, 0x97, ...` en BCD hasta `0x00`. Si sale basura o un valor que no cambia,
probar `0x972B`; si es ése, avisar — significa que `mainmemory` del rig está
byte-swapped respecto de `data.json`, y **todas** las lecturas de 1 byte que ya
existen (`p1_char` 0x81DB, `p2_char` 0x845B, `p1_btn` 0x81E2) estarían leyendo
el byte de al lado.

**Hueco conocido que esto NO cierra:** un TIME OVER con los dos luchadores vivos
en exactamente 0 HP se confunde con el `[0,0]` que la ROM pinta entre rondas, así
que no se declara. Es una **pérdida**, nunca una mala clasificación (el episodio
trunca como TIMEOUT), y requiere que ambas barras estén a 0 exacto en el
timbrazo. No se blindó porque cualquier arreglo pasaría por leer un cuarto valor
de RAM ("ronda en curso") sin evidencia de que el caso ocurra.

**Otros pendientes fuera del alcance de este fix:**

- ~~**`src/envs/league_env.py`** sigue con la lógica vieja.~~ **HECHO** el
  2026-08-26 — ver 6.2. La receta que estaba escrita aquí
  (`resolve_round_result(self.my_ko, self.enemy_ko)`) era **incorrecta** y habría
  invertido cada resultado de la liga; van los crudos `p1_ko`/`p2_ko`.
- **`src/agents/auto_curriculum_callback.py:30`** — `win_rate_threshold = 0.75`.
  Con la clasificación rota el techo real era ~2% y el currículum nunca salió
  del nivel 1/8 (0 de 31 ventanas del Run A llegaron a 0.75). Ahora que
  `win_rate` mide algo, hay que **re-evaluar el umbral con métricas ya
  corregidas** antes del próximo run largo. Ojo con el segundo techo: los
  timeouts también aportan 0 al buffer.
- **`compute_reward` sigue aplicando `gamma*Phi(s') - Phi(s)` en la transición
  terminal**, donde Ng/Harada/Russell exige `Phi(absorbente) = 0`. Es un bono
  real no-invariante de hasta +2.49 en el último step, en la misma dirección que
  el +15 que se acaba de quitar. No se tocó aquí porque cambia el shaping, no la
  semántica de rondas.
