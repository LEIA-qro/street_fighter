# D1-inventario — Inventario funcional exhaustivo del dashboard

Fuente: `/Users/felipe/TEC/LEIA/street_fighter/src/scripts/web_dashboard.py` (2,326 líneas),
HEAD `f65a2932`, rama `stage0-metrics-and-semantics`, build `v1592-unified-r8`.
Leído completo el 2026-08-27. Gradio instalado: **6.25.0** (`.venv/bin/python`).

Método: lectura íntegra del fuente + conteo por tipo de componente + verificación de
huérfanos con grep repo-wide (dos llamadas separadas) + arqueología de `git show` sobre
los commits del rush del stand.

---

## 0. Corrección a la evidencia de campo (nota técnica que pedía verificación)

La evidencia de campo sospechaba que `theme=gr.themes.Soft(primary_hue="blue")` pasado a
`demo.queue().launch(...)` podría estar siendo ignorado. **No lo es.** En gradio 6.25.0:

- `theme` **SÍ** es parámetro de `Blocks.launch()` (verificado por `inspect.signature` y por
  el docstring: *"A Theme object or a string representing a theme…"*).
- `theme` **NO** es parámetro de `gr.Blocks.__init__` en esta versión.

Es decir: el código está bien y **el tema Soft/blue sí se aplica**. La píldora azul de cada
etiqueta viene de ese tema, no del default. Consecuencia para la reconstrucción: cambiar la
paleta se hace en ese único punto (`launch(theme=…)`) más los ~40 hexes literales incrustados
en las f-strings HTML de las 3 tarjetas (`get_league_pool_status_html`,
`get_auto_curriculum_status_html`, `get_live_telemetry_html`).

---

## 1. Conteo global de componentes (entrega principal, parte A)

Conteo literal por `grep -c "gr.<Tipo>("`, cuadrado uno a uno con la lectura del fuente:

| Tipo | N | Nota |
|---|---|---|
| `gr.Dropdown` | **35** | todos verificados por nombre de variable |
| `gr.Button` | **23** | |
| `gr.Number` | **13** | |
| `gr.File` | **13** | 11 de subida + 2 de descarga (`download_curr_file`, `download_json`) |
| `gr.Textbox` | **10** | 4 de entrada, 3 consolas read-only, 1 resultado read-only, 1 anónimo oculto, 1 nombre de estudio |
| `gr.Checkbox` | **8** | |
| `gr.Slider` | **7** | |
| `gr.Markdown` | 24 | títulos y líneas de estado |
| `gr.HTML` | 3 | las 3 tarjetas generadas por f-string |
| `gr.JSON` | 1 | `readonly_params`, sólo lectura |
| `gr.Timer` | 2 | 0.1 s (telemetría) y 5 s (curriculum) — **ambos siempre activos** |
| `gr.Tab` | 10 | 5 de primer nivel + 5 anidadas |

**Controles interactivos que el usuario puede tocar: 35 + 23 + 13 + 11 + 4 + 8 + 7 = 101.**
Repartidos en 5 pestañas → ~20 controles por pantalla, contra 6 acciones reales de lanzamiento.

---

## 2. Agrupación por PREGUNTA que le hace al usuario (entrega principal, parte B)

Esta es la métrica que importa: cuántas veces la app pregunta lo mismo.

### 2.1 "¿QUÉ MODELO?" — **24 controles**
La app pregunta por un modelo veinticuatro veces, en cinco vocabularios distintos
(`.zip` SB3 / `.pkl` de normalización / `.pt` QR-DQN / nombre de salida / subida de archivo).

| Sub-pregunta | N | Controles |
|---|---|---|
| Pesos SB3 `.zip` (dropdown) | 5 | `train_zip_drop`, `tune_zip_drop`, `pbt_zip_drop`, `p1_zip`, `p2_zip` |
| Normalización `.pkl` (dropdown) | 5 | `train_pkl_drop`, `tune_pkl_drop`, `pbt_pkl_drop`, `p1_pkl`, `p2_pkl` |
| Checkpoint Ape-X `.pt` (dropdown) | 2 | `p1_apex_checkpoint`, `p2_apex_checkpoint` |
| Subida de modelo/normalización/JSON | 8 | `ext_zip_upload`, `ext_pkl_upload`, `ext_json_upload`, `upload_json`, `p1_zip_upload`, `p1_pkl_upload`, `p2_zip_upload`, `p2_pkl_upload` |
| Nombre del modelo a producir/atacar | 4 | `model_name_input`, `pbt_model_name_input`, `league_model_name`, `exploiter_model_name` |

Los 10 dropdowns `.zip`/`.pkl` están **acoplados posicionalmente**: `refresh_dropdowns()`
devuelve exactamente 10 `gr.update()` en orden fijo y se re-usa en 5 bindings distintos
(`refresh_files_btn` y los `.then()` de los 3 lanzadores de entrenamiento). Cualquier
reordenamiento rompe silenciosamente el mapeo.

### 2.2 "¿CONTRA QUIÉN?" — **17 controles**

| Contexto | N | Controles |
|---|---|---|
| Matchups (rival P2) | 10 | `p2_algo`, `p2_env`, `p2_device`, `p2_zip`, `p2_pkl`, `p2_apex_checkpoint`, `p2_zip_upload`, `p2_pkl_upload`, `matchup_character`, `cpu_level_cap_slider` |
| League (pool de rivales) | 3 | `league_matchup_mode`, `league_custom_state`, `league_state_upload` |
| Exploiter (rival a explotar) | 4 | `exploiter_matchup_mode`, `exploiter_custom_state`, `exploiter_state_upload`, `exploiter_type` |

`league_*` y `exploiter_*` son **la misma tríada duplicada literalmente**: dos funciones
gemelas idénticas salvo el nombre (`toggle_league_matchup_mode` / `toggle_exploiter_matchup_mode`,
líneas 1365-1371) y el **mismo** handler de subida (`handle_league_state_upload`) para ambas.

### 2.3 "¿QUÉ NIVEL / DIFICULTAD?" — **6 controles + 3 de savestates**

| Control | Pestaña | Semántica |
|---|---|---|
| `train_phase_drop` | Production | **bimodal**: "Start Phase (Manual)" `[0,1,2,3,RYU_ONLY,CUSTOM]` ⇄ "Start Level (Auto)" `[1..8]` según `auto_curr_check` |
| `tune_phase_drop` | Optuna | "Start Phase (States)" `[0,1,2,3,RYU_ONLY,CUSTOM]` |
| `pbt_phase_drop` | PBT | idéntico al anterior |
| `cpu_level_cap_slider` | Matchups | **bimodal**: "CPU Level (exact, 1-8)" ⇄ "CPU Max Level Cap (Infinite Match)" |
| `auto_curr_check` | Production | activa la escalera de 8 niveles |
| `cfg_win_rate` | Config | `WIN_RATE_THRESHOLD` — el umbral que hace avanzar de nivel |
| + `league_custom_state`, `exploiter_custom_state`, `state_upload` | | la dificultad concreta vía savestate |

Tres vocabularios incompatibles para lo mismo: **fase** (0-3 + 2 strings), **nivel** (1-8),
**savestate** (nombre de archivo `.State`). Ninguno explicado en la UI.

### 2.4 "¿EN QUÉ DISPOSITIVO?" — **6 controles**
`train_device`, `tune_device`, `league_device`, `exploiter_device`, `p1_device`, `p2_device`.
Todos con las mismas choices `["auto","cpu","cuda"]` y valor `"auto"`.

**Asimetría real:** PBT es el único lanzador SIN selector de dispositivo — `run_pbt()`
(línea 1048) nunca pasa `--device`. 6 lanzadores, 6 dropdowns de device, pero el reparto es
5+1: uno de ellos es de un flujo (matchups) que ya tiene dos.

### 2.5 "¿QUÉ ENVIRONMENT?" — **5 controles**, con choices INCONSISTENTES
`env_sel` = `["v1","v2","v3"]` · `p1_env`, `p2_env`, `league_env`, `exploiter_env` = `["v2","v3"]`.
`v1` sólo existe en la pestaña de entrenamiento.

### 2.6 "¿QUÉ ALGORITMO?" — **3 controles**
`algo_sel` (`ppo/sac/dqn`), `p1_algo` (+`apex`, `Human Player`), `p2_algo` (+`apex`,
`Human Player`, `CPU (Built-in AI)`). **P1 no ofrece `CPU (Built-in AI)` y P2 sí** — asimetría
que produce código muerto (ver §5.2).

### 2.7 "¿CUÁNTOS PASOS?" — **8 controles numéricos**
`train_steps` (1e6), `tune_steps` (5e4), `pbt_steps` (5e6), `pbt_exploit_steps` (5e5),
`league_steps` (5e6), `exploiter_steps` (1e6), `cfg_steps` (default global), `trials_input`
(nº de trials Optuna). Ninguno con formato de miles ni unidad.

### 2.8 Verbos repetidos
| Verbo | N | Controles |
|---|---|---|
| **Detener** | 5 | `graceful_stop_btn`, `force_kill_btn`, `stop_league_btn`, `kill_league_btn`, `stop_match_btn` → sólo 2 funciones (`graceful_stop_process`, `force_kill_process`) |
| **Refrescar** | 4 | `refresh_files_btn`, `refresh_curr_btn`, `refresh_league_btn`, `refresh_apex_btn` |
| **Copiar logs** | 3 | `copy_btn`, `copy_match_btn`, `copy_league_logs_btn` — los 3 con el mismo JS inline (`navigator.clipboard.writeText` + `alert()`) |
| **Consolas** | 3 | `unified_logs`, `league_logs`, `match_logs` — todas alimentadas por el mismo `stream_logs` |
| **Lanzar** | 6 | `start_train_btn`, `start_tune_btn`, `start_pbt_btn`, `start_league_btn`, `start_exploiter_btn`, `launch_match_btn` |

---

## 3. El hecho estructural más importante: UN SOLO PROCESO GLOBAL

`GlobalState.active_process` (línea 31-39) es **un único slot**. `stream_logs()` (424) toma
`state.process_lock`, y si ya hay algo corriendo emite literalmente
`"Error: A process is already running!"` y termina.

Consecuencias que la reconstrucción DEBE resolver:
- Hay **6 botones de lanzamiento** repartidos en 3 pestañas, pero sólo **1 puede correr**.
- No existe ningún indicador global de "ocupado": ni deshabilitación de los otros 5 botones,
  ni badge, ni banner. El único feedback es una línea de texto dentro de la consola de la
  pestaña donde presionaste.
- Los **5 botones de Stop son globales pero se ven locales**: `stop_league_btn` llama a
  `graceful_stop_process()`, que mata *lo que sea* que esté corriendo — incluido un match
  del stand. Nada en la etiqueta lo dice.
- Las **3 consolas son independientes**: un entrenamiento lanzado en Training no aparece en
  la consola de League ni en la de Matchups, aunque sea el mismo (y único) proceso.

---

## 4. Inventario control por control

Leyenda de efecto: **[SUB]** lanza subproceso · **[FS]** escribe en disco ·
**[CFG]** reescribe `src/core/config.py` · **[RD]** sólo lee estado · **[UI]** sólo UI ·
**[JS]** JavaScript en el cliente. ★ = tarea principal de su pestaña.

### 4.1 Pestaña 1 — 🏋️‍♂️ Training & Tuning (3 sub-pestañas)

**Cabecera global (afecta a las 3 sub-pestañas):**
| Control | Etiqueta | Función | Efecto |
|---|---|---|---|
| `algo_sel` | Algorithm | `update_ui_on_algo` → `.then(get_auto_curriculum_status_html)` | [UI] repuebla 4 dropdowns + reescribe `study_name_input` y `model_name_input` |
| `env_sel` | Environment | `get_auto_curriculum_status_html` | [RD] |
| `tb_main_btn` | 📈 Launch TensorBoard | `launch_tb` | [SUB] `Popen(shell=True)` + `webbrowser.open` **en la máquina servidor** |

**Sub-pestaña A — 🚀 Production Training**
| Control | Etiqueta | Función | Efecto |
|---|---|---|---|
| `model_name_input` | New Model Name | (input de `run_training`) | [CFG] `update_config_var("MODEL_NAME", …)` al lanzar |
| `train_zip_drop` / `train_pkl_drop` | Base Model (.zip) / Base Norm (.pkl) | input | [UI] |
| `ext_zip_upload` / `ext_pkl_upload` / `ext_json_upload` | Upload Model / Normalization / Curriculum State | `handle_model_upload` | [FS] copia a `models/production/{env}/{algo}/` |
| `auto_curr_check` | Enable Auto-Curriculum (Progressive 8-Level) | `toggle_auto_curriculum_ui` | [UI] muta etiqueta+choices de `train_phase_drop` |
| `train_phase_drop` | Start Phase (Manual) / Start Level (Auto) | input | [UI] |
| `train_steps`, `train_device` | Total Timesteps, Compute Device | input | [UI] |
| `train_lr`/`train_ent`/`train_clip` | *Override* (acordeón cerrado) | input | [UI] sólo aplican si > 0.0 |
| `upload_json` | Upload Hyperparameters JSON | `load_hyperparams_from_json` | [FS/RD] rellena los 3 overrides + `readonly_params` |
| **`start_train_btn`** ★ | ▶ Start Training | `run_training` → `.then(refresh_dropdowns)` | **[SUB]+[CFG]** `train.py` |
| `refresh_curr_btn` | 🔄 Refresh Auto-Curriculum Stats | `get_auto_curriculum_status_html` | [RD] — **redundante**: un `gr.Timer(5)` ya lo hace solo |
| `download_curr_btn` | 📥 Download Auto-Curriculum Analytics | `get_auto_curriculum_file` | [RD] revela `download_curr_file` (nace `visible=False`) |
| `auto_curr_card` | (tarjeta HTML) | `get_auto_curriculum_status_html` | [RD] |

**Sub-pestaña B — 🧪 Optuna Tuning**
`study_name_input`, `tune_zip_drop`, `tune_pkl_drop`, `tune_phase_drop`, `tune_steps`,
`tune_device`, `trials_input` → **`start_tune_btn`** ★ (`run_tuning` → `tune.py`) **[SUB]**;
`get_results_btn` (`get_best_tuning_params`) **[SUB]** lanza un `python -c` con un script
formateado por f-string que **interpola `study_name` y rutas dentro de comillas simples de
Python** (línea 712-721) → cualquier apóstrofo en el nombre del estudio rompe o inyecta;
salidas `best_params_output` + `download_json`.

**Sub-pestaña C — 🧬 PBT Training**
`pbt_model_name_input`, `pbt_zip_drop`, `pbt_pkl_drop`, `pbt_phase_drop`, `pbt_steps`,
`pbt_exploit_steps`, `pbt_pop` (4-16), `pbt_concurrent` (1-16), `pbt_envs` (1-8),
`pbt_resume` → **`start_pbt_btn`** ★ (`run_pbt` → `train_pbt.py`) **[SUB]**. Sin device.

**Pie común de la columna izquierda (fuera de las sub-pestañas):**
`graceful_stop_btn` **[SUB/FS]** (escribe `.stop_training`, espera 30 s, `CTRL_BREAK`, luego
`taskkill`), `force_kill_btn` **[SUB]** (`taskkill /F /T`), `refresh_files_btn` **[RD]**,
`stop_status`.

**Columna derecha:** `unified_logs` (35 líneas) + `copy_btn` **[JS]**.

### 4.2 Pestaña 2 — 🏆 Auto-Learning League (2 sub-pestañas)

**A — 🎯 Self-Play League Training:** `league_model_name`, `league_steps`, `league_env`,
`league_device`, `league_matchup_mode` (`toggle_league_matchup_mode` **[UI]** revela 3
controles), `league_custom_state`, `league_resume`, `league_state_upload`
(`handle_league_state_upload` **[FS]+[CFG]** copia a `STATES_DIR` y reescribe
`CUSTOM_STATES` en `config.py` + `importlib.reload`) → **`start_league_btn`** ★
(`run_league` → `train_league.py`) **[SUB]**.

**B — ⚔️ Specialized Exploiter Training:** `exploiter_model_name`, `exploiter_type`
(rusher/spammer/turtle), `exploiter_steps`, `exploiter_env`, `exploiter_device`,
`exploiter_matchup_mode`, `exploiter_custom_state`, `exploiter_state_upload` →
**`start_exploiter_btn`** ★ (`run_exploiter` → `train_exploiter.py`) **[SUB]**.

**Pie:** `refresh_league_btn` **[RD]** (`importlib.reload(config)` + rescan),
`stop_league_btn`, `kill_league_btn`, `copy_league_logs_btn` **[JS]**, `league_logs`.
**Derecha:** `league_analytics_card` **[RD]** — **estático, sin timer**: sólo se refresca
al pulsar Refresh o al terminar un lanzamiento.

### 4.3 Pestaña 3 — 🎮 Model Testing & Matchups

Estructura P1 (7 controles + 2 zonas de carga) **espejada exactamente** en P2 (8 + 2):

| P1 | P2 | Efecto |
|---|---|---|
| `p1_algo` (ppo/sac/dqn/apex/Human) | `p2_algo` (+ `CPU (Built-in AI)`) | [UI] dispara 3 handlers cada uno |
| `p1_env`, `p1_device` | `p2_env`, `p2_device` | [UI] |
| `p1_zip`, `p1_pkl` | `p2_zip`, `p2_pkl` | [UI] |
| `p1_apex_checkpoint` | `p2_apex_checkpoint` | [UI] ocultos salvo `algo == "apex"` |
| `p1_zip_upload`, `p1_pkl_upload` | `p2_zip_upload`, `p2_pkl_upload` | [FS] `handle_model_upload` |

Comunes: `matchup_character` (oculto salvo Ape-X; se fuerza a `["RYU"]` si el rival es un
modelo), `refresh_apex_btn` **[RD]**, **`launch_match_btn`** ★ (`run_matchup` **[SUB]** →
`test_ai_vs_ai_v2.py` | `test_agent_v2.py` | `stand_leia.py` según 3 ramas),
`stop_match_btn` **[SUB/FS]**, `match_profile_checkbox`, `infinite_match_checkbox`
(`update_infinite_match_status` **[UI]** — el texto dice PLAYING/PAUSED **sin escribir
`.agent_state`**; el archivo sólo se escribe en `before_start` al lanzar → el indicador
miente antes del lanzamiento), `rematch_delay_slider`, `cpu_level_cap_slider`,
`toggle_agent_btn` **[FS]** (escribe `.agent_state`), `agent_state_status`,
`match_upload_status`, `match_logs` + `copy_match_btn` **[JS]**.

**Nota de densidad (confirma la evidencia de campo):** 22 controles + 4 zonas de carga
antes del único botón que importa.

### 4.4 Pestaña 4 — 🔮 Observation Telemetry
**Cero controles.** Un `gr.HTML` (`telemetry_html`) + `gr.Timer(0.1, active=True)` que llama
a `get_live_telemetry_html` **10 veces por segundo, siempre**, esté o no abierta la pestaña,
exista o no `.telemetry.json`. La función es la más larga del archivo (1438-1703) y
reconstruye una cadena HTML de ~20 KB en cada tick. Su estado vacío es "🔮 Standby Mode:
Telemetry Offline". La pestaña sólo tiene contenido durante un match test activo.

### 4.5 Pestaña 5 — ⚙️ Core Config Editor
| Control | Etiqueta (literal) | Efecto |
|---|---|---|
| `cfg_n_envs` | `N_ENVS (Parallel Instances)` | |
| `cfg_win_rate` | `WIN_RATE_THRESHOLD (Phase Advance)` | |
| `cfg_steps` | Default Training Steps | |
| `cfg_port` | Base Socket Port | |
| `cfg_input_display` / `cfg_activate_visualization` / `cfg_enable_throttling` | 3 checkboxes | |
| `cfg_throttle_speed` | `Training Throttle Speed % (…)` | |
| **`save_cfg_btn`** ★ | 💾 Save Configuration | **[CFG]** `save_all_config` reescribe 8 variables en `src/core/config.py` por regex + `importlib.reload` |
| `state_upload` | Upload Custom Savestates (.State), múltiple | **[FS]+[CFG]** `handle_state_upload` |

Las etiquetas son **nombres de variable de Python**. Y el botón Guardar **modifica código
fuente del proyecto** — cualquier reconstrucción tiene que decidir si eso sigue siendo así.

---

## 5. Hallazgos de inventario (cosas que existen pero no funcionan / no se ven)

### 5.1 ★ La tarjeta de calidad del checkpoint Ape-X está HUÉRFANA
`refresh_stand_checkpoints()` (línea 398) aparece **exactamente una vez en todo el repo: su
propia definición**. Verificado dos veces en llamadas separadas (grep en `src/` y grep
repo-wide excluyendo `.git`/`.venv`). Nunca se enlaza a ningún componente.
`get_stand_checkpoint_status()` (349) sólo se alcanza desde esa función muerta y desde
`code_testing/pytest/test_telemetry_dashboard.py:436`.

Eso significa que **el único bloque de UI que le diría a un visitante del stand qué tan bueno
es el modelo** — win-rate medio, WR por nivel de la escalera L1..L8, versión de pesos,
arquitectura QR-DQN — **está escrito, probado por pytest, y no se renderiza en ninguna parte.**

### 5.2 Arqueología: 5a852ae2 destruyó el panel del stand
`git show 5a852ae2 -- src/scripts/web_dashboard.py` confirma que ANTES de ese commit existía
un `gr.Accordion("Ape-X QR-DQN vs Human (Viewer)")` con su propio flujo compacto:
`apex_checkpoint`, `apex_device`, `apex_human_character`, **`apex_checkpoint_status`**
(la tarjeta de §5.1, enlazada a `apex_checkpoint.change` y a `refresh_apex_btn`),
`apex_rematch_delay`, `launch_apex_btn` ("🥊 Launch Ape-X vs Human"), `stop_apex_btn`,
`apex_logs`. **8 controles, un solo botón, cero duplicación P1/P2.**

El commit "unificó" ese panel dentro del formulario genérico P1/P2. Resultado neto:
- la duplicación P1/P2 que la evidencia de campo reporta como el caso grave;
- la tarjeta de estado del checkpoint quedó huérfana;
- se perdió el lanzador de un clic para el caso de uso del stand promocional.

**Para la reconstrucción: el flujo del stand tenía una forma buena y se puede recuperar del
git; no hay que inventarla.**

### 5.3 Código muerto por asimetría de choices
`p1_algo` no ofrece `"CPU (Built-in AI)"` (sólo `p2_algo` lo tiene), pero `run_matchup`
contiene dos ramas que lo comprueban:
- línea 829: `opp_type = "cpu" if p1_algo == "CPU (Built-in AI)" else "human"` → siempre `"human"`.
- línea 841: la condición `p1_algo == "CPU (Built-in AI)"` para `--cpu_level_cap` → nunca cierta.
Efecto real: **no se puede montar "CPU vs modelo con el modelo en P2 contra la CPU"**, y el
cap de nivel de CPU se ignora silenciosamente en esa mitad de los casos.

### 5.4 Componente anónimo tirado a la basura
Línea 2222: `tb_main_btn.click(launch_tb, outputs=[gr.Textbox(visible=False)])`. El mensaje
"TensorBoard launched at http://localhost:6006" se escribe en un componente creado al vuelo,
invisible y sin lugar en el layout. **El usuario no recibe ninguna confirmación** de que
TensorBoard arrancó, ni error si falló.

### 5.5 `handle_model_upload` sin guarda de algoritmo
En Matchups se le pasa `p1_algo`/`p2_algo` como carpeta destino
(`models/production/{env}/{algo}/`). Si el algoritmo seleccionado es `"Human Player"`,
`"CPU (Built-in AI)"` o `"apex"`, crea carpetas con esos nombres (incluida una con espacio).

### 5.6 Portabilidad: los 5 botones de Stop son Windows-only
`graceful_stop_process` usa `signal.CTRL_BREAK_EVENT` y `subprocess.run("taskkill /F /T /PID …",
shell=True)`; `force_kill_process` lo mismo. En macOS/Linux fallan en silencio (envueltos en
`try/except` que sólo imprime). Consistente con el arnés Windows del proyecto — **lo registro
como restricción de plataforma, no como defecto de UI**, pero condiciona cualquier demo del
stand fuera de Windows.

### 5.7 Dos timers permanentes
`gr.Timer(0.1)` (telemetría, 10 Hz) y `gr.Timer(5)` (curriculum) corren siempre, para todos
los clientes conectados, independientemente de la pestaña activa. El botón
`refresh_curr_btn` duplica manualmente lo que el timer de 5 s ya hace.

### 5.8 Duplicación literal de lógica
`toggle_league_matchup_mode` y `toggle_exploiter_matchup_mode` (1365-1371) son idénticas
salvo el nombre. Los 3 botones de copiar logs repiten el mismo JS con `alert()`.
`run_league` y `run_exploiter` repiten el mismo `mode_map` de 3 entradas.

### 5.9 Fuera de alcance (registrado, no accionado)
- `get_best_tuning_params` construye un script Python por f-string e interpola `study_name`
  sin escapar dentro de comillas simples (línea 714) — riesgo de inyección/rotura. Es un tema
  de backend, no de UI.
- `update_config_var` / `update_config_list` reescriben código fuente por regex.

---

## 6. Lo que la reconstrucción DEBE poder hacer (lista mínima de capacidades)

6 acciones de lanzamiento + 3 de control de proceso + 1 de configuración:

1. **Entrenar** un modelo SB3 (ppo/sac/dqn) con curriculum manual o automático de 8 niveles.
2. **Tunear** con Optuna (N trials) y recuperar/descargar los mejores hiperparámetros.
3. **PBT** con población, concurrencia, envs por worker y reanudación.
4. **League self-play** contra el pool histórico, con 3 modos de matchup y savestates propios.
5. **Exploiter** especializado (rusher/spammer/turtle) contra el modelo de la liga.
6. **Matchup / stand**: modelo vs humano, vs CPU (nivel 1-8), vs otro modelo — incluyendo el
   camino Ape-X `.pt` con rematch infinito, que es EL caso del stand LEIA.
7. **Detener** (grácil, con guardado de emergencia) y **matar** (inmediato).
8. **Pausar/reanudar** el agente en vivo (`.agent_state`).
9. **Editar** la configuración del proyecto y subir savestates.
10. **Ver**: consola en vivo, tarjeta de curriculum, tarjeta de pool de liga, telemetría de
    observaciones/activaciones, y — hoy ausente — **la tarjeta de calidad del checkpoint**.
