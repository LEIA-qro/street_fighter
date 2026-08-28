# Reconstrucción del Control Center de LEIA

> Documento final del arquitecto (PLAN-final). Sintetiza el Discover (D1/D2/D3), las siete
> propuestas (O1–O7) y el veredicto del panel (J1 viabilidad, J2 valor de usuario, J3 coherencia).
> Donde una propuesta y el panel se contradicen, **gana el panel por mayoría 2-de-3**; los injertos
> rescatados se incorporan aunque su propuesta madre haya perdido.
> Verificaciones propias de hoy (2026-08-27, HEAD `f65a2932`) citadas en línea.
> Documento completo en `/Users/felipe/TEC/LEIA/street_fighter/.scratch-uiux-dashboard-leia/PLAN-final.md`

---

## 1. Veredicto y decisión de stack

**No se reconstruye el dashboard entero, y decir lo contrario sería deshonesto.** El equipo ya no opera esta aplicación: cinco de seis runbooks vivos lanzan y vigilan por CLI y miden en W&B, y el único que abre `src/scripts/web_dashboard.py` es `tools/RUN_STAND_LEIA.md`, para una sola pestaña y una sola rama (O7, confirmado por J1, J2 y J3). Lo que sí vale la pena construir es lo que hoy no existe y tiene fecha: **el Modo Stand**, más la **poda** de todo lo que no puede funcionar. El stack para eso es **el que ya está instalado**: `gr.Blocks.route` existe en el gradio 6.25.0 del venv (`.venv/lib/python3.12/site-packages/gradio/blocks.py:3752`, verificado hoy por mí y antes por O2 y J1), así que las vistas separadas del stand cuestan una función, no una migración. La migración a FastAPI + React + shadcn (O3/O4) **no se descarta: se aplaza y se condiciona** — es el mejor documento técnico del panel y el peor plan de ejecución para tres personas con un evento encima (J1 3/10, J2 6/10, J3 8/10; la mayoría la aplaza, no la mata). La regla que ordena todo el calendario: **nada del stand puede depender de la migración**, y la migración se reevalúa cuando el stand haya pasado y no haya una run crítica en vuelo. El saldo honesto: ~45% del archivo (1,058 líneas) es lógica de negocio que sobrevive intacta, ~28% es pegamento Gradio que se tira y ~26% es HTML incrustado a reemplazar (D2) — esto no es una reescritura, es sacar a la superficie un backend que ya existe, y hacerlo por partes.

---

## 2. Qué sobrevive, qué se mata, qué nace

### 2.1 SOBREVIVE (se conserva o se resucita)

| Funcionalidad | Estado hoy | Por qué sobrevive |
|---|---|---|
| Lanzar y vigilar un **matchup Ape-X vs humano** (`run_stand`) | vivo, enterrado bajo 22 controles | es el único flujo que un runbook abre (`RUN_STAND_LEIA.md`) |
| **Tarjeta de calidad del checkpoint** (`get_stand_checkpoint_status`, :349–395) | **código muerto**: `refresh_stand_checkpoints` (:398) tiene 0 referencias fuera de sí misma; su único consumidor vivo es `code_testing/pytest/test_telemetry_dashboard.py:436` | ya está escrita, ya está probada, y `RUN_STAND_LEIA.md §3` la recita a mano en prosa porque la UI dejó de mostrarla |
| **Inventario QR-DQN** (`_load_stand_checkpoint_meta`, `get_stand_checkpoint_files`, `get_stand_default_checkpoint`, `_resolve_stand_checkpoint`, :186–348) | vivo y limpio | el bloque mejor escrito del archivo; `_resolve_stand_checkpoint` ya es guard de path-traversal |
| **Supervisor de procesos** (`GlobalState`, `graceful_stop_process`, `force_kill_process`) | vivo, Windows-only | es correcto donde importa; se mueve de sitio, no se reescribe |
| **`toggle_agent_btn`** | vivo | corrección de O7 que el panel ratifica: es el único control que reanuda una sesión pausada (`stand_leia.py:881–905`). Muere el indicador que miente (`update_infinite_match_status`), no el botón |
| **Entrenamiento clásico PPO/SB3 + tarjeta de Auto-Curriculum** | vivo (hay una run de PPO en curso) | `02-decisiones.md`: "BizHawk SE QUEDA para evaluación/PvP/humano". Se conserva **un** lanzador, no cinco |
| **MatchSessionLog** (JSONL con `fsync` por evento) | vivo y sin consumidor de UI | marcador, récord y overlay del tablero público son un **lector** de este archivo, no backend nuevo |
| **`GET /status` del learner** (`tools/apex_learner.py`, `src/agents/apex.py:245–259`) | vivo y **sin un solo consumidor** (0 usos de `urllib`/`requests` en el dashboard) | contrato de datos regalado: host, procs, steps_per_s, transitions, last_seen y `age` ya calculado |

### 2.2 SE MATA (evidencia dura, no gusto)

1. **Pestaña Observation Telemetry**, entera. `stand_leia.py` tiene **0 referencias a `write_telemetry`** (verificado hoy), y `.telemetry.json` sólo lo escriben `test_agent_v2.py` y `test_ai_vs_ai_v2.py`. Para el único flujo vivo no es "vacía a veces": es **imposible**. Con ella muere `gr.Timer(value=0.1, active=True)` (`web_dashboard.py:2133`), que repinta 10 veces por segundo, para todos los clientes, una cadena estática.
2. **Sub-pestaña PBT.** `grep -c ray requirements.txt` → **0** (verificado hoy); `agent/handoff.md:100` dice que ray está deliberadamente excluido. 12 controles para un flujo que no arranca.
3. **Sub-pestaña Optuna Tuning.** Cero `.db` en el árbol; su único resultado documentado está marcado como nulo en `03-bugs-cazados.md:8` y `05-runs.md:6`. Además `get_best_tuning_params` interpola `study_name` sin escapar dentro de código Python generado y lo ejecuta con `python -c`.
4. **Pestaña Auto-Learning League (Self-Play + Exploiter).** `models/production/` contiene sólo `league/`, vacío (verificado hoy). Ausente de todo runbook. Su medidor natural, `core/elo.py`, tiene un único importador: su propio test.
5. **Las 8 zonas de subida de modelos y `handle_model_upload`.** La distribución real es `git pull`. Y la función usa el valor del dropdown de algoritmo como nombre de carpeta sin guarda: `"Human Player"` y `"CPU (Built-in AI)"` son valores válidos y crean directorios basura que `get_model_files()` nunca encontrará.
6. **`readonly_params` (`gr.JSON`)**: visor del archivo que el usuario acaba de subir; sin la subida, no tiene productor.
7. **`_DASHBOARD_RELOAD_HEAD` + `DASHBOARD_BUILD_ID`** (watcher JS con polling cada 3 s y contador manual editado seis veces en dos horas). Andamio de una noche de urgencia; obliga a teclear `?build=v1592-unified-r8` en la URL.
8. **`"sac"` en los cuatro dropdowns.** `src/agents/sac/agent.py:113` y `:334` hacen `raise NotImplementedError` en la primera línea de ambos entrypoints. Una opción que sólo produce traceback, en la pantalla de un stand con público, es peor que un bug.
9. **`"v1"` en `env_sel`.** Se entrena y no se puede probar (los otros cuatro dropdowns ofrecen sólo v2/v3; `run_stand` lanza `ValueError` fuera de ese par).
10. **Redundancias verbales**: 4 de 5 Detener, 2 de 3 Copiar, 2 de 3 consolas, `refresh_curr_btn` (duplica un `gr.Timer(5)`), y `toggle_exploiter_matchup_mode` (idéntica a la de league).
11. **El mecanismo de `update_config_var`**, que reescribe `src/core/config.py` versionado con `re.sub`, sin diff ni deshacer. *(Matiz de J2, aceptado: muere el mecanismo, no la pantalla.)*

### 2.3 NACE

1. **`/stand`** — vista de operador, ruta propia vía `gr.Blocks.route`, sin acceso a Force Kill ni a los lanzadores de entrenamiento.
2. **`/tablero`** — vista pública de sólo lectura, **cero elementos interactivos**. No es una restricción que se pueda saltar: es ausencia de superficie (O2, ratificado por J2).
3. **Barra global de ejecución** — la pieza en la que O1, O4 y O6 convergieron de forma independiente (barra global / `RunProvider` / L0). Representación honesta de que existe **un solo** `GlobalState.active_process`: qué corre, desde cuándo, y **un** Detener con confirmación.
4. **Bloque de flota Ape-X** — 4 filas leyendo `GET /status`. Es el único dato realmente nuevo del panel, y el usuario que lo necesita hoy tiene ceguera total.
5. **Consola única** con tee a disco y `run.json` (injerto de O3, Fase 0).
6. **Confirmación destructiva con cuerpo informativo** — "vas a matar `apex-curriculum-lvl6`, 4h12m, 3.4M pasos, último checkpoint hace 22 min → se pierden ~22 min" (injerto de O4).
7. **Tarjeta de credencial resucitada**, en dos superficies: pre-vuelo del stand y selector de checkpoint del laboratorio.

### 2.4 DUDOSO — decisión de Felipe, no muerto por decreto

- **TensorBoard.** `06-pendientes.md:7` pide "W&B sync para PPO", o sea que **hoy PPO no está en W&B** y TB puede ser su único visor. No se mata. Sí se corrige que hoy hace `Popen(shell=True)` + `webbrowser.open` **en la máquina servidor** (con `--host 0.0.0.0`, abre una ventana en otra computadora, `web_dashboard.py:755–756`) y escribe su salida a un `gr.Textbox(visible=False)` creado dentro del propio `.click()`.
- **La pestaña de entrenamiento clásico como superficie web.** El backend se queda; la pregunta es si *lanzarlo desde aquí* sigue siendo la forma de trabajar. Ver §🌅.
- **El matchup SB3 vs SB3.** Sigue en `06-pendientes.md:8`, pero con `models/production/` vacío hoy no hay con qué armarlo. Recomendación: reducir a un caso, no eliminar.
- **La subida de savestates `.State`.** `06-pendientes.md:4` tiene abierto "rellenar estados lvl5–8".
- **`match_profile_checkbox`** (`--profile`): implementado de verdad. Sale de la vista del stand, no del producto.
- **`"models/ está vacío"` como evidencia**: es el estado de la Mac de Felipe, no de la desktop de Santiago (J3 lo descuenta explícitamente). El caso de la poda se sostiene con los runbooks y la memoria versionada, que sí son iguales en todas las máquinas.

---

## 3. La arquitectura de información nueva

Cuatro destinos en el laboratorio, dos rutas separadas para el stand, una barra global permanente. Cero sub-pestañas. El diagnóstico de O1 que el panel ratifica: la navegación actual es un índice del directorio `scripts/` — diez destinos para seis lanzadores, un lector de archivo y un archivo de configuración, y ningún nombre de pestaña es el nombre de algo que una persona quiera *hacer*.

### BARRA GLOBAL DE EJECUCIÓN (permanente, habilita todo lo demás)
- **Contenido**: qué corre ahora (nombre, tipo, tiempo transcurrido), o "Inactivo"; **un** Detener con confirmación de cuerpo informativo; acceso a la única consola.
- **Tarea**: responder "¿está ocupada la máquina?" sin cambiar de pantalla, y hacer imposible el accidente de matar lo que no era.
- **Antes/después**: 6 botones de Launch compitiendo por un slot único sin indicador → los Launch se deshabilitan **con motivo**; 5 Detener globales disfrazados de locales → 1 global + 1 aislado en `/stand`; 3 consolas para 1 proceso → 1; el mensaje `Error: A process is already running!` escrito dentro de un textarea deja de poder ocurrir.

### S1 · VIGILAR (nueva, landing)
- **Contenido**: fila de KPIs del run activo; **bloque de flota** (4 filas: host, procs, steps/s, antigüedad); escalera de curriculum; consola filtrable. Estado vacío diseñado con un CTA.
- **Tarea**: "¿cómo va mi run?" y "¿siguen vivas las cuatro máquinas?".
- **Antes/después**: **0 pantallas** responden hoy esa pregunta (el dato está repartido en 3 consolas independientes, la tarjeta de curriculum y una telemetría imposible) → **1**. Clics para saberlo: 3+ cambios de pestaña, y aun así el dato no queda junto → **0**.

### S2 · ENTRENAR (un lanzador, no cinco)
- **Contenido**: bloque común (nombre de salida, modelo base, environment, dispositivo, timesteps, nivel de inicio) + "Avanzado" colapsado (lr/ent/clip, overrides). La tarjeta de Auto-Curriculum vive aquí.
- **Tarea**: lanzar el entrenamiento clásico que sigue vivo.
- **Antes/después**: la fusión de O1 pasa de **5 formularios (~50 campos)** a **2 (~15)** porque la poda de O7 elimina tres de los cinco (contradicción C3 del panel: el ganador abarata al perdedor). Controles visibles al abrir Training›Production: **19 + consola** → **12–15**.

### S3 · PROBAR (la duplicación P1/P2 muere)
- **Contenido**: un solo bloque de configuración con selector de jugador (P1 | P2); personaje; "Opciones de sesión" colapsado. Defaults que sirven.
- **Tarea**: montar un enfrentamiento en un clic.
- **Antes/después**: **22 controles visibles + 4 zonas de carga**, con "Launch Match" en y≈1150 — 1.6 viewports de scroll — → **6 controles**, Launch siempre visible, **scroll cero**. −73%.

### S4 · AJUSTES (detrás de un engrane)
- **Contenido**: config tipada (etiqueta en lenguaje de usuario, nombre de variable como texto mono secundario), savestates, rutas. Sin reescritura de `config.py` por `re.sub`.
- **Tarea**: la tarea mensual, que hoy tiene pestaña permanente de primer nivel.

### `/stand` · OPERADOR (ruta separada)
- **Contenido**: pre-vuelo mínimo (credencial del checkpoint + "emulador libre") y, en vivo, tres objetivos táctiles ≥88 px: NUEVA PARTIDA · REINICIAR · TERMINAR, más el rival como tarjetas de personaje (AZAR por defecto — `pick_state()` ya re-sortea dentro del bucle de rematch).
- **Tarea**: operar seis horas de pie sin poder romper nada.
- **Antes/después — CLICS PARA LA TAREA DEL STAND**: ~10 interacciones + 2 scrolls dentro de un formulario de 22 controles → **2** (entrar a `/stand` → Empezar), 3 si se elige personaje. **−80%**. Referencia histórica: el flujo previo a `5a852ae2` costaba ~6.

### `/tablero` · PÚBLICO (ruta separada, sólo lectura)
- **Contenido**: banda inferior bajo la ventana de BizHawk — marcador IA / retador, credencial del modelo en lenguaje de visitante, pantalla de espera que **es también** la pantalla de fallo.
- **Tarea**: convertir a un mirón en espectador sin que nadie pueda tocar nada.

### Antes/después en números (recuentos verificados sobre HEAD)

| Métrica | Hoy | Después | Δ |
|---|---|---|---|
| Destinos de navegación | 10 (5 tabs + 5 sub-tabs) | 4 + 2 rutas de stand | −60% |
| `gr.Dropdown` | 35 | ~10 (estimación tras la poda) | ~−71% |
| Controles que preguntan "qué modelo" | 24, en 5 vocabularios | ~5, en 1 | −79% |
| Zonas de carga (`gr.File`) | 11 | 1 (savestates) | −91% |
| Botones de Detener | 5 globales, sin confirmación | 1 global confirmado + 1 aislado | −60% |
| Consolas | 3 (`unified_logs` :1817, `league_logs` :1898, `match_logs` :2004 — **los tres con `elem_id="terminal"`**, id duplicado, verificado hoy) | 1 | −67% |
| Botones de Refrescar | 4 | 4 en Gradio; **0 sólo tras el transporte incremental** | condicionado |
| Altura de consola por defecto | `lines=35` ≈ 880 px, en un viewport útil de 712 | 14 líneas ≈ 252 px | −71% |
| Ancho dado a los controles en Training (1280 px) | 427 px (`Row(scale=1\|2)`) | 1048 px | +145% |
| Pantallas que muestran la calidad del checkpoint | 0 (código muerto) | 2 | +2 |
| Clics para "¿cómo va mi run?" | 3+ cambios de pestaña, dato disperso | 0 | — |
| Clics para la tarea del stand | ~10 | 2 | −80% |

> **Honestidad sobre "4 Refrescar → 0"** (contradicción C9 del panel): esa reducción presupone un canal incremental. En Gradio, sin él, la app efectivamente no conoce su estado y el botón Refrescar es honesto. Se etiqueta **post-transporte**, no se promete ahora.

---

## 4. Composición: pantalla → componentes shadcn

> **Alcance de esta sección**: es el plano de la **fase condicional** (§7 Fase 4), no de lo que se construye esta semana. Hoy no hay React, ni `components.json`, ni tokens: el único consumidor de todo es `src/scripts/web_dashboard.py`. Presentar esta tabla como "lo que vamos a instalar" sería engañoso (riesgo #1 de O4, ratificado por los tres jueces).
> **Todos los nombres provienen de `bunx --bun shadcn@latest search/view @shadcn` ejecutado por D3 y O4 (471 items enumerados). Ninguno inventado.** Hallazgo negativo que hay que respetar: **`@shadcn/data-table` NO existe** (404), aunque `data-table-demo` lo declare como dependencia.

| Pantalla | Registry verbatim | Composición nuestra | Custom de verdad |
|---|---|---|---|
| **Shell** (nav + barra global) | `sidebar` (block base `sidebar-08`; `sidebar-07` para el stand), `breadcrumb`, `separator`, `tabs` (sólo nivel 2, máx. 4), `badge`, `spinner`, `progress`, `button`, `command`, `kbd` | `<RunProvider>` + `<GlobalRunBar>` (badge+spinner+progress+botón+diálogo) | — |
| **S1 Vigilar** | `card`, `badge`, `item` + `ItemGroup`/`ItemSeparator`, `tooltip`, `skeleton`, `empty`, `alert`, `chart` + blocks `chart-area-stacked`, `chart-area-interactive`, `chart-radial-text`, `progress` | `<MetricCard>` (no existe item de KPI/metric/gauge), `<FleetProvider>` (sondeo de `/status` a 1 Hz con backoff) | `<Sparkline>` |
| **S1 · consola** | `message-scroller` (Provider/Viewport/Content/Item/Button; `Item` trae `content-visibility:auto` — render-skipping nativo), `card` (`CardAction` es el slot de la botonera), `button-group`, `badge`, `marker` variant="separator", `empty`, `resizable`, `font-jetbrains-mono` | `<LogConsole>` | `<LogStream>` (transporte + buffer circular), `<LogLine>` (nivel + ANSI) |
| **S2 Entrenar** | `field` (`FieldSet`/`FieldLegend`/`FieldGroup`/`FieldError` — cura estructural de la píldora azul: la etiqueta pasa de bloque de color a slot semántico), `form` (una familia: `form-rhf-*` / `form-tanstack-*` / `form-formisch-*`), `field-choice-card` para Algorithm, `select`, `native-select`, `input` type=number dentro de `input-group` con `InputGroupAddon` para unidades (**no hay item numérico**), `slider`, `switch`, `checkbox`, `radio-group`, `toggle-group`, `collapsible`, `textarea` | `<LaunchFormProvider>` (el discriminante lo posee el proveedor; el subárbol no puede escribirlo) | `<PhaseField>` (un control bimodal son DOS controles) |
| **S3 Probar** | `toggle-group` como selector P1\|P2 con **un** panel de configuración, `field`, `combobox`, `item` como resumen persistente, `slider`, `collapsible` | `<FighterConfig>` con variantes explícitas `<ApexFighter>`/`<SB3Fighter>`/`<HumanFighter>`/`<CpuFighter>` — la asimetría (P2 puede ser CPU y P1 no) va en el **tipo**, no en `visible=` | — |
| **Selector de checkpoint** (10 consumidores) | `combobox` (`ComboboxGroup`/`ComboboxLabel`/`ComboboxEmpty`/`ComboboxSeparator`), `badge`, `hover-card`, `sheet` (ficha completa = tarjeta resucitada), `chart` + block `chart-bar-horizontal` para la escalera L1–L8, `empty` | `<CheckpointCombobox>` | `<LadderStrip>` (8 micro-barras SVG; no hay sparkline en el registry) |
| **Destructivas** | `alert-dialog` (incluye `AlertDialogMedia`), `alert-dialog-demo` como plantilla, `spinner` dentro de la acción, `sonner`, `input`, `field` | `<GracefulStopDialog>` y `<ForceKillDialog>` — **dos componentes, no `<StopDialog dangerous />`**; `<ConfirmPhrase>` reutilizable | — |
| **S4 Ajustes** | `field`, `input`, `input-group`, `slider`, `switch`, `select`, `alert`, `alert-dialog`, `table`, `pagination` | config **tipada**, no JSON crudo | — |
| **Importar archivo** (savestates) | `attachment` (`state: idle\|uploading\|processing\|error\|done`), `input-group`, `item`, `progress`, `button` | — | `<Dropzone>` |
| **`/stand`** | `sidebar-07` colapsado o sin sidebar, `card`, `badge`, `drawer` (controles táctiles), `empty` entre partidas, `kbd` | `<MetricCard>` grande, `<LadderStrip>` | — |

**Rechazado explícitamente**: `scroll-area` para la consola (Radix puro, sin anclaje ni lógica seguir/despegar); `add` a ciegas de `dashboard-01` (arrastra `@dnd-kit/*`, `@tabler/icons-react`, `@tanstack/react-table`, `zod` — se usa como **referencia de ensamblaje**); toda la familia `login-0*`/`signup-0*`; editor de JSON crudo (exigiría CodeMirror/Monaco, fuera del registry).

**Radio de impacto de los componentes compartidos** (regla 5 del contrato):
- `<CheckpointCombobox>` — **10 consumidores**: entrenamiento, tuning, PBT, league, exploiter, matchup P1, matchup P2, modo stand, pantalla de modelos, diálogo de importación. Tras la poda de §2.2 quedan **5**. Su contrato (forma del inventario y del sidecar) se fija **antes** de escribir la primera pantalla.
- `<MetricCard>` — **5 pantallas**: flota, vigilar, ficha de checkpoint, pool de league (si vive), cabecera del stand. API a congelar: `label, valor, unidad, delta, estado, meta`.
- `<RunProvider>` — LaunchButton (×6 usos hoy), StopRunDialog, LogConsole, píldora de cabecera, `SidebarMenuBadge`, flota, modo stand.
- `<PhaseField>` y el vocabulario de "nivel" — **7 lugares**: `train_phase_drop`, `tune_phase_drop`, `pbt_phase_drop`, `cpu_level_cap_slider`, `cfg_win_rate` (`WIN_RATE_THRESHOLD`, que vive en **otra pestaña**), la escalera L1–L8 de la ficha y `win_rate_recent_by_lvl` del learner. **Unificar el token de dominio es prerequisito**: si no, el componente hereda las tres jergas y las cementa.

---

## 5. Contrato técnico

### 5.1 Fase Gradio (lo que se construye ya)

**Rutas.** `gr.Blocks.route(name, path, show_in_navbar=False)` — verificado en `.venv/.../gradio/blocks.py:3752`. Tres superficies desde **un solo proceso**: `/` (laboratorio, en el navbar), `/stand` y `/tablero` (fuera del navbar). `/stand` no enlaza a `/`. **No** se usa `launch(auth=...)`: protegería toda la app y pondría un login delante del proyector.

**Red.** `--host` por defecto pasa de `0.0.0.0` (`web_dashboard.py:2310`) a **`127.0.0.1`**, y la exposición a la tailnet exige bandera explícita. Hoy, cualquiera en la sala con el puerto puede pulsar "Force Kill (No Save)" desde su teléfono. Va en el primer PR o no va nunca (injerto de O3, ratificado por los tres jueces; contradicción C8).

**Persistencia de logs (Fase 0 de O3, prerrequisito de todo lo demás).** `stream_logs` (:424–535) hace hoy `stdout=PIPE` y **no escribe a disco**, y además re-emite el **buffer completo** en cada `yield` (:501–505). Cambio aditivo, dentro del archivo actual: tee a `logs/runs/<run_id>/stdout.log`; `logs/runs/<run_id>/run.json` = `{run_id, kind, cmd, pid, started_at, status, exit_code}`; y `yield` incremental. Esto por sí solo arregla "recargué el navegador y perdí el log", que es la razón de ser del hack `_DASHBOARD_RELOAD_HEAD`, y lo deja borrable.

**Eventos en vivo (Gradio).** No hay SSE en esta fase: se usa el mecanismo nativo (`gr.Timer`) con dos reglas duras — **1 Hz, no 10 Hz**, y **sólo cuando la superficie está activa**. El `gr.Timer(value=0.1, active=True)` de :2133 se elimina con la pestaña que alimenta.

**Flota.** El dashboard **no tiene ningún cliente HTTP** (0 usos de `urllib`/`requests`). Se añade uno mínimo contra `GET http://<learner>:8090/status` cada 1 s, con timeout corto y backoff. El navegador **nunca** habla con la tailnet. Campos ya servidos por `src/agents/apex.py:245–259`: `grad_steps`, `weights_version`, `buffer`, `transitions_in`, `episodes`, `win_rate_cum`, `win_rate_recent200`, `win_rate_recent_by_lvl`, y por actor `{host, procs, steps_per_s, transitions, last_seen, age}` — con `age` **precalculado por el servidor**.

**Tablero público.** Lector del JSONL de `MatchSessionLog` (`stand_leia.py:114–176`, `flush()+fsync()` por evento): `winner`, `ia_wins`, `retador_wins`, `ending` (`ko`/`time_over`), `duration_seconds`, `opponent`; y de `session_start`, `checkpoint_sha256` + sidecar. "Nueva partida" se implementa como **marcador de tanda en el JSONL**: pone el contador a cero sin tocar `stand_leia.py`, sin matar el proceso y sin que el visitante vea arrancar BizHawk.

**Modo forzado del stand.** El lanzador construye siempre `--infinite-match`, `--opponent-type human`, `--rematch-delay` fijo y `--device` resuelto en el pre-vuelo. Ninguno aparece como control. Motivo: `infinite_match_checkbox` está declarado con `value=False` (`web_dashboard.py:1988`, verificado hoy) y sin él `stand_leia.py:878–910` escribe `PAUSE` al terminar cada ronda y se queda sondeando `.agent_state` — **el default congela la demo tras la primera ronda**, y `RUN_STAND_LEIA.md:73` ya tiene que advertirlo en prosa.

**Confirmación de muerte por sonda, no por retorno** (riesgo 4b de O4, **ascendido a requisito** por J3). `graceful_stop_process`/`force_kill_process` usan `CTRL_BREAK_EVENT` y `taskkill` dentro de `try/except` que sólo imprimen al stdout del servidor: fuera de Windows fallan y **la UI reporta éxito igualmente**. El estado "detenido" se afirma sólo tras comprobar que el slot quedó libre; si no se confirma, alerta persistente, no toast verde.

### 5.2 Fase condicional (si se aprueba la migración)

FastAPI + uvicorn con **un solo worker** (`GlobalState`/`process_lock`/`launch_token` viven en memoria del proceso) + Pydantic v2; front Vite + React + TS strict + Tailwind + shadcn, cliente generado del OpenAPI. Transporte **SSE** (el flujo es estrictamente unidireccional; los comandos son POST) con `Last-Event-ID` + ring buffer para replay.

- Lectura: `GET /api/health` · `/api/models` · `/api/checkpoints/apex` (por item: `path, label, wr_mean, ladder L1..L8, weights_version, arch, valid, **error**`) · `/api/states` · `/api/config` · `/api/runs` · `/api/runs/{id}` · `/api/nodes` · `/api/curriculum`.
- Escritura: `POST /api/runs` (unión discriminada por `kind`) → `201 {run_id}` / **`409 {code:"slot_busy", run_id, kind}`** / `422` por campo · `POST /api/runs/{id}/stop {mode: graceful|force}` → `202` · `PATCH /api/config` · `POST /api/agent-state` · `POST /api/tensorboard` → `{url}` (**no abre navegador en el servidor**).
- Streams: `GET /api/events` (**uno solo**, multiplexado — HTTP/1.1 limita ~6 conexiones por origen) y `GET /api/runs/{id}/logs`.
- `_resolve_stand_checkpoint` deja de ser guard interno y pasa a **validador obligatorio** de toda ruta que entre por HTTP.
- El contrato mata cuatro bugs **por construcción**: `409 slot_busy`, `stop` con `{id}`, `algo` como **enum** en cualquier ruta que derive una carpeta, y el campo `error` por checkpoint.

### 5.3 Convivencia con el Gradio actual

**El panel rechaza el plan de cuatro fases con lock `.dashboard_owner`** (J2 y J3 por mayoría): dos supervisores peleando por un slot **re-crean exactamente la clase de bug** que el resto del plan está matando. La convivencia honesta es más simple:

1. Mientras exista el Gradio, **lanzar vive en un solo sitio**: el Gradio.
2. Cualquier superficie nueva es **de sólo lectura** hasta el corte: lee `logs/runs/*/run.json`, hace tail de `stdout.log` (gracias a la Fase 0) y sondea `/status`. Puede abrirse a media run sin tocarla.
3. El corte es por superficie y se hace de una vez, sólo tras usarse en un run real. Orden: Matchups+Stand → Entrenar → Ajustes.
4. Rollback: proceso separado, puerto separado, **sin base de datos ni migración de esquema** → el rollback es "no arranques el nuevo".

**Descartado explícitamente**: base de datos, Redis/Celery, Docker, auth de usuarios, Electron, Tauri en v1 (la pantalla limpia se cubre con `chrome.exe --app=... --kiosk`), y **commitear el bundle compilado** (arrastra a Santiago y Diego a conflictos binarios en cada `git pull`; J1 y J3 lo rechazan).

---

## 6. Sistema visual

**La paleta no se decide aquí.** Las candidatas medidas sobre el ROM del propio repo (122 frames en `benchmarks/state_farm_work/`, extracción con PIL, escalera del DAC de 9 bits como criterio de autenticidad falsable) están en el **artifact de candidatas que produce el orquestador**. Esta sección fija los **roles semánticos** que el sistema necesita y las **restricciones** que cualquier candidata debe cumplir — porque el panel decidió que los gates numéricos deben entregarse **antes** de elegir, no como revisión posterior (injerto de O6, ratificado por los tres jueces).

### 6.1 Roles semánticos requeridos
`--background`, `--foreground`, `--card` / `--card-foreground`, `--popover` / `--popover-foreground`, `--muted` / `--muted-foreground`, **tres tokens de borde distintos** (`--border` decorativo, `--input` contorno de control, `--ring` foco de teclado), `--primary` / `--primary-foreground`, `--secondary`, `--accent`, `--destructive` / `--destructive-foreground`, `--success`, `--warning`, y `--chart-1..5`. Nota técnica: `chart.tsx` genera CSS por tema (`THEMES = { light: "", dark: ".dark" }`), así que las series necesitan valores en **ambos** modos.

### 6.2 Gates que la paleta elegida debe cumplir (no son preferencias)
1. **`--primary` y `--destructive` no pueden compartir familia de matiz.** En una app donde "Force Kill (No Save)" convive con "Launch", eso es peligro real. *(Descarta "Ceniza y Carmín" tal cual, y obliga a condicionar "Barra de Vida": si se adopta, la rampa oro→naranja→rojo queda reservada **en exclusiva** a métricas y `--destructive` sale de esa familia; si esa condición no se acepta, la elección coherente es "Suzaku". Ver J3-C6.)*
2. **Veto al azul** como color de identidad: instrucción explícita del dueño. *(Descarta "Champion Chrome" pese a ser el logo medido del juego.)*
3. **Contraste medido, no juzgado a ojo**: texto normal ≥4.5:1; texto grande ≥3:1; **bordes de control, series, rellenos de barra y puntos de estado ≥3:1** (WCAG 1.4.11). Prueba de que el ojo no sirve de filtro: los cuatro colores que hoy fallan (#3b82f6 → 3.98, #ef4444 → 3.89, #a855f7 → 3.70, #2563eb → 2.83) "se ven bien".
4. **`--ring` no puede ser el mismo color que el relleno primario**: ≥3:1 contra el control **y** contra la página.
5. **Blanco a #E8ECE8, no #FFFFFF.** El hardware Genesis nunca llega a 255; es la decisión de identidad más barata y más fiel, y baja el glare en jornada de 8 h.
6. **Chasis derivado, datos literales**: fondo/card/muted/border no son hexes del juego; el color saturado ocupa <5% del área. Un solo acento por pantalla, máximo 3 usos.
7. **Fuera los emoji de las etiquetas de navegación y botones.** Sin esa decisión **ninguna paleta se percibe**; además cuestan ~20 px cada uno en una tira que ya desborda y los lectores de pantalla los verbalizan.
8. **Una sola dirección**, no dos temas por público (J1). Lo que cambia entre superficies es la **densidad**.

### 6.3 Densidad, escala y jerarquía
- **Tres productos de densidad, no un slider** (J3-C7): laboratorio (raíz 14 px, controles 32 px, objetivos ≥32 px con ≥8 px de separación) · operador del stand (16 px, objetivos ≥88 px de alto con ≥24 px de separación) · tablero público (tipografía de proyector, lectura a 4 m).
- **Escala de espaciado cerrada, base 4**: 0, 2, 4, 8, 12, 16, 20, 24, 32, 48. Etiqueta→control 4; campo→campo 12; grupo→grupo 20; padding de panel 16. Pitch de campo objetivo **64 px** frente a los 78–86 px medidos hoy (20–25% más campos por pantalla).
- **Tipografía**: 6 tamaños, 3 pesos (400/500/600 — fuera el 700 actual de los KPI). `tabular-nums` **obligatorio** en todo número vivo: sin él la tira de flota tiembla cada tick.
- **Jerarquía en 4 niveles con un dominante por pantalla**: L0 barra global sticky (40 px) · L1 objeto dominante (60–70%) · L2 apoyo (20–30%) · L3 metadatos (≤10%, 11 px, sin color). **Exactamente un `primary` alcanzable por pantalla** (hoy Training expone tres). Los destructivos no son `primary` ni comparten fila con la acción primaria.
- **Gate de revisión verificable en 10 segundos**: *la acción primaria de cada pantalla es visible sin scroll a 1280×800*. Hoy Matchups la incumple por ~440 px.
- **Topes duros**: ningún formulario con más de 18 controles visibles; consola por defecto 14 líneas (252 px) **a ancho completo abajo**, no en una columna que se lleva 853 de 1280 px.
- **Navegación de nivel 1 en rail vertical**: la tira ya desborda con menú "…" y un rail no puede desbordar. Nivel 2 en `tabs`, máximo 4.

### 6.4 Accesibilidad
- **Foco**: `:focus-visible` con outline de 2 px y offset 2 px; `outline:none` prohibido. Hoy hay **cero** estilos de foco propios y tres `alert()` inline que roban el foco sin devolverlo.
- **Estado sin color, tres canales en orden forma → texto → color.** Los seis estados de la flota (contribuyendo / ocioso / tasa desconocida / rezagado / perdido / bancado) llevan forma **y** palabra propias, y sólo **tres** tonos (ok/warn/danger): seis matices no se hacen deuteran-seguros, seis formas sí.
- **Las formas van en SVG inline, nunca en Unicode.** El operador está en Windows y ●○◐▲✕▬ renderizan distinto o no renderizan: una garantía que depende de un glifo que puede no existir es una garantía falsa.
- **Honestidad del dato**: `steps_per_s = None` se dibuja `—` con su forma propia, **jamás** una barra de altura cero. Un hueco es honesto; un cero fabricado miente — y el backend ya tomó esa decisión.
- **Actualización**: 1 Hz, y se actualiza sólo el nodo de texto que cambió. Un re-render completo por tick destruye el foco de teclado (WCAG 3.2.5).
- **Regiones vivas**: consola `aria-live="polite"`, `atomic=false`; nada a 10 Hz en región viva. `prefers-reduced-motion` elimina transiciones y pulsos.
- **Corregir de paso**: los tres `elem_id="terminal"` duplicados (:1817, :1898, :2004) son HTML inválido y rompen `aria-labelledby` / `label[for]`.

---

## 7. Plan de construcción por fases

### FASE 0 · Higiene y seguridad (horas, sin diseño) — *hazlo aunque no hagas nada más*
1. `--host` por defecto → `127.0.0.1` (`web_dashboard.py:2310`).
2. Tee de logs a disco + `run.json`; `stream_logs` incremental (:501–505).
3. **`WIN_RATE_THRESHOLD`**: o se conecta, o se etiqueta "sólo aplica con Auto-Curriculum desactivado" (ver §8-R1).
4. Borrar `_DASHBOARD_RELOAD_HEAD` + `DASHBOARD_BUILD_ID` y el `?build=` del runbook.
5. Borrar `gr.Timer(value=0.1, active=True)` (:2133) con su pestaña.
6. `unified_logs` de 35 → 14 líneas e invertir el `Row(scale=1|2)`; `model_name_input` con `lines=1, max_lines=1` dentro de una fila.

**Entrega**: recargar el navegador ya no pierde el log; la demo no se puede matar desde un teléfono; la consola deja de ser más alta que la pantalla; un control deja de mentir.

### FASE 1 · **VERTICAL SLICE MÍNIMO — el que ya sirve en el stand** ⭐
1. `/stand` y `/tablero` como rutas (`gr.Blocks.route`), fuera del navbar.
2. Lanzador de un botón que **fuerza** `--infinite-match` y resuelve device y checkpoint por defecto (`get_stand_default_checkpoint` ya existe).
3. **Resucitar `get_stand_checkpoint_status`** como credencial del pre-vuelo. Coste ~0: está escrita y probada; sólo hay que enlazarla.
4. `/tablero` de sólo lectura sobre el JSONL: marcador, credencial en lenguaje de visitante, pantalla de espera que **es** la pantalla de fallo.
5. `/stand` **sin** Force Kill, sin lanzadores de entrenamiento y sin jerga.

**Entrega**: el stand funciona de punta a punta, con 2 clics en vez de ~10, y un visitante no puede romper nada. **Si sólo cabe una fase antes del evento, es ésta.**

### FASE 2 · Barra global + confirmación destructiva
Barra sticky con el estado del único `active_process`; Launch deshabilitados **con motivo**; 5 Detener → 1 con `alert-dialog` de cuerpo informativo y **confirmación por sonda**; 3 consolas → 1.
**Entrega**: desaparece la clase entera de "maté lo que no era" y el error dentro de un textarea.

### FASE 3 · Poda + Vigilar con flota
Ejecutar §2.2 **junto con el rediseño del wiring** (nunca como borrado quirúrgico: §8-R2). Añadir el cliente HTTP mínimo y el bloque de 4 filas sobre `GET /status`, con estado vacío "coordinador inalcanzable" (no cuatro tarjetas en LOST).
**Entrega**: de 10 destinos a 4, y el usuario que hoy tiene ceguera total ve sus máquinas.

### FASE 4 · CONDICIONAL · Migración a React + shadcn
Sólo si Felipe la aprueba, **después del stand** y **fuera de una run crítica**. Orden: extraer las siete reglas de dominio de los callbacks inline a un módulo Python puro con pytest → API de sólo lectura → superficie por superficie → borrar `web_dashboard.py`.
**Entrega**: los tokens de §6 se aplican de verdad (hoy son 202 sitios literales que `primary_hue` no toca), y la spec de §6.3–6.4 aterriza entera en vez del ~60% que Gradio permite.

---

## 8. Riesgos y cómo se mitigan

**R1 · Un control que miente sobre una run viva.** `AutoCurriculumCallback.__init__` (`src/agents/auto_curriculum_callback.py:30`) recibe `win_rate_threshold` con default 0.75 y usa `self.win_rate_threshold` (:150, :588); **nunca lee `config.WIN_RATE_THRESHOLD`**, y ningún constructor de agente le pasa ese parámetro (verificado hoy). Mientras tanto `02-decisiones.md` tiene un plan fechado: "Plan B si estanca: bajarlo a 65%". Si alguien lo ejecuta moviendo el slider, no pasa nada y no hay aviso. Además hay 4 copias de la constante y `save_all_config` sólo escribe la primera. → **Fase 0, punto 3. Es lo más urgente del documento y no depende de ninguna otra decisión.**

**R2 · Podar rompe el wiring en silencio.** `refresh_dropdowns()` devuelve **10 `gr.update()` posicionales** enlazados en 5 sitios. Quitar una sub-pestaña sin tocar esos bindings desplaza el mapeo, **sin error**. → La poda va con el rediseño del wiring (Fase 3).

**R3 · Perder las reglas de dominio escondidas.** Siete callbacks inline (`update_match_ui`, `update_apex_opponent_controls`, `default_p2_for_apex`, `update_infinite_match_status`, `update_ui_on_algo`, `toggle_auto_curriculum_ui`, `handle_state_upload`) codifican qué rival es legal contra Ape-X y cuándo el nivel de CPU es exacto o tope, atrapadas en tuplas posicionales. → **Extraer la regla a un módulo puro con pytest antes de tirar el widget.** Es el único punto donde se puede perder lógica real.

**R4 · La causa raíz de `88404b0e` sigue viva.** HEAD tiene **tres** `.change` colgados de `p1_algo` (:2015, :2074, :2081) con outputs que se tocan, y lo único que evita la realimentación es un comentario. → Cura estructural: el discriminante lo posee el contenedor y el subárbol **no puede escribirlo**; variantes explícitas en vez de `visible=False`. Vale como regla incluso sin migrar.

**R5 · La UI reporta éxito cuando el kill falla.** Windows-only, fallos en `try/except` mudos. → Sonda posterior obligatoria; y **el test de humo de lanzar/parar se hace en la desktop de Santiago**: en la Mac no se puede validar.

**R6 · Fallos silenciosos en el inventario.** `get_stand_checkpoint_files` descarta con `except Exception: continue` y `_stand_sidecar_metrics` devuelve `{}` tanto si el sidecar falta como si está corrupto. → Devolver el **motivo**; sin ese cambio, no prometer en la UI un grupo de "N descartados".

**R7 · El stand vuelve a fusionarse con el laboratorio en la próxima noche de urgencia.** Es lo que hizo `5a852ae2` en 2h16m. → Separación **estructural** (rutas), no una bandera; y `/stand` no enlaza a `/`.

**R8 · Media spec de accesibilidad aplicada es peor que ninguna**, porque nadie sabe qué regla está viva. En Gradio sólo se aplican `tabular-nums`, tamaños de objetivo y foco visible. El resto espera a la Fase 4.

**R9 · Seguridad que cambia de severidad al exponer HTTP.** `get_best_tuning_params` genera un script Python por f-string con `study_name` y lo ejecuta con `python -c`. Hoy es molesto; con un socket escuchando es ejecución de código por la red. Muere con la sub-pestaña de Optuna; si Optuna vuelve, consulta directa en proceso.

**R10 · Alcance contra calendario.** Sólo la Fase 1 tiene fecha externa. Si algo se cae, se cae en orden inverso: primero la Fase 4, luego la 3.

---

## 🌅 Para Felipe — decisiones que sólo tú puedes tomar

**1 · ¿Se migra a React + FastAPI, o el dashboard se queda en Gradio podado?** *Recomendación del panel (mayoría): quedarse en Gradio ahora y reevaluar después del stand.* El argumento más fuerte a favor de migrar era que el stand necesitaba vista propia — y `gr.Blocks.route` ya la da. **Si migras**: la spec visual y de accesibilidad aterriza entera, los tokens funcionan de verdad, y el bug de outputs solapados se vuelve inexpresable; a cambio es un segundo producto para tres personas. **Si no migras**: entregas el stand esta semana, y aceptas que ~40% de la spec de §6 no aterriza y que "4 Refrescar → 0" no se cumple.

**2 · ¿El entrenamiento clásico PPO/SB3 se sigue lanzando desde la web?** *Recomendación: conservar UN lanzador, no cinco.* Es la única pregunta que cambia el tamaño de la app, y dejarla abierta bloquea el diseño de S2. El backend se queda pase lo que pase. **Si sí**: S2 y la tarjeta de Auto-Curriculum viven. **Si no**: el dashboard es *stand + vigilar*, y se reduce a la mitad.

**3 · Red durante el evento.** *Recomendación del panel: una sola máquina con dos monitores y bind en `127.0.0.1`.* Hoy el default es `0.0.0.0` sin autenticación. **Alternativa (b)**: dos procesos, tablero en `0.0.0.0` y laboratorio en local — más flexible, más piezas que fallan en vivo. Decidir **antes** del evento.

**4 · Paleta.** El artifact del orquestador trae las candidatas medidas. *Recomendación del panel*: "Barra de Vida" **sólo si** aceptas la condición de §6.2-1; si no, "Suzaku" es la elección coherente. "Champion Chrome" está vetada por tu propia instrucción de quitar el azul; "Ceniza y Carmín" por poner primary y destructive en la misma familia en la app del Force Kill.

**5 · ¿Se van los emoji de las etiquetas?** *Recomendación: sí.* Cuesta cero y sin ello ninguna paleta se percibe. Es cambio de contenido, no de estilo, y por eso es tuyo.

**6 · Idioma.** Hoy está mezclado (inglés con textos en español sembrados por `5a852ae2`). *Recomendación: `/stand` y `/tablero` en español, laboratorio en inglés.*

**7 · "Acaba de decidir" en el tablero público** (mostrar la macro que la IA acaba de elegir: `MACRO_NAMES` ya se calcula y se descarta). *Recomendación del panel: NO en v1.* Es el mayor salto de calidad de la demo por la menor cantidad de código, y también el **único** contenido que toca el bucle de tiempo real del proceso que no puede fallar delante de público. Si lo quieres, va con escritura barata y no bloqueante, nunca con `fsync` por decisión, y **medido**.

**8 · TensorBoard: ¿se mata o espera al "W&B sync para PPO"?** *Recomendación: esperar.* Hoy TB puede ser el único visor de una run viva. En cualquier caso hay que corregir que abre el navegador **en la máquina servidor**.

**9 · ¿Se pide nombre al visitante para el récord del día?** *Recomendación: no* (stand público, probablemente con menores) — "Retador #17". Alternativa intermedia: alias opcional que teclea el operador.

**10 · El slot de proceso: ¿sigue siendo uno solo?** Si se vuelve una cola de N trabajos, la barra global pasa a ser una lista y Vigilar necesita un selector. El diseño aguanta ambas, pero hay que elegir **antes** de construir la Fase 2.

**11 · Confirmar el estado real de `models/` en la desktop de Santiago** antes de dar por muerta la ruta SB3. El caso de la poda se sostiene con runbooks y memoria versionada, pero preguntar cuesta cero.

---

### Anexo · Hallazgos fuera de alcance (registrados como línea, sin abrir tarea)
- `get_best_tuning_params` genera código Python por f-string con `study_name` sin escapar y lo ejecuta con `python -c`.
- `run_matchup` tiene dos ramas inalcanzables (~:829, ~:841) porque `p1_algo` no ofrece `"CPU (Built-in AI)"`: `--cpu_level_cap` se omite en silencio.
- `handle_model_upload` crea carpetas con el nombre del algoritmo seleccionado (una con espacio) dentro de `models/production/`.
- Tres componentes comparten `elem_id="terminal"` (:1817, :1898, :2004).
- 69 de los 122 PNG de `benchmarks/state_farm_work/` están completamente en negro.
- `code_testing/pytest/test_telemetry_dashboard.py:436` es un test verde sobre una función que la UI no puede alcanzar; al reconectar la tarjeta deja de serlo.