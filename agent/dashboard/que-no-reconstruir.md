# O7 — Qué NO se reconstruye

Método: `edge-case-hunter` (recorrido mecánico de ramas y fronteras) aplicado no a "qué falla"
sino a "qué rama nunca se alcanza / nunca se ejerce". Alcance: `src/scripts/web_dashboard.py`
@ HEAD `f65a2932`, contrastado con `tools/RUN_*.md`, `agent/memory/*`, `agent/handoff.md`,
el árbol de `models/`, `benchmarks/`, `logs/` y los scripts que el dashboard lanza.

Regla aplicada: toda afirmación de "esto no se alcanza / nadie lo usa" está verificada **dos
veces en llamadas separadas**. Lo que no llegó a ese estándar va marcado `DUDOSO`.

---

## 0. El hecho que ordena todo lo demás

**El equipo NO opera este dashboard.** Los runbooks vivos describen el trabajo real y ninguno
pasa por Gradio salvo uno:

| Runbook | Cómo se opera | ¿Menciona el dashboard? |
|---|---|---|
| `tools/RUN_LARGA_DQN.md` | `tools/apex_learner.py` + `apex_actor.py` por CLI | No. "Monitoreo: Dashboard **DQN v2**" = **una vista de W&B**, no esta app |
| `tools/RUN_LARGA_HELPERS.md` | `python tools/apex_actor.py --learner http://…` | No. "Ver cómo va: `curl http://<DESKTOP>:8090/status`" + W&B |
| `tools/RUN_LARGA_SANTIAGO.md` | `apex_learner.py --port 8090 --wandb-project …` | No |
| `tools/RUN_OMEN_DESDE_CERO.md` | `apex_actor.py …` | Sólo "el dashboard DQN v2" (W&B otra vez) |
| `tools/setup_dqn.md` | CLI + `curl /status` + W&B | Sólo W&B |
| **`tools/RUN_STAND_LEIA.md`** | **`web_dashboard.py` → 🎮 Model Testing & Matchups** | **SÍ — y es el único** |

`agent/memory/02-decisiones.md` (2026-08-25): *"Métricas: **W&B**"*. `agent/memory/08-cola-manana.md:117`
apunta a `wandb.ai/leia-qro-rl/…`. El dashboard no aparece en `06-pendientes.md` ni en la cola.

**Consecuencia para la reconstrucción:** de las 5 pestañas de primer nivel, **una** tiene uso
documentado y vivo (Model Testing & Matchups, y sólo su rama Ape-X). Las otras cuatro son el
dashboard de la era BizHawk/SB3, que el proyecto ya no opera desde aquí. Reconstruir 101
controles para sostener 1 flujo real es exactamente el peso muerto que hay que no portar.

Además hay un techo estructural: `graceful_stop_process` / `force_kill_process` son
**Windows-only** (`signal.CTRL_BREAK_EVENT`, `taskkill /F /T`), y todo lo que el dashboard
lanza excepto nada corre contra BizHawk en Windows. El pipeline vivo (`apex_learner` /
`apex_actor`, stable-retro, multi-máquina Linux/macOS) **no tiene ni una sola representación
en la UI**: no se lanza, no se detiene, no se vigila desde aquí. La app está construida
alrededor del backend que dejó de ser el principal.

---

## 1. MUERTO — código sin ninguna ruta desde la UI

### 1.1 `refresh_stand_checkpoints` + la tarjeta de calidad del checkpoint
Ya levantado por D1/D2; lo confirmo y le pongo el matiz de mi eje: **no se mata, se
reconecta**. Es contenido de producto (WR 94.5%, escalera L1–L8, versión de pesos) que
`RUN_STAND_LEIA.md` §3 recita a mano en prosa porque la UI dejó de mostrarlo. Es lo ÚNICO
demostrable en un stand promocional. Prescindible = NO. Único caso en todo mi informe donde
"muerto" significa "resucitar", no "borrar".

### 1.2 `core/elo.py` — el mismo patrón, y este sí se tira
- **Qué es:** el módulo escrito para medir progreso contra un pool heterogéneo de rivales.
- **Evidencia:** `grep -rnE "(from core.elo import|import core.elo|\belo\.[A-Za-z_]+\()"` sobre
  `src/ tools/ code_testing/` devuelve **un solo hit**: `code_testing/pytest/test_elo.py:14`.
  Verificado dos veces (una pasada ruidosa que hubo que refinar + una pasada con regex exacta).
  `agent/handoff.md:460` lo dice desde antes: *"`core/elo.py` … está cableado a nada"*.
- **Por qué es prescindible:** ningún consumidor, ni en la UI ni en el backend. Su test verde
  es el mismo antipatrón que la tarjeta huérfana: un test que no puede fallar por la razón
  por la que existe.
- Fuera de alcance de UI, lo dejo escrito aquí y no abro tarea.

### 1.3 Ramas de `run_matchup` inalcanzables por asimetría de `choices`
Confirmado por D1 (`p1_algo` sin `CPU (Built-in AI)`, dos ramas que lo comprueban). Añado el
corolario que a D1 le faltó: la **línea 841** condiciona `--cpu_level_cap` a
`(p1_is_ai and not p2_is_ai and p2_algo == "CPU (Built-in AI)") or (p2_is_ai and not p1_is_ai
and p1_algo == "CPU (Built-in AI)")`. La segunda mitad de esa disyunción **nunca puede ser
verdadera**, así que el slider `cpu_level_cap_slider` en su modo "CPU Max Level Cap" tiene la
mitad de sus casos muertos por construcción. No se reconstruye la simetría P1/P2: se
reconstruye "quién es el modelo / quién es el rival", que es la pregunta real.

### 1.4 `readonly_params` (`gr.JSON "Fixed / Read-Only Hyperparameters"`)
- **Qué es:** un panel JSON dentro del acordeón de hiperparámetros avanzados.
- **Evidencia:** su único productor es `load_hyperparams_from_json`, colgado de
  `upload_json.upload`. No se inicializa con nada, no lo escribe ningún otro binding. Arranca
  **vacío y se queda vacío** salvo que el usuario suba un JSON de hiperparámetros a mano.
- **Prescindible:** sí. Es un visor de un archivo que el usuario acaba de subir — el archivo
  que ya tiene abierto. Cero valor informativo propio.

### 1.5 `_DASHBOARD_RELOAD_HEAD` + `DASHBOARD_BUILD_ID`
Andamio de la noche del stand (D2 lo documenta: 6 ediciones manuales del contador en 2h16m,
y el `?build=v1592-unified-r8` que `RUN_STAND_LEIA.md:46` obliga a escribir en la URL). Un
watcher JS que hace polling a `/gradio_api/app_id` cada 3 s para forzar recarga es un
mecanismo de sesión de desarrollo urgente. **No se porta.** Un frontend real recarga por
build hash del bundle, no por una constante editada a mano.

---

## 2. REDUNDANTE — dos (o cuatro) formas de hacer lo mismo

### 2.1 Los 5 "Detener" y los 3 "Copiar logs"
D1 ya los contó. Mi aporte es el criterio de corte: **hay un solo `active_process` global**,
así que hay exactamente **un** Stop conceptual y **un** Force Kill. Los 5 botones no son 5
capacidades: son 5 copias del mismo botón puestas donde el usuario resultó estar mirando. Se
reconstruye **uno**, global, en el chrome de la app, junto al indicador de qué está corriendo.
Idem: **una** consola (hoy 3 independientes para 1 proceso), **un** copiar.

### 2.2 `toggle_league_matchup_mode` ≡ `toggle_exploiter_matchup_mode`
Idénticas salvo el nombre; ambas ramas de subida usan el MISMO `handle_league_state_upload`.
League y Exploiter son el mismo formulario con distinto script destino.

### 2.3 Cuatro copias de `WIN_RATE_THRESHOLD` — y el slider edita la que no manda
- `src/core/config.py:455`, `src/agents/ppo/config.py:22`, `src/agents/dqn/config.py:37`,
  `src/agents/sac/config.py:26`. `save_all_config` escribe **sólo** la de `core/config.py`.
- Ver §3.1: para el curriculum automático ninguna de las cuatro se lee.

### 2.4 El botón "Refresh Auto-Curriculum" duplica un `gr.Timer(5)`
`refresh_curr_btn.click(get_auto_curriculum_status_html, …)` y
`gr.Timer(5).tick(fn=get_auto_curriculum_status_html, …)` invocan lo mismo. Un botón para
forzar lo que ya pasa solo cada 5 s no es una capacidad.

---

## 3. CONTROLES QUE MIENTEN — el label promete algo que la rama no hace

### 3.1 `WIN_RATE_THRESHOLD (Phase Advance)` no afecta al Auto-Curriculum
- **Verificado dos veces, en llamadas separadas.**
  1. `AutoCurriculumCallback.__init__` (`src/agents/auto_curriculum_callback.py:30`) recibe
     `win_rate_threshold: float = 0.75` **por parámetro con default**, y usa
     `self.win_rate_threshold` en `:150` y `:588`. **Nunca lee `config.WIN_RATE_THRESHOLD`.**
  2. Los tres sitios que lo construyen — `ppo/agent.py:176`, `dqn/agent.py:165`,
     `sac/agent.py:243` — **no pasan `win_rate_threshold`**. Confirmado con
     `grep -rn "win_rate_threshold=" src/`: cero resultados fuera del propio callback.
- Quien sí lee `config.WIN_RATE_THRESHOLD` es `manual_curriculum_callback.py:388/455`, la rama
  del checkbox **desactivado**.
- **Por qué importa:** `agent/memory/02-decisiones.md` (2026-08-26) dice literalmente
  *"Plan B si estanca: bajarlo a 65%"*. Si alguien ejecuta ese plan B moviendo este slider
  sobre una run con Auto-Curriculum, **no pasa nada y no hay aviso**.
- **Qué hacer:** no reconstruir el slider como "editor de config.py". O se cablea de verdad
  (parámetro del lanzamiento, no reescritura de fuente), o se borra.

### 3.2 `sac` sigue en 4 dropdowns y es un `NotImplementedError` deliberado
- **Evidencia:** `src/agents/sac/agent.py:113` y `:334` — **ambos** entrypoints
  (`train()` y el de tuning) hacen `raise NotImplementedError(_SAC_DISCRETE_MESSAGE)` en la
  primera línea. `src/agents/sac/config.py:18-19` lo dice en un comentario: *"unreachable
  (both raise NotImplementedError immediately)"*. `agent/handoff.md:595`: *"el andamio existe
  y está deliberadamente muerto"*.
- Aun así `sac` es una opción válida en `algo_sel` (`:1720`), `p1_algo` (`:1919`),
  `p2_algo` (`:1947`), y `run_stand` la acepta como rival SB3 (`:893`).
- **Prescindible:** sí, quitar `sac` de la UI. Un algoritmo que sólo puede producir un
  traceback no es una elección que ofrecerle a nadie — y en un stand con público es peor.

### 3.3 `Environment: v1` sólo existe en Training, y `v4` no existe en ninguna parte
- `env_sel` ofrece `["v1","v2","v3"]`; `p1_env`/`p2_env`/`league_env`/`exploiter_env` ofrecen
  `["v2","v3"]`. Un modelo v1 se entrena y **no se puede probar ni poner en liga desde la UI**.
- Al revés y peor: `src/envs/sf2_v4.py` existe y **v4 + macros es la configuración insignia**
  (es la obs que usa el Ape-X del stand). `agent/handoff.md:460`: *"`--env v4 --macros` lo
  entrena `train.py` y nada más. No lo puedes **ver**, ni **escalerear**, ni **tunear**."*
- **Qué hacer:** el vocabulario de environment de la UI está congelado en 2026-08. No se
  porta la lista literal; se deriva del backend.

### 3.4 El mensaje de TensorBoard va a `gr.Textbox(visible=False)` creado dentro del `.click()`
D1 lo levantó. Añado el motivo por el que **la pestaña de TensorBoard entera es candidata a
no reconstruirse**: `agent/memory/02-decisiones.md` fijó **W&B** como el sistema de métricas
el 2026-08-25, y los cinco runbooks vivos mandan a `wandb.ai`. `logs/` en este árbol contiene
sólo `logs/tuning` (vacío). Ver §5.1 (DUDOSO): en la desktop Windows con PPO/SB3 sí puede
haber TB logs.

### 3.5 `update_infinite_match_status` afirma un estado que no está en disco
D1 lo describió. Mi corrección importante: **`toggle_agent_btn` NO es prescindible.**
`src/scripts/stand_leia.py:881-905` escribe `PAUSE` al terminar un round sin auto-rematch y
luego **hace polling de `.agent_state` esperando `PLAY`** para reanudar; el propio comentario
del código dice *"el botón Toggle puede reanudar esta misma sesión"*, y `RUN_STAND_LEIA.md`
lo confirma. Es el único control del stand que reanuda una sesión pausada. Lo que sí se mata
es el **indicador que miente** (`update_infinite_match_status`), no el botón.

---

## 4. HUÉRFANO tras la era ChatGPT / obsoleto por el pipeline actual

### 4.1 Pestaña "🔮 Observation Telemetry" — estructuralmente inalcanzable desde el stand
- **Escritores de `.telemetry.json`:** exactamente dos, ambos SB3/BizHawk —
  `src/scripts/test_agent_v2.py:257` y `src/scripts/test_ai_vs_ai_v2.py:427`
  (`from core.telemetry import write_telemetry`).
- **`src/scripts/stand_leia.py` no la escribe.** Verificado dos veces en llamadas separadas:
  (a) su bloque de imports (líneas 33-66) **no incluye `core.telemetry`**;
  (b) `grep -c "write_telemetry" src/scripts/stand_leia.py` → **0**.
- **Por tanto:** el único flujo del dashboard que alguien opera hoy (Ape-X vs humano en el
  stand) **jamás puede encender esta pestaña**. La evidencia de campo vio "Standby Mode:
  Telemetry Offline" y ése es su estado permanente para el caso de uso vivo.
- Agravante ya documentado por D2: `gr.Timer(value=0.1, active=True)` en `:2133` la ejecuta
  10 veces/segundo, para siempre, para todos los clientes, para devolver la misma cadena.
- **Prescindible:** la pestaña de primer nivel, sí. El contenido (decodificador de acciones de
  `core/telemetry.py`, coordenadas de `compute_fighter_visual_coords`) es bueno y merece vivir
  **dentro** de la vista de match, apareciendo sólo cuando hay datos — no como una pestaña
  permanente reservada a un estado vacío.

### 4.2 Pestaña "🏆 Auto-Learning League" completa (League + Exploiter)
- **Nunca se ejecutó desde aquí:** `models/production/league/` existe y está **vacío**
  (0 archivos); `models/production/` en conjunto no contiene ni un `.zip`. Los contadores en 0
  que vio la evidencia de campo no son un estado vacío mal diseñado: son **la verdad**.
- **No está en ningún runbook.** `agent/memory/06-pendientes.md:13` la coloca en el futuro:
  *"**Más adelante**: liga/PvP en flota (estados 2P de FightLadder ya catalogados)"* — y en la
  **flota** (stable-retro, multi-máquina), no en el BizHawk que este tab lanza.
- `train_league.py` / `train_exploiter.py` no se tocan desde el 2026-08-25 y sólo por un fix
  transversal de γ (`13dc4a9b`), no por trabajo de liga.
- Su medidor natural (`core/elo.py`) está muerto (§1.2).
- **Prescindible:** sí, en la reconstrucción. Cuando la liga vuelva será sobre la flota
  distribuida y su UI no se parecerá a estos 4 dropdowns + subida de `.State`.

### 4.3 Sub-pestaña "🧬 PBT Training" — lanza un script que este venv no puede correr
- **Evidencia:** `train_pbt.py` importa `agents.pbt.build_orchestrator` →
  `pbt_orchestrator.PBTOrchestrator`, que necesita `ray[tune]`. `requirements.txt` (11 líneas,
  leído completo) **no incluye `ray`**, y `ls .venv/lib/*/site-packages/ | grep -i "^ray"` no
  devuelve nada. `agent/handoff.md:100`: *"`ray`/`ray[tune]` están **deliberadamente
  excluidos** de `requirements.txt`"*.
- Es además el **único lanzador sin selector de dispositivo** (`run_pbt` nunca construye
  `--device`): 12 controles para un flujo que no arranca.
- `train_pbt.py` no se toca desde el 2026-08-24 ("Update train_pbt.py").
- **Prescindible:** sí. Si PBT vuelve, vuelve como job de flota, no como botón local.

### 4.4 Sub-pestaña "🔬 Optuna Tuning" — sin estudios y con el historial en contra
- **No existe ni un `.db` de Optuna** en el árbol (`find . -name "*.db"` fuera de `.venv`/`.git`
  → vacío); `models/tuning/` vacío; `logs/tuning/` vacío.
- El único resultado de Optuna que el proyecto registra es un **desastre documentado**:
  `agent/memory/03-bugs-cazados.md:8` — *"**lr=2.108e-05 de Optuna** (tuneado bajo régimen
  roto) congeló la política en entropía máxima 1M steps"*; `05-runs.md:6` lo marca **NULO**.
  Después de eso el lr se fijó a mano por decisión (`02-decisiones.md`, 2026-08-26: *"3e-4
  desatora … a 16M+ enfriar a 1.5e-4"*).
- `get_best_tuning_params` además construye código Python por f-string con `study_name` sin
  escapar y lo ejecuta con `VENV_PYTHON -c` (D1 ya lo anotó; lo dejo escrito aquí también,
  sin abrir tarea).
- **Prescindible:** sí como pestaña. Optuna es una herramienta de CLI para una campaña
  ocasional, no una superficie permanente del centro de control.

### 4.5 Pestaña "⚙️ Core Config Editor" — 8 controles que reescriben el código fuente
- **Qué hace realmente:** `save_all_config` → `update_config_var`, que abre
  `src/core/config.py`, hace `re.sub` sobre la línea de la variable y **reescribe el archivo**;
  `handle_state_upload` hace lo mismo con la lista `CUSTOM_STATES`. Es decir: la UI edita el
  repositorio versionado en git, en caliente, sin diff y sin deshacer.
- **A quién sirve:** las 8 variables son exclusivamente del arnés BizHawk/SB3:
  `ACTIVATE_VISUALIZATION`, `ENABLE_THROTTLING`, `THROTTLE_SPEED` y `ENABLE_INPUT_DISPLAY` se
  consumen en `lua/v2.0/training_env_client.lua` y `match_test_env_client.lua`; `PORT` es el
  socket TCP de BizHawk; `N_ENVS` es de SB3. **Cero** de ellas toca el pipeline vivo
  (stable-retro / Ape-X). Y `WIN_RATE_THRESHOLD` está roto (§3.1).
- Los tres checkboxes usan `getattr(config, 'X', default)`: si la variable desaparece del
  fuente, el control aparece igual con un valor inventado y `save_all_config` devolvería error.
- **Prescindible:** sí como pestaña de primer nivel. Lo que sobreviva son 3-4 ajustes del
  arnés Windows que pertenecen al lanzador de entrenamiento, no a un editor de config global —
  y ninguno debe seguir escribiendo `.py` versionado.

### 4.6 Las 8 zonas de subida de modelos (`gr.File`) y `handle_model_upload`
- El flujo real de distribución de modelos es **git**: `RUN_STAND_LEIA.md:17` —
  *"`git pull` trae Lua, driver, dashboard y el checkpoint congelado … **No hace falta copiar
  el modelo por separado**"*; `RUN_LARGA_HELPERS.md` §2: *"`git pull` (todas las máquinas,
  siempre)"*; `02-decisiones.md`: *"Regla de oro: `git pull` antes de lanzar cualquier cosa"*.
- El árbol lo confirma: el único modelo SB3 del repo (`models/latest/v3/ppo/*.zip|.pkl`) llegó
  por commit, no por subida; `models/production/` y `models/tuning/` están **vacíos**.
- Y la función es activamente peligrosa: usa el algoritmo seleccionado como nombre de carpeta
  destino sin guarda, de modo que `"Human Player"` o `"CPU (Built-in AI)"` crean directorios
  con esos nombres (uno con espacio) dentro de `models/production/`.
- **Prescindible:** sí, las 8. Un dashboard cuyo inventario se sincroniza por git no necesita
  ser un gestor de archivos.

### 4.7 Sub-pestaña "🚀 Production Training" — obsoleta como *superficie*, no como capacidad
- Lanza `src/scripts/train.py`: PPO/SB3 sobre el puente TCP a BizHawk, un solo proceso local,
  Windows. El entrenamiento vivo del proyecto es `apex_learner` + N `apex_actor` sobre
  stable-retro, multi-máquina, coordinado por CLI y vigilado en W&B — **nada de eso cabe en un
  botón "▶ Start Training" con un `active_process` global**.
- Además choca con una regla operativa escrita: `agent/memory/05-runs.md:75` —
  *"**UN entrenamiento por máquina** (regla de Felipe)"*; el dashboard no la conoce ni la
  expresa (6 botones de Launch compitiendo por un slot, sin indicador de ocupado).
- **DUDOSO/decisión de Felipe:** ver §5.2.

---

## 5. DUDOSO — que decida Felipe, no yo

1. **Botón "📈 Launch TensorBoard".** Contra: las métricas del proyecto son W&B por decisión
   fechada, y aquí `logs/` está vacío. A favor: el PPO de la desktop Windows es SB3 y **sí**
   escribe TB logs, y `agent/memory/06-pendientes.md:7` pide *"W&B sync para PPO (SB3→wandb,
   ~15 líneas en train.py)"* — o sea, **hoy PPO todavía no está en W&B**. Si ese pendiente se
   cierra, TensorBoard se vuelve prescindible; hasta entonces puede ser el único visor de la
   run de PPO. **No lo mato.** (Lo que sí se corrige pase lo que pase: hace `Popen(shell=True)`
   y `webbrowser.open` **en la máquina servidor** — con `--host 0.0.0.0` abre una ventana en
   otra computadora.)
2. **La pestaña de entrenamiento clásico (PPO/SB3 vía BizHawk).** `02-decisiones.md`
   (2026-08-25) es explícita: *"BizHawk **se QUEDA** para evaluación/PvP/humano — nadie lo
   elimina"*, y hay una run de PPO viva rumbo a 31M steps. La pregunta no es si el backend
   sobrevive (sobrevive), sino si **lanzarlo y vigilarlo desde una web local** sigue siendo la
   forma de trabajar, cuando el resto de la flota se opera por terminal + W&B. Decisión de
   producto, no de agente.
3. **Auto-Curriculum: la tarjeta en vivo, el `download_curr_btn` y `refresh_curr_btn`.**
   La tarjeta (`get_auto_curriculum_status_html`, 139 líneas) es información real de una run
   real de PPO. Si la pestaña de training clásico sobrevive, la tarjeta también. Si no,
   se va con ella. **Depende del punto 2.**
4. **`test_agent_v2` / `test_ai_vs_ai_v2` (matchup clásico SB3 vs SB3 / SB3 vs humano).**
   Sigue en `06-pendientes.md:8` (*"head-to-head visual campeón vs retador"*) y en el
   `DEVELOPER_CLI_GUIDE`. Pero con `models/production/` vacío y un solo `.zip` en el repo,
   hoy no hay con qué armar un SB3 vs SB3. **Reducir a un caso, no eliminarlo.**
5. **`state_upload` de savestates (Core Config Editor).** Los 122 `.State` de `states/` llegan
   por git y `tools/farm_states.py` / `forge_states.py` los generan por CLI; la subida por web
   parece sobrante. Pero `06-pendientes.md:4` tiene abierto *"rellenar estados lvl5-8
   faltantes"*, así que el flujo de estados sigue vivo — sólo que probablemente no por aquí.
6. **`match_profile_checkbox` (`--profile`, cProfile).** Está realmente implementado
   (`test_agent_v2.py:51/211`, `test_ai_vs_ai_v2.py:163/334`), así que no es un control
   muerto. Pero es una herramienta de perfilado de desarrollador expuesta en la misma pantalla
   que un visitante del stand toca. **Yo lo sacaría de la vista del stand, no del producto.**

---

## 6. Resumen ejecutivo — la lista

| Veredicto | Qué |
|---|---|
| **NO PORTAR** | Pestaña League + Exploiter · Sub-pestaña PBT · Sub-pestaña Optuna Tuning · Pestaña Core Config Editor como tal · Pestaña Observation Telemetry como pestaña · Las 8 zonas de subida de modelos · `readonly_params` · watcher JS + `DASHBOARD_BUILD_ID` · `core/elo.py` · `sac` en todos los dropdowns · `v1` en `env_sel` · 4 de 5 Stops, 2 de 3 Copiar, 3 de 4 Refrescar · `update_infinite_match_status` |
| **PORTAR, PERO REPARADO** | El flujo del stand Ape-X (uno, no un formulario simétrico P1/P2) · La tarjeta de calidad del checkpoint (reconectar) · `toggle_agent_btn` (es load-bearing) · Un único Stop/Kill global con confirmación · Una única consola con el proceso identificado · La telemetría, dentro de la vista de match y sólo cuando hay datos |
| **DECIDE FELIPE** | TensorBoard · Training clásico PPO/BizHawk y su tarjeta de curriculum · Matchup SB3 clásico · Subida de savestates · `--profile` en la pantalla del stand |

**Nota de radio de impacto:** no hay componentes ni tokens compartidos que auditar — todo vive
en un archivo (`src/scripts/web_dashboard.py`), cuyo único consumidor es él mismo. El acoplamiento
real que sí hay que listar: `refresh_dropdowns` devuelve **10 `gr.update()` posicionales** y está
enlazado en **5 sitios** (`refresh_files_btn` + los `.then()` de los 3 lanzadores de training);
matar cualquiera de los 10 dropdowns (p.ej. los de PBT) rompe el mapeo **en silencio**.

**Fuera de alcance, registrado sin abrir tarea:** (a) `get_best_tuning_params` interpola
`study_name` sin escapar en código Python generado y lo ejecuta; (b) `core/elo.py` está muerto
en producción y su único importador es su test; (c) `handle_model_upload` crea carpetas con el
nombre literal del algoritmo seleccionado dentro de `models/production/`.
