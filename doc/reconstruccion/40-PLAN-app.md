# 40 · PLAN DE PRODUCTO Y UI — consola LEIA

Síntesis final de la corrida del 2026-08-28. Incorpora las 7 propuestas (`20`–`27`) y los
4 veredictos (`30`–`33`). Donde un juez rechazó algo, aquí **no aparece** o aparece con su
corrección; donde varios jueces coincidieron, la regla es dura.

**Toda cifra de esta hoja marcada `[medido]` la verifiqué yo en esta Mac hoy** (comandos en
§0). Lo que no pude verificar va marcado `[NO VERIFICADO]`. Lo que es objetivo de diseño y
no medición va marcado `[objetivo]`.

---

## 0. Re-verificación propia (contrato §4) — 2026-08-28, esta Mac

Dos lecturas de `GET http://desktop-4090-ubuntu-wsl:8090/status` separadas 20.06 s:

| Dato | Valor `[medido]` |
|---|---|
| grad_steps | 1,936,853 → 1,937,417 = **28.11 grads/s** |
| transitions_in | 207,172,800 → 207,201,600 = **1,435.63 trans/s** |
| episodes | **7.58 eps/s** |
| weights_version | 3,808 → 3,809 = **0.05 /s** |
| escalera viva (`win_rate_recent_by_lvl`) | L1 1.00 · L2 .995 · L3 .985 · L4 .925 · L5 .925 · L6 .92 · L7 .92 · **L8 .74** |
| entradas en `actors` | **12**, para **4 hosts** (SSS, PEREA, LAPTOP-1VCCKQL9, mini-fzamorano) |
| actores frescos (`age` < 600 s) | **1** — `SSS-207877`, age 0.7 s |
| el resto | de 3,267 s (54 min, la Mac) a **94,749 s (26.3 h)** |

Y el hecho que ordena el plan entero:

```
$ ps aux | grep -E "apex_selector|night_watch|apex_actor" | grep -v grep
(vacío)
```

**Toda la capa de observabilidad de esta Mac está muerta**, y el actor canario de la Mac
(`--difficulty 1..8`, el único guardián contra el olvido de lvl1-3) lleva **54 min apagado**.
Último renglón de `night_watch_dqn.jsonl`: 09:03. Último del selector: 08:51. El learner sigue
entrenando sin problemas. **Lo que se cayó no es lo observado: es el observador**, por segunda
vez en la misma mañana (`30-judge-arquitectura.md` lo midió a las 09:54 y sigue igual).

Otras verificaciones propias:

| Afirmación | Resultado |
|---|---|
| `find . -name "fleet*.json"` | **vacío** — el censo no existe |
| `grep -rn ray requirements*.txt` | **exit 1** — PBT no puede arrancar |
| `grep -rl web_dashboard code_testing/` | **2 ficheros** (`test_telemetry_dashboard.py`, `test_model_testing_config.py`) — sí hay red parcial |
| `code_testing/pytest/test_telemetry_dashboard.py:436` | llama `get_stand_checkpoint_status` → **no se puede borrar, sólo desconectar** |
| `tools/apex_learner.py:96` | `--batch` default = **256**, no 512 |
| `tools/apex_learner.py:240-241` | el tope de replay es **acumulado desde el resume**, no instantáneo |
| ratio con batch 256 | instantáneo **5.01**, acumulado **2.39** `[medido]` |
| `tools/apex_actor.py:188` | `--name` **ya existe** |
| `tools/apex_actor.py:199` | `socket.gethostname()` → en WSL2 devuelve el nombre de **Windows** |
| `tools/apex_actor.py:271-274` | el actor sólo manda `{procs, steps_per_s, host}` — **ni difficulty ni weights_version** |
| `tools/apex_learner.py:55-66` | sólo `GET /weights`, `GET /status`, `POST /transitions` — **cero escritura** |
| `tools/apex_learner.py:84,143` | bind `0.0.0.0`, **sin auth** |
| `src/scripts/web_dashboard.py:24-28` | `VENV_PYTHON` **NO** está clavado a Windows: hay fallback a `.venv/bin/python` y `sys.executable` |
| conteos del dashboard | **10** `gr.Tab` · **35** `gr.Dropdown` · **13** `gr.File` · **6** "Compute Device" (:1747,:1777,:1836,:1874,:1926,:1954) · **3** `elem_id="terminal"` (:1817,:1898,:2004) · **5** botones de parada (:1810,:1811,:1902,:1903,:1984) · `gr.Timer(0.1, active=True)` en :2133 |
| hexes literales | **107 ocurrencias, 22 valores distintos** (`grep -oE '#[0-9a-fA-F]{6}'`). *Corrijo a `31-judge-producto.md`, que dice 126/24; la conclusión no cambia.* |

**Corrección de premisa que hay que propagar:** `agent/dashboard/PLAN-reconstruccion.md`
(commit `781fe61f`, 2026-08-28 00:02) ya existía cuando arrancó esta corrida, y ninguna de las
7 propuestas lo cita (`grep -rln "PLAN-reconstruccion" .scratch-uiux-reconstruccion/` → 0).
Es el mismo ejercicio 9.5 h antes. **Este plan sí lo cita, y adopta su veredicto sobre el
orden de ejecución** (`33-judge-riesgo.md`): el diseño es completo, la entrega es por etapas.

---

## 1. LA APP EN UN PÁRRAFO

La consola LEIA es el **puesto de mando de un experimento de RL distribuido que hoy no tiene
ninguno**: un learner Ape-X en la desktop 4090 y hasta cuatro máquinas de actores repartidas
entre Querétaro y las casas del equipo, más un rig Windows con BizHawk donde se entrenan y se
juegan los modelos clásicos. Es para **tres personas** (Felipe, Santiago, Diego) que no miran
la pantalla: vuelven cada 30-60 minutos y la pregunta que traen no es "¿cómo va?" sino
**"¿tengo que levantarme?"**. Por eso la app es, en este orden: (1) una **alarma que empuja**
cuando la flota se degrada, (2) una **pantalla de flota de sólo lectura** que dice qué máquina
está muda, desde cuándo y con qué `--difficulty`, y (3) el sitio donde se lanza y se prueba
un modelo sin repetir la misma pregunta cinco veces. No es un dashboard de gráficas: las
series temporales viven en W&B por decisión del equipo con fecha (`agent/memory/04-infra.md:44-47`),
y reimplementarlas es la trampa más cara de esta reconstrucción.

**Criterio de éxito falsable, uno solo** (de `33-judge-riesgo.md`, adoptado): *la flota cae de
4 máquinas a 2 a las 03:00; alguien recibe un aviso en menos de 15 minutos y, al abrir una
página, ve cuáles dos, desde cuándo y con qué difficulty.* Hoy eso **no ocurre**: la flota
lleva ~26 h a 1 de 4 y nadie se enteró `[medido]`.

---

## 2. ARQUITECTURA DE INFORMACIÓN DEFINITIVA

### 2.1 Diagnóstico que la justifica

El dashboard está organizado **por script, no por tarea**: cada `gr.Tab` es un subprocess
(`run_training:731`, `run_tuning:689`, `run_pbt:1048`, `run_league:1063`, `run_exploiter:1085`,
`run_matchup:775`). Los lanzadores piden lo mismo (nombre + base + fase + steps + device) y
ninguno sabe del otro — la receta está copiada cinco veces. Y los seis compiten por **un solo
slot de proceso global** (`GlobalState`, `web_dashboard.py:31-38`; `stream_logs:424-455`, con
`yield "Error: A process is already running!"` en :449). La pestaña es una mentira estructural:
sugiere seis espacios paralelos donde hay un recurso único e invisible.

**Corolario duro: lo que corre no puede vivir dentro de una pestaña.**

### 2.2 Las 5 zonas (no 6)

`20-option-ia.md` propuso 6 y anotó como riesgo propio que `/ahora` y `/flota` responden la
misma pregunta. `31-judge-producto.md` lo rechazó y tomó la alternativa que la propia propuesta
dejó al juez. **Cinco zonas de un solo nivel, cada una con URL propia**, más dos elementos
transversales que no son zonas.

| # | Zona | Responde | Origen |
|---|---|---|---|
| **T1** | **Barra de Ejecución** (persistente, ~44 px) | ¿qué corre AQUÍ, desde cuándo, y su log? | transversal, no es zona |
| **T2** | **Paleta ⌘K** | ir a cualquier sitio / abrir cualquier artefacto | transversal |
| **1** | **`/flota`** (landing) | ¿está viva la flota? ¿tengo que levantarme? | **100 % nuevo** |
| **2** | **`/entrenar`** | lanzar y vigilar una corrida clásica | fusiona 5 tabs |
| **3** | **`/probar`** | quién contra quién: jugar contra un modelo, examinarlo | fusiona Matchups + Telemetría |
| **4** | **`/biblioteca`** | ¿qué artefactos tengo y qué son? | absorbe 13 `gr.File` + escaneos |
| **5** | **`/ajustes`** | config del rig BizHawk + censo `fleet.json` | Core Config (:2137) |

**T1 no lleva acciones de flota.** Su sujeto es el job **local** (capacidad 1) y por eso **sí**
lleva su botón *Detener*, con la condición de que **el botón nombre a su víctima** — hoy
"Graceful Stop League" (:1902) mata un PPO lanzado desde otra pestaña sin avisar. La **barra de
estado de FLOTA es otra cosa** y no lleva ninguna acción, porque su sujeto son N máquinas y
ningún botón puede nombrarlas todas (regla de `21-option-flota-ux.md`, cierre de conflicto de
`31-judge-producto.md`).

### 2.3 Antes / después, en números

Sólo van cifras que medí. Lo que es proyección se etiqueta.

| Métrica | Hoy `[medido]` | Plan | Δ |
|---|---|---|---|
| Contenedores de primer nivel | 10 `gr.Tab` (2 niveles) | 5 zonas (1 nivel) | −50 %, profundidad 2→1 |
| Rutas deep-linkables | 0 | 5 | — |
| `gr.Dropdown` | **35** | **9 instancias** de 6 tipos `[objetivo]` | −74 % |
| "Compute Device" idénticos | **6** (`choices=["auto","cpu","cuda"], value="auto"`) | **0** (→ selector de MÁQUINA en T1) | −100 % |
| `gr.File` | **13** | 1 dropzone | −92 % |
| Consolas de log | **3**, las tres con `elem_id="terminal"` duplicado (HTML inválido; el CSS `#terminal textarea` :2321 sólo aplica a una) | **1** | −67 % |
| Botones de parada | **5** sobre el MISMO proceso | 1 *Detener* + 1 *Forzar* tras `alert-dialog` | −60 % |
| Hexes literales en el código de UI | **107 ocurrencias / 22 valores** | 0 (todo token) | −100 % |
| `gr.Timer` a 0.1 s repintando oculto | 1 (:2133) | 0 | — |
| Superficie de flota | **0** (única mención de Ape-X en 2,326 líneas: :125) | 1 zona | — |

**Clics por tarea: NO SE PUBLICAN CIFRAS.** `20-option-ia.md` presentó "48 → 22 clics (−54 %)"
y "117 → ~36 controles visibles" en su tabla de resultados y en sus propios riesgos admitió que
nunca operó la app. `31-judge-producto.md` lo rechazó y este plan respeta el rechazo. Lo que sí
se afirma, cualitativamente y con evidencia:

- *Lanzar una corrida continuando el último modelo*: hoy exige elegir a mano algoritmo, entorno,
  zip, pkl y device — cinco preguntas cuya respuesta el artefacto ya sabe (§3). El plan las
  elimina, no las mueve.
- *"¿sigue avanzando?"*: hoy hay que recordar **en qué pestaña** se lanzó, porque dos de las tres
  consolas están siempre rancias (muestran fragmentos del mismo proceso único). Con T1 es coste 0.
- *"¿está completa la flota?"* y *"¿hay campeón nuevo?"*: hoy **imposibles**, no caras. Doce
  coronaciones ocurrieron sin testigo.

---

## 3. EL PATRÓN QUE MATA LOS SELECTORES REPETIDOS

El argumento no es de conteo — es de **tipos**. Cada pregunta repetida ya **divergió**, porque
son listas literales independientes en vez de un vocabulario cerrado:

- **entorno**: `["v1","v2","v3"]` en :1722 vs `["v2","v3"]` en :1924/:1952. **Ninguno ofrece `v4`**,
  que es el contrato vivo (`src/core/env_tools.py:22-30` ramifica v4/v3/else→v2; elegir `v1`
  **entrena en v2 en silencio**).
- **algoritmo**: 3 / 5 / 6 opciones en :1720 / :1916 / :1944, con `sac` presente en las tres y
  muerto en el código (`src/agents/sac/agent.py:113` y `:334` levantan `NotImplementedError` en la
  primera línea).

Un enum cerrado no puede divergir de sí mismo; seis `choices=[...]` sí, y ya lo hicieron.

### R1 — El artefacto lleva su identidad. Algo/env/norm dejan de ser preguntas.

Un `.zip` ya sabe su algoritmo; un `.pt` ya trae sidecar (`_load_stand_checkpoint_meta:186` **ya
lo lee y lo tira**); un `.pkl` sólo existe junto a SU `.zip`. Pasan a **badges derivados**,
editables sólo en *Avanzado*. Elimina ~13 de los 35 dropdowns `[objetivo]`.

> **Condición dura puesta por `32-judge-construible.md`, aceptada:** los badges **no son
> construibles hoy**. 454 de 500 filas de `benchmarks/bench_12rivals.jsonl` dicen
> `apex_eval_tmp.pt` (`bench_12rivals.py:366` guarda el basename), hay 3 esquemas de sidecar
> incompatibles y 4 `.pt` huérfanos. Sin un **índice unificado de artefactos** (trabajo de
> backend, no de UI) el picker mostrará "desconocido" en la mayoría de los casos, que es
> exactamente el estado que R1 prometía eliminar. **R1 se implementa cuando exista el índice;
> mientras tanto el badge dice `procedencia desconocida` y el control manual sigue visible.**

### R2 — El device no se elige; se elige la MÁQUINA.

Los 6 "Compute Device" son **byte-idénticos** (verificado, §0). En un híbrido de 4 máquinas la
pregunta correcta no es cpu/cuda, es **"¿en cuál?"**, y se responde **una vez, en T1**. Elimina 6.
*Nota de deuda:* `run_pbt:1048-1061` nunca pasa `--device` — irrelevante, porque PBT se mata (§7).

### R3 — Un solo componente responde "¿qué artefacto?"

**`<ResourcePicker>` como PRESET sobre `@shadcn/combobox`, no como wrapper opaco.** Verificado
por CLI que el registry ya exporta el compuesto completo (`Combobox`, `Input`, `Content`, `List`,
`Item`, `Group`, `Label`, `Collection`, `Empty`, `Separator`, `Chips`, `Trigger`, `Value`,
`useComboboxAnchor`); envolverlo tiraría API que el canon ya provee. Sólo se añaden `.Refresh` y
`.Meta`, que el registry no trae.

**Parametrizado por DESCRIPTOR, jamás por props booleanas.** `<ResourcePicker isFile isApex
showRefresh allowUpload multiple/>` son 6 booleanos = 64 estados.

```ts
type ResourceKind<T> = {
  id: string
  label: string
  list: () => Promise<T[]>          // la MISMA list() alimenta el picker y el ⌘K
  key: (t: T) => string
  render: (t: T) => ReactNode
  search: 'none' | 'client' | 'server'   // discriminante, NO booleano
  empty: ReactNode                        // @shadcn/empty, nunca una cadena vacía
}
```

`search` es el discriminante porque las 11 preguntas de hoy son sólo **3 formas de dato**:
enum corto (→ `select`), listado del FS largo y cambiante (→ combobox con búsqueda obligatoria),
jerárquico (→ árbol, **decisión de Felipe**, §8).

**8 presets explícitos**, no configuración inline: `MachinePicker`, `AlgoPicker`, `EnvPicker`,
`PhasePicker`, `ModelZipPicker`, `NormPklPicker`, `ApexCheckpointPicker`, `SavestatePicker`.
Instanciados en **3 ranuras** (base, P1, P2) + 1 selector de máquina en T1.

**Vocabularios cerrados, corregidos** (de `27-option-matar.md`, aceptado por dos jueces):

```ts
type Algo = 'ppo' | 'dqn' | 'apex'      // sin 'sac'
type Env  = 'v2'  | 'v3'  | 'v4'        // sin 'v1', CON v4
```

**Efecto colateral que importa:** mata los 5 `.change` encadenados con orden implícito sobre
`p1_algo`/`p2_algo` (:2021, :2029, :2074, :2080, :2085 — el nudo donde `p2_algo` era entrada y
salida de sí mismo). Se sustituyen por `derive(runTarget, p1, p2) -> MatchViewModel`: función
pura, sin orden implícito, testeable sin DOM.

**`ResourcePicker.Meta` restituye el dato que `5a852ae2` borró**: la tarjeta de identidad del
checkpoint (`in_dim` / `n_actions` / `quantiles` / `weights_version` / escalera L1-L8 del sidecar)
que hoy calcula `get_stand_checkpoint_status:349`. **Ojo: esa función tiene un test verde**
(`code_testing/pytest/test_telemetry_dashboard.py:436`) — se **desconecta** de Gradio, **no se
borra**. `27-option-matar.md` afirmó "cero llamadores" y "cero tests tocan web_dashboard.py";
ambas son **falsas** y `33-judge-riesgo.md` las cazó. Verificado por mí (§0).

### R4 — "rancio ≠ dato" (injerto de `21`/`24`/`32` a TODAS las superficies)

Toda celda derivada de un POST viejo se pinta `stale` usando `age`, que **sí** viene del server
(`src/agents/apex.py:258`). No es sólo para la tabla de actores: aplica a la Biblioteca y al
`ResourcePicker`. Motivo medido hoy: `PEREA-602` reporta **46,479.4 steps/s** con **17.6 h** de
rancio, y dos actores más reportan `procs=40` congelados desde hace 16 h `[medido]`. Una tabla que
pinta eso sin marcarlo es **una mentira renderizada**.

---

## 4. LA ZONA DE FLOTA

Diseño **100 % nuevo**: la única mención del mundo Ape-X en 2,326 líneas es
`benchmarks/apex_milestones` como raíz de búsqueda de checkpoints (`web_dashboard.py:125`).

### 4.1 Los primeros 3 segundos — `FleetStatusBar`

Persistente en toda la app, ~56 px, `aria-live` con la política de §6.4, cada celda con **su
propio semáforo** (un semáforo global habría dicho "rojo" y borrado la información útil):

| Celda | Contenido | Hoy diría `[medido]` |
|---|---|---|
| **LEARNER** | píldora `training_state` + grads/s | `TRAINING · 28.1 grads/s` |
| **FLOTA** | **máquinas frescas / esperadas** + procs | **`1/4`** en rojo |
| **DATOS** | trans/s medido vs esperado de flota llena | `1,436 /s` vs ~12k → degradado |
| **ESCALERA** | 8 micro-barras + tier más débil nombrado | `L8 0.74` |
| **CAMPEÓN** | wr_media + **antigüedad** | (bloqueado, §8) |

**El KPI de portada es `n/N máquinas` + `trans/s vs esperado`. NO es el replay ratio.**
`21-option-flota-ux.md` propuso `replay_ratio_real 9.29 / STARVED` como tile de portada;
`31-judge-producto.md` lo rechazó y yo lo reproduje:

- El 9.29 sale de usar `--batch 512`; el default real es **256** (`apex_learner.py:96`).
- El tope que el learner aplica (`apex_learner.py:240-241`) es **acumulado desde el resume**, no
  instantáneo.
- Mis dos lecturas dan **5.01 instantáneo / 2.39 acumulado** `[medido]`.
- Y es **imposible por construcción** que el ratio se sostenga por encima del tope: el learner
  duerme 50 ms en cuanto lo supera (`:242`). El tile habría pintado en rojo un estado que el
  propio código impide.

**Corolario que cambia el diagnóstico:** con ratio acumulado 2.39 el learner **nunca throttlea**
→ está limitado por **cómputo** a ~28 grads/s, **no hambriento**. Así que "un tile cierra E1 y E4
a la vez" es falso: **E4 (colapso parcial) sólo lo cierra el censo**. El replay ratio baja a dato
secundario y **sólo se muestra con el `batch` LEÍDO del servidor** (§4.6, campo aditivo #1).

Reglas de la barra: **ningún número sin su edad de lectura visible** (HTTP 200 no es liveness —
incidente del 2026-08-27: HTTP vivo con el loop congelado 1.5 h). Fuera de la portada: `buffer`
(clavado en capacidad, señal muerta), `win_rate_cum` (denominador 1.2 M), `episodes`, `loss`.

### 4.2 La unidad de pantalla es la MÁQUINA DEL CENSO, no la entrada del dict

Medido hoy: **12 entradas para 4 hosts, 1 fresca**. El dict nunca se poda (`apex.py:182`) y la
identidad es `hostname-pid` (`apex_actor.py:199`). Por eso:

- Se renderiza **una tarjeta por máquina ESPERADA de `fleet.json`**, siempre las 4, aunque tres
  estén mudas. **La fila fantasma es literalmente el detector de E4.**
- Los ~136 procesos **nunca** se listan en la vista principal: agregación por máquina,
  drill-down por `sheet`.
- Los zombis se colapsan en un cajón "histórico de procesos" con el **corte de `age` expuesto
  como control** (600 s es convención de un script de scratchpad, `night_watch_dqn.py:32`).
- `procs` se rotula **"declarado"**, nunca "medido", porque miente cuando el actor muere.
- El **throughput del learner (Δ`transitions_in`) es la AUTORIDAD**; el `steps_per_s` del actor
  es un **RECLAMO**, y la discrepancia se muestra, jamás se promedian. Hoy: actor fresco 3,021.5
  st/s vs learner 1,435.6 trans/s `[medido]`. **Prohibido sumar `steps_per_s` crudo**, que es lo
  que hace `night_watch_dqn.py:70` y lo que produce el tile ridículo de PEREA-602.

**Bloqueador de identidad (de `23-option-fleet-agent.md`, confirmado por dos jueces y por mí):**
`actors[*].host` da `SSS` / `PEREA` / `LAPTOP-1VCCKQL9` — nombres de las máquinas **Windows** —
mientras la tailnet las llama `desktop-4090-ubuntu-wsl` / `legion-wsl` /
`laptop-omen5080-ubuntu-wsl`, porque `socket.gethostname()` dentro de WSL2 hereda el nombre del
anfitrión. Sólo `mini-fzamorano` (la Mac, no-WSL) coincide. **`fleet.json` NO puede llavear por
hostname ni por nodo de Tailscale**: hace falta un `node_id` opaco, propagado por el flag
**`--name` que ya existe** (`apex_actor.py:188`) → **cambio de wire de cero líneas**.

### 4.3 Monitoreo vs operación — la línea se dibuja en el ESPACIO

Criterio: **efecto sobre la run**. Regla dura: **ninguna acción vive en `FleetStatusBar`**; cada
acción cuelga del sujeto que afecta y **nombra su alcance en el propio botón**.

**v1 de la zona de flota es SÓLO LECTURA, y eso va en la portada del proyecto, no en un anexo.**
Verificado: `apex_learner.py:55-66` expone únicamente `GET /status`, `GET /weights` y
`POST /transitions` — **cero superficie de escritura**. `31-judge-producto.md` puso la condición
que este plan adopta: **v1 no embarca un solo botón que parezca actuar y no actúe** — o se dibuja
deshabilitado con su razón visible, o no se dibuja.

Catálogo de operaciones (diseñado ahora por decisión #1 de Felipe; **encendido después**, §8):

| Operación | Riesgo | Patrón |
|---|---|---|
| Relanzar actor | bajo | botón en la tarjeta de la máquina |
| Ajustar `--procs` | bajo | campo + confirmar |
| Podar zombis | bajo | acción del cajón, con `undo` |
| Cambiar `--difficulty` | **alto — redefine el experimento vivo** | `alert-dialog` + aviso de que la escalera viva quedará sesgada ~20 min |
| Parar / arrancar actor | medio | `alert-dialog` que **nombra al dueño de la máquina** |
| Forzar checkpoint | medio | confirmar |
| Lanzar examen del banco | medio | ocupa cómputo |
| Adoptar `.pt` huérfano | bajo | formulario de procedencia |
| Cambiar `replay_ratio` / `lr` | **crítico** | confirmación escrita |
| Pausar learner | **crítico** | confirmación escrita |
| Coronar a mano | **prohibido sin examen** | no se ofrece |

Ninguna sin `undo` o sin registro. **Y el supervisor nunca arranca, para ni reinicia un
learner** (rechazo duro de `30-judge-arquitectura.md`): un reinicio destruye estado que ningún
`previous_ref` restaura — el buffer de 1 M transiciones en memoria y el dict `actors` entero
(comprobado: `transitions_in` es la suma exacta de las 12 entradas, o sea la historia de flota
es *in-memory*). El learner se reinicia **a mano, en ventana planeada**.

### 4.4 Jerarquía de alarma — 4 niveles y un no-nivel

| Nivel | Significado | Forma |
|---|---|---|
| **L0 Nominal** | todo bien | `success` |
| **L1 Aviso** | vigilar | `warning` |
| **L2 Alerta** | **se pierde capacidad** | `destructive` **en contorno** + toast |
| **L3 Crítico** | **la run muere** | `destructive` **en relleno** + banner fijo |
| **DESCONOCIDO** | dato no fresco | `muted-foreground` **rayado** |

Cuatro reglas:

1. **DESCONOCIDO nunca se pinta de verde.** Es el error que hace que `procs=40` en una máquina
   muerta hace 26 h parezca salud.
2. **L2 y L3 comparten color** y los separa la **forma** y el **emplazamiento**, porque la paleta
   es cerrada (`00-DECISIONES.md:7-17`) y **inventar un quinto token de severidad sería cambiar
   el contrato de color = decisión de Felipe**, no de un agente.
3. La severidad se calcula **por sujeto**, nunca global.
4. Toda alarma trae **su evidencia numérica y su desmentido**.

Asignación: **L3** = learner congelado, learner inalcanzable, flota totalmente muda, olvido de
lvl1-3 confirmado, disco lleno. **L2** = colapso **parcial** de flota (*ocurriendo ahora*),
canario caído, examen del selector sin correr, hub mudo, sampler muerto. **L1** = zombis
acumulándose, `weights_version` sin avanzar, ratio fuera de banda, máquina en horas silenciosas.
**L0 y en lima**: **campeón nuevo coronado**. Meter lo bueno en la escala de lo malo es cómo una
UI se vuelve un muro de rojo que nadie lee; las 12 coronaciones sin testigo son un fallo de
**observabilidad**, pero son una **noticia**, no una alarma.

Anti-fatiga: **una alarma por causa raíz**, **histéresis 2/2 visible**, `ack` ≠ `resolver`,
ventana de mantenimiento por máquina.

### 4.5 La regla del velador, corregida (el cambio de una condición)

`night_watch_dqn.py:81-83` alerta si **NINGÚN** actor está fresco. Con 1 de 4 vivo **nunca
dispara** — es ciego por construcción al modo de fallo que de hecho ocurre. Debe alertar si
**hay menos máquinas frescas que las esperadas**. Ese cambio de una condición **habría avisado
hace 26 horas**. Y el velador **no debe suicidarse**: hoy `alert()` termina en `raise SystemExit(1)`
(`night_watch_dqn.py:46`) — el detector se mata al detectar, que es la razón por la que ahora
mismo no hay ninguno corriendo `[medido]`.

### 4.6 Cinco pantallas, y qué de ellas es v1

| Pantalla | Contenido | v1 |
|---|---|---|
| **Máquinas** (default) | grid por máquina esperada, fila fantasma, drill-down | **SÍ** |
| **Learner** | máquina de 6 estados `WARMUP / TRAINING / THROTTLED / STARVED / FROZEN / UNREACHABLE`; el desambiguador entre THROTTLED y FROZEN es **`transitions_in` avanzando mientras `grad_steps` no**; serie con eje X de **reloj**, no de `grad_steps` | parcial (la píldora sí, la serie no) |
| **Escalera** | **DOS escaleras jamás fundidas**: la **VIVA** (deque 200/nivel, termómetro, sesgada) y la **EXAMINADA** (48 eps/tier, juez), superpuestas en bullet chart de 8 filas con referencia 0.95. Nivel sin episodios recientes se dibuja **HUECO** con "sin datos hace 42 min", **nunca como 0** — el contrato ya codifica esa confusión: `win_rate_cum` devuelve `0.0` y no `null` sin episodios (`apex.py:251`). **Prohibido el guion mudo "—"** | la VIVA sí; la EXAMINADA cuando el histórico esté en el repo |
| **Campeones** | timeline de coronaciones; la "era" (4 tiers vs 8) es **separador visual duro** | **NO** (§7, bloqueado aguas arriba) |
| **Eventos** | bitácora persistente **en el repo** que reemplaza al velador | **SÍ** (es la alarma) |

**Presupuesto de poll como decisión de diseño**, no como buena intención: `status()` toma el
**mismo lock** que `train_tick` usa para `sample()` (`apex.py:229` vs `:246`). Barra 10 s ·
pantalla visible 5 s · **pestaña oculta 0** · campeones 60 s · **`GET /weights` NUNCA desde la
UI** (5,335,856 bytes sin ETag ni HEAD). Un solo poller compartido.

> **Corrección importante de `30-judge-arquitectura.md`, adoptada:** el argumento "si cada pestaña
> poléa se multiplica la carga sobre el proceso que entrena" es **falso por dos órdenes de
> magnitud** — midió 12 clientes a 0.5 s = −1.6 % de gradientes, dentro del ruido. El presupuesto
> de poll se mantiene por las razones que **sí** resisten: las derivadas sólo existen diferenciando
> dos lecturas en cadencia fija, cada navegador calcularía una tasa distinta, y cerrar la pestaña
> borraría la historia. **El peligro real es un cliente desbocado** (205 req/s cuestan −31 %), y
> este repo tiene precedente literal: `gr.Timer(0.1, active=True)` en :2133. Por eso el cerrojo va
> **del lado del learner**: `GET /status` sirve un snapshot cacheado ≤1 s **sin tomar el lock**,
> más límite de tasa por IP. Son unas líneas, es aditivo, y hace irrelevante la disciplina del cliente.

### 4.7 Campos aditivos al wire (precondición de la UI, no mejora)

Todos **aditivos y compatibles hacia atrás**: `apex.py:184-186` ya filtra `stats` por llaves
conocidas, así que un actor viejo simplemente manda menos campos. Ninguno toca la matemática ni
exige reiniciar la run de 1.9 M grads.

| # | Cambio | Dónde | Coste |
|---|---|---|---|
| C1 | pasar `--name <node_id>` a los actores | **cero código** (`apex_actor.py:188` ya lo acepta) | 0 |
| C2 | el actor reporta `node_id`, `difficulty`, `weights_version` en uso, hijos vivos | `apex_actor.py:271-274` | ~5 líneas |
| C3 | `/status` expone `server_time`, `training_state`, `batch`, `buffer_capacity`, `uptime`, y **poda `actors` por edad** | `apex_learner.py` (`:258-277` ya calcula grads/s, trans/s, loss y beta — **sólo van a stdout y a W&B**) | ~10 líneas |
| C4 | `GET /status` cacheado ≤1 s sin lock + rate-limit por IP | `apex_learner.py:55-66` | ~10 líneas |
| C5 | bind de `0.0.0.0` a la IP de tailnet | `apex_learner.py:84` | **1 línea**, cierra el agujero de auth |

**`batch` y `buffer_capacity` son obligatorios antes de mostrar cualquier ratio.** Sin ellos, todo
cociente que pinte la UI está escalado por un factor constante supuesto — que es exactamente el
error que produjo el 9.29.

---

## 5. MAPEO A COMPONENTES

Leyenda: **[R]** item del registry, verificado por CLI · **[P]** preset sobre un item **[R]** ·
**[C]** custom con hueco verificado.

`32-judge-construible.md` barrió los 41 items que los documentos nombran: **41 de 41 existen**
con el nombre y las deps exactas. Y los 3 huecos negativos son ciertos: `@shadcn/data-table`
**no existe** como item instalable (y `data-table-demo` declara la regDep rota `"data-table"`),
`toast` no existe (es `@shadcn/sonner`), y `log-viewer` da 404 en todos los registries.

| Superficie | Componentes |
|---|---|
| Shell / rail de 5 zonas | `@shadcn/sidebar` **[R]** |
| **T1 Barra de Ejecución** | `@shadcn/badge` + `button-group` + `progress` + `alert-dialog` **[R]** |
| **T2 ⌘K** | `@shadcn/command` + `kbd` **[R]** |
| Selectores (§3) | `ResourcePicker` **[P]** sobre `@shadcn/combobox` **[R]** (+ `.Refresh`, `.Meta` **[C]**) |
| Consola de log | **`LogConsole` [C]** — `scroll-area` + `card` + `button-group` + `badge` + `empty` + `sonner` **[R]** + `@tanstack/react-virtual` + stick-to-bottom propio |
| Tablas densas | **`DataGrid` [C]** = `@shadcn/table` **[R]** + `@tanstack/react-table` + `dropdown-menu` + `checkbox` + `input` + `pagination` **[R]**. `data-table-demo` se **lee como plantilla, jamás se instala** |
| Tarjeta de máquina | `card` + `badge` + `tooltip` + `sheet` (drill-down) **[R]** + `HealthDot` **[C]** |
| KPIs de la barra | **`StatTile` [C]** (búsquedas `stat`/`kpi` → 0 items) |
| Escalera de 8 tiers | `@shadcn/chart` **[R]** (bullet horizontal) + `WinRateRow` **[C]** |
| Alarmas | `sonner` (L2) + `alert` fijo (L3) **[R]** |
| Confirmaciones destructivas | `@shadcn/alert-dialog` **[R]**, siempre nombrando al sujeto |
| Subida de artefactos | `UploadField` **[C]** sobre `input` + `card` **[R]** (1 dropzone sustituye 13 `gr.File`) |
| Formulario de lanzamiento | `@shadcn/field` + `input` + `select` + `toggle-group` (método) **[R]** |
| Estados vacíos | `@shadcn/empty` **[R]** — **prohibido** hand-rollear |

**Prohibido hand-rollear** (existe item **[R]**): toast → `sonner`; split → `resizable`;
paleta → `command`; Start/Stop/Kill → `button-group`; vacío → `empty`; gráficas → `chart`;
campo → `field`.

**Inventario custom real** — corrigiendo la subcuenta de "9 componentes" que
`32-judge-construible.md` señaló: son **9 componentes + 8 presets de picker + 3 configuraciones de
`DataGrid`** (actores agrupados por host sin paginación / banco con facetas / campeones con
columna `schema`) **+ 5 pantallas de flota de diseño nuevo + el shell**. Es una **aplicación**, no
un puñado de piezas sobre el registry.

**Tres precios que hay que declarar antes del primer `shadcn add`:**

1. `@shadcn/combobox` — pieza central del colapso de dropdowns — depende de **`@base-ui/react`**
   mientras el resto depende de `radix-ui`. Verificado que `r/base/combobox.json`,
   `r/radix/combobox.json` y `r/styles/radix/combobox.json` dan **404**: aun con `--base radix`
   el combobox trae `@base-ui/react`. **Dos librerías de primitivas en el mismo bundle es un
   hecho, no un riesgo evitable** — hay que presupuestar probar juntos los dos modelos de portal
   y foco.
2. `@shadcn/sonner` arrastra **`next-themes`** en un proyecto Vite sin Next.
3. `@kibo-ui/table` arrastraría **`jotai`** (gestor de estado global que ningún documento costeó).

**Estimación de `LogConsole` corregida:** `24-option-composicion.md` dijo "~40 líneas"; con
`WrapToggle` las alturas de fila son **variables**, lo que exige `measureElement` y rompe la
aritmética simple del stick-to-bottom — además de ring buffer, `truncated`/`droppedCount`
**visibles**, `connected:false` pintado distinto de `lines:[]`, y reanudación por offset de byte.
**Real: 3–5 jornadas con pruebas de comportamiento**, no de revisión visual.

---

## 6. CONTRATO DE TOKENS — Champion Chrome

Forma **oficial shadcn Tailwind v4**: variables sin prefijo en `:root` / `.dark`, más un bloque
`@theme inline` que las mapea a `--color-*`. **Sin esa línea la utilidad `bg-<token>` no existe y
falla en silencio.** Valores en **hex, no oklch**: los 26 están pinneados en hex por Felipe
(`00-DECISIONES.md:10-17`) y convertir a oklch y volver introduce deriva de ±1 por canal, lo que
dejaría todos los ratios de abajo sin respaldo.

**Los 26 valores pinneados NO se tocan.** Los fallos de AA se arreglan con **slots derivados
nuevos** (`--destructive-fg`, `--secondary-fg`, `--warning-fg`), no pisando el valor original: un
token derivado no reabre una decisión cerrada, sólo nombra un uso que el valor pinneado no cubría.

**Todos los ratios de esta sección los calculé yo hoy** con un script de luminancia WCAG 2.x
`[medido]`. Donde difiero de `25-option-tokens.md` lo digo.

### 6.1 Ratios del polo OSCURO

| Token | /card `#0D1020` | /background `#05070F` | /muted `#161B30` |
|---|---|---|---|
| foreground `#E8ECE8` | 15.82 | 16.86 | 14.26 |
| muted-foreground `#9AA3BE` | 7.51 | 8.01 | 6.77 |
| primary/ring `#40A8C8` | 6.88 | 7.33 | 6.20 |
| secondary `#3E85C0` | 4.78 | 5.10 | **4.31 ✗** |
| accent `#C8EC40` | 13.96 | 14.88 | 12.59 |
| destructive `#E83A2A` | 4.56 | 4.86 | **4.11 ✗** |
| success `#40CC88` | 9.18 | 9.78 | 8.28 |
| warning `#E8CC00` | 11.75 | 12.52 | 10.59 |
| input `#525E8C` | 3.00 | 3.20 | **2.71 ✗** |
| border `#262C48` | 1.38 | 1.47 | 1.25 |

### 6.2 Ratios del polo CLARO

| Token | /card `#F8F9FC` | /background `#EEF0F6` | /muted `#E0E4EE` |
|---|---|---|---|
| foreground `#0F1424` | 17.41 | 16.08 | 14.40 |
| muted-foreground `#4E5670` | 6.90 | 6.38 | 5.71 |
| primary/ring `#206488` | 6.16 | 5.69 | 5.09 |
| secondary `#00688C` | 5.94 | 5.49 | 4.91 |
| accent `#4A6B00` | 5.88 | 5.43 | 4.86 |
| destructive `#A81400` | 7.21 | 6.67 | 5.97 |
| success `#00684A` | 6.48 | 5.99 | 5.36 |
| warning `#886400` | 5.15 | 4.76 | **4.26 ✗** |
| input `#7B84A0` | 3.53 | 3.26 | **2.92 ✗** |
| border `#C9CFDE` | 1.48 | 1.37 | 1.23 |

### 6.3 Los 5 fallos y sus parches (todos sobre `--muted`)

**Los cinco caen sobre `--muted`, que es exactamente la superficie de zebra de tabla y de fila
seleccionada — o sea donde vivirá la tabla de actores de la flota.** Ninguno de los parches mueve
el matiz ni pasa del ~4 % de luminancia.

| Fallo | Original | Parche | Ratio `[medido]` |
|---|---|---|---|
| oscuro destructive / muted | 4.11 | `--destructive-fg: #EA4B3C` | **4.52** |
| oscuro secondary / muted | 4.31 | `--secondary-fg: #4389C2` | **4.54** |
| oscuro input / muted | 2.71 | `--input-strong: #596697` | **3.06** |
| claro warning / muted | 4.26 | `--warning-fg: #826000` | **4.56** (*el doc dice 4.54*) |
| claro input / muted | 2.92 | `--input-strong: #78819E` | **3.04** |

### 6.4 Cuatro reglas sistémicas que salen de la medición

1. **Tinta sobre relleno cromático: `#05070F` en oscuro SIEMPRE, `#FFFFFF` en claro SIEMPRE.**
   Verificado y contraintuitivo: en oscuro la tinta **oscura** gana en los **seis** rellenos —
   destructive **4.86** con `#05070F` vs **4.14** con blanco; primary 7.33 vs 2.74; accent 14.88
   vs 1.35; success 9.78 vs 2.06; warning 12.52 vs 1.61; secondary 5.10 vs 3.95. La intuición
   "botón azul → texto blanco" es exactamente el error, y es el que produce hoy los
   `<div style='color: red'>` de :1176, :1357, :1466.
2. **`--border` (1.38 / 1.48) queda PROHIBIDO como único límite de un control interactivo.** Ése
   es el rol de `--input` / `--input-strong` (WCAG 1.4.11 pide 3:1). Corolario de layout: **un
   input nunca se coloca sobre `--muted`**.
3. **Foco = doble anillo con offset**, no `outline` simple:
   `box-shadow: 0 0 0 2px var(--background), 0 0 0 4px var(--ring)`. Motivo medido: `--ring` **es**
   `--primary` (mismo hex), así que el anillo sobre un botón primary lleno tiene contraste
   **1.00 — literalmente invisible**. `25-option-tokens.md` dice "--ring 7.33/5.69, holgado" y
   omite ese caso; **el contrato de tokens absorbe la regla de `26-option-layout.md`** (injerto
   de `32-judge-construible.md`).
4. **El `-subtle` NUNCA es señal por sí solo.** Los diez `subtle`/`card` que medí caen entre
   **1.14 y 1.37**, imperceptibles. Todo chip lleva **las tres capas**: fondo + punto + etiqueta
   de texto. Un chip "bonito" sin punto ni etiqueta es una violación de 1.4.1 que se ve perfecta
   en el mockup.

### 6.5 Tokens de estado — 5 estados × 3 slots, cero hues nuevos

Convención: `base` = punto/borde · `-subtle` = fondo del chip · `-foreground` = texto sobre
`-subtle`. Todos derivados de colores ya pinneados.

**OSCURO** (`card #0D1020`) `[medido]`:

| Estado | base | subtle | fg/subtle | subtle/card | punto/card |
|---|---|---|---|---|---|
| `running` ← success | `#40CC88` | `#142A2F` | **7.28** | 1.26 | 9.18 |
| `idle` ← muted-fg | `#9AA3BE` | `#212536` | **6.04** | 1.24 | 7.51 |
| `degraded` ← warning | `#E8CC00` | `#2C2A1C` | **8.98** | 1.31 | 11.75 |
| `champion` ← accent | `#C8EC40` | `#272F24` | **10.23** | 1.37 | 13.96 |
| `alarm` ← destructive | **`#EE5F51`** | `#2C1824` | **5.06** | 1.14 | 5.75 |

> Nota propia: con `#EA4B3C` (el parche del doc) el par fg/subtle da **4.41 ✗**, por debajo de AA.
> Subo el token de estado a **`#EE5F51`** (5.06) — sigue siendo el mismo matiz, y `--destructive`
> pinneado no se toca. Es un fallo que `25-option-tokens.md` no reporta.

**CLARO** (`card #F8F9FC`) `[medido]`:

| Estado | base | subtle | fg/subtle | subtle/card | punto/card |
|---|---|---|---|---|---|
| `running` | `#00684A` | `#D5E5E3` | 5.24 | 1.24 | 6.48 |
| `idle` | `#4E5670` | `#E0E2E8` | 5.61 | 1.23 | 6.90 |
| `degraded` | `#826000` | `#E7E4D9` | 4.55 | 1.21 | 5.51 |
| `champion` | `#4A6B00` | `#E0E5D9` | 4.83 | 1.22 | 5.88 |
| `alarm` | `#A81400` | `#EDD9D9` | 5.62 | 1.28 | 7.21 |

**Los 10 pares fg/subtle pasan AA.** Todos los puntos pasan 1.4.11 (mínimo 5.06).

**`running` = verde, no primary cian**, porque `primary` es el color de lo que se puede pulsar
(botones, links, ring): un badge "corriendo" en cian haría que un estado y un control fueran el
mismo color, ilegible en una tabla de 12 actores.

**Hueco declarado, no tapado:** `warmup` del learner no mapea limpio — no es `running` (no hay
gradientes) ni `degraded` (es correcto y esperado). Se pinta como `idle` con la etiqueta
"calentando". Es un parche; §8 lo lleva a Felipe.

### 6.6 Rampa de tiers — secuencial, no categórica

`win_rate_recent_by_lvl` es **ordinal**, y la paleta no tiene 8 hues distinguibles. Rampa sobre
el matiz de `primary`, medida por mí `[medido]`:

```
oscuro:  --tier-1 #2D6E88 (3.32) · #33829E (4.34) · #3A96B4 (5.58) · #40A8C8 (6.88)
         · #5EB4CE (8.00) · #7CC0D4 (9.30) · #9ECEDA (11.06) · --tier-8 #C3DDE1 (13.26)
claro:   --tier-1 #81A7BC (2.44 ✗) · #6594AD (3.12) · #4B829F (4.00) · #367394 (4.94)
         · #206488 (6.16) · #1C506F (8.22) · #183C56 (10.96) · --tier-8 #13283D (14.26)
```

**Los 8 pasan 3:1 en oscuro. En claro `tier-1` (2.44) no llega** y no es corregible sin perder el
rango → lleva trazo de 1 px en `--input` (3.53 sobre card) + **etiqueta numérica**.

### 6.7 Asimetría honesta entre polos

**El polo oscuro es el default** y el claro es "soportado y degradado", no un par:

- El accent claro pinneado `#4A6B00` es **olivo oscuro**, no lima: funciona (5.88) pero no brilla.
- **`#C8EC40` sobre card claro da 1.28:1 — invisible** `[medido]`.
- El polo claro es **casi isoluminante** para series de gráfica (`25-option-tokens.md` midió 9 de
  10 pares por debajo de 1.30, peor 1.04; en oscuro 3 de 10). **Consecuencia obligatoria: ninguna
  gráfica multi-serie puede depender del color** — dash + marcador + etiqueta directa al final de
  la línea; la leyenda de recharts sola no basta. **Tope duro: 5 series.**

Es decir: **la firma lima que justifica la paleta entera sólo existe en oscuro.** Si el equipo
acaba usando el polo claro (proyector, impresión, preferencia), queda una consola azul competente
pero anónima. Está declarado, no escondido.

### 6.8 Disciplina del lima `#C8EC40` — tesis cuantitativa

Es la **única** entrada de la paleta fuera del sector azul-cian, y sobre background da **14.88**
contra 16.86 del foreground: ópticamente es el **segundo elemento más brillante de la pantalla**,
al 88 % del cuerpo de texto. **Su significado está en su rareza, no en su matiz.**

**SÍ (lista cerrada, máx. 1 por viewport):** chip del campeón vigente (`apex_escalera_best.pt`) ·
mejor valor de columna del banco de 12 rivales · serie del campeón sobre la rampa de escalera ·
récord histórico en un KPI · wordmark.

**JAMÁS:** CTAs (eso es `primary`) · tab activo · **hover** · focus ring · **cualquier estado
operativo** (lima **no** es "bueno", es "el mejor" — confundirlos rompe el uso legítimo) ·
rellenos grandes · texto de cuerpo · visor de logs · config · uploads.

**Presupuesto auditable por grep: ≤5 componentes, ninguno `Button` / `Tab` / `Input` / log.**

> ### 🚫 BLOQUEADOR DE TOKEN — no se puede ejecutar `shadcn add` hasta cerrarlo
>
> `25-option-tokens.md:134` fija el lima en **`--accent`** y `:385` prohíbe el lima en hover.
> **Las dos cosas no pueden ser verdad.** `32-judge-construible.md` leyó el código que sirve el
> registry hoy: **`bg-accent` ES el token de hover/focus/highlight** — `dropdown-menu` ×5
> (`focus:bg-accent`), `button` ×3 (`hover:bg-accent`, `hover:bg-accent/50`), `command`, `select`,
> `item`, `toggle`, y `combobox` (`data-highlighted:bg-accent`). Con el lima ahí, **el día de la
> instalación** cada botón fantasma, cada item de menú enfocado, cada opción del ⌘K y cada fila
> del `ResourcePicker` bajo el cursor se pintan de `#C8EC40`, y el presupuesto de ≤5 componentes
> pasa de *auditable-por-grep* a *violado-por-construcción*.
>
> **Forma mínima de arreglo (propuesta, requiere el sí de Felipe):** `--accent` vuelve a
> superficie neutra derivada de `--muted`, y el lima se declara como token propio
> **`--champion` / `--color-champion`** con sus 5 usos cerrados. Es un **cambio al contrato de
> tokens ⇒ decisión de Felipe** (contrato §5), no de un agente. → §8.

### 6.9 Tipografía

Tres roles. **Display Chakra Petch** (600/700) — wordmark, eyebrows, nombre del campeón, título de
diálogo; **nunca** texto corrido, **nunca** <14 px, **nunca dígitos**. **UI IBM Plex Sans**
(400/500/600). **Datos IBM Plex Mono** (400/500/600) para **todos** los números, IDs, hashes,
rutas, celdas, KPIs y logs. Plex Sans + Mono es superfamilia real (misma métrica vertical), así
que una fila que mezcla etiqueta y valor no hace escalón.

- **`tabular-nums` OBLIGATORIO en todo número.** Medido hoy: `grad_steps` avanza **28.11/s** y
  `transitions_in` **1,435.63/s** `[medido]`; sin cifras tabulares la fila entera vibra
  horizontalmente en cada tick. Es el defecto de legibilidad más caro de la pantalla y **una
  línea de CSS**.
- **Los KPI grandes son MONO, no display**: Chakra Petch no garantiza cifras tabulares.
- **PROHIBIDO: contadores animados** (un tween sobre 28 cambios/s garantiza que el número nunca
  esté quieto), interpolación de las barras de tiers, shimmer en bucle, GIF.
- **`html { font-size: 16px }` NUNCA se toca** (rompe el zoom del usuario). Base de app 13 px
  = `0.8125rem` en `body`, todo en `rem`. 7 pasos de texto (10/11/12/13/15/18/22) + **escala
  numérica separada** (16/22/30/40), la única autorizada a saltar, con cupo por viewport.
- **El visor de logs impone 12/18 px enteros** (no `rem`): `@tanstack/react-virtual` necesita
  altura de fila fija y un `line-height: 1.5` sobre 12 px acumula deriva a las mil líneas.
  Monocromo (foreground + muted-foreground para timestamp + `alarm` sólo para `ERROR`).
- `prefers-reduced-motion` con reset global **más dos cosas que el reset no cubre**:
  `scroll-behavior: auto` en el log e **`isAnimationActive={false}` en Recharts**, que anima por
  JS e **ignora la media query** — el fallo de reduced-motion más común con `@shadcn/chart`.

### 6.10 Densidad, movimiento y accesibilidad — reglas de v1

- **Una sola densidad, alta, SIN toggle.** La jerarquía la da la posición. Un toggle duplica la
  matriz de prueba de cada tabla; el eje que la gente sí usa es el zoom del navegador, y funciona
  porque todo está en `rem`. Escala de 4 px con medio paso de 2; alturas de control 24/28/32
  (default 28), fila de tabla 28, barra global 44. **24 px es exactamente el mínimo de WCAG 2.2
  AA 2.5.8** → piso absoluto.
- **Objetivos táctiles a 44 px por `@media (pointer: coarse)`, NO por ancho.** A 200 % de zoom en
  1920 el viewport CSS cae en `sm`: si el disparador fuera el ancho, un usuario con baja visión y
  ratón recibiría controles inflados sin motivo (WCAG 1.4.10 Reflow).
- **Presupuesto de movimiento: 5 gestos, sólo `transform`/`opacity`, 80–160 ms.**
- **`aria-live` POR DATO.** El error por defecto sería envolver el panel de flota en
  `aria-live="polite"`: con `transitions_in` cambiando **1,435 veces por segundo**, eso genera una
  cola infinita de anuncios y hace la app inusable con lector de pantalla. Los números de alta
  frecuencia van en `aria-live="off"` + `aria-hidden`, con un `sr-only` compacto refrescado ≤1 vez
  cada 30 s. **Sólo las TRANSICIONES se anuncian** (`polite`, debounce ≥5 s, coalescencia); los
  fallos en `assertive`.
- **Se eliminan los emoji de los títulos de pestaña** (🏋️‍♂️🚀🧪🧬🎯⚔️🎮⚙️): un lector de pantalla lee
  "hombre levantando pesas" antes de cada título.
- **El color nunca es canal único**: forma + texto + icono. Hoy el semáforo de tiers es puro color
  y las etiquetas **divergen** — `"CRITICAL WEAKNESS"` (:1145-1156) vs `"WEAKNESS"` (:1319-1330)
  **para el mismo umbral**. Vocabulario unificado.
- **Prohibido mostrar éxito antes de confirmar la muerte del PID** (§7).
- **Responsive: sólo `/flota`**, hasta 360 px y de **sólo lectura** en táctil. Las otras 4 zonas
  muestran un bloque honesto "necesita ≥1024 px". No se rediseña para un pulgar un formulario de
  20 campos que lanza 12 h de cómputo. *Y el vigía nocturno no necesita web responsive: necesita
  que el aviso le llegue al teléfono* — push primero, pantalla móvil después.

---

## 7. QUÉ SE MATA

De las 37 capacidades del inventario: **9 se matan · 3 familias se fusionan · 2 salen de la app ·
6 salen de v1 · 17 sobreviven.** El corte no es estético: **cinco de las nueve que se matan no
pueden ejecutarse hoy en ninguna máquina del equipo**, y otras cuatro **reportan éxito cuando
fallan**.

### 7.1 PROBADO MUERTO — no puede ejecutarse

| # | Qué | Evidencia |
|---|---|---|
| 1 | **PBT entero** (sub-pestaña + 6 controles, :1787-1800, `run_pbt:1048-1061`) | `grep -rn ray requirements*.txt` → **exit 1** en los 4 requirements, contra **7 imports** en `src/agents/pbt/pbt_orchestrator.py` (:54, :168, :177-181). El botón lanza un `ImportError`. `[medido]` |
| 2 | **`sac` en los 3 vocabularios** (:1720, :1916, :1944) | `src/agents/sac/agent.py:113` y `:334` levantan `NotImplementedError` **en la primera línea del cuerpo**; el propio repo lo documenta muerto en `sac/config.py:16-22`. Vocabulario nuevo: `{ppo, dqn, apex}` |
| 3 | **La opción `"v1"` de entorno** (:1722) | `src/core/env_tools.py:22-30` sólo ramifica v4/v3/else→v2: **elegir v1 entrena en v2 en silencio**. Espejo del mismo bug: **`v4`, el contrato vivo, no está en ningún dropdown** — y eso es más grave, porque la UI **esconde lo que sí existe** |
| 4 | **`normalize_apex_p2_selection`** (:768) | tras `c365ae98` su condición es siempre falsa; el guard `"Invalid Ape-X matchup"` (:784) es **inalcanzable** y **su docstring es hoy literalmente falsa** |
| 5 | **La implementación de Force Kill** (`taskkill`, :616 y dentro de `force_kill_process`) | `taskkill` no existe fuera de Windows, `capture_output=True` **traga el error**, y la función devuelve `"⚡ Force Kill Executed"` **incondicionalmente**. Síntoma: el operador ve "matado", el emulador sigue vivo, y el siguiente Launch se rechaza por "ya hay un proceso" — **culpando al botón equivocado**. Es el gemelo POSIX del bug que `37e8a6e3` sí arregló (:467-469 ya tiene `start_new_session`). **La capacidad vive; la implementación muere** |

### 7.2 FALLOS SILENCIOSOS que no se portan

| # | Qué | Evidencia |
|---|---|---|
| 6 | **Botón TensorBoard** (:752-755) | `Popen` a DEVNULL + return incondicional, **y su salida va a un `gr.Textbox(visible=False)`** (:2222): el usuario **nunca ve ni la confirmación mentirosa**. Y `find -name "events.out.tfevents*"` → **0 resultados** en todo el repo → **fuera de la app**: un enlace, no un lanzador |
| 7 | **`get_best_tuning_params`** (:705-727) | script Python en f-string con `study_name` venido de un Textbox, ejecutado por `python -c`. **Inyección trivial**; síntoma benigno frecuente: un nombre con apóstrofo = `SyntaxError` invisible |
| 8 | **Config editor por `re.sub` sobre `src/core/config.py`** (:663-687, :990-1006) + `importlib.reload` en caliente | deja **el repo sucio** sin que el operador lo sepa. La capacidad vive (las 8 vars tienen consumidores reales: `lua/v2.0/training_env_client.lua:22,25,27`, `base_env.py:51`) pero **en JSON/TOML**, y etiquetada como **superficie del rig BizHawk**, no de la flota |
| 9 | **`gr.Timer(0.1, active=True)`** (:2133) | repinta 272 líneas de f-strings **10 veces por segundo, siempre, con la pestaña oculta** `[medido]` |
| 10 | **`_DASHBOARD_RELOAD_HEAD`** (:44-79) | 35 líneas de JS que parchean el cacheo de schema de Gradio. Workaround de una herramienta que se abandona: **no se porta** |

### 7.3 FUSIONAR, FUERA DE LA APP, FUERA DE v1

**FUSIONAR** (§2.3 y §3): 6 device + 5 zip + 5 pkl + 5 env + 3 algo → 9 instancias · 5 "nombre de
corrida" + 5 "timesteps" (+ `cfg_steps:2142`, hoy un default global sin relación visible) → un
formulario único · 3 consolas / 3 copiar / 4 refrescar / 5 parar → **uno de cada** · duplicados
literales: `toggle_league_matchup_mode` y `toggle_exploiter_matchup_mode` (:1365-1371,
**byte-idénticas**), `handle_league_state_upload:1373` vs `handle_state_upload:2287`,
`refresh_curr_btn:1762` (redundante con el `gr.Timer(5)` de :2197).

> **`.stop_training` SÍ es contrato real** — lo leen `auto_curriculum_callback.py:528`,
> `manual_curriculum_callback.py:419`, `stand_leia.py:676`, `test_agent_v2.py:220`,
> `test_ai_vs_ai_v2.py:343`. **Se conserva la parada suave; mueren los cinco botones.**

**FUERA DE LA APP:** series temporales, curvas, loss, beta, grads/s histórico → **W&B**, decisión
del equipo con fecha (`agent/memory/04-infra.md:44-47`, `02-decisiones.md:5`), y
`apex_learner.py:258-277` **ya lo publica ahí**. Dos excepciones nombradas: (a) W&B tira filas en
silencio con la cola llena y su eje X `grad_steps` **retrocede** al reiniciar sin `--resume-ckpt`
→ la consola debe distinguir *"hueco en W&B"* de *"hueco en el entrenamiento"*; (b) **PPO/BizHawk
no reporta a W&B** (`04-infra.md:20`), y ese hueco se cierra en `train.py` (~15 líneas), **no en
la UI**. Corolario: la consola muestra **estado instantáneo + liveness derivada** (Δ`grad_steps`),
no gráficas.

**FUERA DE v1 — con candado, no como eufemismo de "muerto":**

| Qué | Señales `[medido]` | Candado |
|---|---|---|
| **Optuna / tuning** | `models/tuning/` y `logs/tuning/` con **0 entradas**, sin BD de estudio; única mención en memoria es **en negativo** (`03-bugs-cazados.md:8`: *"lr=2.108e-05 de Optuna (tuneado bajo régimen roto) congeló la política en entropía máxima 1M steps"* — **el último resultado documentado de este flujo fue un daño**) | reentra con criterio escrito |
| **League + Exploiter** (~20 controles) | `models/production/league/` **vacío**; `grep -ril exploiter agent/memory/*.md` → **vacío**; y `06-pendientes.md:11` los clasifica: *"más adelante: liga/PvP **en flota**"* — la liga que renace no es ésta | idem |
| **Matchup SB3-vs-SB3** | `find models -type f` → **3 ficheros** (un `.zip`, su `_vecnorm.pkl`, un json de curriculum): **un solo modelo SB3**, y la rama IA-vs-IA (`run_matchup:820`) necesita dos. **Se conserva** la rama modelo-vs-humano/CPU (:826/:832) | no se borra la rama; se deja de exponer |
| **Telemetría de observación** (272 líneas, :1438-1710) | `write_telemetry` sólo lo llaman `test_agent_v2.py` y `test_ai_vs_ai_v2.py`; **`stand_leia.py` NO la llama** y `.telemetry.json` no existe en disco. Muere la **implementación**; sobrevive el **contrato**, y el panel vuelve el día que el motor emita telemetría | instrumentar `stand_leia.py` es trabajo de motor |
| **Banco de 12 rivales como tabla histórica** | **454 de 500 filas** dicen `apex_eval_tmp.pt` (`bench_12rivals.py:366` guarda el basename), **sin timestamp, sin id de corrida, sin dificultad** | v1 expone **lanzar** el banco y ver **el resultado de esa corrida** |
| **Campeones como ranking histórico** | 3 esquemas de sidecar incompatibles (uno con `wr_lvl1..4` cuya `wr_media` **no es comparable** con la de 8 tiers), 4 `.pt` huérfanos, uno con `weights_version` como **string**, y la coronación **no es atómica** (`torch.save` y luego `json.dump` sin tmp+rename) | **bloqueo aguas arriba**: se arregla en `tools/`, no en la UI |

> **Advertencia de honestidad, exigida por `31-judge-producto.md`:** la evidencia de desuso viene
> de git, del **disco de ESTA Mac** y de los runbooks. Diego y Santiago corren en
> Windows/Legion/desktop-4090 con sus propios `models/`. Es *"el equipo dejó de usarlos"*,
> **nunca** *"están rotos"*. **Preguntar en el canal es requisito previo a congelar la lista**, y
> "fuera de v1" sólo es válido si trae **criterio de reentrada o fecha escritos** — sin eso, en
> tres meses es "muerto" sin que nadie lo haya decidido.
>
> **Y dos correcciones a `27-option-matar.md` que hay que propagar** (las cazó
> `33-judge-riesgo.md`, las verifiqué yo, §0): (a) **`get_stand_checkpoint_status` NO tiene cero
> llamadores** — tiene un **test verde** en `test_telemetry_dashboard.py:436`, así que se
> **desconecta**, no se borra; (b) **es falso que ningún test toque `web_dashboard.py`**: son
> **dos** ficheros y ejercitan justo lo que hay que preservar (`get_model_files`,
> `stream_logs`, `run_matchup`, `get_stand_checkpoint_files`…).

### 7.4 Lo que renace, y EN QUÉ ORDEN

Para que el corte no se lea como demolición — y porque el riesgo medido nº1 de este proyecto
**no es una mala arquitectura, es re-derivar el mismo plan cada noche sin memoria del anterior**
(`agent/dashboard/PLAN-reconstruccion.md`, commit `781fe61f` de hace 9.5 h, puntuó **3/10** la
viabilidad de exactamente la migración que `22-option-arquitectura.md` volvió a proponer, y
**ninguna propuesta de hoy lo cita**).

**El diseño de esta hoja es completo. La entrega es por etapas, y las etapas están ordenadas por
"qué habría evitado el incidente de hoy".**

| Etapa | Contenido | Coste | Riesgo |
|---|---|---|---|
| **v1.0 — OJOS** *(sin UI)* | (1) `fleet/fleet.json` con `expected[]` · (2) **rescatar al repo el histórico** que hoy vive en `/private/tmp/claude-502/.../scratchpad/*.jsonl` con la ruta hardcodeada — una purga de `/tmp` borra las 12 coronaciones **antes** de que exista la UI que las dibujaría · (3) `tools/fleet_sampler.py` (stdlib, poll 10 s → JSONL **en el repo**), productor único de las derivadas · (4) **alarma que EMPUJA**, con la regla corregida de §4.5, histéresis 2/2, y **que no se suicide** | **~1 día** | **cero** |
| **v1.1 — WIRE** | los 5 campos aditivos C1-C5 de §4.7 | **~20 líneas** | aditivo; en reinicio planeado |
| **v1.2 — UNA PANTALLA** | `/flota` de sólo lectura: `FleetStatusBar` + grid por máquina esperada con fila fantasma + escalera viva | 3-5 días | bajo |
| **v2 — el resto de la app** | `/entrenar`, `/probar`, `/biblioteca`, `/ajustes`, `LogConsole`, `ResourcePicker` | ver §8 | medio |
| **v3 — plano de escritura** | fleet-agent con supervisión | 10 j+ | **alto**, §8 |

`grep -rn "ntfy|telegram|slack|webhook|smtp"` sobre `tools/` y `apex.py` → **cero**. **Dos líneas
ahí entregan más observabilidad real que las cinco pantallas juntas**, y la propia
`21-option-flota-ux.md` admite que *"el operador no mira la pantalla"*. Un observatorio que exige
presencia humana **no cierra E4**.

**El dashboard actual sigue vivo e intacto** (corre en :7861): nadie se queda sin herramienta,
porque no se le quita ninguna. **El jugable se queda en el Gradio actual** — funciona, tiene
tests, y es el único flujo que un runbook abre (`tools/RUN_STAND_LEIA.md:37`; de 7 runbooks vivos,
**sólo ése** menciona el dashboard).

**Parches al Gradio ACTUAL que no esperan a nada** (horas, sin reconstruir): vocabularios cerrados
`{ppo,dqn,apex}` y `{v2,v3,v4}` (mata hoy dos trampas que producen corridas fantasma) · 5 botones
de parada → 1 que **nombra a su víctima** · `gr.Timer(0.1)` → `active=False` cuando la pestaña no
está visible · bind del learner de `0.0.0.0` a la IP de tailnet (**1 línea**,
`apex_learner.py:84`) · pasar `--name` a los actores (**0 líneas de código**).

### 7.5 Correcciones de hecho que deben propagarse antes de implementar

| Documento | Dice | Verdad `[medido]` |
|---|---|---|
| `10-`, `11-`, `13-discover-*` | *"`theme=` en `.launch()` es un no-op; va en `gr.Blocks()`"* | **FALSO y peligroso**: en gradio 6.25.0 `Blocks.__init__` **NO** acepta `theme`, `launch()` **SÍ** acepta `theme` y `css`. **`web_dashboard.py:2320` está CORRECTO** — seguir a Discover **rompe el arranque** |
| `24-`, `26-option-*` | *"`VENV_PYTHON` está clavado a `.venv/Scripts/python.exe`, en esta Mac nada puede ejecutarse"* | **FALSO**: :24-28 es una cadena de fallback → `.venv/bin/python` → `sys.executable`. El impedimento real es **BizHawk/la ROM**, no el intérprete |
| `21-option-flota-ux.md` | replay ratio **9.29 / STARVED** | batch real **256** (`:96`), tope **acumulado** (`:240-241`); medido **5.01 / 2.39** → el learner está **limitado por cómputo**, no hambriento |
| `27-option-matar.md` | *"cero tests tocan `web_dashboard.py`"* | **FALSO**: dos (§0) |
| `31-judge-producto.md` | 126 ocurrencias / 24 hexes | mido **107 / 22** en `web_dashboard.py` |
| `25-option-tokens.md` | `alarm` oscuro `#EA4B3C` sobre su `-subtle` | mido **4.41 ✗** → subo el token de estado a `#EE5F51` (5.06) |

---

## 8. 🌅 Para Felipe

Ocho decisiones. Las cuatro primeras **bloquean trabajo**; las cuatro últimas se pueden dejar
correr pero se van a cobrar.

### 🔴 1. El lima `#C8EC40` en `--accent` es un bloqueador de token

**Es tu decisión, no de un agente** (contrato §5: cambiar el contrato de color es tuyo). En shadcn
`accent` **es** el token de hover/focus/highlight — verificado en el código que sirve el registry
hoy: `dropdown-menu` ×5, `button` ×3, `command`, `select`, `item`, `toggle`, `combobox`. Con el
lima ahí, **el día del primer `shadcn add`** cada botón fantasma y cada fila bajo el cursor se
pinta de lima, y la disciplina de "la firma se usa poco y donde importa" muere por construcción.

**Propuesta:** `--accent` → superficie neutra derivada de `--muted`; el lima pasa a token propio
`--champion` con sus 5 usos cerrados. **Ninguno de los 26 hexes cambia de valor**, sólo de nombre
de rol. ¿Lo autorizas?

**Y hay una segunda mitad de la misma pregunta:** el bloque real de tokens introduce **~58
valores** donde `00-DECISIONES §2` pinneó **26** — 16 de rampa de tier, 10 de `-subtle`, 6 de
`-fg`, 10 de chart. Todos **derivados** de tus hues, ninguno inventado, pero §2 es una decisión
cerrada **sobre valores**. ¿Los ratificas como derivación autorizada?

### 🔴 2. `fleet.json` no existe y todo depende de él

`find . -name "fleet*.json"` → **vacío** `[medido]`. Las siete propuestas lo declaran dependencia
dura. **Sin censo esperado, el colapso parcial de flota es indetectable por construcción** — y
está ocurriendo ahora mismo: **1 máquina fresca de 4, desde hace ~26 h**. Cuesta **30 minutos y
cero código**. ¿Lo escribimos hoy, antes de dibujar nada?

Va con una pregunta pegada: **el `node_id`**. La llave **no puede** ser el hostname (en WSL2
devuelve el nombre de Windows: `SSS`, `PEREA`, `LAPTOP-1VCCKQL9`) ni el nodo de Tailscale (no
viaja en el wire). Propuesta: **id opaco elegido por humano en el alta**, propagado por el flag
`--name` que ya existe. ¿Nombres? (sugerencia: `desktop-4090`, `legion`, `omen`, `mini`).

### 🔴 3. El histórico de la flota vive en `/tmp` y se puede borrar en cualquier momento

Las 48 filas del selector y las 197 del velador están en
`/private/tmp/claude-502/-Users-felipe/<id-de-sesión>/scratchpad/*.jsonl`, **con la ruta
hardcodeada en los scripts** `[medido]`. **Cualquier limpieza de `/tmp` borra la historia entera
de la flota** — las 12 coronaciones incluidas — antes de que exista la UI que las dibujaría.
Copiarlas al repo es trabajo de **minutos** y no debe esperar a ninguna fase. ¿Lo hago ya?

### 🔴 4. ¿Dónde vive el hub? (`desktop-4090` está rechazado)

`30-judge-arquitectura.md` lo rechazó con un hecho que verifiqué: **`ssh desktop-4090-ubuntu-wsl`
→ Connection refused**. No existe ningún camino remoto de administración a la máquina que
hospedaría el hub: si muere ahí, **sólo Santiago, físicamente frente a esa máquina, puede
revivirlo**. Además es WSL2 bajo el Windows de un tercero, su llave de tailnet expira el
**2027-02-23**, y hoy esa máquina ya es learner **+ único actor vivo**. Y la mitigación propuesta
("sampler espejo en la Mac") **ya falló en vivo antes de existir** — es exactamente la clase de
proceso lanzado a mano que hoy lleva 54 min muerto `[medido]`.

Opciones: (a) hub en la Mac, **supervisado por `launchd`** (no lanzado a mano); (b) esperar a que
haya una caja dedicada; (c) v1.0 sin hub — sólo sampler local + push. **Recomendación: (c) ahora,
(a) cuando haya más de un consumidor.** `madre` está excluida por ser EC2 (`00-DECISIONES:22`).

### 🟡 5. ¿Encendemos el plano de ESCRITURA del fleet-agent?

Diseñado (decisión #1 tuya, cumplida). **Recomendación de dos jueces: NO todavía.** Corre sobre
**laptops personales de compañeros** y sobre una run viva de **1.9 M grads** `[medido]`, y el
consentimiento local (`.fleet/node.json`) es **una promesa de código, no un mecanismo**: el agente
corre como el dueño y puede escribir el archivo que dice respetar. **El 80 % del valor (VER que la
flota cayó) no necesita ni una escritura.** Criterio de reentrada propuesto: v1.0 con **2 semanas
sin falsos positivos** + la conversación de confianza que pide `08-cola-manana.md:66`.

### 🟡 6. ¿Español o inglés en la superficie?

**Recomendación: español**, con un léxico técnico inglés **congelado** (~35 términos: learner,
actor, checkpoint, savestate, grad_steps, weights_version, replay ratio, win rate, Ape-X, PPO/DQN,
matchup, wandb…) envuelto en `translate="no"`. La mezcla de hoy no es aleatoria, es **por capa**:
los 119 labels son inglés al 100 % pero los **errores y tarjetas** —lo que más significado carga—
**ya son español** (:809, :865, :889, :900, :913; :354-384; :1997), y hay líneas bilingües dentro
de una sola cadena (:809 abre en inglés y sigue en español). Traducir 119 labels es mecánico;
retraducir los errores a inglés destruye matiz ya ganado. **Riesgo declarado:** la frontera se
erosiona en 3 meses si la lista cerrada no vive en **un archivo único de strings versionado**.

### 🟡 7. ¿Vale la pena el toolchain de build, y quién lo mantiene?

**No existe proyecto de frontend**: 0 `package.json`, sin `components.json`, sin `styles.css`, sin
`components/ui/`. Estimación honesta de `32-judge-construible.md`: **v1 completo ≈ 45 jornadas**
sin fleet-agent, **60+** con él — y **ninguna de las propuestas de UI costeó el backend** (B1 ≈ 5 j,
B2 ≈ 8-12 j y es el **único punto de no retorno**). Contexto duro: `git log --format=%an` da
**Perea 299 · Felipe 85 · Santiago 47** commits (+14 `LEIA-Tec`, +6 `Perea094`) `[medido]`, y la persona con 2/3 del repo **hace RL, no
frontend**.

Y un problema práctico: **node/bun sólo existen en esta Mac**; los 3 rigs son WSL2/Windows. Un
pipeline Vite ata cada corrección de UI a tu portátil, que es la máquina de presencia más frágil
de la flota. **Recomendación: v1.0 y v1.2 sin toolchain** (HTML+CSS vars planas sirviendo los
tokens de §6.5, que funcionan sin build); el toolchain entra con v2, cuando ya sepamos **qué
pantallas se miran de verdad**.

### 🟡 8. Cuatro huecos menores que necesitan tu palabra

1. **`warmup` del learner** no mapea a ninguno de los 5 estados (§6.5). ¿Se pinta `idle` con
   etiqueta "calentando", o abrimos un sexto estado? *(Abrirlo rompe el "la lista es cerrada", que
   es lo que evita la proliferación.)*
2. **~188 checkpoints**: ¿`DataGrid` plano o árbol? Recomendación: **grid**, porque el problema
   real es de **atribución**, no de jerarquía (454/500 filas del banco dicen `apex_eval_tmp.pt`) —
   un árbol sólo refleja el filesystem.
3. **`@kibo-ui`** como segundo registry (cerraría 4 huecos custom). **No es gratis**:
   `@kibo-ui/table` arrastra `jotai`. Añadir un registry cambia el canon del proyecto → tuyo.
4. **El polo claro**: ¿"soportado y degradado" o **fuera de v1**? En claro el lima es invisible
   (**1.28:1** `[medido]`), `tier-1` no llega a 3:1, y las series de gráfica son casi
   isoluminantes. Mantenerlo como par cuesta el doble de auditoría por una firma visual que ahí
   no existe.

---

### Anexo — riesgos que este plan asume conscientemente

- **El arnés oculta conducta.** Nada del rig BizHawk se ejecutó: los veredictos de §7.1 son
  estáticos (grep / lectura), no dinámicos. Son sólidos donde son **ausencias de dependencia o de
  sintaxis** (`ray`, `sac`, `taskkill`); "v1 cae al else" es lectura del `if/elif/else` y no lo
  probé corriendo.
- **Cero red de pruebas en la capa que se reescribe.** Hay **2** ficheros de test que tocan
  `web_dashboard.py` y cubren parcialmente lo que hay que preservar. Una regresión en el jugable,
  la parada suave o el lanzador clásico se detectará **a mano**.
- **`fleet.json` sólo aporta censo y estado deseado.** Regla dura de `30-judge-arquitectura.md`:
  **todo valor de runtime en pantalla viene del wire o se dibuja como "sin dato"**. `--difficulty`
  se declara en el plano de control y **no viaja en el wire** (`apex_actor.py:271-274`), y
  relanzar un actor a mano es la operación **normal** (`RUN_OMEN_DESDE_CERO.md §7`): si la UI
  pintara el valor declarado, **mentiría sobre el experimento en cuanto alguien haga lo que
  siempre hace**.
- **SSE no es camino probado.** `apex_actor.py:190-195` documenta que en red ajena *"el POST
  /transitions se queda en timed out con los GET pasando bien"* — un stream de larga vida es justo
  lo que un NAT de WSL2 corta en silencio, y `EventSource` no distingue "conectado y callado" de
  "conectado y sin novedad". **Latido explícito y fallback con ETag son obligatorios, no
  opcionales.**
- **Dead-man's switch.** La UI trata "sin muestra del sampler desde hace X" como **estado de
  primera clase**, nunca como página vacía; y sabe degradar a lectura directa de `/status`. Sin
  esto, el fallo observado hoy se repite con más código.
- **Fuera de alcance, anotado y no delegado a un `spawn_task`** (prohibido por contrato §8): el
  normalizador de procedencia de checkpoints (3 esquemas de sidecar + 4 `.pt` huérfanos + `sha256`
  en sólo uno de nueve) y la coronación atómica (`tmp` + `rename`) son trabajo de `tools/` que
  **bloquea** las pantallas de Campeones y del banco. Nadie las verá si nadie toca `tools/`.
