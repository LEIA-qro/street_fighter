# HANDOFF — street_fighter (LEIA)

**Escrito el 2026-09-11, al cerrar la limpieza. Si acabas de llegar a este repo, lee
esto completo antes de tocar nada. Son diez minutos y te ahorran una semana.**

Autores del trabajo que se describe aquí: Felipe (FelipeJackFox), Diego (Perea094),
Santiago (SantiagoSaldanaS), con agentes de Claude Code. Todo lo que se afirma abajo
está medido y tiene acta; cuando algo es opinión o sospecha, lo dice.

---

## 1. En qué quedó el proyecto

Enseñar a una IA a jugar *Street Fighter II' — Special Champion Edition* (Sega Genesis).
Hay **ocho niveles de dificultad × doce rivales**. El juego está esencialmente **resuelto**.

**El campeón es un Ape-X DQN.** `benchmarks/apex_milestones/apex_v3291_media990.pt`:

| lvl1 | lvl2 | lvl3 | lvl4 | lvl5 | lvl6 | lvl7 | lvl8 |
|---|---|---|---|---|---|---|---|
| 100% | 100% | 100% | 100% | 99.0% | 99.0% | 97.9% | 93.8% |

Eso es win rate de **rounds de apertura**, con desfase aleatorio de arranque (la vara
honesta), n=768 episodios. En **peleas COMPLETAS al mejor de tres en lvl8** hace
**~90%**: dos mediciones independientes de n=360 con semillas distintas dieron
91.7% [88–94] y 89.7% [86–92]. Sus únicos muros reales: **Balrog ~50%** (el único
rival sin proyectil, embestida pura), **Guile y E.Honda 65–77%**.

**El otro modelo es un OpenES en flota** — lo que en las conversaciones se llamó
"el eggroll primigenio". Vive en `src/es/`: una MLP numpy de ~14k–22k parámetros
entrenada por evolución (Salimans 2017, pares antitéticos, centered ranks, Adam),
con un coordinador en la nube y workers que jalan trabajo. Llegó a ~75–82% honesto
en el banco de 12 rivales de nivel 1. **Perdió la carrera contra el DQN**, con el
mismo entorno y el mismo reward, por un margen grande.

> **Por qué se llama "eggroll" y por qué NO es EGGROLL.** EGGROLL es OpenES con
> perturbaciones *low-rank*, pensado para redes enormes. A 14k–22k parámetros el
> low-rank restringe la exploración sin ahorrar nada: el cuello de botella es el
> emulador (~99.9% del tiempo), no el álgebra. La decisión de quedarse en OpenES
> está fechada y argumentada en `agent/memory/01-arquitectura.md` y `02-decisiones.md`.
> El swap vive contenido en `src/es/openes.py` si algún día la red crece. Hay un
> reporte de reconocimiento completo en `agent/recon-2026-08-25-rumbo-a-eggroll.html`.

**La tabla maestra**, siete modelos × tres condiciones, banco de 12 rivales lvl1
(win rate %, promedio de las tres condiciones):

| Modelo | Limpio | Desfase | Ruido 5% | Promedio |
|---|---|---|---|---|
| **DQN Ape-X, 90k gradientes (2.5 h)** | **100** | 91.7 | 95.8 | **95.8** |
| PPO campeón, 39.7M steps | 90.6 | **93.1** | 95.8 | 93.2 |
| "Glaber Xtreme", PPO 104.7M steps | 85.4 | 79.2 | 90.3 | 85.0 |
| PPO 31M | 74.0 | 80.6 | 81.9 | 78.8 |
| ES run 1, gen 95 | 83.3 | 63.9 | 76.4 | 74.5 |
| ES run 3, gen 113 | 83.3 | 79.2 | 56.9 | 73.1 |
| ES run 2, gen 754 | 75.0 | 75.0 | 65.3 | 71.8 |

El DQN ganó con **~50 veces menos cómputo** que el campeón PPO. Y ese Ape-X de 90k
gradientes es el ABUELO del v3291 de arriba: lo que sigue después es la run del
curriculum, que llevó el modelo de dominar el nivel 1 a dominar los ocho.

**Y un resultado que nadie pidió: el campeón CAMINA.** Todos los modelos históricos
saltaban ~45% del tiempo (saltar es óptimo contra los CPU de nivel bajo). El campeón
del curriculum tiene `air_frac` 0.20, y modula por rival (0.23 contra Balrog). Nadie
puso un incentivo para caminar: los anti-aéreos de los niveles 5–8 le quitaron el
vicio solos, y el estilo de piso bajó con él a todos los niveles.

---

## 2. Empieza aquí

```bash
git clone git@github.com:LEIA-qro/street_fighter.git
cd street_fighter
git checkout main          # ver §8: main es la verdad desde el 2026-09-11
python3.12 -m venv .venv
# torch NO esta en requirements: se instala aparte segun tu hardware
# (https://pytorch.org/get-started/locally/). CPU basta para actores y bancos.
.venv/bin/pip install torch
.venv/bin/pip install -r requirements.txt -r requirements-retro.txt
.venv/bin/python -m pytest code_testing/pytest -q     # deben pasar 630
```

Verificado el 2026-09-11 sobre un clon recién hecho, en una máquina **sin BizHawk en
ninguna parte**: los 630 pasan. Antes de esta limpieza no era así — `core/config.py`
lanzaba una excepción al importarse si no encontraba `EmuHawk.exe` en el directorio
padre, y nueve módulos de test morían en la recolección sin tocar BizHawk siquiera.

En Windows nativo **no** se puede instalar stable-retro (no publica wheels para
`win_amd64`): las dos laptops y la desktop lo corren dentro de WSL2. macOS y Linux
instalan nativo. Ver `requirements-retro.txt` y `tools/setup_worker.md`.

Necesitas el ROM en `roms/` con sha1 `a5aad1d108046d9388e33247610dafb4c6516e0b`
(cualquier otro dump NO sirve para stable-retro), e importarlo con
`python -m stable_retro.import roms/` — es `stable_retro`, no `retro`.

**Los cinco comandos que valen:**

```bash
# 1. Ver al campeón pelear, y grabarlo en video
.venv/bin/python tools/grabar_gauntlet.py --difficulty 8

# 2. Medirlo contra los 12 rivales (el banco de verdad, con desfase)
#    OJO: el brazo del Ape-X se llama "rainbow" (comparten la red QR-DQN)
.venv/bin/python tools/bench_12rivals.py --arm rainbow \
  --ckpt benchmarks/apex_milestones/apex_v3291_media990.pt \
  --difficulty 8 --desync-max 30

# 3. La interfaz local (dashboard Gradio) -- escucha SOLO en esta maquina
.venv/bin/python src/scripts/web_dashboard.py

# 4. Los ojos de la flota: censo, alarmas y consola web en :8099
.venv/bin/python tools/leia_hub.py --serve 8099

# 5. Jugar TU contra la IA (requiere el rig BizHawk en Windows)
#    Runbook completo: tools/RUN_STAND_LEIA.md
.venv\Scripts\python.exe src\scripts\stand_leia.py --opponent RANDOM
```

---

## 3. El mapa del repo

| Ruta | Qué es |
|---|---|
| `src/envs/` | Los entornos. `retro_env.py` (stable-retro, headless, el que entrena hoy) y `sf2_v1..v4.py` (BizHawk por TCP). `reward.py` y `action_macros.py` son compartidos y puros. |
| `src/agents/` | PPO, DQN, Rainbow-QR, Ape-X, liga, PBT. **`sac/` está deliberadamente muerto** (lanza `NotImplementedError` en la primera línea de sus dos entrypoints). |
| `src/es/` | La pista de evolución: coordinador, worker, OpenES, política numpy, protocolo. |
| `src/scripts/` | Lanzadores y `web_dashboard.py` (la interfaz). `stand_leia.py` = humano contra la IA. |
| `tools/` | Lo que se opera de verdad hoy: `apex_learner.py`, `apex_actor.py`, `bench_12rivals.py`, `grabar_gauntlet.py`, `leia_hub.py`, `forge_states.py`, y los runbooks `RUN_*.md`. |
| `benchmarks/` | Los modelos congelados (`apex_milestones/`) y las actas crudas de cada medición. |
| `retro_integration/` | La integración custom de stable-retro: 25 variables de RAM y **96 savestates** verificados (12 rivales × 8 niveles). |
| `fleet/` | `fleet.json` = el censo de qué DEBERÍA estar corriendo. `history/` = lo que pasó. |
| `agent/memory/` | **La memoria del proyecto. Ocho archivos densos. Léelos.** |
| `code_testing/pytest/` | 629 tests. Cada bug histórico tiene su regresión. |
| `infra/` | Terraform de la "madre" (EC2 coordinador del ES). |
| `doc/reconstruccion/`, `agent/dashboard/` | Análisis de la interfaz. **Planes CANCELADOS**, conservados como radiografía. Ver §6. |

**`agent/memory/INDEX.md` es el mapa vivo.** `agent/handoff.md` (69k) y
`agent/stage0-runbook.md` son de la era 2026-08-25: profundos y válidos como
referencia, pero anteriores a casi todo lo de arriba.

---

## 4. Dos backends, un contrato

Esto es lo menos obvio del proyecto y lo que más confunde al llegar.

- **stable-retro** (`src/envs/retro_env.py`) — Linux/macOS/WSL2, headless, ~3,700 fps
  por proceso. **Es donde se entrena.** Un emulador por proceso (límite de libretro).
- **BizHawk** (`src/envs/base_env.py` + `lua/v2.0/`) — Windows, un puente TCP lock-step
  con el emulador. **Se queda** para evaluación visual, PvP y humano-contra-IA. Nadie
  lo va a eliminar.

Los dos hablan el **contrato v4 exacto** (23 floats por frame × 4 frames), con paridad
validada bit a bit. Una política entrenada en uno funciona en el otro — el campeón PPO
ganó 88% en retro sin haber visto retro nunca. **La diferencia que hay que recordar:
BizHawk entrega la observación con un paso de retraso** (pipelining deliberado del
protocolo, no un bug).

---

## 5. El bug de seis meses, y por qué importa todavía

Del 2026-03-01 al 2026-08-26, **perder pagaba +55 de reward y ganar pagaba −12**.
La causa: se decidía "muerto" con `hp <= 0` sobre un entero sin signo, y se calculaba
el daño por diferencia de HP en el frame terminal. La verdad de la ROM (300k+ frames
medidos) es que **HP va de −27 a 176**, que `hp == 0` es una lectura VIVA que persiste
cientos de frames, y que el marcador de muerte es **el signo** (65535 = −1).

Se arregló con una única función de decodificación (`envs/reward.py: hp_to_signed`)
compartida por ambos backends, más un `RoundTracker` común. Después del fix, medido
sobre 150 episodios: ganar +138, perder −96, sin traslape.

**Por qué te importa:** todo resultado del proyecto anterior al 2026-08-26 se midió
bajo un régimen roto. Y `main` cargó con ese bug hasta esta limpieza. Los otros nueve
bugs cazados, todos con test de regresión, están en `agent/memory/03-bugs-cazados.md`.
**No los reintroduzcas.** `07-gotchas.md` tiene las trampas que ya costaron horas.

---

## 6. La interfaz: qué se mantiene y qué se canceló

**Se mantiene: `src/scripts/web_dashboard.py`** (Gradio). Es la interfaz local del
proyecto, cinco pestañas, y la única que algún runbook manda abrir. En esta limpieza
recibió once arreglos (commit `3575547c`): el campeón por defecto estaba desactualizado,
la tarjeta que muestra la escalera del modelo estaba huérfana, cuatro dropdowns ofrecían
un environment que sus scripts rechazan, servía en `0.0.0.0` sin autenticación, y
"Force Kill" no mataba nada fuera de Windows. Hay tests que impiden que vuelvan
(`code_testing/pytest/test_dashboard_contratos.py`).

**Se canceló: la consola React** (`consola-app/`, React 19 + shadcn + Tailwind, más su
bundle en `web/app/`). Decisión de Felipe, 2026-09-11. Salió del árbol; sigue en la
historia de git bajo el tag **`consola-react-cancelada`**:

```bash
git checkout consola-react-cancelada -- consola-app web/app
```

**Se queda lo que NO era interfaz:** `tools/leia_hub.py` son los **ojos de la flota** —
muestrea el `/status` del learner, compara contra el censo de `fleet/fleet.json`, alarma
cuando faltan máquinas y guarda la historia en el repo. Su pantalla es ahora
`web/consola.html`: una sola página, sin build, sin `node_modules`, servida en `/` y en
`/simple`. Es el único productor de las tasas derivadas (grads/s, trans/s, replay ratio).

Los planes de reconstrucción (`doc/reconstruccion/`, `agent/dashboard/`) llevan un aviso
de CANCELADO arriba. Se conservan porque el análisis que contienen — qué controles
mienten, qué ramas nunca se alcanzan, qué pestañas no pueden funcionar — es la mejor
radiografía que existe del dashboard, y sigue siendo cierta.

---

## 7. La flota y la infraestructura

Cuatro máquinas, coordinadas por Tailscale sobre el tailnet de la organización
(`leia-qro.org.github`, los tres owners son admins — no hay humano que sea punto único
de fallo):

| Máquina | Dueño | Papel |
|---|---|---|
| desktop "SSS" (i9-13900K / RTX 4090, Windows + WSL2) | Santiago | Hospeda el **learner**. 24/7. |
| Legion (275HX / 5070Ti, WSL2) | Diego | Actor, 40 procesos. |
| Omen (275HX / 5080M, WSL2) | Santiago | Actor, 40 procesos. |
| Mac mini M4 | Felipe | Actor **canario** (12 procesos) + observabilidad. |

**El canario importa:** cuando las máquinas grandes se movieron a los niveles 4–8, la
Mac quedó como la única que alimenta los niveles 1–3. Si se cae, el detector de olvido
queda ciego y las ventanas de lvl1-3 pasan a ser datos rancios.

Métricas en **Weights & Biases**, equipo `leia-qro-rl`, dos proyectos que no se mezclan:
`leia-sf2-dqn` (pista Ape-X) y `leia-sf2-es` (pista evolución). Solo la madre necesita
el API key.

**⚠️ Lo primero que alguien tiene que verificar: la "madre".** Es una EC2 `t3.small`
en `us-east-1`, cuenta AWS de educación (perfil `awsedu`, `800407728644`), creada con
el terraform de `infra/`. Es el coordinador del ES y **lleva ociosa desde el
2026-08-27**, esperando una run 4 que nunca se lanzó. No pude comprobar si sigue
encendida (el token de SSO estaba vencido). Si ya no se va a correr ES, **está cobrando
por nada**:

```bash
aws sso login --sso-session focaltec
AWS_PROFILE=awsedu aws ec2 describe-instances --region us-east-1 \
  --filters "Name=tag:Project,Values=leia-sf2-es" \
  --query "Reservations[].Instances[].[InstanceId,State.Name,LaunchTime]" --output text
# Para tirarla sin rastro: cd infra && terraform destroy
# (después, borrar a mano el nodo "madre" en la consola de Tailscale)
```

Todo lo de operación — cómo entrar a la madre, cómo lanzar un run fresco, dónde viven
los secretos, qué máquina falta dar de alta — está en `agent/memory/04-infra.md`.

---

## 8. Estado del repositorio

**`main` es la verdad desde el 2026-09-11.** Hasta esta limpieza no lo era: la rama
`stage0-metrics-and-semantics` llevaba **128 commits** de ventaja y `main` todavía
contenía el bug de reward de seis meses. Quien clonara el repo se llevaba la versión
rota. Esta limpieza se hizo sobre `stage0-metrics-and-semantics` y se integró a `main`.

Ramas viejas que puedes ignorar: `sf2-sota-rl-upgrade` y `docs/recon-rumbo-a-eggroll`
son ancestros de la rama de trabajo; `origin/Old` y `origin/refactoring` son de la era
anterior.

**El campeón ya está en git.** No lo estaba: `apex_v3291_media990.pt` — el modelo
ganador, resultado de tres días de cuatro máquinas — vivía **solo en la Mac de Felipe**,
igual que v1212, v781 y v511. Un disco muerto y se perdía. Ahora los cuatro hitos
versionados están en el repo, cada uno con su acta `.json` al lado.

Lo que NO se versiona, a propósito: `apex_escalera_best.pt`, `apex_curriculum_best.pt`
y `apex_best_desync.pt` son **alias móviles** que el selector reescribe mientras hay una
run viva. Verificado por sha256: cada uno es byte a byte idéntico a un hito que sí está
versionado. `leia_hub.py` cae al campeón congelado cuando el alias no existe, así que un
clon recién hecho tiene campeón que mostrar.

En la máquina de Felipe quedan dos directorios `.scratch-uiux-*` (~1 MB) con el trabajo
en bruto de los agentes que auditaron la interfaz. Están en `.gitignore` y **no viajan**;
lo que valía de ahí ya está versionado en `agent/dashboard/` y `doc/reconstruccion/`.
Se pueden borrar.

---

## 9. Qué sigue

La cola completa y priorizada vive en **`agent/memory/08-cola-manana.md`**. Lo grande:

**1. La run 2 del curriculum, con mezcla por MESH.** Es la mejora central que Felipe
especificó y que todavía no existe. Hoy la dieta de dificultades se define por máquina
("la desktop en 4–8, la Mac en 1–8"), lo que la hace invisible y frágil a relanzamientos
con la banda equivocada. La idea: un **vector de porcentajes por tier definido a nivel
flota**, que todos los actores muestrean por episodio, con **decaimiento por dominio**
(cuando el modelo le gana a un tier por encima de cierto umbral durante suficientes
evaluaciones, su proporción baja y el cómputo se va a donde todavía hay pelea) y un
**piso que nunca llega a cero** — los canarios dejan de ser una máquina y pasan a ser
una propiedad de la mezcla, que es más robusto: no mueren cuando muere una laptop.
Implementación natural: la mezcla viaja en la config de `/weights`, el controlador lee
los win rates por tier del banco del selector, los pisos y umbrales viven en `fleet.json`.

**2. Acelerar el learner. Vale más que sumar máquinas.** Medido el 2026-08-27: **8.7
gradientes/s** con 2,031 transiciones/s entrando, o sea un **replay ratio real de 1.10**
contra un tope configurado de 8. Cada transición se entrena una vez y el buffer la
recicla: la flota entera está sub-explotada. 115 ms por paso de gradiente en una 4090
con una MLP de ~1M parámetros y batch 256 — el tiempo no se va en la GPU. El sospechoso
principal es `ApexLearner._featurize`, que corre `expand_char_onehot` sobre 256 muestras
en Python puro por paso. Ganancia esperada: **3–8× más gradientes con el mismo hardware**.
Perfila primero (cProfile alrededor de `train_tick`, 200 pasos), y prueba en un learner
de juguete: esto toca el corazón de una run viva.

**3. El fix numpy del actor.** Los hijos actúan con un forward de torch por paso
individual: ~0.5–1 ms de overhead de despacho por consulta. Convertir el *acting* a
numpy puro (como ya hace `src/es/policy.py`; torch solo para cargar el `state_dict`)
vale ~2–3× de throughput por core.

**4. Los sentidos que faltan.** Balrog al ~50% es el mejor argumento que ha habido para
añadir canales a la observación: medidor de **stun** (no está), **Y de los proyectiles**
(solo hay X), **fase/frames del movimiento del rival** (hay ID, no timing). Ojo: cada
canal nuevo es una observación nueva, o sea runs nuevas. Va en paquete, no de a uno.

**5. El end-to-end del modo stand en Windows.** Está construido, revisado por 16 agentes
y commiteado; le falta una corrida real con BizHawk, ~5 minutos con el runbook.

---

## 10. Decisiones que esperan a un humano

1. **La madre (EC2).** ¿Se va a correr la run 4 del ES, o se destruye? Ver §7.
2. **Las pestañas muertas del dashboard.** Cuatro de las cinco pestañas no tienen uso
   documentado: League y Exploiter nunca se ejecutaron desde ahí (`models/production/league/`
   está vacío), PBT necesita `ray` (excluido de requirements a propósito), Optuna no
   tiene ni un estudio en el árbol y su único resultado histórico fue un desastre
   documentado. **No las borré**: quitarlas es una decisión de producto, no un arreglo
   de bug, y `refresh_dropdowns` devuelve diez `gr.update()` posicionales enlazados en
   cinco sitios — quitar un dropdown rompe el mapeo en silencio. Si se decide podar,
   es un commit y el análisis ya está hecho en `agent/dashboard/que-no-reconstruir.md`.
3. **El slider `WIN_RATE_THRESHOLD` no afecta al auto-curriculum.** `AutoCurriculumCallback`
   recibe el umbral por parámetro con default 0.75 y nunca lee `config.WIN_RATE_THRESHOLD`.
   Como `02-decisiones.md` dice literalmente "Plan B si estanca: bajarlo a 65%", alguien
   podría ejecutar ese plan moviendo el slider y **no pasaría nada, sin aviso**. O se
   cablea de verdad (como parámetro del lanzamiento, no reescribiendo el fuente), o se
   borra. Lo dejo señalado, no decidido.
4. **`core/elo.py` está cableado a nada.** Su único importador es su propio test.
5. **El README de la raíz** describe el proyecto de la era BizHawk/SB3. Sigue siendo
   cierto para ese backend, pero no es el retrato del proyecto. Le puse un encabezado
   que apunta aquí; reescribirlo entero es trabajo pendiente.

---

## 11. Las reglas de la casa

- **Identidad de git: `FelipeJackFox` / `felipaupz@gmail.com`, siempre.** El historial
  ya quedó partido en tres identidades una vez. Verifica `git config user.email` antes
  de tu primer commit.
- **`git pull` antes de lanzar cualquier cosa**, en todas las máquinas.
- **Un entrenamiento por máquina.** Regla de Felipe.
- **Tras desplegar código nuevo, RELANZA los workers y actores.** La banca por
  fingerprint deja esperando al código viejo, no lo mata.
- **`pkill` siempre acotado al venv del repo** (`street_fighter/.venv`). Hubo dos
  incidentes de daño colateral.
- **Lanza con `nohup` o `tmux`, nunca en la terminal del editor.** Cerrar el editor mató
  una run.
- **Los modelos guardados deben seguir cargando.** Cambiar la observación o el espacio
  de acciones = entorno nuevo, nunca modificación en sitio.
- **El campeón se SELECCIONA, no se toma de los pesos vivos.** El modelo churea: entre
  90k y 192k gradientes el número limpio cayó de 100 a 83 mientras el honesto se
  mantenía. Por eso existe el selector y por eso lo que va al stand es un archivo
  congelado con nombre de versión.
- **El número honesto es el de desfase**, no el limpio. Un banco greedy con estado fijo
  produce episodios idénticos: repetir no añade muestra. Toda acta lleva n e intervalo
  de Wilson.
