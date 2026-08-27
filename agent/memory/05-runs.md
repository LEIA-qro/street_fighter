# Bitácora de runs (números medidos)

| Run | Config | Resultado |
|---|---|---|
| "83%" histórico (pre-agosto) | 39.7M steps, curriculum viejo | Llegó lvl4 con conteo roto. ES EL CAMPEÓN actual: 88% wr / fitness +1.108 en banco retro (Guile lvl1-real... ojo: el estado shipped es dif 4). Checkpoint: models/latest/v3/ppo/. |
| Run A (2026-08-25 am) | lr Optuna 2.108e-05, 1M | Política congelada en entropía máx. NULO — dio el diagnóstico del lr. |
| Run A' (25 pm) | --lr 3e-4, 1M | Optimizador vivo; reveló el bug terminal (rew↑ con wr↓, double_ko 55%). |
| Run B (25 noche) | + fix semántica + --ground_gate --no_anneal_lr, 1M | **Primer run exitoso**: wr real 56→64% subiendo, timeouts 0, outcomes suman 1.0, dist mediana 86→71. air_frac ~0.43 plano (PBRS no cambia ruta óptima). |
| Overnight (26) | resume B, +15M (a 16M) | **Promoción legítima a lvl2** (+8 de lvl3). wr rolling 71-72%. dist ~59-61. air ~0.45 plano incluso vs lvl2-3. KL subió 0.019→0.043 ⇒ enfriar lr. ~1,200 steps/s. En banco retro Guile: 58% (el estilo saltarín sufre vs proyectiles). Checkpoint: models/production/v3/ppo/ppo_v3_autocurrTest27_lvl2_plus8_final_WR71pct_16023552steps.zip |
| 15M-cont (26, EN CURSO en desktop) | resume lvl2, --lr 1.5e-4 | rumbo a ~31M. Preguntas: ¿lvl3-4? ¿air_frac cae con anti-airs? ¿KL baja? |
| ES fine-tune última capa (26, M4) | campeón congelado, 30 gens vs Guile | NULO instructivo: baseline en techo ⇒ ES degrada (+1.108→+0.887). Receta: headroom + más eps/miembro + élite. tools/es_finetune_lastlayer.py |
| **ES flota gen-cero (26, EN CURSO)** | madre + M4 sola (--cpu-share 0.5), 12 rivales lvl1, pop 256, 2 eps | gen0: mean −0.113, best +1.393, 128s/gen, 2,832 steps/s. Rescate especulativo funcionó en producción. Curva mean: −0.11→0.57 (gen13)→~0.83 (gen50s), gens acortándose 128→44s (gana más rápido). W&B: leia-qro-rl/leia-sf2-es |
| Banco 12 rivales (26, gen 64) | tools/bench_12rivals.py, verificado adversarial (3 lentes PARITY_OK) | **ES θ: +0.835 / wr 66.7%** — bimodal puro: domina 8/12 con fitness ~1.2-1.49 (Blanka, EHonda, Dhalsim, Guile, MBison, Ryu, Vega, Zangief), pierde 0% vs Balrog/ChunLi/Ken/Sagat. **Campeón PPO en el MISMO aro: +1.152 / wr 90.6%** (débil: EHonda 4/8, Sagat y MBison 6/8). ES ya le gana en 5 matchups (EHonda por paliza +1.483 vs +0.512). ⚠️ El ancla histórica 88%/+1.108 era el Guile shipped a dificultad 4 — NO comparable con estos lvl1; la comparación limpia es brazo-vs-brazo de este banco. Fingerprint rotación 06132d47fd5b7b22 == madre (verificado contra su startup log). Repetir `--arm es` cada ~100 gens para graficar la persecución. |
| Banco gen 74 — **CICLADO de estrategias** | mismo banco, 10 gens después | wr limpio IGUAL (66.7%) pero rotó rivales: craqueó Balrog/Ken/Sagat y PERDIÓ MBison/Ryu/Vega (ChunLi sigue invicta). El mean poblacional sube (~0.95) por victorias más limpias, no por más rivales. **Hipótesis arquitectónica: el char ID del rival entra como escalar /15 (frame v4 idx 22) — el MLP 64x64 no puede ramificar por matchup; el PPO lo recibe one-hot de 16.** Cambio propuesto: policy ES con char one-hot (frame 23→53, obs 212, ~22k params) = run nueva; momento natural, cuando entre la flota. |

Ritual de análisis: la desktop exporta tfevents a ~/Downloads de la M4; script de análisis por cuartos en la historia de la sesión (pandas + EventAccumulator).

## [2026-08-26 tarde] Cierre run 1 + arranque run 2 + Glaber

- **Run 1 (escalar) CERRADA en gen 96.** Acta final (banco gen 95): **+1.079 / 83.3% (10/12)** — salió del ciclo 8/12: recuperó Ryu y Vega sin soltar Balrog/Ken/Sagat; solo Chun-Li y M.Bison invictos. El ciclado NO era techo duro, era interferencia lenta (Bison entra/sale, Dhalsim degradó 1.48→1.01). De −0.11 a +1.08 en ~2h de una M4 sola.
- **Modelo sorpresa del equipo: "Glaber Xtreme V1"** — PPO v3 de **104.7M steps, llegó a lvl5**, WR rolling 76% (archivos en ~/Downloads de la M4). En el banco 12 rivales lvl1: **+1.044 / 85.4%** — PEOR que el campeón de 39.7M (+1.152/90.6%) en lvl1 (se especializó arriba; este banco solo mide el piso). ES gen 95 quedó a 0.07 de fitness del 100M.
- **Run 2 lanzada: policy v4onehot** (char IDs one-hot, 21887 params, obs de wire igual 92). Fresh start verificado, S3 `es-run2-onehot/`, W&B run `pious-sea-6`. M4 al 80% (8 procs, ~3,700 steps/s). Hipótesis a validar: sin interferencia de matchup, debería acumular rivales en vez de rotarlos y pasar 10/12.
- Pendientes de la verificación adversarial (menores): theta_cache del worker puede cruzar runs si colisionan números de gen en swap de coordinator (mitigado hoy: worker reiniciado a mano; fix real = nonce de run en el wire); load_checkpoint no valida dim vs policy (falla ruidoso en worker de todos modos).

## [2026-08-26 tarde] Prueba de robustez — el 10/12 de run 1, bajo el microscopio

bench_12rivals.py ahora tiene `--action-noise P` y `--desync-max K` (perturbaciones seedeadas por episodio, mismas para ambos brazos; con perturbación=0 reproduce lo limpio bit a bit). Resultados (θ final run 1 vs campeón 39.7M, 72 eps/condición):

| Condición | ES run1 | PPO campeón |
|---|---|---|
| Limpio | +1.079 / 83.3% | +1.152 / 90.6% |
| Ruido 5% | +0.894 / 76.4% | +1.254 / 95.8% |
| Ruido 10% | +0.861 / 73.6% | +1.144 / 90.3% |
| Desfase ≤30f | **+0.785 / 63.9%** | +1.188 / 93.1% |

**El PPO es a prueba de balas** (entrenó estocástico; ninguna perturbación lo mueve). El ES pierde 19 pts de wr con solo desfasar el arranque ≤30 frames. Por rival bajo desfase: 7 victorias son ESTRATEGIA real (Balrog, Blanka, Guile, Ken, Ryu, Vega, Zangief: 6/6 todas), 3 eran coreografía total o parcial (**Sagat 0/6, Dhalsim 1/6, EHonda 3/6** — la "paliza" a EHonda era mitad guión). Implicación: NO tocar la run 2 (experimento de una variable: one-hot); si run 2 también sale frágil, la palanca de run 3 es **desync aleatorio en entrenamiento** (domain randomization barata). Protocolo de banco de ahora en adelante: reportar limpio + desync en cada checkpoint.

## [2026-08-26 tarde-2] Run 2 gen 42 + fix de chunks + flota 2 máquinas

- **Banco run 2 gen 42 (v4onehot): 66.7% limpio / 65.3% desync — brecha ~1pt, robusto de fábrica.** ¡CHUN-LI Y M.BISON CAYERON! (+1.424 y +1.371, los imposibles históricos del escalar). Pierde: Balrog, Dhalsim, Ryu, Sagat. En números ROBUSTOS, gen 42 de run 2 ≈ gen 95 de run 1. Protocolo permanente: doble curva (limpio+desync) por checkpoint; la métrica de fiteo es LA BRECHA entre ambas (caveat de Felipe: puede ser general ahora y fitearse después).
- **Cuello de botella cazado: chunk_size 8 limitaba a CUALQUIER worker a 8 miembros concurrentes** — la Legion (22 procs) trabajaba al 36% (4.2k steps/s de sus 16k). Fix: `--chunk-size 24` en el unit (resume del checkpoint sin pérdida; chunk size NO es identidad del run). Resultado: **15s/gen, 12.4k steps/s de flota** (Legion 8.6k + Mac 3.8k), reparto 160/96. Pendiente el fix estructural: worker que tome ceil(procs/chunk) leases concurrentes para autollenarse en cualquier máquina.
- El env NO cambió con v4onehot (pregunta de Diego): obs 92 y MultiDiscrete([9,7]) congelados; el one-hot es feature map interno de la política.

## [2026-08-26 tarde-3] Run 2 gen 152 + track Rainbow armado

- **Banco run 2 gen 152: 75.0% limpio / 73.6% desync (brecha ~1pt, sigue robusto).** 9/12: sumó Dhalsim y Ryu, SOLTÓ EHonda (acumulación neta con rotación residual — mejor que el escalar puro pero no monótona). En pie: Balrog, EHonda, Sagat. En números HONESTOS (desync) la run 2 a gen 152 (73.6%) ya supera el mejor honesto de run 1 (~64%).
- **Track Rainbow-lite (QR) construido** (pedido del equipo; política de flota de Felipe: UN entrenamiento a la vez por máquina): agents/rainbow.py (QR+Double+Dueling+PER+n-step, puro), envs/discrete_sf2.py (Discrete(63) mismo divmod del ES, sorteo de estado por episodio, desync de entrenamiento default 30, char one-hot), tools/train_rainbow.py, brazo `--arm rainbow --ckpt` en el banco. QR en vez de C51 a propósito (no exige v_min/v_max). 513 tests.
- **Gotcha cazado en vivo: gymnasium 1.3 usa autoreset NEXT_STEP** — final_observation/final_info YA NO EXISTEN; el step terminal trae obs terminal + info real (dict de arrays con máscaras `_key`), y el step siguiente es el reset cuya transición hay que SALTAR (just_reset flag). El primer smoke reportaba wr 0.00 por leer las keys viejas. Probe con spawn exige `if __name__ == "__main__"`.

## [2026-08-26 noche] Run 2 gen 407 — meseta robusta; protocolo corregido

- Curva honesta (desync): 65.3 (g42) → 73.6 (g152) → **72.2 (g407) = MESETA**. Limpio se desplomó a 58.3% pero es ruido de trayectoria fija: **desync>limpio (72.2 vs 58.3) enseñó que el banco "limpio" es UNA película por rival (varianza alta de moneda fija); EL NÚMERO BUENO ES EL DE DESFASE.** Protocolo corregido: reportar desync como métrica primaria.
- Paredes duras tres bancos seguidos: **Balrog, EHonda, Sagat (0 victorias)**; Bison/Ryu bailan en la frontera. ‖θ‖ crece sin freno (17.6→23.0→32.3; wd 0.005 no contiene). Mean poblacional oscila 0.6-0.8 sin tendencia desde ~g120.
- **Palancas run 3 (orden de convicción): eps/miembro 2→4 (gradiente ruidoso), desync EN entrenamiento (worker), pop 512 cuando entre la desktop, wd mayor.** Todo flags existentes salvo el desync del worker + fix multi-lease.
- Plan de Felipe: ES llegó a 400 → Rainbow se prueba en la Legion (3M steps, 16 envs, GPU); desktop sigue su 31M y al acabar entra de worker. Mac sigue ES sola mientras.

## [2026-08-26 noche-2] Run 2 CERRADA en gen 754 — acta de randomness final

- La "meseta" de g407 era transitoria: cerró subiendo (mean poblacional ~0.82; banco g751: desync 81.9%, Sagat cayó, EHonda regresó). Archivo: benchmarks/run2_final/ (theta gen 754 + checkpoint), S3 es-run2-onehot/ intacto, coordinator y workers detenidos limpio (madre viva para run 3).
- **Matriz de randomness θ754**: limpio 75.0 / desync 75.0 (**brecha CERO — cero coreografía**, vs −19pts de run 1) / ruido 5% 65.3 / ruido 10% 66.7 (**sin dosis-respuesta**: el primer ruido rompe combos, más ya no empeora) / combinada 66.7. Número honesto de run 2: ~75-80% (varianza muestral ±5). El PPO campeón sigue siendo inmune al ruido (90.3% con 10%) por entrenar estocástico.
- Palanca extra run 3 confirmada por esto: ruido de acciones EN los episodios de evaluación del ES (además del desync), para presionar la motricidad fina.

## [2026-08-26 noche-3] RUN 3 LANZADA — "ni una partida igual"

- **Config**: v4onehot + eval_desync_max 30 + eval_action_noise 0.05 + 4 eps/miembro + wd 0.01, S3 `es-run3-perturbed/`, W&B `sandy-pine-8`. Mac SOLA (8 procs, ~3.9k steps/s, ~166s/gen con los 4 eps) mientras la Legion corre Rainbow. La desktop se suma al terminar su 31M.
- **Mecanismo nuevo (commit 5b3b1bda)**: perturbaciones sorteadas del seed del PAR con stream por episodio (gemelos antitéticos sufren idénticas; el ruido se cancela en la diferencia del par). Identidad de run anclada en checkpoint, viaja en el lease ("eval"), echo obligatorio (eval_fingerprint) — worker viejo = rechazo ruidoso. Validado en producción con el primer chunk aceptado.
- **OJO al comparar curvas**: el fitness de run 3 es la función objetivo ROBUSTA (bajo ruido+desfase) — sus números serán más bajos que los de run 2 A IGUAL HABILIDAD (el 65-67% de run 2 bajo ruido es la vara inicial, no el 75 limpio). El banco sigue midiendo con las mismas 5 condiciones para comparar de verdad.

## [2026-08-27 00:20] Run 3 gen 53 — RÉCORD del proyecto en limpio

Banco triple de θ53 (Mac sola, ~2h): **limpio +1.179 / 91.7% (11/12) — ARRIBA del campeón PPO (90.6%)** y de todo lo previo; desfase +0.920/76.4% (≈ run 2 final en 1/14 de las generaciones); ruido 5% +0.610/59.7% (su eje débil, pero es EL que entrena — debe subir). theta/norm 18.2 contenida (wd 0.01 funciona). Lección clave: entrenar con "ni una partida igual" ACELERA (cada episodio perturbado es información nueva, el gradiente ve el matchup entero). Próximo banco: gen 150.

## [2026-08-27 ~01:00] Track Ape-X construido y verificado (commit f93e8780)

Rainbow-QR **distribuido** listo: `tools/apex_learner.py` (máquina GPU: replay PER central + HTTP; config del run viaja en /weights) + `tools/apex_actor.py` (una por máquina: emuladores + epsilon escalera + n-step local + lotes fp16 comprimidos + pesos frescos cada 5s). Verificado adversarial OK×3; smoke e2e 10.4k transiciones/311 grads. **Gotchas cazados**: fp16 clip anti-inf (canales >u2 sin clip upstream), seeds únicos por máquina (sin duplicados M-plicados), POSTs de tamaño fijo + cola acotada (caídas largas), recarga ruidosa + config check (drift de objetivo), Queue maxsize ≤30k en macOS (SEM_VALUE_MAX 32767). Despliegue: learner en Legion (`--wandb-project leia-sf2-es`, tag dqn automático), actores `tools/apex_actor.py --learner http://legion-wsl:8090`. UN entrenamiento por máquina (regla de Felipe); Mac de Felipe reservada al ES. Dashboards separados por vista W&B: ES nw=kmy9403tzga · DQN nw=cbbffbvwg3t.

## [2026-08-27 ~02:30] APE-X DESPEGÓ — nuevo subcampeón honesto (87.5%)

A ~79k gradientes (~2h en la Legion sola, learner+actor local): **limpio +1.199/91.7% (11/12, fitness ARRIBA del campeón) · desfase 87.5%** — 2º lugar honesto del proyecto, aplastando al ES (~75-82) con el MISMO env. Chun-Li (+1.444) y Balrog cayeron; solo Sagat resiste. wr de comportamiento 64.6% (con epsilon; el greedy es lo de arriba). SIN macros aún. Truco operativo: los pesos vivos se jalan de GET /weights del learner por la tailnet y se arman en ckpt benchable sin tocar la máquina de Diego (script en historial; meta desde la config del payload). La tesis "es el env" pierde fuerza: mismo env, algoritmo distinto, +12pts honestos — el replay/off-policy exprime la experiencia como ES/PPO-onpolicy no pudieron.

## [2026-08-27 ~03:00] 🏆 PLENO PERFECTO — Ape-X 12/12 a 90k gradientes

**Primer modelo en la historia del proyecto que barre los 12 rivales limpios: 24/24, fitness +1.313 (récord absoluto del banco). Desfase honesto: 91.7% (+1.181) — a 1.4pts del campeón (93.1%).** ~2.5h de entrenamiento en la Legion sola, SIN macros. Sagat (+1.034) fue el último en caer. Checkpoint archivado: benchmarks/apex_milestones/apex_grads90k_PLENO12de12.pt (ojo: los .pt de torch son zips — las descargas a veces les añaden .zip; se renombra y listo). Taildrop entre usuarios NO funciona ("peer owned by different user") — los checkpoints se comparten por http.server en la tailnet o el /weights vivo del learner.

## [2026-08-27 ~04:00] TABLA MAESTRA FINAL — ES run 3 cerrada, todo medido

ES run 3 CERRADA en gen 113 (acta en benchmarks/run3_final/: theta+ckpt+metrics.jsonl); madre ociosa lista para run 4; Mac libre. **Matriz completa 7 modelos × 3 condiciones (wr / fitness):**
| Modelo | Limpio | Desfase | Ruido5% | Prom |
|---|---|---|---|---|
| **DQN Ape-X 90k (2.5h)** | **100/+1.313** | 91.7/+1.181 | 95.8/+1.247 | **95.8** |
| PPO campeón 39.7M | 90.6/+1.152 | **93.1**/+1.188 | 95.8/+1.254 | 93.2 |
| Glaber 104.7M | 85.4/+1.044 | 79.2/+0.986 | 90.3/+1.154 | 85.0 |
| PPO 31M | 74.0/+0.858 | 80.6/+0.931 | 81.9/+0.975 | 78.8 |
| ES3 g113 | 83.3/+1.015 | 79.2/+0.985 | 56.9/+0.570 | 73.1 |
| ES1 g95 | 83.3/+1.079 | 63.9/+0.785 | 76.4/+0.894 | 74.5 |
| ES2 g754 | 75.0/+0.924 | 75.0/+0.909 | 65.3/+0.691 | 71.8 |

**DQN líder en promedio (95.8) con ~50× menos cómputo que el campeón; campeón conserva solo la corona del desfase (+1.4pts).** Glaber pierde con desfase (79.2 — tiene algo de coreografía). ES3: robusto al desfase (−4pts) pero débil al ruido (113 gens no alcanzaron a entrenarlo). Reporte para el equipo (artifact): https://claude.ai/code/artifact/cec6dc95-28b1-4cb6-9c62-0a40fbad8531 — siguiente: Ape-X a dificultades altas + estrenar macros.

## [2026-08-27 ~05:00] Mac de actor DQN + selección automática + lección de churn

- **La Mac entró de actor Ape-X** (8 procs, ~3.5k steps/s) — flota de actuación ~7k steps/s hacia el learner de la Legion. Regla de un-entrenamiento-por-máquina intacta (mismo entrenamiento, más cuerpos).
- **Lección DQN medida**: entre 90k y 192k grads el limpio cayó 100→83.3 pero el desfase se mantuvo 91.7 (fit subió) — la política CHUREA con el entrenamiento continuo; el checkpoint bueno se SELECCIONA. El pico sigue siendo 90k (archivado).
- **Selector automático corriendo en la Mac**: cada 30 min jala /weights del learner, banco desfase (72 eps), bitácora en scratchpad/apex_selector.jsonl y guarda el mejor en benchmarks/apex_milestones/apex_best_desync.pt(.json). El campeón DQN se elige solo.
- `win_rate_recent200` agregado al learner (visible al relanzarlo). air_frac del DQN: 0.432 medio pero MODULADO por matchup (0.23 vs Balrog — primer modelo que decide cuándo no brincar).

## [2026-08-27 ~00:00] LA RUN DEL CURRICULUM VIVA — cierre de la noche

**Learner en la desktop 4090 (24/7, Santiago la onboardeó): 72 acciones (MACROS ESTRENADOS), buffer 1M, wandb `rainbow-apex-curriculum`.** Actores: desktop 44p + Legion 30p + Mac 12p (sobre-suscritos: los hijos cargan ~35-60%/core por el overhead de torch por paso — diagnóstico y fix numpy en 08-cola). Curriculum UNIFORME 1,2,3,4 en todos. **Escalera a 15 min de nacer: lvl1 85.5 / lvl2 74.5 / lvl3 65.5 / lvl4 36 — perfectamente ordenada y toda subiendo; con macros aprende ~5-10× más rápido que la run lvl1 a la misma edad.** Selector v2 multi-tier corriendo en la Mac. Actas archivadas: pleno 90k + final 286k de la run lvl1. Runbooks por rol en tools/RUN_LARGA_*.md. La cola completa de mañana: **08-cola-manana.md**.

## [2026-08-27 ~02:30] LA MADRUGADA DE LA FORJA — malla 12×8 completa + learner congelado

**La malla de savestates está COMPLETA: 12 rivales × lvl1-8 = 96 estados** (28 forjados esta madrugada, 240/240 en verify_states). La historia científica en tres actos:

1. **El poke ingenuo FALLÓ como debía**: `0xFE45` (el byte del menú) persiste en pelea pero es COSMÉTICO — la escena copia la dificultad a parámetros AI-locales al cargar. La validación conductual (tools/validate_forged_states.py: 3 rivales × 4 brazos × 40 eps, campeón DQN + desfase, tests de permutación) lo cachó de inmediato: el forjado jugaba IDÉNTICO a su donante (Zangief forjado-lvl7 wr 100% = donante lvl1 100%; p vs donante 0.5-0.97). Sin ese experimento habríamos envenenado el curriculum con 28 etiquetas falsas.
2. **La caza del estado real**: diff masivo de blobs sobre los 68 estados auténticos (12 rivales, 8 niveles) → los ÚNICOS bytes función pura del nivel: **`0x97B2` (la copia en-pelea, = nivel−1 en los 68), `0x96B8` (derivado no lineal {0,0,0,32,48,96,128,255}), `0xBA35/38/58` (derivado lineal triplicado)** + el 0xFE45 del menú.
3. **Poke de los 6 bytes → PASS rotundo**: forjado ≈ etiqueta (Zangief 47.5% vs 45.0% auténtico, p=.89; Chun-Li 12.5% vs 17.5%, p=.76) y forjado ≠ donante (p<.0001) en 3 rivales × 2 direcciones. --fill forjó las 28 celdas (donante = nivel auténtico más alto del rival), manifest con procedencia + `difficulty_ram_in_fight` documentado.

Herramientas nuevas: `tools/forge_states.py` (forja gateada por el veredicto del reporte), `tools/validate_forged_states.py` (el experimento, reusa el banco), +19 tests. Dato regalado por el control del experimento: **el campeón del curriculum ya le gana 45% a Zangief lvl7 y 80% a Dhalsim lvl7** — transferencia hacia arriba real.

**⚠️ Y el incidente: el learner se CONGELÓ en grads 33103 (~00:40)**, poco después de llenarse el buffer — HTTP vivo, ingesta corriendo, cero gradientes (bloqueado a media iteración fuera del lock; sospechoso #1: wandb.log colgado). SSH a la desktop: rechazado — irrecuperable hasta que despierte Santiago. **Fix commiteado** (wandb a hilo daemon con cola drop-on-full), reinicio documentado en RUN_LARGA_SANTIAGO.md §5, velador v2 armado. Escalera al momento del congelamiento: lvl1 .96 / lvl2 .93 / lvl3 .90 / lvl4 .56 — 20 puntos arriba del arranque en lvl4, cero mínimo cobarde.

## [2026-08-27 ~07:30] EL PLENO TRIPLE — v511 barre el banco canónico

**El campeón del curriculum (v511, ~289k grads) hace 100% EN LAS TRES CONDICIONES del banco canónico lvl1** (216/216 episodios): limpio 100/+1.454 (récord de fitness), desfase ≤30 100/+1.415 (le quita al PPO la última corona), ruido 5% 100/+1.420. Y el mismo modelo hace 85.4% en lvl4 (media multi-tier 96.4, selector). Congelado en `benchmarks/apex_milestones/apex_v511_media9640.pt`; acta jsonl en `benchmarks/bench_v511_canonico.jsonl`. La noche completa del learner: congelamiento por captura de consola de wandb (~1.5h zombi, 0 grads perdidos, 3 parches) y de ahí ~12/s sin interrupciones hasta 302k grads. lvl1-3 saturados en el banco → recomendación en pie: extender actores a --difficulty 1..8 (estados listos y validados).

## [2026-08-27 ~08:45] La escalera del v511 — la estrategia VIAJA

Banco canónico (3 condiciones) × niveles 1-6 con el v511 (wr promedio por nivel): **lvl1 100.0 / lvl2 99.7 / lvl3 92.7 / lvl4 74.3 / lvl5 66.3 / lvl6 61.8** — monótona, sin acantilados. En lvl5-6 (JAMÁS entrenados) desfase ≈ ruido ≈ limpio: generalización real, no coreografía. vs el campeón lvl1-only de ayer (50% lvl5, 31% lvl6): el curriculum enseñó a pelear, no a memorizar. Actas: benchmarks/bench_v511_escalera.jsonl (+ canónico en bench_v511_canonico.jsonl). Artifact del equipo actualizado (misma URL). Felipe pidió esta prueba ANTES de soltar los tiers altos → luz verde técnica para extender actores a 1-8.
