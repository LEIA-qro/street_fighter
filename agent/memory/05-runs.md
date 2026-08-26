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
