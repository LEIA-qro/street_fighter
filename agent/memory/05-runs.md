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
| **ES flota gen-cero (26, EN CURSO)** | madre + M4 sola (--cpu-share 0.5), 12 rivales lvl1, pop 256, 2 eps | gen0: mean −0.113, best +1.393, 128s/gen, 2,832 steps/s. Rescate especulativo funcionó en producción. W&B: leia-qro-rl/leia-sf2-es |

Ritual de análisis: la desktop exporta tfevents a ~/Downloads de la M4; script de análisis por cuartos en la historia de la sesión (pandas + EventAccumulator).
