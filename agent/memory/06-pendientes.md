# Pendientes (prioridad aproximada)

1. **Enlistar Omen, Legion y desktop-WSL como workers** (tools/setup_worker.md) — multiplica la flota ×4-6. Solo falta lo humano.
2. **Leer el 15M-cont de la desktop** al terminar (~31M total): promociones, air_frac vs anti-airs, KL con lr 1.5e-4. Luego banco retro vs campeón (es_finetune_lastlayer.py --generations 0).
3. **Edición Lua de 3 líneas** (runbook §6.5, con auto-verificación de paridad): darle al rig BizHawk los contadores + reloj → detecta TIME OVER/DRAW en la desktop. Recomendado antes de runs >20M.
4. Rellenar estados lvl5-8 faltantes (granja con el mejor checkpoint nuevo como piloto: tools/farm_states.py).
5. Re-intentar **ES fine-tune** con headroom (retador vs estados donde gane 40-60%).
6. head-to-head visual campeón vs retador (test_ai_vs_ai_v2, comando en historia/05-runs).
7. W&B sync para PPO (SB3→wandb, ~15 líneas en train.py).
8. Vendorizar LICENSE de FightLadder junto a los estados (LOW del validador).
9. Vigilar workers version-mixta al activar rotaciones nuevas (bancados por fingerprint = esperado).
10. Diego: aceptar invitación W&B.
11. Más adelante: liga/PvP en flota (estados 2P de FightLadder ya catalogados), harness C/EnvPool-style si 25k steps/s se vuelve meta real, EGGROLL si crece la red.
