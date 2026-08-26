# Arquitectura (2026-08-26)

## Dos backends, un contrato
- **BizHawk** (Windows desktop, D:\GitHub\Street\street_fighter dentro de la carpeta BizHawk-2.8): EmuHawk + lua/v2.0/training_env_client.lua ↔ TCP lock-step ↔ src/envs/base_env.py. Payload CSV ASCII; anchos aceptados (13,24,26,27) con bloques `>=` (base_env.ACCEPTED_PAYLOAD_WIDTHS). Sirve: entrenamiento PPO, evaluación visual, PvP/liga, humano-vs-agente. ~350 agent steps/s/env, agregado ~1,160 con 16 envs (medido; NO escala más).
- **stable-retro** (Linux/WSL2/macOS): src/envs/retro_env.py (RetroSF2Env) + integración custom en retro_integration/ (data.json con 25 vars RAM, 212 savestates). Habla el contrato v4 EXACTO (23 floats/frame ×4; paridad bit-a-bit validada contra el parser real). ~3,700 fps/proc en M4 = ~900 agent steps/s/proc. Un emulador por proceso (límite libretro).
- **Diferencia de fase documentada**: BizHawk entrega la obs con 1 paso de retraso (pipelining deliberado del protocolo, ver nota en base_env.step); retro entrega la obs actual. Transferencia de políticas entre backends FUNCIONA (el campeón PPO ganó 88% en retro) pero con ese shift.
- Módulos compartidos puros: envs/reward.py (compute_reward + RoundTracker + hp_to_signed), action_macros, macro_wrapper, sf2_extractor, elo (sin conectar), selective_norm.

## La flota ES
- **Madre** (coordinador): EC2 en AWS cuenta educación, servicio systemd `leia-coordinator`, src/es/coordinator.py (stdlib http). Endpoints: /work /result /theta /status. Reparte chunks de miembros con arriendo+expiración+robo de trabajo+re-arriendo especulativo en la cola. Checkpoint por generación a disco + S3 (con RESTORE desde S3 al bootear). W&B por generación.
- **Workers** (src/es/worker.py): stateless, pull-based. `tools/run_worker.sh --coordinator http://madre:8080`. Auto-sizing (`--procs auto` = físicos−2; `--cpu-share 0.5`; `nice 10` en cada proceso emulador — ESA es la perilla de "máquina usable", no bajar procs). Reportan stats (steps/s, procs) que /status y W&B muestran por máquina.
- **OpenES** (src/es/openes.py): Salimans 2017 — pares antitéticos por semilla (el wire solo lleva θ+semillas), centered ranks, Adam. Política: MLP numpy [92→64→64→63] ~14k params (src/es/policy.py), argmax→divmod a MultiDiscrete.
- **Rotación de estados**: coordinador `--states manifest --difficulty N`; la lista viaja en cada lease y es identidad del run (pinneada en checkpoint, gana al CLI en resume). Estado por episodio derivado de la SEMILLA DEL PAR (openes.states_for_member, stream separado) → ambos lados del par antitético pelean la misma secuencia de rivales (common random numbers). Workers viejos: bancados por fingerprint (protocol.states_fingerprint) — actualizar workers antes de activar rotación.
- **Fitness** (openes.fitness_from_episode): win(1.0) + margen_hp/176·0.5 − min(0.0001·steps, 0.5). Lee info["win"] — depende de la semántica de rondas correcta.

## Por qué OpenES y no EGGROLL (no re-litigar sin datos nuevos)
EGGROLL = OpenES + perturbaciones low-rank para redes ENORMES; con 14k params el low-rank restringe exploración sin ahorrar nada (el cuello es el emulador ~99.9% del tiempo, no el álgebra). El swap vive contenido en openes.py si algún día crece la red. Paper: parity con OpenES en RL, no superioridad.
