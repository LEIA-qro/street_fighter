# Decisiones (fecha + porqué). No re-litigar sin datos nuevos.

- 2026-08-25 **Rumbo**: flujos A(instrumentar)+E(semántica) primero; ES en flota condicionado a backend rápido. Guile-solo como run insignia: CANCELADO por Felipe — sin curriculum parejo no hay comparación válida.
- 2026-08-25 **Backend headless = stable-retro** (Genesis Plus GX). BizHawk se QUEDA para evaluación/PvP/humano — nadie lo elimina. El proyecto había empezado en gym-retro y lo abandonó en marzo; volvimos con contratos de paridad.
- 2026-08-25 **Madre en la nube** (VPS chico cuenta edu, terraform desechable) — pedido explícito de Felipe. Métricas: W&B. Red: tailnet de ORG (no personal) para no tener humano-SPOF.
- 2026-08-25 **OpenES, no EGGROLL** a 14k params (ver 01-arquitectura). EGGROLL entra si/cuando la red crezca.
- 2026-08-26 **Carrera PPO vs ES** con ambientes casi idénticos (mismo reward, misma obs v4/contrato, mismos rivales por nivel). PPO = baseline compuerta 5 del handoff.
- 2026-08-26 **Pipeline etapa 2** (ES afina última capa de PPO congelado): probado en miniatura — NULO con baseline en techo (88% vs Guile lvl1); prerequisito real = estados con headroom (baseline 40-60%). Ya hay estados; re-intentar con el retador cuando toque.
- 2026-08-26 **lr para PPO largo**: 3e-4 desatora (Run B); a 16M+ steps enfriar a 1.5e-4 constante (KL subía con clip al tope). --no_anneal_lr siempre (anneal a cero mata la cola del run).
- 2026-08-26 **Curriculum threshold 75%** se respetó (promocionó legítimo a lvl2 en ~10M steps). Plan B si estanca: bajarlo a 65%.
- Convenciones duras: git identity FelipeJackFox/felipaupz@gmail.com SIEMPRE; commits con Co-Authored-By Claude; AGENT_GAMMA única literal de descuento (guard AST); obs float32; modelos guardados deben seguir cargando (cambios de obs/action = env nuevo, nunca in-place).
