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

## [2026-09-11] Limpieza de handoff — decisiones de Felipe

- **La consola React se CANCELA, no se pausa.** `consola-app/` (React 19 + shadcn) y su
  bundle `web/app/` salen del arbol. La UI que el proyecto mantiene es el dashboard
  Gradio `src/scripts/web_dashboard.py`. Motivo: el proyecto queda en manos de otro
  compañero y un frontend a medias es una decision que hay que volver a tomar, no un
  activo. Recuperable: tag `consola-react-cancelada`.
  Lo que NO era interfaz se queda: `tools/leia_hub.py` (los ojos de la flota) con
  `web/consola.html` de pantalla, sin build ni node_modules.
- **`main` pasa a ser la verdad.** `stage0-metrics-and-semantics` se integra. Antes de
  esto, clonar el repo te daba una version con el bug de reward de 6 meses.
- **El campeon entra a git.** v3291/v1212/v781/v511 con sus actas. Los alias moviles del
  selector no (son duplicados exactos, verificado por sha256).
- **No se podan las pestañas muertas del dashboard** (League, Exploiter, PBT, Optuna).
  Es una decision de producto, no un arreglo, y `refresh_dropdowns` acopla 10 dropdowns
  posicionalmente en 5 enlaces: quitar uno rompe el mapeo en silencio. Queda listada en
  HANDOFF.md seccion 10 con el analisis ya hecho en `agent/dashboard/que-no-reconstruir.md`.

- 2026-09-11 **Tirar la infra de AWS** (decisión de Felipe). Las tres runs de ES estaban
  cerradas desde el 26 de agosto y la madre llevaba 16 días encendida sin trabajo. Se
  destruyeron los seis recursos que manejaba terraform; `infra/` se conserva ENTERO a
  propósito, "por si se vuelve a necesitar". Antes de destruir se rescató de S3 lo no
  reproducible (últimas generaciones de las runs 1 y 2, que solo vivían ahí) y se
  documentó en `benchmarks/LEEME-runs-ES.md`. W&B y Tailscale NO se tocaron: son el
  registro científico y la malla del equipo, no infraestructura de cómputo.
  Lo aprendido: el tfstate vive solo en una Mac, y la cuenta AWS es compartida con otros
  proyectos — al limpiar, filtrar siempre por el tag `Project = leia-sf2-es`.
