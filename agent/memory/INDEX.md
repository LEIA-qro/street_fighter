# Memoria de agentes — LEIA street_fighter

**Para cualquier agente (Claude/Codex/Gemini) trabajando en este proyecto: LEE ESTO PRIMERO.**
Sistema de memoria compartida del proyecto, versionado en git. Convención:

- `INDEX.md` (este archivo): el mapa + snapshot del estado. Se actualiza al cerrar sesiones importantes.
- Un archivo por tema, denso, sin narrativa — hechos, números, rutas, comandos.
- Al aprender algo no-obvio o tomar una decisión: APÉNDALO al archivo del tema con fecha. No borres historia; marca `[OBSOLETO: por qué]` si algo dejó de ser cierto.
- Los documentos históricos (`agent/handoff.md` de 2026-08-25, `agent/stage0-runbook.md`) siguen siendo válidos como referencia profunda; esta memoria es el estado VIVO.

## Archivos

- [01-arquitectura.md](01-arquitectura.md) — los dos backends, contratos, flota ES, la madre.
- [02-decisiones.md](02-decisiones.md) — decisiones con fecha y porqué (no re-litigar).
- [03-bugs-cazados.md](03-bugs-cazados.md) — los bugs históricos y sus fixes. NO REINTRODUCIR.
- [04-infra.md](04-infra.md) — madre/AWS, Tailscale, W&B, S3, terraform: cómo operar todo.
- [05-runs.md](05-runs.md) — bitácora de entrenamientos con números.
- [06-pendientes.md](06-pendientes.md) — cola de trabajo abierta.
- [07-gotchas.md](07-gotchas.md) — trampas conocidas que cuestan horas.

## Snapshot (2026-09-11, al cerrar la limpieza de handoff)

- **EL DOCUMENTO DE ENTRADA ES `HANDOFF.md` EN LA RAIZ.** Esta memoria es el detalle;
  el handoff es el mapa. El README de la raiz es de la era BizHawk/SB3 y lo dice.
- **Rama: `main`.** Hasta el 2026-09-11 la verdad vivia en `stage0-metrics-and-semantics`
  (128 commits de ventaja) y `main` todavia cargaba el bug de reward de 6 meses: quien
  clonara se llevaba la version rota. Ya se integro. Suite: **630 tests**.
- **El juego esta resuelto.** Campeon: `benchmarks/apex_milestones/apex_v3291_media990.pt`
  (Ape-X DQN, 72 acciones con macros). ~99% de rounds de apertura sobre los 8 tiers y
  **~90% de peleas COMPLETAS al mejor de 3 en lvl8** (n=360 x2 semillas, el banco
  replica). Muros: BALROG ~50% (sin proyectil, embestida pura), GUILE/EHONDA 65-77%.
- **El campeon YA ESTA EN GIT.** No lo estaba: v3291, v1212, v781 y v511 vivian solo en
  la Mac de Felipe. Los alias moviles del selector (escalera_best, curriculum_best,
  best_desync) NO se versionan -- son byte a byte identicos a hitos que si estan.
- **Nada corriendo, y nada costando.** La run 1 del curriculum se cerro a proposito el
  2026-08-28; el learner esta tumbado y los actores parados. **La infra de AWS se
  DESTRUYO el 2026-09-11** (la madre llevaba 16 dias encendida sin trabajo): instancia,
  bucket, IAM y security group, seis recursos, cero sobrantes. El terraform de `infra/`
  se conserva para volver a levantarla. Lo de S3 que no era reproducible se rescato al
  repo: `benchmarks/LEEME-runs-ES.md`. Tailscale y W&B siguen vivos; queda por borrar a
  mano el nodo `madre` de la consola de Tailscale.
- **UI: la consola React se CANCELO** (Felipe, 2026-09-11). Lo que se mantiene es el
  dashboard Gradio `src/scripts/web_dashboard.py` (11 arreglos el mismo dia) y
  `tools/leia_hub.py` con `web/consola.html` como pantalla. Lo borrado vive en el tag
  `consola-react-cancelada`. Los planes de reconstruccion llevan aviso de CANCELADO.
- Equipo: Felipe (FelipeJackFox), Diego (Perea094, diegop00dx@gmail.com), Santiago
  (SantiagoSaldanaS, sssubias@gmail.com). Los 3 owners de la org GitHub LEIA-qro =
  admins de tailnet y W&B.
- **La cola viva es `08-cola-manana.md`** (sustituye a 06 en lo operativo). Lo grande:
  el curriculum por MESH para la run 2, acelerar el learner (replay ratio real 1.10
  contra un tope de 8 -- vale mas que sumar maquinas), el fix numpy del actor, y los
  sentidos que faltan (stun, Y de proyectiles, fase del move rival).
