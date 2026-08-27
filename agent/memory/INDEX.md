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

## Snapshot (2026-08-26, tarde)

- Rama de trabajo: `stage0-metrics-and-semantics` (sobre `sf2-sota-rl-upgrade`, ninguna mergeada a main). Suite: **480 tests**.
- **Dos entrenamientos vivos**: PPO en la desktop (rumbo a 31M steps, curriculum lvl2→3, BizHawk) y **ES desde cero en la flota** (madre EC2 + M4 como único worker por ahora, 12 rivales lvl1, ~128s/generación).
- Retro tiene **212 savestates verificados** (lvl1-4 completos ×12 rivales). El bug histórico del reward (perder pagaba más que ganar, 6 meses) está muerto y doble-validado.
- Equipo: Felipe (FelipeJackFox), Diego (Perea094, diegop00dx@gmail.com), Santiago (SantiagoSaldanaS, sssubias@gmail.com). Los 3 owners de la org GitHub LEIA-qro = admins de tailnet y W&B.

- **08-cola-manana.md** — LA COLA VIVA post noche-de-los-tres-algoritmos (sustituye a 06 en lo operativo): fix numpy del actor, inferencia de savestates lvl5-8 por poke de 0xFE45, fleet-agent, vigilancia de la run del curriculum, experimentos en cola.
