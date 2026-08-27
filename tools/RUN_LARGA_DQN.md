# Run larga DQN — índice

La primera run seria del Ape-X: indefinida, multi-máquina, dificultades 1-4
del curriculum de 212 savestates, estrenando los macros. El runbook está
dividido por rol:

- **[RUN_LARGA_SANTIAGO.md](RUN_LARGA_SANTIAGO.md)** — la desktop `sss`:
  setup restante, encender el learner, su actor local, y anunciar el nodo.
- **[RUN_LARGA_HELPERS.md](RUN_LARGA_HELPERS.md)** — los actores (Legion,
  Mac, Omen): matar la run vieja, actualizar, lanzar, señales.

## Común a todos

- **Monitoreo**: Dashboard DQN v2
  (https://wandb.ai/leia-qro-rl/leia-sf2-es?nw=zd7dgnfgz3s) con
  `win_rate_recent200` y la escalera `win_rate_recent/lvl1..4`; checkpoints
  cada 10k grads en `models/rainbow_apex/` de la desktop; el selector
  automático (en la Mac) examina los pesos vivos cada 30 min y guarda el
  mejor en `benchmarks/apex_milestones/`.
- **Criterio de paro**: sin duración fija — se detiene cuando la escalera por
  nivel lleve ~2 evaluaciones nocturnas plana. Cambiar de tiers = relanzar
  ACTORES con otro `--difficulty`; el learner no se toca.
- **Regla de oro**: `git pull` antes de lanzar cualquier cosa; los guards
  ruidosos ("stale", "config cambió") se curan igual: pull + relanzar.
