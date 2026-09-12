# Las tres runs del ES — lo que queda, ahora que S3 no existe

El bucket `leia-sf2-es-ckpt-d4a2b8dc99bbd0b480e7520f5e` guardaba un checkpoint por
generación de cada run (1,949 objetos, 222 MB). **Se destruyó el 2026-09-11** junto con
el resto de la infraestructura AWS del proyecto: las tres runs estaban cerradas desde
agosto y el coordinador llevaba dieciséis días encendido sin trabajo.

Antes de destruirlo se bajó lo que no era reproducible. Esto es lo que sobrevive:

| Run | Política | Carpeta | Última generación |
|---|---|---|---|
| 1 — escalar | `v4`, 14,207 params | `run1_final/` | gen 96 (el acta) **+ gen 105** |
| 2 — one-hot | `v4onehot`, 21,887 params | `run2_final/` | gen 754 (el acta) **+ gen 755** |
| 3 — perturbada | `v4onehot` + desfase y ruido | `run3_final/` | gen 113 + `metrics.jsonl` |

**Por qué hay dos generaciones por run.** El acta de cada run se escribió cuando el
equipo la dio por cerrada; el coordinador siguió guardando unas generaciones más antes
de que alguien apagara el servicio. `gen_000105` y `gen_000755` son el **último estado
real** que existió de esas runs, y entraron al repo el día de la destrucción. La de la
run 3 ya coincidía byte a byte con la que estaba archivada, así que no se duplicó.

Cada `.npz` trae θ, el estado de Adam y la configuración completa (sigma, lr, weight
decay, master seed, dimensión, lista de estados): con eso se reanuda una run o se
examina la política en el banco. `theta_final.npz` es solo el vector de pesos.

**Lo que NO está aquí y no se perdió:**

* Las curvas de fitness generación a generación viven en **Weights & Biases**, equipo
  `leia-qro-rl`, proyecto `leia-sf2-es`, runs `es-run1-scalar`, `es-run2-onehot` y
  `es-run3-perturbed`. W&B no se tocó.
* La run 3 además trae su `metrics.jsonl` local, que era precisamente la observabilidad
  a prueba de fallos de W&B.
* Los resultados medidos de las tres runs, con su contexto, están en
  `agent/memory/05-runs.md`.

**Para examinar cualquiera de estas políticas sin resucitar nada:**

```bash
.venv/bin/python tools/bench_12rivals.py --arm es \
  --theta-npz benchmarks/run3_final/gen_000113.npz --policy v4onehot \
  --desync-max 30
```

**Para volver a levantar la flota ES:** el terraform sigue en `infra/` y se conserva a
propósito. Ver `HANDOFF.md`, sección 7.
