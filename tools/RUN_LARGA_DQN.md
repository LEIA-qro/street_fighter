# Run larga DQN — curriculum completo con macros (runbook de arranque)

**Qué es**: la primera run seria del Ape-X — indefinida, multi-máquina, contra
las dificultades 1-4 del curriculum de 212 savestates, estrenando los macros
(los specials de Ryu como acciones atómicas). Sin fecha de fin: corre hasta que
las curvas por dificultad se aplanen; el mejor checkpoint se selecciona solo.

**Roles**: el **learner** vive en la máquina 24/7 (la desktop `sss`); todas las
demás son **actores** (puro CPU, se suman y caen en caliente sin drama).

---

## 0. Pre-vuelo (una vez, TODAS las máquinas)

```bash
cd ~/street_fighter && git pull
```

Sin excepción: un actor con código viejo queda esperando con un mensaje de
"actualízate" (a propósito).

## 1. Matar la run actual (la de lvl1, ya cumplió)

**Orden: actores primero, learner al final** (el learner guarda su checkpoint
final al recibir Ctrl+C).

1. Mac (actor de Felipe/Claude): lo detiene Claude.
2. Legion (actor local de Diego): `Ctrl+C` en su terminal del actor.
3. Legion (learner de Diego): `Ctrl+C` en la terminal del learner — la línea
   final dice dónde quedó su checkpoint (`models/rainbow_apex/apex_final_*.pt`).
   **Ese archivo es el acta de la run lvl1** — no borrarlo.

## 2. Encender el learner (desktop `sss`, dentro de su Ubuntu WSL)

```bash
cd ~/street_fighter && source .venv/bin/activate
python tools/apex_learner.py --port 8090 --macros --buffer 1000000 --beta-anneal-grads 1000000 --wandb-project leia-sf2-es --wandb-id rainbow-apex-curriculum
```

Notas:
- `wandb login` una vez si esa máquina nunca lo ha hecho (key en
  https://wandb.ai/authorize).
- La primera línea debe decir `acciones=72` (macros activos) y `device=cuda`
  si torch ve la 4090 (si dice cpu, no pasa nada grave: la red es diminuta,
  pero `pip install torch --index-url https://download.pytorch.org/whl/cu128`
  lo arregla).
- Esa terminal queda abierta: es el dashboard de consola (línea cada 30 s con
  `wr acum`, `reciente200`, buffer, actores).

## 3. Encender los actores (cada máquina, en su propia terminal)

El hostname del learner es el nodo tailscale del WSL de la desktop — verlo con
`tailscale status` (algo como `sss-1` o similar) y sustituirlo abajo.

```bash
cd ~/street_fighter && source .venv/bin/activate
python tools/apex_actor.py --learner http://<DESKTOP-WSL>:8090 --difficulty 1,2,3,4 --procs <N>
```

`--procs` por máquina:

| Máquina | procs |
|---|---|
| Desktop (actor local; usar `--learner http://127.0.0.1:8090`) | 28 |
| Legion (Diego) | 20 |
| Omen (Santiago) | 12 |
| Mac (Felipe) — la lanza Claude | 8 |

Todas con `--difficulty 1,2,3,4`: **el curriculum es uniforme en todos los
actores** (decisión de diseño: por-máquina sesgaría el buffer hacia el tier de
la CPU más fuerte; con mezcla uniforme el PER hace el enfoque solo).

Primera línea sana del actor: `procs=N | epsilons=[0.4, ...] | estados=48`.

## 4. Verificar que respira (cualquier máquina en el tailnet)

```bash
curl -s http://<DESKTOP-WSL>:8090/status
```

- `actors`: deben aparecer las 4 máquinas con su `steps_per_s`.
- `buffer` subiendo; primeros gradientes al cruzar 20k transiciones (~1 min).
- `win_rate_recent_by_lvl`: el termómetro segmentado — al inicio bajo en todo,
  lvl1 despega primero.

## 5. Monitoreo (nadie tiene que cuidarla)

- **Dashboard DQN v2**: https://wandb.ai/leia-qro-rl/leia-sf2-es?nw=zd7dgnfgz3s
  — `win_rate_recent200`, la escalera `win_rate_recent/lvl1..4`, loss, buffer,
  throughput. (Ojo: W&B pinta con ~1 log de retraso.)
- Checkpoints: cada 10k grads en `models/rainbow_apex/` de la desktop.
- **Selector automático** (lo corre Claude en la Mac): cada 30 min examina los
  pesos vivos con el banco honesto por tier y guarda el mejor en
  `benchmarks/apex_milestones/`.

## 6. Criterio de paro / promoción

No hay duración fija. Se detiene (o se celebra) cuando la escalera por nivel
lleve ~2 evaluaciones nocturnas plana. Si un tier estorba o falta, se cambia el
`--difficulty` de los ACTORES y se relanzan — **el learner no se toca** (en
DQN la rotación no es identidad del run; el buffer digiere el cambio solo).

## 7. Problemas conocidos (y su fix en una línea)

| Síntoma | Fix |
|---|---|
| Actor: "actor stale" / "config del learner cambió" | `git pull` + relanzar el actor |
| Actor: "learner inalcanzable; reintento en 10s" | Normal si el learner aún no arranca; si persiste, checar tailscale en ambos lados |
| `Imported 0 games` al importar ROM | La ruta: es `python -m retro.import roms/` desde `~/street_fighter` (y el sha1 debe ser `a5aad1d...`) |
| Taildrop "peer owned by different user" | No funciona entre cuentas: compartir archivos con `python3 -m http.server` en la tailnet |
| El .pt descargado llega como .zip | Es el mismo archivo (los .pt de torch son zips): renombrar y listo |
