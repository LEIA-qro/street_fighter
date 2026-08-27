# Run larga DQN — parte de los HELPERS (actores: Legion, Mac, Omen…)

Los actores son puro CPU: emuladores generando experiencia para el learner de
la desktop. Se suman y se caen en caliente — nada se rompe por apagar uno.

## 1. Matar la run vieja (solo Diego, una vez)

En la Legion, **en este orden**:

1. `Ctrl+C` a la terminal del **actor**.
2. `Ctrl+C` a la terminal del **learner** — su última línea dice dónde guardó
   el checkpoint final (`models/rainbow_apex/apex_final_*.pt`).
   **Ese archivo NO se borra: es el acta de la run lvl1.**

## 2. Actualizar (todas las máquinas, siempre)

```bash
cd ~/street_fighter && git pull
```

## 3. Lanzar el actor

Sustituir `<DESKTOP>` por el nodo WSL que pase Santiago (`tailscale status`):

```bash
source .venv/bin/activate && python tools/apex_actor.py --learner http://<DESKTOP>:8090 --difficulty 1,2,3,4 --procs <N>
```

| Máquina | `--procs` |
|---|---|
| Legion (Diego) | 20 |
| Mac (Felipe) — la lanza Claude | 8 |
| Omen (Santiago, cuando esté) | 12 |

Todos con `--difficulty 1,2,3,4`: el curriculum es uniforme en cada actor a
propósito (por-máquina sesgaría el buffer hacia el tier de la CPU más fuerte;
con mezcla uniforme el PER enfoca solo).

## 4. Señales

- Arranque sano: `procs=N | epsilons=[0.4, ...] | estados=48`.
- "learner inalcanzable; reintento en 10s": normal si Santiago aún no enciende
  — el actor espera solo.
- "actor stale" / "config del learner cambió": `git pull` + relanzar (es el
  guard trabajando, no un bug).

## 5. Ver cómo va (cualquiera, sin tocar nada)

```bash
curl -s http://<DESKTOP>:8090/status
```

y la escalera por dificultad en vivo: Dashboard DQN v2 →
https://wandb.ai/leia-qro-rl/leia-sf2-es?nw=zd7dgnfgz3s
