# Setup del track DQN (Rainbow single-machine y Ape-X distribuido)

Dos roles: **learner** (LA máquina con GPU: Legion) y **actores** (las demás;
solo CPU). Regla de flota: UN entrenamiento por máquina — si tu máquina corre
el worker ES, detenlo antes (Ctrl+C; la madre re-lease sus chunks sola).

## 1. Entorno (todas las máquinas, dentro de WSL2 en Windows)

```bash
cd ~/street_fighter && git pull
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-es.txt -r requirements-retro.txt -r requirements-dqn.txt
```

(Si recreaste el venv, el ROM importado se perdió con él — reimportar:
la sección de ROM de `tools/setup_worker.md`, `python -m retro.import ...`.)

**Solo el learner** (GPU): verificar que torch vea la gráfica:

```bash
python -c "import torch; print(torch.cuda.is_available())"
# False con RTX 50xx -> pip install torch --index-url https://download.pytorch.org/whl/cu128
```

**Solo la primera vez**: `wandb login` (cuenta del team leia-qro-rl).

## 2. Tailscale (los actores hablan con el learner por el tailnet)

```bash
sudo tailscale up
```

Entrar con GitHub → tailnet **leia-qro.org.github**. Si sale "User approval
required", pedirle a Felipe que apruebe en la consola (una vez por usuario).

## 3. Arrancar

**Learner (Legion):**

```bash
.venv/bin/python tools/apex_learner.py --port 8090 --wandb-project leia-sf2-es
```

**Actor (cada máquina extra; --procs ≈ cores físicos − 2):**

```bash
.venv/bin/python tools/apex_actor.py --learner http://legion-wsl:8090 --procs 12
```

El orden no importa (el actor espera al learner con backoff), los actores se
pueden sumar/caer en caliente, y el learner es el único con estado (sus
checkpoints en `models/rainbow_apex/`). Ver progreso: la consola del learner
(línea cada 30s), `curl http://legion-wsl:8090/status`, o el Dashboard DQN
del proyecto en W&B. Examinar un checkpoint:

```bash
.venv/bin/python tools/bench_12rivals.py --arm rainbow --ckpt models/rainbow_apex/apex_grads_XXXXXXXX.pt
```

## Gotchas conocidas

- Windows nativo NO puede (stable-retro es Linux-only): todo dentro de WSL2.
- Un actor con código viejo truena ruidoso al cargar pesos (a propósito):
  `git pull` + relanzar.
- Si el learner se reinicia con otra config, los actores truenan solos
  pidiendo relanzarse (también a propósito).
