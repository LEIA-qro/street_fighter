# Run larga DQN — parte de SANTIAGO (desktop `sss` = el learner)

La desktop es el cerebro de la run: recibe la experiencia de todos los
actores, entrena en la 4090 y guarda los checkpoints. Es la máquina 24/7.

## 0. Terminar el setup (si falta algo; todo dentro de Ubuntu-24.04)

```bash
cd ~/street_fighter && git pull
```

**ROM** (si no está importado): copiarlo del lado Windows —

```bash
mkdir -p ~/roms && ls /mnt/d/GitHub/Street/street_fighter/*.md
```

(ajustar la ruta a donde viva el `.md` del BizHawk, copiarlo a `~/roms/`)

```bash
sha1sum ~/roms/*.md
```

debe dar `a5aad1d108046d9388e33247610dafb4c6516e0b` — si no, es otro dump y no
sirve. Luego:

```bash
python -m retro.import ~/roms
```

→ "Imported 1 games". Verificación final:

```bash
.venv/bin/python -c "import sys; sys.path.insert(0,'src'); from envs.retro_env import RetroSF2Env; e=RetroSF2Env(); print('ROM OK'); e.close()"
```

**Tailscale DENTRO de Ubuntu** (el de Windows no le sirve a WSL):

```bash
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
```

Entrar con GitHub → tailnet `leia-qro.org.github` (entra solo, ya estás
aprobado).

**W&B** (una vez; key en https://wandb.ai/authorize):

```bash
wandb login
```

## 1. Esperar la señal

Diego mata su run vieja primero (actor y learner de la Legion). Cuando avise,
sigue el paso 2.

## 2. Encender el learner

```bash
cd ~/street_fighter && source .venv/bin/activate && python tools/apex_learner.py --port 8090 --macros --buffer 1000000 --beta-anneal-grads 1000000 --wandb-project leia-sf2-es --wandb-id rainbow-apex-curriculum
```

Checks de la primera línea: **`acciones=72`** (macros activos) y
**`device=cuda`**. Si dice `cpu`, no es grave (la red es chica), pero se
arregla con `pip install torch --index-url https://download.pytorch.org/whl/cu128`
y relanzando. Esta terminal se queda abierta: imprime el estado cada 30 s.

## 3. Tu actor local (OTRA terminal de Ubuntu)

```bash
cd ~/street_fighter && source .venv/bin/activate && python tools/apex_actor.py --learner http://127.0.0.1:8090 --difficulty 1,2,3,4 --procs 28
```

## 4. Avisar al grupo tu nodo

```bash
tailscale status
```

El nombre de tu nodo WSL (algo como `sss-1`) es la dirección que los helpers
usan en su `--learner http://<ese-nodo>:8090`.
