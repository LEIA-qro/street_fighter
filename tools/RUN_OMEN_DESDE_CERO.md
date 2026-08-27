# Omen desde CERO → helper de la flota DQN

De Windows pelón a actor mandando experiencia al learner. Todo va DENTRO de
WSL (Ubuntu); Windows solo pone la máquina. Tiempo estimado: ~30-40 min, la
mitad son descargas. Cada bloque es copy-paste en orden.

## 0. Windows: WSL + ajustes de energía (una vez)

En **PowerShell como administrador**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reinicia cuando lo pida, abre "Ubuntu 24.04" del menú inicio, crea tu
usuario/contraseña de Linux. (Si truena con error de virtualización:
habilitar "Virtualization Technology" en el BIOS.)

**Energía** (para que la lap trabaje sola): Configuración → Sistema →
Energía y batería → pantalla y suspensión → **"Nunca" suspender conectado**;
y en Panel de control → Energía → "elegir el comportamiento al cerrar la
tapa" → **No hacer nada (con corriente)**. Déjala SIEMPRE enchufada: WSL se
congela si Windows se suspende.

Todo lo que sigue es dentro de Ubuntu.

## 1. Básicos de Ubuntu

```bash
sudo apt update && sudo apt install -y git python3.12-venv python3-pip curl
```

## 2. Tailscale (DENTRO de Ubuntu — el de Windows NO le sirve a WSL)

```bash
curl -fsSL https://tailscale.com/install.sh | sh && sudo tailscale up
```

Abre el link que imprime → **entrar con GitHub** → tailnet
`leia-qro.org.github`. Si no entra solo, avisa: Felipe aprueba el nodo
nuevo en el panel. Verifica que ves a la flota:

```bash
tailscale status
```

(debes ver `desktop-4090-ubuntu-wsl`, `mini-fzamorano`, etc.)

## 3. Clonar el repo

```bash
cd ~ && git clone https://github.com/LEIA-qro/street_fighter.git && cd street_fighter
```

(Repo privado: te pedirá login de GitHub — si no tienes credenciales a la
mano, `sudo apt install -y gh && gh auth login` y repites el clone.)

## 4. Entorno de Python (solo lo que un actor necesita)

```bash
cd ~/street_fighter && python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements-es.txt -r requirements-retro.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

El torch **CPU** es a propósito: un actor no usa GPU, y el wheel de CUDA
pesa gigas de más.

## 5. El ROM (jálalo de tu propia desktop por la tailnet)

En la **desktop** (WSL de la 4090), una terminal temporal:

```bash
cd ~/roms && python3 -m http.server 8099
```

En la **Omen**:

```bash
mkdir -p ~/roms && cd ~/roms && curl -O "http://desktop-4090-ubuntu-wsl:8099/$(curl -s http://desktop-4090-ubuntu-wsl:8099/ | grep -o 'Street[^"]*\.md' | head -1)"
sha1sum ~/roms/*.md
```

**El sha1 DEBE ser `a5aad1d108046d9388e33247610dafb4c6516e0b`** — si da otro,
es otro dump y el retro no lo acepta (ya nos pasó: el de 6 meses de BizHawk
de la desktop es OTRO dump; el bueno es el de `~/roms` que importaste ayer).
Ya copiado, mata el http.server de la desktop (Ctrl+C).

Importar — **con ruta relativa, parado en home** (con ruta equivocada dice
"Imported 0 games" sin error, la trampa clásica):

```bash
cd ~ && ~/street_fighter/.venv/bin/python -m retro.import roms
```

Debe decir **"Imported 1 games"**. Verificación final de toda la tubería:

```bash
cd ~/street_fighter && .venv/bin/python -c "import sys; sys.path.insert(0,'src'); from envs.retro_env import RetroSF2Env; e=RetroSF2Env(); print('ROM + emulador OK'); e.close()"
```

## 6. Lanzar el actor 🥊

```bash
cd ~/street_fighter && git pull && source .venv/bin/activate && python tools/apex_actor.py --learner http://desktop-4090-ubuntu-wsl:8090 --difficulty 1,2,3,4,5,6,7,8 --procs 12
```

La primera línea sana dice: `procs=12 | epsilons=[0.4, ...] |` **`estados=96`**
— el 96 confirma la escalera completa de dificultades. Si dice menos, te
faltó el `git pull`.

Con el actor corriendo, checa el CPU (`htop`, `sudo apt install -y htop`):
si el total anda **debajo de ~80%, súbele** — relanza con `--procs 16` (los
hijos no saturan un core cada uno; sobre-suscribir es la doctrina).

## 7. Señales y reglas de la casa

- **"learner inalcanzable; reintento en 10s"**: normal si el learner anda
  reiniciando — el actor espera solo. No hagas nada.
- **"actor stale" / "config del learner cambió"**: es el guard, no un bug —
  `git pull` y relanzar el actor.
- **Tras cualquier deploy de código**: relanzar el actor (el viejo se queda
  esperando con código viejo, no se muere solo).
- Los actores se caen y se suman EN CALIENTE: apagar la Omen no rompe nada;
  al volver, mismo comando del paso 6.
- Monitoreo sin tocar nada: `curl -s http://desktop-4090-ubuntu-wsl:8090/status`
  y el dashboard DQN v2:
  https://wandb.ai/leia-qro-rl/leia-sf2-es?nw=zd7dgnfgz3s

## Apéndice: si WSL se siente ahogado

`C:\Users\<tu-usuario>\.wslconfig` (archivo nuevo, lado Windows):

```
[wsl2]
memory=12GB
processors=12
```

y en PowerShell: `wsl --shutdown` + reabrir Ubuntu. Solo si hace falta —
el default (todos los cores, 50% de la RAM) suele bastar.
