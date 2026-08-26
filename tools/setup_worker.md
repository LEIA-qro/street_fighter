# Setup de un worker ES (por maquina)

Cada maquina del equipo (desktop 13900K, laptops 275HX, MacBook M4) corre **workers**:
procesos que piden seeds a la madre (`http://madre:8080`), evaluan episodios de SF2 con
el backend headless `stable-retro` y regresan el fitness. La madre vive en AWS y solo es
alcanzable por Tailscale — ver `infra/README.md`.

**Regla de oro: los workers son 100% stateless.** Toda la verdad (politica, generacion,
checkpoints) vive en la madre. Puedes cerrar la laptop, matar el proceso, reiniciar
Windows o desconectarte del wifi **en cualquier momento y no se pierde nada** — la madre
simplemente reasigna esos seeds a otro worker. Tambien al reves: conectar una maquina
nueva a media corrida suma throughput al instante. No hay que "avisar" ni hacer drain.

---

## Windows (desktop y laptops)

`stable-retro` no corre nativo en Windows; el worker vive dentro de **WSL2 Ubuntu**.

### 1. WSL2 + Ubuntu

```powershell
# PowerShell como administrador; reinicia si te lo pide
wsl --install -d Ubuntu-24.04
```

Abre "Ubuntu" desde el menu inicio, crea tu usuario, y todo lo que sigue va **dentro**
de esa terminal.

### 2. Dependencias y repo

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
git clone -b stage0-metrics-and-semantics https://github.com/LEIA-qro/street_fighter.git ~/street_fighter
cd ~/street_fighter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-retro.txt -r requirements-es.txt
# el ROM de SF2 se importa una vez (pideselo a quien ya lo tenga; no va en el repo):
python -m retro.import /ruta/al/directorio/con/el/rom
```

### 3. Tailscale (elige UNA de las dos)

- **Opcion A — tailscale dentro de WSL (la mas directa):**

  ```bash
  curl -fsSL https://tailscale.com/install.sh | sh
  sudo tailscale up      # abre el link de login en el navegador de Windows
  ```

  WSL no trae systemd siempre activo; si `tailscale up` reclama que no hay demonio:
  `sudo tailscaled &` o habilita systemd en `/etc/wsl.conf` (`[boot] systemd=true`) y
  `wsl --shutdown` desde PowerShell para reiniciar.

- **Opcion B — tailscale de Windows + mirrored networking:** si ya usas la app de
  Tailscale en Windows, WSL2 puede compartir esa red con el modo *mirrored*: en
  `%UserProfile%\.wslconfig` pon `[wsl2]` y `networkingMode=mirrored` (Windows 11
  22H2+), luego `wsl --shutdown`. Con eso `madre` resuelve desde adentro de WSL sin
  instalar nada extra. Si `curl http://madre:8080/status` no responde, vuelve a la
  opcion A, que falla menos.

### 4. Correr el worker

```bash
cd ~/street_fighter && source .venv/bin/activate
tools/run_worker.sh --coordinator http://madre:8080 --procs 12
```

`--procs` = procesos emulador en paralelo. Regla practica: nucleos fisicos menos 2-4
para que la maquina siga siendo usable (13900K: 16-20; 275HX: 12-16; pruebalo con
`tools/retro_bench.py`). Verifica que trabaja: la madre lo lista en
`curl http://madre:8080/status`.

### 5. Autostart (opcional)

- **Dentro de WSL con systemd habilitado:** unit de usuario en
  `~/.config/systemd/user/leia-worker.service` con
  `ExecStart=/home/TU_USUARIO/street_fighter/tools/run_worker.sh --coordinator http://madre:8080 --procs N`,
  `Restart=always`, y `systemctl --user enable --now leia-worker`
  (+ `loginctl enable-linger $USER` para que arranque sin sesion abierta).
- **Desde Windows (Task Scheduler):** tarea "At startup" que corra
  `wsl -d Ubuntu-24.04 -- bash -lc "cd ~/street_fighter && tools/run_worker.sh --coordinator http://madre:8080 --procs N"`.
  Marca "Run whether user is logged on or not".

Como los workers son stateless, el autostart es puro confort — matar la tarea o el
proceso nunca corrompe nada.

## macOS (MacBook M4)

Igual que arriba pero sin WSL:

```bash
brew install python git    # si no estan ya
git clone -b stage0-metrics-and-semantics https://github.com/LEIA-qro/street_fighter.git ~/street_fighter
cd ~/street_fighter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-retro.txt -r requirements-es.txt
python -m retro.import /ruta/al/rom

# Tailscale: app de la Mac App Store (o brew install --cask tailscale) y sesion iniciada

tools/run_worker.sh --coordinator http://madre:8080 --procs 8
```

Autostart en macOS: LaunchAgent en `~/Library/LaunchAgents/com.leia.worker.plist` con
`KeepAlive=true` apuntando al mismo comando, o simplemente una pestana de terminal —
recuerda: apagarlo no cuesta nada.

## Problemas tipicos

| Sintoma | Causa probable |
|---|---|
| `curl http://madre:8080/status` no responde | Tailscale no esta arriba en ESTA maquina (o en WSL con opcion B sin mirrored). `tailscale status` primero. |
| `retro.make` truena con "Game not found" | Falta importar el ROM en esta maquina: `python -m retro.import ...` |
| El worker conecta pero el fps es bajisimo | Demasiados `--procs` para la maquina (thermal throttling) o corriendo en bateria. Baja `--procs`. |
| La madre no existe / DNS falla | Nadie ha hecho `terraform apply`, o se destruyo. Ver `infra/README.md`. |
