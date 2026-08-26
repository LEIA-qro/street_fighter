# Setup de un worker ES — guía paso a paso por máquina

Cada máquina del equipo (desktop 13900K, Omen, Legion, MacBook M4) corre **workers**:
procesos que piden seeds a la madre (`http://madre:8080`), evalúan episodios de SF2 en
el backend headless `stable-retro` y regresan el fitness. La madre vive en AWS y solo
es alcanzable por Tailscale — ver `infra/README.md`.

**Regla de oro: los workers son 100% stateless.** Toda la verdad (política, generación,
checkpoints) vive en la madre. Puedes cerrar la laptop, matar el proceso, reiniciar
Windows o quedarte sin luz **en cualquier momento y no se pierde nada** — la madre
reasigna esos seeds a otro worker solita. Y al revés: conectar una máquina a media
corrida suma throughput al instante. No hay que avisar ni hacer drain.

**Estado actual:** la madre ya está viva (`madre` = `100.90.13.19` en la tailnet
`leia-qro.org.github`) y el ciclo completo ya se probó de punta a punta desde la M4.

> ⏳ **Antes de correr workers "de verdad":** esperen el aviso de que el fix del reward
> terminal está mergeado (el bug de "perder paga más que ganar"). Mientras, TODO lo
> demás de esta guía se puede y se debe hacer ya — incluido el benchmark del paso 4,
> que no entrena nada.

---

## Parte A — Tailscale (la red). Aplica a TODOS.

Somos una tailnet de **organización de GitHub**: `leia-qro.org.github`. Los tres somos
owners de la org LEIA-qro, así que **los tres somos admins de la tailnet
automáticamente** al entrar con GitHub — nadie tiene que invitar a nadie.

### A1. Instala el cliente de Windows (para acceso remoto a la máquina)

1. Descarga e instala <https://tailscale.com/download/windows>.
2. Ícono de Tailscale en la bandeja → **Log in** → **Sign in with GitHub**.
3. **⚠️ LA PANTALLA IMPORTANTE:** GitHub te va a pedir autorizar la app "Tailscale".
   En esa pantalla, junto a la organización **LEIA-qro** hay un botón **Grant** —
   dáselo. Sin ese grant, Tailscale no ve la org y solo te ofrecerá tu tailnet
   personal.
4. Después del login, Tailscale muestra **"Select a tailnet"**: vas a ver tu personal
   (`tuusuario.github`) y **`LEIA-qro`**. **Elige LEIA-qro.** Si eliges la personal
   por error, no pasa nada: logout desde el ícono y vuelve a entrar.
5. Verifica en PowerShell: `tailscale status` — debes ver `madre 100.90.13.19` en la
   lista.

Con esto tu máquina Windows ya es alcanzable desde cualquier lado (RDP/SSH sobre la
tailnet) y tú puedes alcanzar las demás. Si en el paso 3 ya no te aparece el Grant
(porque otro owner ya lo dio), directo al paso 4.

### A2. Ponle nombre legible a tu nodo

En <https://login.tailscale.com/admin/machines> → los tres puntos de tu máquina →
**Edit machine name** → `omen`, `legion` o `desktop`. Ayuda muchísimo cuando estemos
depurando a distancia.

---

## Parte B — El worker (Windows: dentro de WSL2)

`stable-retro` no publica wheels para Windows nativo; el worker vive en **WSL2 Ubuntu**
(la emulación ahí corre a velocidad prácticamente nativa).

### B1. WSL2 + Ubuntu

```powershell
# PowerShell como administrador; reinicia si te lo pide
wsl --install -d Ubuntu-24.04
```

Abre "Ubuntu" del menú inicio, crea tu usuario de Linux, y todo lo que sigue va
**dentro de esa terminal de Ubuntu**.

Opcional pero recomendado en las laptops de 32 GB — dale suficiente RAM/CPU a WSL:
crea `%UserProfile%\.wslconfig` (desde Windows) con:

```
[wsl2]
memory=20GB
processors=20
```

y `wsl --shutdown` en PowerShell para aplicarlo.

### B2. Tailscale DENTRO de WSL

WSL es su propia "máquina" en la red. La forma que menos falla es darle su propio
nodo:

```bash
curl -fsSL https://tailscale.com/install.sh | sh
sudo tailscale up --hostname omen-wsl     # o legion-wsl / desktop-wsl
```

`tailscale up` imprime un link — ábrelo en el navegador de Windows, login con GitHub,
y **otra vez: elige la tailnet LEIA-qro**, no tu personal.

Si reclama que no hay demonio corriendo: habilita systemd —
`sudo nano /etc/wsl.conf`, agrega:

```
[boot]
systemd=true
```

luego `wsl --shutdown` desde PowerShell, reabre Ubuntu y repite `sudo tailscale up`.

**Prueba de fuego** (si esto responde, tu red está lista):

```bash
curl http://madre:8080/status
# → {"generation": ..., "pop_size": 256, ...}
# si el nombre no resuelve, prueba con la IP: curl http://100.90.13.19:8080/status
```

### B3. Repo + dependencias

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip git
git clone -b stage0-metrics-and-semantics https://github.com/LEIA-qro/street_fighter.git ~/street_fighter
cd ~/street_fighter
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-retro.txt -r requirements-es.txt
```

**El ROM ya viene en el repo** (`roms/`). Se importa a stable-retro UNA vez:

```bash
python -m stable_retro.import roms/
# → "Importing StreetFighterIISpecialChampionEdition-Genesis-v0 / Imported 1 games"
```

(Ojo: es `stable_retro.import`, no `retro.import` — el módulo cambió de nombre en
v0.9.7.)

### B4. Valida la máquina con el benchmark (no entrena, solo mide)

```bash
python tools/retro_bench.py
```

Corre solo, imprime la tabla de fps y appendea el resultado a
`benchmarks/retro_bench.jsonl` — **manden ese archivo (o la tabla) al grupo**: con eso
sabemos cuánto aporta cada máquina a la flota. Referencia: la M4 dio ~3,700 fps por
proceso y ~19,700 agregados con 8 procesos.

### B5. Correr el worker (cuando se dé el banderazo del fix)

```bash
cd ~/street_fighter && source .venv/bin/activate
tools/run_worker.sh --coordinator http://madre:8080 --procs 12
```

`--procs` = emuladores en paralelo (uno por proceso, límite del API de libretro).
Regla práctica: núcleos físicos menos 2-4 para que la máquina siga usable —
13900K: 16-20; 275HX (Omen/Legion): 12-16; ajusta con lo que diga tu benchmark del
paso B4. Laptops: **conectadas a la corriente** y modo de energía en "Best
performance", o el thermal throttling se come la mitad del throughput.

Verifica que la madre te ve: `curl http://madre:8080/status` → tu nodo aparece en
`"workers"`. Para parar: `Ctrl+C` y ya — stateless, sin ceremonia.

### B6. Autostart (opcional, puro confort)

- **Dentro de WSL (con systemd):** unit de usuario en
  `~/.config/systemd/user/leia-worker.service`:

  ```ini
  [Unit]
  Description=LEIA ES worker

  [Service]
  ExecStart=/home/TU_USUARIO/street_fighter/tools/run_worker.sh --coordinator http://madre:8080 --procs 12
  Restart=always
  RestartSec=10

  [Install]
  WantedBy=default.target
  ```

  ```bash
  systemctl --user enable --now leia-worker
  loginctl enable-linger $USER   # para que corra sin sesión abierta
  ```

- **Desde Windows (Task Scheduler):** tarea "At startup" con acción
  `wsl -d Ubuntu-24.04 -- bash -lc "cd ~/street_fighter && tools/run_worker.sh --coordinator http://madre:8080 --procs 12"`,
  marcando "Run whether user is logged on or not".

---

## Parte C — macOS (la M4; ya está hecha, referencia)

Igual que B pero sin WSL: Tailscale de brew/App Store con login a la tailnet de la
org, `git clone`, venv, `pip install -r requirements-retro.txt -r requirements-es.txt`,
`python -m stable_retro.import roms/`, y `tools/run_worker.sh --coordinator
http://madre:8080 --procs 8`.

---

## Problemas típicos

| Síntoma | Causa / arreglo |
|---|---|
| `curl http://madre:8080/status` no responde | Tailscale no está arriba EN WSL (`tailscale status` ahí adentro, no en Windows) o entraste a tu tailnet personal — logout y re-login eligiendo LEIA-qro. |
| `madre` no resuelve pero la IP sí | MagicDNS a medias en WSL; usa `--coordinator http://100.90.13.19:8080` y sigue con tu vida. |
| No me apareció la opción LEIA-qro al hacer login | Falta el **Grant** de la org: GitHub → Settings → Applications → Authorized OAuth Apps → Tailscale → Organization access → Grant en LEIA-qro. Luego logout/login en Tailscale. |
| `retro.make` truena con "Game not found" | Falta el import del ROM en ESTA máquina: `python -m stable_retro.import roms/` desde la raíz del repo. |
| El toggle de la extensión/permiso se revierte (macOS) | El app de Tailscale debe estar corriendo cuando apruebas la extensión en System Settings. |
| Worker conecta pero fps bajísimos | Demasiados `--procs`, laptop en batería, o power mode en balanced. |
| La madre no existe / todo muerto | Alguien hizo `terraform destroy`, o la instancia está apagada. Ver `infra/README.md` — se relevanta en ~3 min. |
| Se fue la luz a media corrida | Nada que hacer: la madre reasignó tus chunks. Reconecta cuando vuelvas y ya. |
