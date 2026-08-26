# Gotchas — trampas que ya costaron tiempo

- **retro.make SIEMPRE con render_mode=None** — si no, abre ventana pyglet y mides el vsync del monitor (60fps), no el emulador.
- Import del ROM: `python -m stable_retro.import roms/` (es stable_retro, NO retro; el ROM SÍ viene en el repo).
- stable-retro NO tiene wheels Windows → workers Windows viven en WSL2 (velocidad casi nativa). Tailscale DENTRO de WSL como nodo propio (omen-wsl...).
- .gitignore tiene `*.State` case-insensitive en mac/win → los .state de retro_integration viven bajo excepción `!retro_integration/**/*.state`.
- core.config revienta al importar sin EmuHawk.exe en el dir padre — en la M4 hay stub en ~/TEC/LEIA/EmuHawk.exe. retro_env/es NO importan core.config a propósito.
- Consola Tailscale: los toggles del form de keys se resetean al editar otros campos. El unit de systemd de la madre es 0600 (sudo para leerlo). /opt/leia es de root (sudo git).
- Windows: `time.monotonic` puede dar ticks gruesos — tests de timing con relojes inyectados, jamás contra el reloj real (ya mordió una vez).
- SF2: primeros ~12 agent steps del episodio los inputs NO hacen nada (freeze de "FIGHT!"). Spawns 205/307. Walk ~2.8px/step; salto NO es más rápido que caminar (+129 vs +139px/50 steps).
- Dificultad en RAM: 0xFE45 = nivel−1 (0-7). El estado shipped "Level1" es dificultad 4 real.
- pytest en Windows: 1 test skipped (renice POSIX) — esperado; el total debe coincidir con mac−1.
- La desktop pega logs con [Step N] Command Sent... intercalados: es el debug de rank0, inofensivo.
- PowerShell: cuidado al pegar comandos multilinea — se concatenan (ya pasó); Esc antes de pegar.
- Los benchmarks/eval en la M4: nice 10 + --cpu-share dejan la máquina usable; es la config por defecto de run_worker.
