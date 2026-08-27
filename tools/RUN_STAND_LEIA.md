# Stand LEIA — reta a la IA 🥊 (humano vs campeón, en BizHawk)

El modo promoción: un visitante agarra el control y pelea contra el campeón
DQN en el rig BizHawk de siempre (ventana del juego real, sonido, marcador en
pantalla, rematch automático). P1 = la IA (Ryu). P2 = el humano.

Corre en cualquier máquina **Windows** con el rig BizHawk ya montado (la
desktop, o una laptop para el stand). Piezas: `lua/v2.0/stand_env_client.lua`
(payload extendido + passthrough del pad 2) y `src/scripts/stand_leia.py`
(el campeón + obs v4 idéntica al entrenamiento + reproducción de macros).

## 0. Requisitos (una vez)

1. El rig BizHawk del proyecto funcionando (el mismo de `test_ai_vs_ai_v2` /
   entrenamientos: BizHawk 2.8, ROM en `roms/`, venv de Windows con torch).
2. `git pull` (trae Lua + driver + el checkpoint NO — ver 3).
3. **El checkpoint del campeón** en `benchmarks\apex_milestones\`:
   `apex_v781_escalera831.pt` (4 MB — está en la Mac de Felipe; pedirlo por
   la tailnet: en la Mac `python3 -m http.server 8099` dentro de
   `benchmarks/apex_milestones/` y en Windows bajarlo con el navegador o
   `curl -O http://mini-fzamorano:8099/apex_v781_escalera831.pt`).
4. **El control del retador como Player 2**: BizHawk → Config → Controllers
   → Genesis 3-button/6-button → asignar el pad USB al **puerto 2** y
   guardar. (La IA inyecta P1 por socket; el pad 2 pasa directo.) Probar los
   6 botones: X/Y/Z = puños débil/medio/fuerte, A/B/C = patadas.
5. **Limpiar TODOS los bindings de Player 1** (teclado incluido) en ese
   mismo menú, y dejar Start/Mode SIN asignar en el pad del retador: el Lua
   ya blinda el pad de la IA, pero un Start suelto pausando el juego a media
   demo no se lo deseamos a nadie.
6. **Ruta del proyecto SIN acentos ni caracteres raros** (p.ej. `C:\LEIA\`):
   el protocolo del socket cuenta bytes y una ruta con acentos en los
   comandos RESET es buscarle ruido a la feria.

## 1. Lanzar el stand

Desde la raíz del repo, en PowerShell:

```powershell
.venv\Scripts\python.exe src\scripts\stand_leia.py --opponent RANDOM
```

- `--opponent RANDOM` rota el PERSONAJE del retador cada round (los 12
  estados `RYU_<PERSONAJE>_R1_PvP.State` — la IA siempre es Ryu, que es como
  entrenó); un personaje fijo: `--opponent KEN`.
- `--ckpt <ruta>` para otro campeón; `--rematch-delay 4.0` la pausa de KO.
- BizHawk abre solo (ventana 4×, sonido ON) y el marcador vive en pantalla:
  IA abajo-izquierda, RETADOR abajo-derecha.

Se termina con **Ctrl+C en la consola** (cierra el emulador limpio).

## 2. El flujo del visitante

1. El round arranca solo al cargar el estado (ambos con vida completa).
2. Pelea normal con el pad; al KO, pantalla de resultado ~4 s y **rematch
   automático** con el marcador acumulado. Nadie toca teclado ni menús.
3. Con `RANDOM`, cada round el retador recibe personaje nuevo — la IA
   siempre es Ryu (así entrenó).

## 3. Señales y trucos

- La consola canta cada round: `[round N] GANA LA IA | IA 3 - 1 Retador`.
- **La IA juega en serio** (es el campeón que barre lvl1-3 al 100%). Para
  visitantes casuales considera un checkpoint más tierno (cualquier
  `apex_grads_*.pt` temprano con `--ckpt`) — el driver acepta cualquier
  checkpoint CON macros (72 acciones).
- Payload de 13 campos / error "¿Lua viejo?": BizHawk cargó
  `match_test_env_client.lua` — el driver ya apunta al Lua nuevo; `git pull`
  y relanzar.
- Si el emulador se congela >120 s sin Python, su dead-man's-switch lo
  cierra solo (heredado del match test): relanzar el comando y ya.
- Rondas por tiempo: si el reloj llega a 0, gana quien tenga más vida (se
  anuncia "por tiempo"); empate exacto no suma a nadie.
- Nota técnica: gracias al re-primado del reset, la IA ve el estado generado
  por su propia acción anterior (cero lag extra) — paridad plena con su rig
  de entrenamiento.

## 4. Qué NO tocar

- La ventana de BizHawk durante sesión (los menús pausan el lock-step; si
  alguien lo hace, el dead-man's-switch o Ctrl+C + relanzar).
- `states/RYU_*_R1_PvP.State`: son savestates 2P alineados — el inventario
  del stand.
