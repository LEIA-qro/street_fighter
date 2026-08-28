# Modo exhibición — reta a la IA 🥊 (humano vs campeón, en BizHawk)

El modo promoción: un visitante agarra el control y pelea contra el campeón
DQN en el rig BizHawk de siempre (ventana del juego real, sonido, marcador en
pantalla, rematch automático). P1 = la IA (Ryu). P2 = el humano.

Corre en cualquier máquina **Windows** con el rig BizHawk ya montado (la
desktop, o una laptop para exhibición). Piezas internas:
`lua/v2.0/stand_env_client.lua`
(payload extendido + passthrough del pad 2) y `src/scripts/stand_leia.py`
(el campeón + obs v4 idéntica al entrenamiento + reproducción de macros).

## 0. Requisitos (una vez)

1. El rig BizHawk del proyecto funcionando (el mismo de `test_ai_vs_ai_v2` /
   entrenamientos: BizHawk 2.8, ROM en `roms/`, venv de Windows con torch).
2. `git pull` (trae Lua + driver + el checkpoint NO — ver 3).
3. **El checkpoint del campeón** en `benchmarks\apex_milestones\`:
   `apex_escalera_best.pt` (4 MB — es el alias del mejor selector vigente y
   está en la Mac de Felipe; pedirlo por
   la tailnet: en la Mac `python3 -m http.server 8099` dentro de
   `benchmarks/apex_milestones/` y en Windows bajarlo con el navegador o
   `curl -O http://mini-fzamorano:8099/apex_escalera_best.pt`). Bajar también
   `apex_escalera_best.pt.json` para que el dashboard muestre versión y WR.
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

## 1. Abrir el viewer en Model Testing

Desde la raíz del repo, en PowerShell:

```powershell
.venv\Scripts\python.exe src\scripts\web_dashboard.py
```

Si se acaba de reemplazar `web_dashboard.py`, cerrar primero cualquier pestaña
vieja de `localhost:7860`: un frontend Gradio ya abierto conserva los IDs de
los componentes anteriores aunque el backend haya reiniciado. Abrir una carga
fresca y versionada:

```powershell
Start-Process "http://127.0.0.1:7860/?build=v1404-additive-r3"
```

Entrar a **🎮 Model Testing & Matchups**. La
parte superior conserva el probador clásico P1-vs-P2 (PPO/SAC/DQN, Human y
CPU). Para el campeón nuevo, abrir el acordeón **Ape-X QR-DQN vs Human
(Viewer)** que está debajo. Ahí elegir checkpoint/personaje/delay y pulsar
**Launch Ape-X vs Human**. **Terminate Ape-X Match** pausa el Lua y cierra
BizHawk. Si se copia un checkpoint nuevo con el viewer ya abierto, pulsar
**Refresh Ape-X checkpoints**.

- `RANDOM` rota el PERSONAJE del retador cada round (los 12
  estados `RYU_<PERSONAJE>_R1_PvP.State` — la IA siempre es Ryu, que es como
  entrenó); también se puede fijar, por ejemplo, `KEN`.
- El dropdown permite elegir otro campeón compatible; el slider controla la
  pausa de KO.
- BizHawk abre solo (ventana 4×, sonido ON) y el marcador vive en pantalla:
  IA abajo-izquierda, RETADOR abajo-derecha.

### Fallback técnico por consola

Para diagnosticar sin Gradio se puede llamar al driver interno directamente:

```powershell
.venv\Scripts\python.exe src\scripts\stand_leia.py --opponent RANDOM
```

Sin `--ckpt`, usa `benchmarks\apex_milestones\apex_escalera_best.pt`.
Admite `--opponent KEN`, `--ckpt <ruta>` y `--rematch-delay 4.0`; se termina
con **Ctrl+C**.

## 2. El flujo del visitante

1. El round arranca solo al cargar el estado (ambos con vida completa).
2. Pelea normal con el pad; al KO, pantalla de resultado ~2 s por defecto
   (ajustable en el slider) y **rematch automático** con el marcador
   acumulado. Nadie toca teclado ni menús.
3. Con `RANDOM`, cada round el retador recibe personaje nuevo — la IA
   siempre es Ryu (así entrenó).

## 3. Señales y trucos

- La consola canta cada round: `[round N] GANA LA IA | IA 3 - 1 Retador`.
- **La IA juega en serio** (selector robusto lvl1-3: 100/100/97.9%). Para
  visitantes casuales usa un checkpoint anterior que también declare
  `macros=true` y 72 acciones. Los `apex_grads_*.pt` viejos de 63 acciones
  **no son compatibles** con este viewer.
- Payload de 13 campos / error "¿Lua viejo?": BizHawk cargó
  `match_test_env_client.lua` — el driver ya apunta al Lua nuevo; `git pull`
  y relanzar.
- Errores `FileData meta` o `Value: apex is not in the list of choices`: la
  pestaña es del dashboard anterior y está llamando al backend nuevo. Cerrarla
  y abrir la URL versionada de arriba (o usar Ctrl+Shift+R/ventana InPrivate).
  No se arregla reinstalando Torch, Gradio ni el modelo. Desde el build r3,
  una pestaña ya cargada se autocorrige en futuros reinicios al detectar el
  `app_id` nuevo; la pestaña anterior a r3 sí debe cerrarse una vez.
- Si el emulador queda congelado y se cierra ~120 s después, el
  dead-man's-switch detectó que Python y Lua perdieron la fase del socket:
  conservar la consola del viewer para diagnóstico antes de relanzar.
- Rondas por tiempo: si el reloj llega a 0, gana quien tenga más vida (se
  anuncia "por tiempo"); empate exacto no suma a nadie.
- Nota técnica: el arranque drena un payload inicial; cada rematch consume
  exactamente un payload post-reset. Durante combate la IA ve el estado
  generado por su propia acción anterior, sin lag extra.

## 4. Qué NO tocar

- La ventana de BizHawk durante sesión (los menús pausan el lock-step; si
  alguien lo hace, el dead-man's-switch o Ctrl+C + relanzar).
- `states/RYU_*_R1_PvP.State`: son savestates 2P alineados — el inventario
  del viewer.
