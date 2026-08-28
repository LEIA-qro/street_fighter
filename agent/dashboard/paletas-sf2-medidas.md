# O5-identidad-sf2 — Identidad visual con raíz en Street Fighter II (Genesis)

Agente: **O5-identidad-sf2**. Fecha 2026-08-27. Repo `~/TEC/LEIA/street_fighter`, HEAD `f65a2932`.
Alcance: sólo lectura + este .md. No se tocó ningún archivo del proyecto.

---

## 0. Método — de dónde salen los hexes (esto NO es de memoria)

El repo contiene **122 frames PNG capturados del ROM real** por el propio arnés del proyecto, en
`benchmarks/state_farm_work/` (320×224, RGB, salida nativa de Genesis). Toda la sección 1 de este
documento son colores **medidos con PIL sobre esos frames**, no recordados. Frames usados:

| Archivo | Qué muestra |
|---|---|
| `benchmarks/state_farm_work/dbg2_full_title.png` | Pantalla de título de *Special Champion Edition* (logo completo + menú) |
| `benchmarks/state_farm_work/RYU_RYU_R1_lvl1_live.png` | Combate en vivo, escenario de Ryu (Suzaku Castle) con HUD completo |
| `benchmarks/state_farm_work/RYU_SAGAT_R1_lvl1_live.png` | Combate en vivo, escenario de Sagat (templo de Tailandia, diurno) |
| `benchmarks/state_farm_work/RYU_MBISON_R1_lvl1_live.png` | Combate en vivo, escenario de M. Bison |
| `benchmarks/state_farm_work/map_opts_exit.png` | Pantalla de Options |
| `benchmarks/state_farm_work/ps_after_start.png` | Frame de arranque de partida |

Aviso operativo para quien reuse este material: **69 de los 122 frames están completamente en negro**
(capturas tomadas durante transiciones). Los frames útiles son los `*_live.png`, los `dbg*`, los
`map_*` y `ps_after_start.png`. Los `*_capture.png` de la escalera L1–L8 son casi todos negros.

### 0.1 El hallazgo técnico que gobierna toda la paleta: la escalera del DAC

Barriendo los 122 frames, **cada canal de cada píxel cae en 8 niveles y sólo 8**:

```
R y B: 0x00 0x20 0x40 0x60 0x88 0xA8 0xC8 0xE8      (0, 32, 64, 96, 136, 168, 200, 232)
G:     0x00 0x20 0x44 0x64 0x88 0xA8 0xCC 0xEC      (0, 32, 68, 100, 136, 168, 204, 236)
```

Es la tabla de conversión de color de 9 bits (8×8×8 = **512 colores posibles**, ~61 simultáneos en
pantalla en Mega Drive) que usa el emulador del proyecto. Dos consecuencias de diseño, no de trivia:

1. **El blanco de Street Fighter II no es `#FFFFFF`. Es `#E8ECE8`.** Ningún canal llega nunca a 255.
   El blanco del juego es 232/236/232: un blanco de fósforo, apenas cálido, 6% por debajo del máximo.
   Adoptar `#E8ECE8` como `--foreground` en oscuro es gratis, es fiel, y quita al dashboard el filo
   de "texto blanco puro sobre negro puro" que cansa en una jornada de ocho horas.
2. **Existe una regla de snap objetiva.** Cualquier color "del juego" en esta identidad debe caer en
   la escalera. Es un criterio verificable que sustituye al gusto: si un hex no está en la escalera,
   no viene del juego, y hay que decir de dónde viene. Lo aplico abajo y marco cada excepción.

### 0.2 Corrección a la premisa del encargo: el logo NO es naranja-rojo

El brief pedía "el logo con su naranja-rojo". **En esta versión no es así.** Medido sobre
`dbg2_full_title.png`: el logotipo de *Street Fighter II′ Special Champion Edition* en Genesis es un
**cromado azul→cian→verde-lima→crema sobre índigo profundo `#000060`**. El naranja-rojo pertenece
al logo de arcade/SNES y, aquí, sólo al subtítulo "SPECIAL CHAMPION EDITION" (`#E80000`) y al menú
(`#E8CC00` / `#E88800`). Cualquier dirección que quiera "el logo" tiene que ser cian-sobre-índigo,
no naranja. Lo digo porque es exactamente el tipo de detalle que un agente habría inventado al revés.

### 0.3 Cómo se leyó el estado actual

`grep -oE '#[0-9a-fA-F]{6}' src/scripts/web_dashboard.py` → **21 hexes distintos, 111 ocurrencias**.
Los tres más usados son `#94a3b8` (33), `#60a5fa` (15) y `#3b82f6` (14): **slate-400, blue-400 y
blue-500 de Tailwind, literales**. `#ef4444`, `#22c55e`, `#f59e0b`, `#a855f7`, `#e2e8f0`, `#1e293b`
completan el cuadro: la paleta actual es el default de Tailwind copiado a mano dentro de f-strings de
Python. **Ni uno solo de esos 21 hexes cae en la escalera del DAC.** Es decir: hoy el dashboard de un
proyecto de Street Fighter II no comparte un solo color con Street Fighter II.

---

## 1. Colores medidos, agrupados por fuente

Porcentajes = cobertura de píxeles del frame. Todos verificados dos veces (conteo global + conteo
por bandas de filas sobre el mismo frame).

### 1.A HUD de combate — `RYU_RYU_R1_lvl1_live.png`, banda y=12–30

El HUD es el objeto de diseño más relevante del juego para un dashboard: es literalmente un panel de
instrumentos de alta legibilidad diseñado para leerse de reojo, a distancia, en movimiento.

| Hex | Rol en el juego | % de la banda |
|---|---|---|
| `#E8CC00` | **Barra de vida llena** (amarillo-oro) | 7.6% |
| `#E88800` | Vida en zona media / naranja del degradado de la barra | 2.4% |
| `#E80000` | **Vida crítica + subtítulo "SPECIAL CHAMPION EDITION"** | 2.6% |
| `#E8ECE8` | Texto blanco del HUD (nombres, "KO", "PRESS") | 3.6% |
| `#000000` | Fondo del HUD | 72.1% |
| `#000020` | Sombra/marco interior de la caja de la barra | 9.5% |

Nota de lectura: **la barra de vida no usa verde en ningún momento**. La escala es
oro → naranja → rojo. Cualquier dirección que tome el HUD como raíz hereda ese vacío y tiene que
importar el verde de otra fuente (lo marco donde ocurre).

### 1.B Logo y pantalla de título — `dbg2_full_title.png` (34 colores)

| Hex | Rol |
|---|---|
| `#000060` | **Campo índigo del título** (59.6% del frame — el color dominante de la pantalla) |
| `#0088C8` | Cian medio del cromado del logo |
| `#40A8C8` · `#20A8C8` | Cian claro del cromado |
| `#0088E8` · `#2088E8` | Azul del cromado |
| `#C8EC40` · `#A8EC40` · `#88EC60` · `#60EC88` · `#40CC88` | Rampa verde-lima→verde del reflejo del logo |
| `#E8ECC8` · `#E8EC88` | Crema del filo superior de las letras |
| `#E80000` · `#E82000` · `#E84400` · `#E86400` · `#E88800` | Rampa roja→naranja de "SPECIAL CHAMPION EDITION" |
| `#E8CC00` | Oro de las opciones del menú (CHAMPION / HYPER / OPTIONS) |
| `#404440` · `#606460` · `#A8A8A8` | Grises de UI |

### 1.C Escenarios (gamas por etapa)

**Suzaku Castle / etapa de Ryu (nocturno)** — `RYU_RYU_R1_lvl1_live.png`:
- Cielo nocturno en cuatro pasos: `#000000` → `#000020` → `#000040` → `#000060`, más `#0000A8`.
- Bambú y tejados, rampa verde apagada: `#002020` `#204440` `#406460` `#608860` `#88A860` `#88A840` `#C8EC60`.
- Suelo de madera, rampa cálida: `#400000` `#602000` `#884400` `#A86400` `#886440` `#604420`.
- Farolillos / acentos cálidos: `#E8A820` `#E8A860` `#E88800`.
- Gi de Ryu (blanco sucio): `#E8ECE8` `#C8CCA8` `#A88860`; cinta roja: `#E80000` `#C84420` `#882000`.
- Gi del Ryu espejo (P2, azul): `#4064C8` `#206488` `#40A8E8` `#88CCE8`.

**Templo de Sagat (diurno)** — `RYU_SAGAT_R1_lvl1_live.png`:
- Cielo diurno: `#60CCE8` (8.6%), `#A8ECE8`.
- Follaje: `#A8CC60` (9.9%), `#60A840`, `#406420`, `#006400`.
- Buda / piedra dorada: `#E8CC88` (6.0%), `#E8A840`, `#C88820`, `#C8CCA8`.
- Terracota de la túnica: `#C84420`, `#E88860`, `#E8A888`.
- Tierra: `#602000`, `#884400`, `#A86400`, `#404420`, `#606440`.

**Etapa de M. Bison** — `RYU_MBISON_R1_lvl1_live.png`:
- Grises de piedra: `#C8CCC8` (5.6%), `#A8A8A8`, `#888888`.
- Carmín apagado / sangre seca: `#600000` (4.3%), `#400000`, `#A82000`, `#C84420`.
- Oliva militar: `#404420`, `#606440`, `#888860`, `#202000`, `#408820`.
- Oro de las charreteras: `#E8CC00`, `#E8CC60`, `#E88800`, `#E8ECA8`.

**Pantalla de Options** — `map_opts_exit.png` (la superficie de "herramienta" del propio juego):
- Fondo lavanda-piedra `#A8A888` (16.5%), `#A8A8C8`, `#8888A8`, `#606488`, `#404460`.
- Texto/realces: `#E8CC60`, `#C8A840`, `#E8ECE8`, `#88A8E8`.

---

## 2. Las reglas que hacen que esto NO se vea chillón

Antes de las direcciones, el criterio compartido. Las cinco lo cumplen; sin él, cualquiera de las
cinco se vuelve un juguete.

1. **Chasis derivado, datos literales.** `background`, `card`, `muted`, `border` y `input` **no** son
   hexes del juego: son derivaciones de baja croma del *tinte* de la fuente. Sólo `primary`,
   `secondary`, `accent`, `destructive`, `success` y `warning` usan hexes literales de la escalera.
   El color saturado del juego aparece entonces en **menos del 5% del área de pantalla**, que es
   exactamente la proporción que tiene en el juego (la barra de vida es 7.6% de su propia banda, y
   ~1% del frame). Ésa es la diferencia entre "inspirado en SF2" y "disfrazado de SF2".
2. **Techo de luminancia 0xE8/0xEC.** Ningún blanco es `#FFFFFF`, ni en claro ni en oscuro. Es fiel
   al hardware y baja el glare en jornada larga.
3. **Piso de negro `#0B0C0D`, no `#000000`.** El juego usa negro puro porque es un CRT; un monitor
   moderno con texto encima produce halo. Todos los fondos oscuros están 4–6% por encima del negro.
4. **Un solo acento por pantalla.** Las direcciones traen dos acentos porque la app necesita
   distinguir "acción primaria" de "estado en curso", pero **nunca los dos en el mismo bloque**.
5. **Sin neón por construcción.** El neón nace de croma alta *más* luminancia alta. Los hexes del
   juego con esas dos propiedades a la vez (`#60EC88`, `#C8EC40`, `#60CCE8`, `#88EC60`) están
   **prohibidos como superficie o como texto de cuerpo**; se permiten sólo como trazo de 1–2 px
   (serie de gráfica, borde de foco) y lo digo en cada dirección donde aparecen.
6. **Emoji fuera.** La evidencia de campo reporta emoji en todas las etiquetas de pestaña. El emoji
   es color que no se puede gobernar con tokens: mete verdes, azules y amarillos ajenos a la paleta
   en cada etiqueta. Reemplazo por iconografía monocroma que herede `currentColor`.

### 2.1 Contraste — números medidos, no afirmados

Calculé WCAG 2.1 sobre **cada par relevante de las diez variantes** (5 direcciones × claro/oscuro).
Resultado: `foreground/background` entre **13.20 y 16.86**, `muted-foreground/background` entre
**5.36 y 8.22**, y todos los pares `*-foreground` sobre su relleno ≥ **4.5**, con tres excepciones
declaradas abajo. La tabla completa por dirección va en cada bloque.

**Regla de borde** (el único "fallo" sistemático y es intencional): `--border` queda en ~1.3–1.6:1
en las diez variantes. Es correcto — un separador decorativo no es objeto de UI y WCAG no le exige
3:1. Pero por eso **separo tres tokens que el default de shadcn confunde**:
- `--border` decorativo (~1.5:1), sólo válido cuando las superficies ya se distinguen entre sí;
- `--input` = contorno de control, **medido ≥ 2.85:1** (objetivo 3:1) en las diez variantes;
- `--ring` = foco de teclado, **siempre = el `primary` de la variante, ≥ 4.5:1**.
En un dashboard con 101 controles esto no es purismo: el contorno del control ES la retícula.

---

## 3. Las cinco direcciones

Cada una: nombre, raíz medida, roles semánticos, bloque de tokens claro+oscuro, por qué no es
chillona, y su riesgo real. Están ordenadas de más segura a más arriesgada.

---

### D1 · **Suzaku Nocturno** — bambú, madera y farol

**Raíz:** el escenario de Ryu de noche (`RYU_RYU_R1_lvl1_live.png`). Cielo `#000060`→`#000000`,
bambú `#204440`/`#608860`/`#88A860`, tarima `#884400`/`#A86400`, farol `#E8A820`.

**Roles (8 hexes del juego):**

| Rol | Oscuro | Claro | Origen medido |
|---|---|---|---|
| Acento primario (lanzar, confirmar) | `#E8A820` | `#884400` | farol / tarima |
| Acento secundario (en curso, liga) | `#608860` | `#406460` | bambú medio / tejado |
| Éxito | `#88A860` | `#3F5A3A`* | bambú claro |
| Alerta | `#E8CC00` | `#886400` | oro del HUD |
| Error | `#E83A2A`* | `#A81400`* | rojo de vida crítica, reencuadrado |
| Texto | `#E8ECE8` | `#14201B`* | blanco del gi |

\* fuera de la escalera: derivado para cumplir contraste. Se marca siempre.

```css
/* D1 Suzaku Nocturno — CLARO */
:root {
  --background: #EDF0EA;  --foreground: #14201B;
  --card: #F7F9F4;        --card-foreground: #14201B;
  --popover: #F7F9F4;     --popover-foreground: #14201B;
  --muted: #E0E5DB;       --muted-foreground: #55655C;
  --border: #CDD6C8;      --input: #77896F;   --ring: #884400;
  --primary: #884400;     --primary-foreground: #E8ECE8;
  --secondary: #406460;   --secondary-foreground: #E8ECE8;
  --accent: #3F5A3A;      --accent-foreground: #E8ECE8;
  --destructive: #A81400; --destructive-foreground: #E8ECE8;
  --success: #3F5A3A;     --warning: #886400;   /* extensiones: shadcn no las trae */
}
/* D1 Suzaku Nocturno — OSCURO */
.dark {
  --background: #0A0F0F;  --foreground: #E8ECE8;
  --card: #121918;        --card-foreground: #E8ECE8;
  --popover: #121918;     --popover-foreground: #E8ECE8;
  --muted: #1A2321;       --muted-foreground: #9BAAA1;
  --border: #2A3532;      --input: #556661;   --ring: #E8A820;
  --primary: #E8A820;     --primary-foreground: #0A0F0F;
  --secondary: #608860;   --secondary-foreground: #0A0F0F;
  --accent: #88A860;      --accent-foreground: #0A0F0F;
  --destructive: #E83A2A; --destructive-foreground: #0A0F0F;
  --success: #88A860;     --warning: #E8CC00;
}
```

Contraste medido — oscuro: fg/bg **16.17**, muted-fg/bg **7.96**, primary/bg **9.24**,
destructive/bg **4.66**, input/bg **3.18**. Claro: fg/bg **14.58**, muted-fg/bg **5.36**,
primary/bg **6.35**, input/bg **3.26**.

**Por qué no es chillona:** el eje es ámbar-sobre-verde-oscuro, dos familias que en el juego ya
conviven apagadas — el bambú de SF2 es `#608860`, un verde de 53% de luminosidad y croma baja, no un
verde de UI. El ámbar `#E8A820` es el único punto caliente y ocupa el ~1% de la pantalla. En claro,
el primario cae a `#884400`, un marrón de madera; no hay un solo color de croma alta en modo claro.

**Riesgo:** el ámbar como primario compite visualmente con `--warning` (`#E8CC00`): dos amarillos
separados sólo por 40 puntos de rojo. Mitigación: `warning` sólo aparece con icono y borde, nunca
como relleno de botón. Si el equipo no puede sostener esa disciplina, esta dirección se degrada.

---

### D2 · **Barra de Vida** — el HUD como panel de instrumentos

**Raíz:** exclusivamente el HUD (`RYU_RYU_R1_lvl1_live.png` y=12–30). Oro `#E8CC00`, naranja
`#E88800`, rojo `#E80000`, blanco `#E8ECE8`, negro `#000000`.

**La idea:** el HUD de SF2 ya es un panel de telemetría en tiempo real de alta legibilidad, con una
escala oro→naranja→rojo que **codifica degradación**. El dashboard tiene exactamente eso: win-rate,
progreso de escalera, salud del proceso, nivel de CPU. La paleta no es decoración; es un mapeo
directo de la semántica del juego a la semántica de la herramienta.

| Rol | Oscuro | Claro | Origen |
|---|---|---|---|
| Primario (barra llena / acción) | `#E8CC00` | `#E8CC00` (relleno) / `#6E5A00`* (texto) | vida llena |
| Secundario (degradación / en curso) | `#E88800` | `#8A5000`* | zona media de la barra |
| Acento | `#E8A820` | `#6E5A00`* | farol |
| Error / KO | `#E83A2A`* | `#B41200`* | vida crítica |
| Éxito | `#88A860` | `#3F5A3A`* | **importado del bambú — el HUD no tiene verde** |

```css
/* D2 Barra de Vida — CLARO */
:root {
  --background: #F4F3EF;  --foreground: #16171A;
  --card: #FBFAF7;        --card-foreground: #16171A;
  --popover: #FBFAF7;     --popover-foreground: #16171A;
  --muted: #E7E6E1;       --muted-foreground: #5B5D64;
  --border: #D6D5CE;      --input: #8B8A84;   --ring: #8A5000;
  --primary: #E8CC00;     --primary-foreground: #16171A;
  --secondary: #8A5000;   --secondary-foreground: #E8ECE8;
  --accent: #6E5A00;      --accent-foreground: #E8ECE8;
  --destructive: #B41200; --destructive-foreground: #E8ECE8;
  --success: #3F5A3A;     --warning: #8A5000;
}
/* D2 Barra de Vida — OSCURO */
.dark {
  --background: #0B0B0D;  --foreground: #E8ECE8;
  --card: #14151A;        --card-foreground: #E8ECE8;
  --popover: #14151A;     --popover-foreground: #E8ECE8;
  --muted: #1C1E24;       --muted-foreground: #9DA0A9;
  --border: #2B2E36;      --input: #5B5F6A;   --ring: #E8CC00;
  --primary: #E8CC00;     --primary-foreground: #0B0B0D;
  --secondary: #E88800;   --secondary-foreground: #0B0B0D;
  --accent: #E8A820;      --accent-foreground: #0B0B0D;
  --destructive: #E83A2A; --destructive-foreground: #0B0B0D;
  --success: #88A860;     --warning: #E88800;
}
```

Contraste medido — oscuro: fg/bg **16.48**, muted-fg/bg **7.52**, primary/bg **12.24**,
primary-fg sobre primary **12.24**, input/bg **3.08**. Claro: fg/bg **16.14**,
primary-fg sobre primary **11.15**, input/bg **3.12**.

**Por qué no es chillona:** porque el amarillo es **relleno, no texto**, y el chasis es gris
neutro-cálido casi sin croma. Un amarillo saturado sobre un gris muerto lee como instrumento
(tacómetro, cinta de seguridad), no como fiesta. El juego hace exactamente esto: `#E8CC00` sobre
`#000000`.

**Riesgo — y es un riesgo duro, medido:** en modo claro, `primary/background` da **1.45:1**. El oro
**no puede ser texto ni borde fino sobre fondo claro, nunca**. Es un token de relleno con
`--primary-foreground` oscuro encima. Todo enlace o texto "primario" en claro tiene que usar
`--accent: #6E5A00` (5.63:1). Si esa regla no se codifica en los componentes, el modo claro de esta
dirección es ilegible. Es la dirección más fuerte de identidad y la que más disciplina exige.

---

### D3 · **Champion Chrome** — el título: cian sobre índigo

**Raíz:** `dbg2_full_title.png`. Campo `#000060`, cromado `#0088C8`/`#40A8C8`, reflejo verde-lima
`#C8EC40`, crema `#E8ECC8`, oro del menú `#E8CC00`.

| Rol | Oscuro | Claro | Origen |
|---|---|---|---|
| Primario | `#40A8C8` | `#206488` | cromado del logo |
| Secundario | `#3E85C0`* | `#00688C`* | azul del logo |
| Acento (chispa) | `#C8EC40` | `#4A6B00`* | reflejo verde-lima |
| Éxito | `#40CC88` | `#00684A`* | verde del logo |
| Alerta | `#E8CC00` | `#886400` | oro del menú |
| Error | `#E83A2A`* | `#A81400`* | rojo del subtítulo |

```css
/* D3 Champion Chrome — CLARO */
:root {
  --background: #EEF0F6;  --foreground: #0F1424;
  --card: #F8F9FC;        --card-foreground: #0F1424;
  --popover: #F8F9FC;     --popover-foreground: #0F1424;
  --muted: #E0E4EE;       --muted-foreground: #4E5670;
  --border: #C9CFDE;      --input: #7B84A0;   --ring: #206488;
  --primary: #206488;     --primary-foreground: #E8ECE8;
  --secondary: #00688C;   --secondary-foreground: #E8ECE8;
  --accent: #4A6B00;      --accent-foreground: #E8ECE8;
  --destructive: #A81400; --destructive-foreground: #E8ECE8;
  --success: #00684A;     --warning: #886400;
}
/* D3 Champion Chrome — OSCURO */
.dark {
  --background: #05070F;  --foreground: #E8ECE8;
  --card: #0D1020;        --card-foreground: #E8ECE8;
  --popover: #0D1020;     --popover-foreground: #E8ECE8;
  --muted: #161B30;       --muted-foreground: #9AA3BE;
  --border: #262C48;      --input: #525E8C;   --ring: #40A8C8;
  --primary: #40A8C8;     --primary-foreground: #05070F;
  --secondary: #3E85C0;   --secondary-foreground: #05070F;
  --accent: #C8EC40;      --accent-foreground: #05070F;
  --destructive: #E83A2A; --destructive-foreground: #05070F;
  --success: #40CC88;     --warning: #E8CC00;
}
```

Contraste medido — oscuro: fg/bg **16.86**, muted-fg/bg **8.01**, primary/bg **7.33**,
secondary-fg sobre secondary **5.10**, input/bg **3.20**. Claro: fg/bg **16.08**,
primary/bg **5.69**, input/bg **3.26**.

**Por qué no es chillona:** el cian del logo es `#0088C8` — **matiz ~219° con el rojo en cero**, un
cian de teléfono, no el azul-cielo de Tailwind. Y el fondo `#000060` puro es índigo-tinta, no
`slate-900`. La distancia perceptual con `#3b82f6` es grande aunque ambos sean "azules". El
verde-lima `#C8EC40` se usa **sólo como filo de 1–2 px** (foco, serie de gráfica), jamás como
superficie: es el color más peligroso de todo el documento.

**Riesgo — el político, y es serio:** el dueño pidió literalmente **quitar el azul**. Ésta es la única
de las cinco que sigue siendo, en una descripción de una palabra, "azul". Su defensa es que es
*el azul del título del juego*, medido, y no el default heredado. Su debilidad es que en una
demostración de treinta segundos nadie percibe la diferencia entre "índigo de SF2" y "el azul de
antes". **La incluyo por completitud del abanico, no como recomendación.**

---

### D4 · **Templo Diurno** — la única dirección de claro primero

**Raíz:** `RYU_SAGAT_R1_lvl1_live.png`. Cielo `#60CCE8`, follaje `#A8CC60`/`#006400`, piedra dorada
`#E8CC88`/`#C88820`, terracota `#C84420`.

**La idea:** las otras cuatro son oscuras nativas. Un stand promocional se monta bajo iluminación de
salón, muchas veces con proyector o pantalla a máximo brillo y luz ambiente alta, donde un tema
oscuro se lava y refleja caras. Ésta nace en claro y **su modo oscuro es el derivado**, no al revés.

| Rol | Claro | Oscuro | Origen |
|---|---|---|---|
| Primario | `#006400` | `#A8CC60` | verde profundo del templo / follaje claro |
| Secundario | `#8A5A00`* | `#E8A840` | piedra dorada del Buda |
| Acento | `#A83214`* | `#E88860` | terracota de la túnica |
| Éxito | `#006400` | `#A8CC60` | follaje |
| Alerta | `#8A5A00`* | `#E8A840` | oro |
| Error | `#A81400`* | `#E83A2A`* | rojo, reencuadrado |

```css
/* D4 Templo Diurno — CLARO (modo nativo) */
:root {
  --background: #E9E7DC;  --foreground: #1E2114;
  --card: #F5F3EC;        --card-foreground: #1E2114;
  --popover: #F5F3EC;     --popover-foreground: #1E2114;
  --muted: #DEDBCC;       --muted-foreground: #5A5C48;
  --border: #CAC7B4;      --input: #84816D;   --ring: #006400;
  --primary: #006400;     --primary-foreground: #E8ECE8;
  --secondary: #8A5A00;   --secondary-foreground: #E8ECE8;
  --accent: #A83214;      --accent-foreground: #E8ECE8;
  --destructive: #A81400; --destructive-foreground: #E8ECE8;
  --success: #006400;     --warning: #8A5A00;
}
/* D4 Templo Diurno — OSCURO (derivado) */
.dark {
  --background: #0E120B;  --foreground: #E8ECE8;
  --card: #161C11;        --card-foreground: #E8ECE8;
  --popover: #161C11;     --popover-foreground: #E8ECE8;
  --muted: #1E2618;       --muted-foreground: #9BA68C;
  --border: #2E3A24;      --input: #586A47;   --ring: #A8CC60;
  --primary: #A8CC60;     --primary-foreground: #0E120B;
  --secondary: #E8A840;   --secondary-foreground: #0E120B;
  --accent: #E88860;      --accent-foreground: #0E120B;
  --destructive: #E83A2A; --destructive-foreground: #0E120B;
  --success: #A8CC60;     --warning: #E8A840;
}
```

Contraste medido — claro: fg/bg **13.20**, muted-fg/bg **5.53**, primary/bg **6.00**,
primary-fg sobre primary **6.23**, input/bg **3.17**. Oscuro: fg/bg **15.85**,
primary/bg **10.33**, input/bg **3.21**.

**Por qué no es chillona:** el fondo `#E9E7DC` es piedra caliza con tinte verde-oliva, **no** el crema
`#F4F1EA` que es el default reconocible de la IA generativa; el verde primario `#006400` es un hex
literal del juego, casi negro (luminancia 20%), que lee como tinta de sello, no como "verde de éxito".
El cielo `#60CCE8` — el color más brillante del frame, 8.6% de cobertura — está **deliberadamente
excluido de los tokens**: es exactamente el tipo de cian caramelo que volvería juguete la pantalla.
Su lugar es una serie de gráfica, y nada más.

**Riesgo:** `primary` y `success` son el mismo color. En una app que muestra "corriendo / terminado /
ganó" eso es una colisión real. Mitigación: `success` se expresa con `--success` en fondo tenue +
icono, y el primario sólo como relleno de botón; si el equipo no quiere esa distinción, hay que
mover `success` a `#406420` (follaje medio, también literal del juego).

---

### D5 · **Ceniza y Carmín** — la más sobria de las cinco

**Raíz:** `RYU_MBISON_R1_lvl1_live.png`. Grises de piedra `#C8CCC8`/`#A8A8A8`/`#888888`, carmín
apagado `#600000`/`#A82000`, oliva militar `#404420`/`#606440`, oro `#E8CC00`.

**La idea:** es la dirección para las ocho horas. El chasis es un gris cálido casi acromático y el
único color de la pantalla es un carmín de ladrillo. Todo lo que no es acción es ceniza.

| Rol | Oscuro | Claro | Origen |
|---|---|---|---|
| Primario | `#DC5238`* | `#A82000` | carmín de la etapa, aclarado en oscuro por contraste |
| Secundario | `#888888` | `#4C4E38`* | piedra / oliva militar |
| Acento | `#E8CC00` | `#886400` | oro de las charreteras |
| Texto suave | `#A8A8A8` | `#5C5A56`* | gris de piedra (literal del juego en oscuro) |
| Error | `#E80000` | `#C80000` | rojo puro del HUD |
| Éxito | `#88A860` | `#3F5A3A`* | **importado del bambú — esta etapa no tiene verde legible** |

```css
/* D5 Ceniza y Carmín — CLARO */
:root {
  --background: #EFEEEC;  --foreground: #1A1917;
  --card: #F8F7F5;        --card-foreground: #1A1917;
  --popover: #F8F7F5;     --popover-foreground: #1A1917;
  --muted: #E2E0DD;       --muted-foreground: #5C5A56;
  --border: #D2CFCA;      --input: #848179;   --ring: #A82000;
  --primary: #A82000;     --primary-foreground: #E8ECE8;
  --secondary: #4C4E38;   --secondary-foreground: #E8ECE8;
  --accent: #886400;      --accent-foreground: #E8ECE8;
  --destructive: #C80000; --destructive-foreground: #E8ECE8;
  --success: #3F5A3A;     --warning: #886400;
}
/* D5 Ceniza y Carmín — OSCURO */
.dark {
  --background: #0D0C0B;  --foreground: #E8ECE8;
  --card: #161514;        --card-foreground: #E8ECE8;
  --popover: #161514;     --popover-foreground: #E8ECE8;
  --muted: #201E1C;       --muted-foreground: #A8A8A8;
  --border: #302D2A;      --input: #666158;   --ring: #DC5238;
  --primary: #DC5238;     --primary-foreground: #0D0C0B;
  --secondary: #888888;   --secondary-foreground: #0D0C0B;
  --accent: #E8CC00;      --accent-foreground: #0D0C0B;
  --destructive: #E80000; --destructive-foreground: #E8ECE8;
  --success: #88A860;     --warning: #E8CC00;
}
```

Contraste medido — oscuro: fg/bg **16.37**, muted-fg/bg **8.22**, primary/bg **4.94**,
accent/bg **12.16**, input/bg **3.18**. Claro: fg/bg **15.15**, muted-fg/bg **5.93**,
primary/bg **6.31**, input/bg **3.36**.

**Por qué no es chillona:** es la que menos color tiene, punto. El 92% del área es gris cálido de
croma casi nula. `#A82000` es un carmín de óxido con 6% de luminancia relativa; el oro sólo aparece
en `--accent`, reservado para el estado "corriendo".

**Riesgo — el más grave de las cinco:** `primary` (carmín) y `destructive` (rojo) son **la misma
familia de matiz**, separados sólo por luminancia y saturación. En un dashboard donde
"Force Kill (No Save)" convive con "Launch", eso es peligroso. Sólo es aceptable si toda acción
destructiva pasa por un `alert-dialog` (que D3-registry ya identificó como `@shadcn/alert-dialog`,
hoy inexistente en `web_dashboard.py:1810-1811, :1902-1903, :1984`) y lleva icono propio. **Si esa
condición no se cumple, esta dirección se descarta.** Alternativa si se descarta: intercambiar los
roles — oro `#E8CC00` como primario, carmín reservado exclusivamente a destructivo.

---

## 4. Cuadro comparativo para decidir

| | D1 Suzaku | D2 Barra de Vida | D3 Champion Chrome | D4 Templo Diurno | D5 Ceniza y Carmín |
|---|---|---|---|---|---|
| Modo nativo | oscuro | oscuro | oscuro | **claro** | oscuro |
| Familia dominante | ámbar / verde bambú | oro / naranja sobre gris | cian sobre índigo | verde tinta / piedra | ceniza / carmín |
| Reconocible como SF2 | media | **muy alta** (es el HUD) | **muy alta** (es el logo) | media-alta | baja-media |
| Jornada de 8 h | muy buena | buena | buena | buena bajo luz alta | **la mejor** |
| Dignidad en demo | buena | **la mejor** | muy buena | buena | sobria, poco memorable |
| Cumple "nada de azul" | sí | sí | **no** | sí | sí |
| Disciplina que exige | media | **alta** (oro nunca es texto en claro) | alta (lima sólo 1 px) | media | **alta** (dos rojos) |

**Mi lectura, si se me pide una:** **D2 Barra de Vida** como dirección principal y **D1 Suzaku** como
la segura. D2 es la única cuya paleta *significa* algo — la escala oro→naranja→rojo es el mapeo del
juego a la telemetría del dashboard, y hace que la identidad no sea un tema encima sino una decisión
de producto. D1 es la que menos puede salir mal. D5 sirve si alguien vetara el amarillo. D4 sólo si
el stand se monta bajo luz fuerte. D3 va contra la instrucción explícita del dueño.

---

## 5. Radio de impacto (obligatorio por contrato)

Componente/token compartido que tocaría cualquiera de estas direcciones. **Consumidores completos:**

1. **`theme=gr.themes.Soft(primary_hue="blue")`** en `src/scripts/web_dashboard.py:2320`
   (dentro de `demo.queue().launch(...)`). Único consumidor: ese archivo. D1-inventario y D2-deuda
   confirmaron por introspección del venv (`gradio==6.25.0`) que `launch()` sí acepta `theme` y
   `Blocks.__init__` no; el azul se aplica de verdad. Este punto cubre **sólo** los componentes
   nativos de Gradio (píldoras de etiqueta, botones, dropdowns).
2. **Los 21 hexes literales / 111 ocurrencias** dentro de f-strings de HTML del mismo archivo,
   más los **121 atributos `style=` inline** que reportó D2-deuda. Ninguno obedece a `primary_hue`.
   Se concentran en la cubeta C de D2-deuda: `get_stand_checkpoint_status` (349-395, hoy muerta),
   `get_league_pool_status_html` (1104-1177), `get_auto_curriculum_status_html` (1220-1358),
   `get_live_telemetry_html` (1438-1704) y `compute_fighter_visual_coords` (1395-1437).
3. **`compute_fighter_visual_coords`** merece mención aparte: dibuja las posiciones de los
   luchadores sobre una representación del escenario. Es el único lugar del dashboard donde el color
   **debe** coincidir con el juego, porque representa el juego. Cualquier dirección elegida tiene que
   darle una serie de colores explícita.
4. **No hay más consumidores.** No existe React, `components.json`, capa de tokens ni CSS externo;
   `src/scripts/web_dashboard.py` es el único archivo con color en todo el repo (verificado por
   `grep` de hexes sobre `src/`). El radio de impacto de un cambio de paleta es **un archivo**.

Consecuencia práctica: **cambiar la paleta no es cambiar una línea.** Es 1 línea (el tema) + 202
sitios literales. Ese número es el argumento más fuerte a favor de reemplazar la cubeta C por
componentes con tokens en vez de hacer buscar-y-reemplazar de hexes — coincide con lo que concluyó
D2-deuda por otra vía.

---

## 6. Notas sueltas y hallazgos fuera de alcance (registrados, no accionados)

- Los `--chart-1..5` de shadcn necesitan valor en **ambos** temas o las series se ven mal en uno
  (lo señaló D3-registry leyendo `chart.tsx`). Serie sugerida por dirección, todos hexes del juego:
  D1 `#E8A820 #608860 #4064C8 #C8EC60 #E80000` · D2 `#E8CC00 #E88800 #E80000 #88A860 #E8ECE8` ·
  D3 `#40A8C8 #C8EC40 #E8CC00 #E80000 #E8ECC8` · D4 `#006400 #C88820 #C84420 #60CCE8 #606440` ·
  D5 `#A82000 #E8CC00 #888888 #606440 #C8CCC8`. Aquí sí se permite `#60CCE8` y `#C8EC40`: son trazos
  de 2 px, no superficies.
- **El emoji de las etiquetas de pestaña es contaminación de paleta**, no sólo ruido de estilo: cada
  emoji inyecta colores fuera del sistema en un elemento de navegación. Cualquier dirección de estas
  se rompe visualmente si los emoji se quedan.
- `#94a3b8` (slate-400) aparece **33 veces** — es el color más usado del dashboard y es el texto
  secundario. Reemplazarlo por `--muted-foreground` de la dirección elegida es, por sí solo, el
  cambio de mayor superficie visible con menor esfuerzo.
- 69 de los 122 PNG en `benchmarks/state_farm_work/` están totalmente en negro (capturas de
  transición). No es un asunto de UI, pero si alguien vuelve a usar ese directorio como evidencia,
  conviene saberlo antes de sacar conclusiones de un frame vacío.
- Los tres frames "en vivo" (`*_live.png`) son de mucha mejor calidad como evidencia que los
  `*_capture.png`. Si el proyecto quiere material gráfico para el stand, el capturador debería
  guardar frames en vivo por defecto.
