# _MASTER — Reconstrucción del Control Center de LEIA

**Corrida:** `wvs1t0vow` / `wf_c467ca47-f30` · 2026-08-27
**Skill:** `uiux-workflow-architect-v2` · **Archetype B — Planning / integration**
**Objetivo:** `LEIA-qro/street_fighter` @ `f65a2932`, rama `stage0-metrics-and-semantics`
**Artefacto auditado:** `src/scripts/web_dashboard.py` (2,326 líneas, build `v1592-unified-r8`)

---

## Encuadre y desviaciones declaradas del spine de la skill

La skill está escrita contra **`focaltec/orchestrator`** (React + shadcn + Tailwind, canon
instalado en `apps/dashboard`). El objetivo de esta corrida es **otro repo, otro stack y otro
proyecto**, y las reglas globales del usuario prohíben mezclar ambos contextos. Desviaciones:

| Punto de la skill | Aquí | Por qué |
|---|---|---|
| Step 0: leer `components.json` / `styles.css` / `modes.md` | **No aplica** | Ese canon es del otro repo. El canon local, leído hoy: **no existe** — colores hexadecimales literales en f-strings de Python + `gr.themes.Soft(primary_hue="blue")`. |
| shadcn MCP (`get_project_registries`, `search_items_in_registries`) | **CLI en su lugar** | El MCP existe pero está *project-scoped* en `/Users/felipe/projects/orchestrator/.mcp.json` con `-c apps/dashboard`; no está cargado en esta sesión y apuntarlo aquí importaría el canon ajeno. Verificado funcionando: `bunx --bun shadcn@latest search @shadcn -q "tabs"` → 16 items. |
| Paleta/tokens: dimensión **cerrada permanentemente** | **RE-ABIERTA** | No hay canon que las posea, y el dueño las pidió explícitamente. Precedente del propio §5: un token autorizado por el dueño no es un agente inventando paleta. |
| `frontend-design` / `taste-skill` gateados [D] | **Cargados** (sólo en O5) | Misma autorización del dueño. |
| React correctness | **Sustituido** | No hay React en el objetivo; se reemplaza por corrección Python/Gradio y por arquitectura de front en la propuesta. |
| Driving en vivo con N pestañas paralelas (`04`) | **Capturado por el orquestador** | Evita el riesgo de contaminación entre pestañas: la evidencia viva se tomó una sola vez y se dejó por escrito en `_EVIDENCIA-CAMPO.md`. |
| PR contra `main` + revisión de Facundo | **No aplica** | Este repo va en `stage0-metrics-and-semantics`; el entregable es plan + artifact, no PR. |

---

## Hechos verificados antes de repartir (no de memoria)

- El dashboard **sí corre en macOS** (`core.config` y `scripts.web_dashboard` importan): se
  levantó en `127.0.0.1:7861`. Lo que no funciona fuera de Windows son los subprocesos que
  lanza — arnés, no defecto.
- Inventario UI **antes vs ahora** (`b338c906` → `HEAD`): 1,667 → 2,326 líneas · `gr.Tab`
  10 → 10 · `gr.Dropdown` 32 → 35 · `gr.Button` 22 → 23. **La sobrecarga de pestañas y
  dropdowns es HEREDADA, no la introdujo la era ChatGPT.**
- La paleta azul tiene dos capas: el tema (`primary_hue="blue"`) y hexes literales
  (`#3b82f6`, `#60a5fa`, `#93c5fd`, `#a855f7`) repartidos por el HTML incrustado.
- Pendiente de verificar por agente: `theme=` se pasa a `.launch()`, no a `gr.Blocks()` —
  puede estar siendo ignorado por Gradio.

---

## Agentes despachados

| Fase | Agente | Rol | Skills cargadas |
|---|---|---|---|
| Discover | `D1-inventario` | inventario funcional exhaustivo | `design-audit` |
| Discover | `D2-deuda` | historia b338c906→HEAD + cubetas de código Python | `code-health-review` |
| Discover | `D3-registry` | mapeo de necesidades → items reales del registry (CLI) | `composition-patterns` |
| Options | `O1-arquitectura-informacion` | la IA nueva, la compactación | `ui-ux-pro-max` (mitad UX) |
| Options | `O2-modo-stand` | el flujo del stand público | `bencium-controlled-ux-designer` |
| Options | `O3-stack` | stack, API Python, transporte en vivo, transición | `react-best-practices` |
| Options | `O4-composicion` | pantalla → componentes | `composition-patterns` |
| Options | `O5-identidad-sf2` | identidad visual SF2 → direcciones de paleta | `frontend-design` + `taste-skill` |
| Options | `O6-layout-a11y` | densidad, jerarquía, accesibilidad | `web-design-guidelines` + `ui-refactor` |
| Options | `O7-que-matar` | funcionalidad muerta / redundante / huérfana | `edge-case-hunter` |
| Judge | `J1-viabilidad` | ¿lo puede construir este equipo sin quedarse sin herramienta? | — |
| Judge | `J2-valor-usuario` | ¿mejora la vida de las 4 personas que lo usan? | — |
| Judge | `J3-coherencia` | ¿encajan las siete propuestas entre sí? | — |
| Plan | `PLAN-final` | el documento integrado | — |

Cada agente escribe su propio `.md` en este directorio.

---

## Resultados

**Corrida completa: 14/14 agentes, 0 errores, ~33 min, 1.67M tokens de subagente.**
Documento final: **`PLAN-final.md`** (42.7 KB, 8 secciones + decisiones para Felipe).

### El veredicto que no esperábamos

El panel concluyó, por unanimidad de sus tres lentes, que **NO se reconstruya el dashboard
entero** — y lo sostuvo con evidencia, no con opinión: **el equipo ya no lo usa**. De los 5
runbooks vivos, sólo `RUN_STAND_LEIA.md` lo abre, y para una sola pestaña. Lo que sí vale
construir es lo que no existe y tiene fecha (el Modo Stand) más la poda de lo que no puede
funcionar. La migración a React+shadcn **no se descarta: se aplaza y se condiciona** — mejor
documento técnico del panel (J3 8/10), peor plan de ejecución para tres personas con un evento
encima (J1 3/10).

### Afirmaciones code-truthed por el orquestador (todas confirmadas)

| Afirmación del panel | Verificación |
|---|---|
| `ray` no está en requirements → la pestaña PBT no puede arrancar | `grep -ci '^ray' requirements*.txt` → **0** ✓ |
| SAC lanza `NotImplementedError` en ambos entrypoints | `src/agents/sac/agent.py:113` y `:334` ✓ |
| `models/production/league/` está vacío → league nunca produjo nada | directorio vacío ✓ |
| `stand_leia.py` no llama `write_telemetry` → la pestaña Telemetría es imposible para el flujo vivo | **0 referencias** ✓ |
| `gr.Blocks.route` existe en el gradio instalado → las vistas del stand cuestan una función | gradio **6.25.0**, `def route` presente ✓ |
| Sólo 1 de 5 runbooks abre el dashboard | `RUN_STAND_LEIA.md`, único ✓ |
| No hay ni un `.db` de Optuna en el árbol | **0** ✓ |
| `gr.Timer(value=0.1)` repinta 10 veces/s una cadena estática | `web_dashboard.py:2133` ✓ |

### Los números del rediseño propuesto

| Métrica | Hoy | Después |
|---|---|---|
| Destinos de navegación | 10 | 4 + 2 rutas de stand |
| Controles que preguntan «qué modelo» | **24**, en 5 vocabularios | ~5, en 1 |
| `gr.Dropdown` | 35 | ~10 |
| Zonas de carga | 11 | 1 |
| Botones de Detener | 5 globales sin confirmación | 1 confirmado + 1 aislado |
| Consolas (las 3 con el MISMO `elem_id`) | 3 | 1 |
| Clics para la tarea del stand | ~10 + 2 scrolls | **2** |

### Entregables

- `PLAN-final.md` — el plan completo, con fases; la **Fase 1 es el vertical slice** que ya sirve
  en el stand y es lo único indispensable antes del evento.
- `O5-identidad-sf2.md` — cinco paletas **muestreadas de píxeles reales** del juego, con
  contrastes calculados y riesgos honestos (una se auto-descalifica si no se cumple su condición).
- Artifact de paletas: https://claude.ai/code/artifact/53cfb6ff-717c-43b5-800d-cbbe06a91747

## 🌅 Para Felipe

Las 11 decisiones están al final de `PLAN-final.md`, cada una con recomendación del panel y qué
implica cada opción. Las tres que bloquean todo lo demás:

1. **¿Migrar a React+FastAPI o quedarse en Gradio podado?** — panel: quedarse ahora, reevaluar
   después del stand. El argumento más fuerte para migrar (el stand necesita vista propia) lo
   resolvió `gr.Blocks.route`.
2. **¿El entrenamiento clásico PPO/SB3 se sigue lanzando desde la web?** — panel: conservar UN
   lanzador, no cinco. Es la pregunta que cambia el tamaño de la app.
3. **Paleta** — panel: «Barra de Vida» si aceptas su condición (el oro nunca es texto en claro);
   si no, «Suzaku». «Champion Chrome» vetada por tu propia instrucción de quitar el azul.

Y una de higiene que el panel marcó como *hazla aunque no hagas nada más*: el dashboard se sirve
hoy en `0.0.0.0` **sin autenticación** — cualquiera en la red del evento puede matar la demo desde
un teléfono.
