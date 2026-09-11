> # ⛔ CANCELADO — 2026-09-11
>
> **Este documento es HISTORICO. No lo ejecutes.**
>
> La reconstruccion de la interfaz se cancelo por decision de Felipe: la
> consola nueva (React + shadcn, `consola-app/`) salio del arbol y la UI que el
> proyecto mantiene es el dashboard Gradio, `src/scripts/web_dashboard.py`.
> El codigo borrado sigue en la historia de git bajo el tag
> `consola-react-cancelada`.
>
> Lo que SI sobrevivio de este trabajo y sigue vigente:
>
> * Los arreglos de higiene al Gradio, ya aplicados (commits `6955cafa` y
>   `3575547c`) — la lista de trampas que este analisis encontro es real y se
>   uso para arreglarlas.
> * `tools/leia_hub.py` + `fleet/fleet.json` + `web/consola.html`: los ojos de
>   la flota. Eso no era interfaz, era observabilidad, y se queda.
> * `design/champion-chrome.css` y `agent/dashboard/paletas-sf2-medidas.md`:
>   la paleta medida de pixeles del juego, que usa la consola del hub.
>
> Se conserva porque el analisis (que controles mienten, que ramas no se
> alcanzan, que pestañas no pueden funcionar) es la mejor radiografia que
> existe del dashboard, y quien lo retome la va a querer. Ver `HANDOFF.md`.

# Decisiones del dueño (Felipe) — pinneadas, NO re-litigables

1. **Alcance: AMBAS.** El fleet-agent (sistema de actualización automática de la
   flota) y su UI se diseñan JUNTOS en una sola corrida, para que la interfaz
   nazca pensada para recibir el sistema de flota. No existe todavía: es el
   pendiente #4 de agent/memory/08-cola-manana.md.
2. **Paleta DECIDIDA: "Champion Chrome"** (D3 del artifact "Cinco Paletas
   Medidas"). No se proponen alternativas. Tokens abajo. Disciplina añadida por
   el orquestador: el lima #C8EC40 es la FIRMA, se usa poco y donde importa.
   dark:  background #05070F · foreground #E8ECE8 · card #0D1020 · muted #161B30
          muted-foreground #9AA3BE · border #262C48 · input #525E8C
          primary/ring #40A8C8 · secondary #3E85C0 · accent #C8EC40
          destructive #E83A2A · success #40CC88 · warning #E8CC00
   light: background #EEF0F6 · foreground #0F1424 · card #F8F9FC · muted #E0E4EE
          muted-foreground #4E5670 · border #C9CFDE · input #7B84A0
          primary/ring #206488 · secondary #00688C · accent #4A6B00
          destructive #A81400 · success #00684A · warning #886400
3. **Arquitectura HÍBRIDA, sin AWS, accesible para todo el equipo.**
   - LOCAL en cada máquina: los modelos, el jugable (BizHawk + emulador + ROM +
     torch). No puede vivir en un servidor.
   - COMPARTIDO: tracking de la flota + onboarding de máquinas nuevas.
   - PROHIBIDO AWS. Ojo: la "madre" es EC2 = AWS = EXCLUIDA como host.
   - Hipótesis a batir (del orquestador, NO decidida): ya existe la tailnet
     `leia-qro.org.github` con las 4 máquinas dentro y la desktop 4090 está
     24/7 — eso da acceso privado para todos sin infra nueva ni AWS.
4. **El "modo stand" NO existe.** Fue un término inventado por un agente. Esa
   sección es sólo un SELECTOR DE QUIÉN CONTRA QUIÉN dentro de pruebas, y el
   dashboard lo usa EL EQUIPO para probar modelos jugando contra ellos.
   `src/scripts/stand_leia.py` se conserva como MOTOR (probado, funciona); lo
   que le faltó fue la UI encima — eso es lo que se reconstruye.
5. El desmadre de dropdowns/pestañas YA existía antes de la era ChatGPT
   (1,667 → 2,326 líneas, 10 gr.Tab iguales, 32 → 35 gr.Dropdown). Lo que
   aporta el conteo no es la noticia: es dimensionar cuánto es selector
   repetido y colapsable.
