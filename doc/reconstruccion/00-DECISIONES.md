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
