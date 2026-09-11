# Bugs cazados — NO REINTRODUCIR (todos con test de regresión)

1. **EL bug de 6 meses (nació 2026-03-01, muerto 2026-08-26)**: 'muerto' se decidía con `hp<=0` sobre u16 + daño por diff de HP en el frame terminal ⇒ PERDER pagaba +55 y GANAR −12 (medido). Verdad de la ROM (300k+ frames medidos): HP ∈ [−27,176]; hp==0 es lectura VIVA (persiste cientos de frames); el marcador de muerte es el SIGNO (65535=−1). Fix: envs/reward.py hp_to_signed (única decodificación, ambos backends; el Lua NO se toca) + RoundTracker compartido. Tras fix (150 eps): WIN +138 / LOSS −96, sin traslape.
2. **Tres finales de round, no dos**: KO (signo negativo, ventana ≥33 frames, HP del ganador congelado), TIME OVER (contadores matches_won, payload 26), DRAW GAME (time over con vidas iguales: NINGÚN contador tica; detectado con el reloj del round en RAM 0xFF972A, hallado por barrido de 65,536 direcciones — resolvió el "round timer never located" del handoff). Empate paga −50 (antes el double-KO pagaba +15 neto).
3. **Payload gate `== 24`** tiraba en silencio los campos extendidos al ensanchar a 26 ⇒ ground_gate muerto sin aviso. Fix: ACCEPTED_PAYLOAD_WIDTHS (13,24,26,27) + bloques `>=`.
4. **trainable=False pagaba el terminal por siempre** (1,773 pagos en 2,500 steps medidos) — RoundTracker lo consume una vez por ronda.
5. **La liga clasificaba en perspectiva P2** (parser_p2 corre al último) ⇒ todo resultado invertido. Fix: resolver con p1_ko/p2_ko crudos. La prescripción original del runbook ERA el bug.
6. **lr=2.108e-05 de Optuna** (tuneado bajo régimen roto) congeló la política en entropía máxima 1M steps — el "se acerca saltando" del Run A era random puro (3/9 direcciones son salto). Entropía máx MultiDiscrete([9,7]) = 4.1431: si entropy_loss está clavado ahí, el optimizador está muerto.
7. **Frame terminal / sentinels**: HP>200 = frame ilegible (se salta reward y terminación); el frame [0,0] entre rondas dura 1 frame y NO termina nada.
8. **Reset del protocolo BizHawk**: siempre hay exactamente 1 payload en vuelo; reset() drena el rancio, lee el fresco, y re-arma el offset con comando neutro (mantiene el pipelining). El lag de 1 paso DURANTE el episodio es deliberado — quitarlo ingenuo serializa y pierde throughput.
9. **atexit en workers de SubprocVecEnv** mataba los 16 emuladores al morir uno (sniper PowerShell) — solo se registra en MainProcess.
10. **PBRS es policy-invariant** (el ground_gate anti-salto no puede cambiar la ruta óptima, solo la señal de aprendizaje). 15M steps lo confirmaron: air_frac plano ~0.45 aunque spacing mejoró. Si caminar es objetivo en sí: término no-invariante o niveles con anti-air.

## [2026-09-11] Los de la limpieza de handoff (todos con regresion salvo donde se diga)

11. **El campeon por defecto se quedo atras.** `stand_leia.DEFAULT_CHECKPOINT` apuntaba
    a `apex_v1592_benchmarked.pt` (mejor modelo de media jornada del 27 de agosto). La
    run cerro al dia siguiente con v3291 y el default no se movio: durante dos semanas
    el dashboard y el CLI arrancaban un modelo peor que el que ya existia. Fix: apunta a
    v3291 y el test verifica ademas el acta del sidecar (3291 / 0.99).
12. **La trampa de v4, segunda parte.** El parche del 28-ago quito `v1` (que se degradaba
    a v2 en silencio) y metio `v4` en los CINCO dropdowns de environment. Solo `train.py`
    acepta v4: tune, matchups, liga y exploiter lo rechazan en su argparse. Se cambio una
    trampa silenciosa por cuatro traceback. Fix: cuatro constantes de vocabulario en
    web_dashboard + `test_dashboard_contratos.py`, que lee el argparse REAL de cada
    script con `ast` y falla si se separan.
13. **Force Kill no mataba nada fuera de Windows.** `taskkill /F /T` por shell: en la Mac
    y en los actores Linux el boton se veia igual y no hacia nada. Fix: `os.killpg` del
    grupo del hijo (nacen con `start_new_session=True`), con guarda por si alguien cambia
    eso -- matar el grupo propio mataria al dashboard.
14. **El dashboard se servia en 0.0.0.0 sin autenticacion.** Esta pantalla mata procesos,
    reescribe `src/core/config.py` y sube archivos. Fix: default `127.0.0.1`, `--auth
    USUARIO:CLAVE`, y aviso en consola al exponerla sin el.
15. **El nombre del estudio de Optuna se ejecutaba.** `get_best_tuning_params` lo
    interpolaba en un fuente Python que luego corre con `-c`: un textbox convertido en
    consola. Fix: viaja por argv, el script es constante. (Sin regresion: no hay estudios
    en el arbol con que probarlo.)
16. **La tarjeta del campeon estaba huerfana.** `get_stand_checkpoint_status` -- escalera
    L1-L8, win rate y version de pesos -- no la llamaba ningun evento de la UI, y
    RUN_STAND_LEIA.md recitaba los numeros a mano. Fix: enlazada bajo los selectores de
    checkpoint de P1 y P2.
17. **Una corrida parcial del gauntlet pisaba el acta de registro.** El fix anterior
    separo video de medicion, pero `--sin-video --rivales BALROG` (n=1) seguia
    escribiendo `benchmarks/gauntlet_lvl8.json`, que es la medicion de n=360 que la
    consola muestra. Fix: el registro solo lo escribe una corrida COMPLETA que no encoja
    la muestra; lo demas va a `_parcial_<ts>.json`. Verificado en vivo: el acta de 360
    sobrevivio a los dos intentos.
18. **TensorBoard abria el navegador del SERVIDOR.** `Popen(shell=True)` +
    `webbrowser.open()`: con el dashboard servido en red, el clic de un usuario abria una
    ventana en la computadora de otro. Fix: lista de argumentos y devuelve la URL.
19. **La subida de modelos creaba carpetas con el nombre del rol.** `handle_model_upload`
    usaba el valor del dropdown como nombre de directorio, asi que "Human Player" o
    "CPU (Built-in AI)" creaban `models/production/v2/Human Player/` al primer clic.
    Fix: guarda que solo acepta algoritmos SB3 reales.
