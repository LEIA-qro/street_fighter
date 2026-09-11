"""El dashboard y los scripts que lanza tienen que hablar el mismo idioma.

Un dropdown de la pantalla es una PROMESA: "esta opcion se puede elegir". El
argparse del script hijo es quien la cumple o no. Cuando las dos listas se
separan, el usuario elige algo que la pantalla acepta y el proceso muere en el
hijo con un traceback que no dice cual de los dos manda.

Ya paso dos veces en este proyecto:

  * `v1` se ofrecia en el selector de entrenamiento y `env_tools.py` lo
    degradaba a v2 EN SILENCIO -- entrenabas otra cosa sin aviso.
  * `v4` se agrego a los cinco selectores al arreglar lo anterior, pero solo
    `train.py` lo acepta: matchups, liga, exploiter y Optuna lo rechazan.

Estos tests leen el argparse REAL de cada script con `ast` (no lo importan: eso
arrastraria torch y el emulador) y lo comparan con las constantes del
dashboard. Si alguien vuelve a separar los vocabularios, esto se pone rojo
antes de que un compañero pierda una tarde.
"""

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPTS = REPO / "src" / "scripts"
sys.path.insert(0, str(REPO / "src"))


def choices_de(script, *nombres_de_flag):
    """Las `choices=[...]` que declara argparse para una bandera de un script.

    Devuelve None si la bandera existe pero no declara choices (acepta todo).
    """
    arbol = ast.parse((SCRIPTS / script).read_text(encoding="utf-8"))
    for nodo in ast.walk(arbol):
        if not isinstance(nodo, ast.Call):
            continue
        if not (isinstance(nodo.func, ast.Attribute)
                and nodo.func.attr == "add_argument"):
            continue
        if not nodo.args or not isinstance(nodo.args[0], ast.Constant):
            continue
        if nodo.args[0].value not in nombres_de_flag:
            continue
        for kw in nodo.keywords:
            if kw.arg == "choices":
                try:
                    return list(ast.literal_eval(kw.value))
                except ValueError:
                    return None
        return None
    pytest.fail(f"{script} no declara ninguna de {nombres_de_flag}")


@pytest.fixture(scope="module")
def dashboard():
    from scripts import web_dashboard
    return web_dashboard


@pytest.mark.parametrize("constante, script, flags", [
    ("TRAIN_ENV_CHOICES", "train.py", ("--env",)),
    ("TUNE_ENV_CHOICES", "tune.py", ("--env",)),
])
def test_el_vocabulario_del_dashboard_cabe_en_el_del_script(
        dashboard, constante, script, flags):
    ofrecido = set(getattr(dashboard, constante))
    aceptado = set(choices_de(script, *flags))
    assert ofrecido <= aceptado, (
        f"{constante} ofrece {sorted(ofrecido - aceptado)}, que "
        f"{script} rechaza en su argparse")


@pytest.mark.parametrize("script, flags", [
    ("test_ai_vs_ai_v2.py", ("--env_p1", "--env_p2")),
    ("test_agent_v2.py", ("--env",)),
    ("train_league.py", ("--env_version",)),
    ("train_exploiter.py", ("--env_version",)),
    ("stand_leia.py", ("--p2-env",)),
])
def test_los_selectores_sb3_caben_en_todos_sus_scripts(dashboard, script, flags):
    ofrecido = set(dashboard.SB3_ENV_CHOICES)
    for flag in flags:
        aceptado = set(choices_de(script, flag))
        assert ofrecido <= aceptado, (
            f"SB3_ENV_CHOICES ofrece {sorted(ofrecido - aceptado)} y "
            f"{script} {flag} no lo acepta")


def test_sac_no_se_ofrece_en_ninguna_parte(dashboard):
    """SACAgent.train y .tune lanzan NotImplementedError en su PRIMERA linea.

    Ofrecerlo es un boton que solo puede tronar. Se quito de la pantalla el
    2026-08-28 y de stand_leia.py el 2026-09-11; esto evita que vuelva.
    """
    assert "sac" not in dashboard.SB3_ALGO_CHOICES
    assert "sac" not in dashboard.APEX_P2_ALGO_TO_TYPE
    assert "sac" not in set(choices_de("stand_leia.py", "--p2-algo"))


def test_el_dashboard_escucha_local_por_defecto():
    """Esta pantalla mata entrenamientos y reescribe src/core/config.py.

    Estuvo sirviendose en 0.0.0.0 sin autenticacion: cualquiera en la red del
    lugar podia tumbar una run desde un telefono. Exponerla tiene que ser una
    decision que alguien escriba, no el default.
    """
    fuente = (SCRIPTS / "web_dashboard.py").read_text(encoding="utf-8")
    arbol = ast.parse(fuente)
    defaults = {}
    for nodo in ast.walk(arbol):
        if (isinstance(nodo, ast.Call)
                and isinstance(nodo.func, ast.Attribute)
                and nodo.func.attr == "add_argument"
                and nodo.args
                and isinstance(nodo.args[0], ast.Constant)
                and nodo.args[0].value in ("--host", "--auth")):
            for kw in nodo.keywords:
                if kw.arg == "default":
                    defaults[nodo.args[0].value] = ast.literal_eval(kw.value)
    assert defaults["--host"] == "127.0.0.1"
    assert defaults["--auth"] is None
    assert "auth=auth" in fuente, "launch() debe recibir el --auth parseado"


def test_la_tarjeta_del_campeon_esta_enlazada(dashboard):
    """get_stand_checkpoint_status existia y no la llamaba NADIE desde la UI.

    Es la escalera L1-L8, el win rate y la version de pesos: lo unico
    demostrable que produce este proyecto. Estuvo huerfana y el runbook la
    recitaba a mano.
    """
    componentes = dashboard.demo.get_config_file()["components"]
    tarjetas = [
        c for c in componentes
        if c["type"] == "markdown"
        and "Campeón vigente" in str(c.get("props", {}).get("value", ""))
    ]
    assert len(tarjetas) == 2, "una tarjeta por lado (P1 y P2)"
    for tarjeta in tarjetas:
        assert "99.0%" in tarjeta["props"]["value"]
        assert "L8" in tarjeta["props"]["value"]


def test_importar_config_no_exige_bizhawk():
    """`import core.config` tiene que funcionar en una maquina sin BizHawk.

    Estuvo haciendo `raise FileNotFoundError` a nivel de MODULO si no encontraba
    EmuHawk.exe en el directorio padre. Medido el 2026-09-11 sobre un clon recien
    hecho: `pytest code_testing/pytest` moria en la recoleccion de NUEVE modulos
    de test, ninguno de los cuales toca BizHawk. El remedio que circulaba era
    crear un EmuHawk.exe VACIO al lado del repo -- un archivo falso para enganiar
    a un guard es la señal de que el guard esta en el lugar equivocado.

    La comprobacion vive ahora donde se lanza el emulador. BizHawk solo corre en
    Windows y el backend de entrenamiento es stable-retro: exigirlo para importar
    una constante cerraba el repo a las tres plataformas de la flota.
    """
    config_src = (REPO / "src" / "core" / "config.py").read_text(encoding="utf-8")
    arbol = ast.parse(config_src)
    for nodo in arbol.body:                      # solo nivel de modulo
        for hijo in ast.walk(nodo):
            if isinstance(hijo, ast.Raise):
                pytest.fail(
                    "core/config.py vuelve a lanzar una excepcion al importarse; "
                    "eso rompe el import en cualquier maquina sin BizHawk")

    base_src = (REPO / "src" / "core" / "bizhawk_base.py").read_text(encoding="utf-8")
    assert "os.path.exists(self.bizhawk_path)" in base_src, (
        "la comprobacion de EmuHawk.exe tiene que seguir existiendo, en el sitio "
        "donde de verdad se lanza el emulador")
    assert base_src.index("os.path.exists(self.bizhawk_path)") < \
        base_src.index("subprocess.Popen(launch_args)"), (
        "la comprobacion va ANTES del Popen, no despues")
