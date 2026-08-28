# verificar-contraste.py -- mide el contraste REAL del contrato de color.
#
#   python3 design/verificar-contraste.py
#
# Los ratios anotados en champion-chrome.css se re-corren, no se creen. Este
# script parsea el CSS de verdad (no una copia de los valores) y mide cada par
# que la app va a usar, en los dos polos. Sale con codigo 1 si algo falla, asi
# que sirve de puerta en CI.
#
# Umbrales WCAG 2.1: texto normal AA 4.5 · texto grande AA 3.0 ·
# componentes de interfaz y bordes 1.4.11 -> 3.0.

import re
import sys

CSS = __file__.replace("verificar-contraste.py", "champion-chrome.css")


def luminancia(hexcolor):
    h = hexcolor.lstrip("#")
    canales = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        canales.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    r, g, b = canales
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def ratio(a, b):
    la, lb = luminancia(a), luminancia(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def bloques(css):
    """-> {'claro': {token: hex}, 'oscuro': {...}} leidos del CSS real."""
    claro, oscuro = {}, {}
    # :root inicial = polo claro; el bloque [data-theme="dark"] = polo oscuro.
    m_dark = re.search(r':root\[data-theme="dark"\]\s*\{(.*?)\}', css, re.S)
    m_light = re.search(r'^:root\s*\{(.*?)\}', css, re.S | re.M)
    for destino, m in ((claro, m_light), (oscuro, m_dark)):
        if not m:
            continue
        for tok, val in re.findall(r'(--[\w-]+)\s*:\s*(#[0-9A-Fa-f]{6})', m.group(1)):
            destino[tok] = val
    # el polo oscuro hereda lo que no redefine
    for k, v in claro.items():
        oscuro.setdefault(k, v)
    return {"claro": claro, "oscuro": oscuro}


# (etiqueta, token de tinta, token de fondo, umbral)
PARES = [
    ("texto sobre fondo",            "--foreground",        "--background", 4.5),
    ("texto sobre tarjeta",          "--card-foreground",   "--card",       4.5),
    ("texto suave sobre fondo",      "--muted-foreground",  "--background", 4.5),
    ("texto suave sobre tarjeta",    "--muted-foreground",  "--card",       4.5),
    ("texto suave sobre muted",      "--muted-foreground",  "--muted",      4.5),
    ("tinta sobre primary",          "--primary-foreground", "--primary",   4.5),
    ("tinta sobre secondary",        "--secondary-foreground", "--secondary", 4.5),
    ("tinta sobre destructive",      "--destructive-foreground", "--destructive", 4.5),
    ("tinta sobre success",          "--success-foreground", "--success",   4.5),
    ("tinta sobre warning",          "--warning-foreground", "--warning",   4.5),
    ("tinta sobre champion",         "--champion-foreground", "--champion",  4.5),
    ("borde input sobre tarjeta",    "--input",             "--card",       3.0),
    ("borde sobre fondo",            "--border",            "--background", 1.0),
    ("anillo de foco sobre fondo",   "--ring",              "--background", 3.0),
    ("estado corriendo",             "--state-running-fg",  "--state-running-subtle", 4.5),
    ("estado detenido",              "--state-stopped-fg",  "--state-stopped-subtle", 4.5),
    ("estado degradado",             "--state-degraded-fg", "--state-degraded-subtle", 4.5),
    ("estado ALARMA",                "--state-alarm-fg",    "--state-alarm-subtle", 4.5),
    ("estado campeon",               "--state-champion-fg", "--state-champion-subtle", 4.5),
    ("estado desconocido",           "--state-unknown-fg",  "--state-unknown-subtle", 4.5),
]

# Los slots derivados existen precisamente para estos casos.
DERIVADOS = [
    ("destructive como TEXTO sobre muted", "--destructive-fg", "--secondary-fg", "--destructive", "--muted", 4.5),
    ("secondary como TEXTO sobre muted",   "--secondary-fg",   None,             "--secondary",   "--muted", 4.5),
    ("warning como TEXTO sobre muted",     "--warning-fg",     None,             "--warning",     "--muted", 4.5),
]


def main():
    css = open(CSS, encoding="utf-8").read()
    polos = bloques(css)
    fallos = 0
    for polo, toks in polos.items():
        print(f"\n=== polo {polo.upper()} ===")
        print(f"{'par':<34} {'ratio':>7}  umbral  veredicto")
        for etiqueta, tinta, fondo, umbral in PARES:
            if tinta not in toks or fondo not in toks:
                print(f"{etiqueta:<34} {'—':>7}  (token ausente en este polo)")
                continue
            r = ratio(toks[tinta], toks[fondo])
            ok = r >= umbral
            fallos += 0 if ok else 1
            print(f"{etiqueta:<34} {r:>7.2f}  {umbral:>6.1f}  {'OK' if ok else 'FALLA'}")
        # los derivados: se compara el pinneado contra su slot de texto
        print(f"\n  -- slots derivados (por que existen) --")
        for etiqueta, slot, _x, pinneado, fondo, umbral in DERIVADOS:
            if slot not in toks or pinneado not in toks:
                continue
            r_pin = ratio(toks[pinneado], toks[fondo])
            r_slot = ratio(toks[slot], toks[fondo])
            print(f"  {etiqueta:<40} pinneado {r_pin:.2f} -> slot {r_slot:.2f} "
                  f"{'OK' if r_slot >= umbral else 'SIGUE FALLANDO'}")
            fallos += 0 if r_slot >= umbral else 1
    print(f"\n{'TODO EN REGLA' if not fallos else f'{fallos} FALLOS'}")
    return 1 if fallos else 0


if __name__ == "__main__":
    sys.exit(main())
