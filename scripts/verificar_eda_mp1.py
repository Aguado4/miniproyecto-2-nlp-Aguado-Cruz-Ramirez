#!/usr/bin/env python3
"""Verifica que el bloque heredado del Miniproyecto 1 (Sección 2 "El corpus" y Sección 3
"Análisis exploratorio") sea idéntico, celda por celda, al original.

Por qué es un script y no una celda del notebook: el notebook debe poder ejecutarse en
Colab sin más que el propio archivo (ver CLAUDE.md §4, "notebook autocontenido"), y el
Miniproyecto 1 vive en OTRO repositorio privado que Colab no tiene forma de clonar sin
credenciales. La verificación de identidad, por tanto, es un control de autoría que se
corre en local antes de commitear, no una celda que se ejecute como parte del notebook.

Uso:
    python scripts/verificar_eda_mp1.py --mp1 /ruta/al/miniproyecto1_restmex.ipynb

Si no se pasa --mp1, se buscan rutas candidatas comunes (ver CANDIDATOS_MP1).
"""
import argparse
import json
import sys
from pathlib import Path

# Rango de celdas del bloque heredado (Sección 2 + Sección 3 completas).
# Ver docs/DECISIONS.md §D-201: MP1 cells[12:54) == MP2 cells[13:55) (42 celdas).
MP1_RANGO = (12, 54)
MP2_RANGO = (13, 55)

CANDIDATOS_MP1 = [
    "../miniproyecto 1/notebooks/miniproyecto1_restmex.ipynb",
    "../miniproyecto1-nlp-Aguado-Cruz-Ramirez/notebooks/miniproyecto1_restmex.ipynb",
    "../NLP-2026/notebooks/miniproyecto1_restmex.ipynb",
]


def cargar_celdas(ruta):
    nb = json.loads(Path(ruta).read_text(encoding="utf-8"))
    return nb["cells"]


def fuente(celda):
    return "".join(celda["source"])


def encontrar_mp1(arg):
    if arg:
        return Path(arg)
    aqui = Path(__file__).resolve().parent.parent
    for candidato in CANDIDATOS_MP1:
        ruta = (aqui / candidato).resolve()
        if ruta.exists():
            return ruta
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mp1", help="Ruta al miniproyecto1_restmex.ipynb")
    ap.add_argument(
        "--mp2",
        default=str(Path(__file__).resolve().parent.parent / "notebooks" / "miniproyecto2_restmex.ipynb"),
        help="Ruta al miniproyecto2_restmex.ipynb (por defecto, el de este repo)",
    )
    args = ap.parse_args()

    ruta_mp1 = encontrar_mp1(args.mp1)
    if ruta_mp1 is None:
        print("No se encontró el notebook del Miniproyecto 1 en ninguna ruta candidata.")
        print("Pasa la ruta explícita con --mp1 /ruta/al/miniproyecto1_restmex.ipynb")
        return 2

    celdas_mp1 = cargar_celdas(ruta_mp1)[MP1_RANGO[0]:MP1_RANGO[1]]
    celdas_mp2 = cargar_celdas(args.mp2)[MP2_RANGO[0]:MP2_RANGO[1]]

    print(f"MP1: {ruta_mp1}")
    print(f"MP2: {args.mp2}")
    print(f"Comparando {len(celdas_mp1)} celdas de MP1 (índices {MP1_RANGO}) contra "
          f"{len(celdas_mp2)} celdas de MP2 (índices {MP2_RANGO})\n")

    if len(celdas_mp1) != len(celdas_mp2):
        print(f"ADVERTENCIA: número de celdas distinto ({len(celdas_mp1)} vs {len(celdas_mp2)}). "
              "El rango de alguno de los dos notebooks cambió; hay que reajustar MP1_RANGO/MP2_RANGO.")

    diffs = []
    for i, (a, b) in enumerate(zip(celdas_mp1, celdas_mp2)):
        if a["cell_type"] != b["cell_type"]:
            diffs.append((i, "tipo de celda distinto", a["cell_type"], b["cell_type"]))
            continue
        fa, fb = fuente(a), fuente(b)
        if fa != fb:
            diffs.append((i, "contenido distinto", fa, fb))

    if not diffs:
        print("OK — las", len(celdas_mp1), "celdas son byte-idénticas.")
        return 0

    print(f"Se encontraron {len(diffs)} celdas con diferencias (de {len(celdas_mp1)}):\n")
    for i, motivo, a, b in diffs:
        print(f"--- celda relativa #{i} (MP1[{MP1_RANGO[0]+i}] / MP2[{MP2_RANGO[0]+i}]) — {motivo} ---")
        print("MP1:", repr(a[:180]))
        print("MP2:", repr(b[:180]))
        print()

    print(
        "Nota (ver docs/DECISIONS.md §D-217): a fecha de la última corrida, las únicas "
        "diferencias conocidas son 3 sustituciones cosméticas de guión largo (—) por coma "
        "o guión corto, sin cambios de código, cifras ni conclusiones. Cualquier diferencia "
        "nueva o de otro tipo debe tratarse como una regresión real."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
