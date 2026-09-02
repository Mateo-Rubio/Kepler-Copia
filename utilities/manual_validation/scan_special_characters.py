#!/usr/bin/env python3
"""
scan_special_characters.py

Recorre todos los escenarios de cada modelo y revisa, para cada archivo
'ollama_prompt_TASK_GEN_N.txt', si contiene AL MENOS UNO de los caracteres
especiales definidos en SPECIAL_CHARS (más abajo). Estos caracteres no
suelen aparecer en una solicitud redactada en prosa natural, así que su
presencia puede indicar restos de formato markdown, artefactos de la
plantilla del sistema, o texto mal formado generado por el LLM.

Solo se comprueba PRESENCIA/AUSENCIA por archivo (no se cuenta cuántos
caracteres especiales hay, ni cuántas veces aparece cada uno). Al final se
genera un gráfico de barras con el número de prompts por modelo que
contienen al menos uno de estos caracteres.

Estructura de datos esperada:
    <data-root>/constellation_dataset_<modelo>/scenario_<n>/ollama_prompt_TASK_GEN_*.txt

Requiere matplotlib:
    pip install matplotlib --break-system-packages

Ejemplo de uso:
    python3 scan_special_characters.py --data-root data
    python3 scan_special_characters.py --data-root data --list-matches
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("[!] Este script requiere matplotlib. Instálalo con:\n"
          "    pip install matplotlib --break-system-packages", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------ #
# Lista de caracteres especiales a detectar.
#
# Se consideran "especiales" aquellos caracteres que NO son de esperarse
# en una solicitud redactada en prosa natural en inglés (a diferencia de
# la puntuación normal: . , ! ? ' - : ; que sí es válida en una oración).
# Edita esta lista según lo que quieras detectar.
# ------------------------------------------------------------------ #
SPECIAL_CHARS = [
    "`",   
    "~",   
    '"',   
    "[",   
    "]",   
    "{",   
    "}",   
    "<",   
    ">",   
    "|",   
    "^",   
    "\\",  
    "*",  
    "_",  
    "@",   
    "#",   
    "$",  
    "%",   
    "&",   
]


def build_pattern(chars: list) -> re.Pattern:
    """Construye una expresión regular tipo [caracteres] escapando cada uno."""
    char_class = "".join(re.escape(c) for c in chars)
    return re.compile(f"[{char_class}]")


def discover_models(data_root: Path):
    models = []
    for p in sorted(data_root.glob("constellation_dataset_*")):
        if p.is_dir():
            models.append(p.name.replace("constellation_dataset_", "", 1))
    return models


def parse_scenario_range(spec: str):
    """Parsea '1-25', '1,3,5' o '1-3,7,10-12' en una lista de enteros."""
    scenarios = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-")
            scenarios.update(range(int(start), int(end) + 1))
        else:
            scenarios.add(int(part))
    return sorted(scenarios)


def scan_model(data_root: Path, model: str, scenarios, pattern: re.Pattern):
    """
    Recorre todos los .txt de prompt de un modelo y devuelve:
      - total_files: número total de archivos de prompt encontrados
      - matched_files: lista de rutas cuyo contenido contiene AL MENOS UNO
        de los caracteres especiales (solo presencia/ausencia por archivo)
    """
    total_files = 0
    matched_files = []

    for scenario in scenarios:
        scenario_dir = data_root / f"constellation_dataset_{model}" / f"scenario_{scenario}"
        if not scenario_dir.is_dir():
            continue

        for txt_path in sorted(scenario_dir.glob("ollama_prompt_TASK_GEN_*.txt")):
            total_files += 1
            try:
                content = txt_path.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError) as e:
                print(f"  [!] No se pudo leer {txt_path}: {e}", file=sys.stderr)
                continue

            if pattern.search(content):  # solo presencia/ausencia, no conteo
                matched_files.append(txt_path)

    return total_files, matched_files


def plot_bars(counts_by_model: dict, totals_by_model: dict, output_path: Path):
    models = sorted(counts_by_model.keys())
    if not models:
        print("[!] No se encontraron modelos/archivos para graficar.")
        sys.exit(0)

    values = [counts_by_model[m] for m in models]

    x = range(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.4), 6))
    bars = ax.bar(x, values, color="#8172B2")

    ax.set_xlabel("Modelo")
    ax.set_ylabel("Nº de prompts con al menos un carácter especial")
    ax.set_title("Prompts por modelo que contienen caracteres especiales")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bar, model in zip(bars, models):
        height = bar.get_height()
        total = totals_by_model[model]
        pct = (height / total * 100) if total else 0
        ax.annotate(f"{int(height)}/{total} ({pct:.1f}%)",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\n[SUCCESS] Gráfico guardado en: {output_path.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Detecta caracteres especiales (definidos en SPECIAL_CHARS) en los "
                    "archivos de prompt generados y grafica cuántos prompts por modelo "
                    "contienen al menos uno."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                         help="Carpeta que contiene constellation_dataset_<modelo>/ (default: ./data)")
    parser.add_argument("--scenarios", type=str, default="1-25",
                         help="Rango de escenarios a considerar, ej. '1-25' o '1,3,5' (default: 1-25)")
    parser.add_argument("--output", type=Path, default=Path("special_chars_by_model.png"),
                         help="Ruta del archivo PNG de salida (default: special_chars_by_model.png)")
    parser.add_argument("--list-matches", action="store_true",
                         help="Además del gráfico, imprime la ruta y los caracteres especiales "
                              "encontrados en cada archivo")
    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"[!] La carpeta de datos '{args.data_root}' no existe.", file=sys.stderr)
        sys.exit(1)

    pattern = build_pattern(SPECIAL_CHARS)

    scenarios = parse_scenario_range(args.scenarios)
    models = discover_models(args.data_root)
    if not models:
        print(f"[!] No se encontraron carpetas 'constellation_dataset_*' en "
              f"'{args.data_root}'.", file=sys.stderr)
        sys.exit(1)
    print(f"Modelos encontrados ({len(models)}): {', '.join(models)}")
    print(f"Caracteres especiales considerados: {SPECIAL_CHARS}\n")

    counts_by_model = {}
    totals_by_model = {}
    all_matches = defaultdict(list)

    for model in models:
        total, matched = scan_model(args.data_root, model, scenarios, pattern)
        totals_by_model[model] = total
        counts_by_model[model] = len(matched)
        all_matches[model] = matched
        print(f"  {model:20s}  {len(matched):4d} / {total:4d} prompts contienen "
              f"caracteres especiales")

    if args.list_matches:
        print("\nArchivos con caracteres especiales y cuáles se encontraron:")
        for model in models:
            for path in all_matches[model]:
                content = path.read_text(encoding="utf-8")
                found = sorted(set(c for c in SPECIAL_CHARS if c in content))
                print(f"  [{model}] {path}  ->  {found}")

    plot_bars(counts_by_model, totals_by_model, args.output)


if __name__ == "__main__":
    main()