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

Estructura de datos esperada (cualquiera de las dos):
    <data-root>/<fecha>/constellation_dataset_<modelo>/scenario_<n>/ollama_prompt_TASK_GEN_*.txt
    <data-root>/<fecha>/<estrategia>/constellation_dataset_<modelo>/scenario_<n>/...

Si no se usa --date, se escanea <data-root> directamente (estructura antigua).

Requiere matplotlib:
    pip install matplotlib --break-system-packages

Ejemplo de uso:
    python3 scan_special_characters.py --list-dates
    python3 scan_special_characters.py --data-root data --date 2026-09-14
    python3 scan_special_characters.py --data-root data --date 2026-09-14 --list-matches
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

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def build_pattern(chars: list) -> re.Pattern:
    """Construye una expresión regular tipo [caracteres] escapando cada uno."""
    char_class = "".join(re.escape(c) for c in chars)
    return re.compile(f"[{char_class}]")


def list_available_dates(data_root: Path):
    """Carpetas YYYY-MM-DD presentes bajo data-root, de más antigua a más reciente."""
    if not data_root.exists():
        return []
    return sorted(d.name for d in data_root.iterdir()
                  if d.is_dir() and DATE_PATTERN.match(d.name))


def pretty_label(rel_path: Path) -> str:
    """
    constellation_dataset_llama3_1_8b            -> llama3_1_8b
    few_shot/constellation_dataset_llama3_1_8b   -> few_shot | llama3_1_8b
    """
    parts = list(rel_path.parts)
    model = parts[-1].replace("constellation_dataset_", "", 1)
    prefix = parts[:-1]
    return " | ".join(prefix + [model]) if prefix else model


def discover_model_dirs(scan_root: Path):
    """
    Devuelve [(etiqueta, ruta)] de cada carpeta de modelo encontrada bajo
    scan_root, a cualquier profundidad. Usar rglob permite que funcione
    igual con data/<fecha>/<modelo>/ que con
    data/<fecha>/<estrategia>/<modelo>/.
    """
    found = []
    for p in sorted(scan_root.rglob("constellation_dataset_*")):
        if p.is_dir():
            found.append((pretty_label(p.relative_to(scan_root)), p))
    return found


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


def scan_model_dir(model_dir: Path, scenarios, pattern: re.Pattern):
    """
    Recorre todos los .txt de prompt de una carpeta de modelo y devuelve:
      - total_files: número total de archivos de prompt encontrados
      - matched_files: lista de rutas cuyo contenido contiene AL MENOS UNO
        de los caracteres especiales (solo presencia/ausencia por archivo)
    """
    total_files = 0
    matched_files = []

    for scenario in scenarios:
        scenario_dir = model_dir / f"scenario_{scenario}"
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


def plot_bars(counts_by_model: dict, totals_by_model: dict, output_path: Path,
              date: str = None):
    models = sorted(counts_by_model.keys())
    if not models:
        print("[!] No se encontraron modelos/archivos para graficar.")
        sys.exit(0)

    values = [counts_by_model[m] for m in models]

    x = range(len(models))
    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.6), 6))
    bars = ax.bar(x, values, color="#8172B2")

    ax.set_xlabel("Modelo")
    ax.set_ylabel("Nº de prompts con al menos un carácter especial")
    title = "Prompts por modelo que contienen caracteres especiales"
    if date:
        title += f"\nCorrida: {date}"
    ax.set_title(title)
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
                         help="Carpeta raíz de datos (default: ./data)")
    parser.add_argument("--date", type=str, default=None,
                         help="Carpeta de fecha YYYY-MM-DD bajo data-root. Si se omite y "
                              "existen carpetas de fecha, se usa la más reciente.")
    parser.add_argument("--list-dates", action="store_true",
                         help="Solo listar las fechas disponibles y salir.")
    parser.add_argument("--scenarios", type=str, default="1-25",
                         help="Rango de escenarios a considerar, ej. '1-25' o '1,3,5' (default: 1-25)")
    parser.add_argument("--output", type=Path, default=None,
                         help="Ruta del archivo PNG de salida. Por defecto incluye la fecha "
                              "en el nombre.")
    parser.add_argument("--list-matches", action="store_true",
                         help="Además del gráfico, imprime la ruta y los caracteres especiales "
                              "encontrados en cada archivo")
    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"[!] La carpeta de datos '{args.data_root}' no existe.", file=sys.stderr)
        sys.exit(1)

    available_dates = list_available_dates(args.data_root)

    if args.list_dates:
        if available_dates:
            print("Fechas disponibles:")
            for d in available_dates:
                print(f"  {d}")
        else:
            print(f"No hay carpetas de fecha bajo '{args.data_root}'.")
        sys.exit(0)

    # ---- Resolver qué carpeta escanear ---------------------------- #
    selected_date = args.date
    if selected_date is None and available_dates:
        selected_date = available_dates[-1]
        print(f"[INFO] No se especificó --date; usando la más reciente: {selected_date}")

    if selected_date:
        scan_root = args.data_root / selected_date
        if not scan_root.exists():
            print(f"[!] No existe la carpeta de fecha '{selected_date}' en "
                  f"'{args.data_root}'.", file=sys.stderr)
            print(f"[!] Fechas disponibles: {available_dates if available_dates else 'ninguna'}",
                  file=sys.stderr)
            sys.exit(1)
    else:
        scan_root = args.data_root
        print(f"[INFO] Sin carpetas de fecha; escaneando '{scan_root}' directamente.")

    output_path = args.output
    if output_path is None:
        suffix = f"_{selected_date}" if selected_date else ""
        output_path = Path(f"special_chars_by_model{suffix}.png")

    pattern = build_pattern(SPECIAL_CHARS)
    scenarios = parse_scenario_range(args.scenarios)

    model_dirs = discover_model_dirs(scan_root)
    if not model_dirs:
        print(f"[!] No se encontraron carpetas 'constellation_dataset_*' en "
              f"'{scan_root}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Modelos encontrados ({len(model_dirs)}): "
          f"{', '.join(label for label, _ in model_dirs)}")
    print(f"Caracteres especiales considerados: {SPECIAL_CHARS}\n")

    counts_by_model = {}
    totals_by_model = {}
    all_matches = defaultdict(list)

    for label, model_dir in model_dirs:
        total, matched = scan_model_dir(model_dir, scenarios, pattern)
        if total == 0:
            print(f"  {label:28s}  sin archivos de prompt, se omite")
            continue
        totals_by_model[label] = total
        counts_by_model[label] = len(matched)
        all_matches[label] = matched
        pct = len(matched) / total * 100
        print(f"  {label:28s}  {len(matched):4d} / {total:4d} prompts contienen "
              f"caracteres especiales ({pct:.1f}%)")

    if args.list_matches:
        print("\nArchivos con caracteres especiales y cuáles se encontraron:")
        for label in sorted(all_matches):
            for path in all_matches[label]:
                content = path.read_text(encoding="utf-8")
                found = sorted(set(c for c in SPECIAL_CHARS if c in content))
                print(f"  [{label}] {path}  ->  {found}")

    plot_bars(counts_by_model, totals_by_model, output_path, date=selected_date)


if __name__ == "__main__":
    main()