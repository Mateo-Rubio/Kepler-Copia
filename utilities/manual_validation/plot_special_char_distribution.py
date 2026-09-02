#!/usr/bin/env python3
"""
plot_special_char_distribution.py

Calcula, EXCLUSIVAMENTE para los modelos phi (8b) y qwen (7b), la
distribución de apariciones de cada carácter especial (definidos en
SPECIAL_CHARS) a lo largo de todos los archivos
'ollama_prompt_TASK_GEN_N.txt' de cada modelo, y grafica DOS BARRAS
SEPARADAS por carácter: una por modelo.

Regla de conteo (importante):
    Por cada PROMPT, un mismo carácter especial se cuenta COMO MÁXIMO UNA
    VEZ, sin importar cuántas veces aparezca dentro de ese prompt (si
    aparece 1 vez o 5 veces dentro del mismo archivo, cuenta igual como 1).
    Esa cuenta de "1 por prompt" SÍ se va sumando a través de todos los
    prompts de cada modelo, para obtener el total de apariciones por
    carácter y por modelo.

Los caracteres cuya suma total (entre ambos modelos) sea 0 se excluyen del
gráfico.

Estructura de datos esperada:
    <data-root>/constellation_dataset_<modelo>/scenario_<n>/ollama_prompt_TASK_GEN_*.txt

Requiere matplotlib:
    pip install matplotlib --break-system-packages

Ejemplo de uso:
    # Autodetecta las carpetas de phi-8b y qwen-7b por nombre
    python3 plot_special_char_distribution.py --data-root data

    # O especifica explícitamente los sufijos de carpeta si el autodetect falla
    python3 plot_special_char_distribution.py --data-root data --models phi3_5_8b,qwen2_7b
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("[!] Este script requiere matplotlib y numpy. Instálalos con:\n"
          "    pip install matplotlib numpy --break-system-packages", file=sys.stderr)
    sys.exit(1)


# ------------------------------------------------------------------ #
# Lista de caracteres especiales a analizar (reducida según lo pedido).
# Edita esta lista según lo que quieras detectar.
# ------------------------------------------------------------------ #
SPECIAL_CHARS = [
    "[",   # corchete de apertura
    '"',   # comilla doble
    "`",   # backtick (markdown / bloques de código)
]

# Grupos de palabras clave usados para AUTODETECTAR las carpetas de phi-8b
# y qwen-7b entre 'constellation_dataset_<modelo>/', si no se pasa --models
# explícitamente. Cada tupla es un conjunto de substrings que TODOS deben
# aparecer (en minúsculas) en el nombre de la carpeta del modelo.
TARGET_MODEL_KEYWORDS = [
    ("phi", "8b"),
    ("qwen", "7b"),
]


def discover_models(data_root: Path):
    models = []
    for p in sorted(data_root.glob("constellation_dataset_*")):
        if p.is_dir():
            models.append(p.name.replace("constellation_dataset_", "", 1))
    return models


def autodetect_target_models(all_models: list, keyword_groups: list):
    matched = []
    for model in all_models:
        lower = model.lower()
        for keywords in keyword_groups:
            if all(k in lower for k in keywords):
                matched.append(model)
                break
    return matched


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


def count_char_occurrences_per_model(data_root: Path, model: str, scenarios: list, chars: list):
    """
    Recorre todos los .txt de prompt de UN modelo y devuelve:
      - char_counts: dict carácter -> número de prompts en los que aparece
        (máximo 1 por carácter por archivo, sumado a través de todos los
        archivos del modelo)
      - total_files: número total de archivos de prompt escaneados para
        este modelo
    """
    char_counts = defaultdict(int)
    total_files = 0

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

            # Máximo 1 por carácter por archivo, sin importar cuántas veces
            # aparezca dentro de ese mismo archivo.
            for ch in chars:
                if ch in content:
                    char_counts[ch] += 1

    return char_counts, total_files


def plot_grouped_distribution(counts_by_model: dict, totals_by_model: dict, output_path: Path, chars: list):
    # Se excluyen caracteres cuya suma entre ambos modelos sea 0
    combined_totals = {ch: sum(counts_by_model[m].get(ch, 0) for m in counts_by_model) for ch in chars}
    filtered_chars = [ch for ch in chars if combined_totals[ch] >= 1]

    if not filtered_chars:
        print("[!] Ningún carácter especial tuvo al menos 1 aparición en ningún modelo. "
              "No hay nada que graficar.")
        sys.exit(0)

    models = sorted(counts_by_model.keys())

    # Etiquetas legibles para caracteres que se confunden visualmente en el eje X
    display_labels = {"`": "` (backtick)", '"': '" (comilla)', "[": "[ (corchete)"}
    x_labels = [display_labels.get(c, c) for c in filtered_chars]

    x = np.arange(len(filtered_chars))
    width = 0.35
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    fig, ax = plt.subplots(figsize=(8, 7.5))

    for i, model in enumerate(models):
        model_total = sum(counts_by_model[model].get(ch, 0) for ch in filtered_chars)
        raw_counts = [counts_by_model[model].get(ch, 0) for ch in filtered_chars]
        percentages = [(c / model_total * 100) if model_total else 0 for c in raw_counts]

        offset = (i - (len(models) - 1) / 2) * width
        bars = ax.bar(x + offset, percentages, width, label=model, color=colors[i % len(colors)])

        for bar, raw in zip(bars, raw_counts):
            height = bar.get_height()
            ax.annotate(f"{raw}\n({height:.1f}%)",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Carácter especial")
    ax.set_ylabel("Porcentaje de apariciones (%)")
    ax.set_title("Distribución de caracteres especiales por modelo")
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.legend(title="Modelo")
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n[SUCCESS] Gráfico guardado en: {output_path.resolve()}")

    print("\nResumen (nº de prompts en los que aparece cada carácter, máx. 1 por prompt):")
    for model in models:
        print(f"  {model:20s}  ({totals_by_model[model]} prompts escaneados)")
        for ch in filtered_chars:
            n = counts_by_model[model].get(ch, 0)
            pct = (n / totals_by_model[model] * 100) if totals_by_model[model] else 0
            print(f"    {display_labels.get(ch, ch):15s}  {n:4d} prompts  ({pct:.1f}% de sus prompts)")


def main():
    parser = argparse.ArgumentParser(
        description="Grafica, con una barra por modelo, la distribución (%) de caracteres "
                    "especiales para phi-8b y qwen-7b, contando como máximo 1 aparición por "
                    "prompt por carácter."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                         help="Carpeta que contiene constellation_dataset_<modelo>/ (default: ./data)")
    parser.add_argument("--scenarios", type=str, default="1-25",
                         help="Rango de escenarios a considerar, ej. '1-25' o '1,3,5' (default: 1-25)")
    parser.add_argument("--models", type=str, default=None,
                         help="Sufijos EXACTOS de carpeta de los modelos a incluir, separados por "
                              "coma (ej. 'phi3_5_8b,qwen2_7b'). Si se omite, se autodetectan "
                              "modelos cuyo nombre contenga 'phi'+'8b' o 'qwen'+'7b'.")
    parser.add_argument("--output", type=Path, default=Path("special_char_distribution.png"),
                         help="Ruta del archivo PNG de salida (default: special_char_distribution.png)")
    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"[!] La carpeta de datos '{args.data_root}' no existe.", file=sys.stderr)
        sys.exit(1)

    all_models = discover_models(args.data_root)
    if not all_models:
        print(f"[!] No se encontraron carpetas 'constellation_dataset_*' en "
              f"'{args.data_root}'.", file=sys.stderr)
        sys.exit(1)

    if args.models:
        target_models = [m.strip() for m in args.models.split(",") if m.strip()]
        missing = [m for m in target_models if m not in all_models]
        if missing:
            print(f"[!] Los siguientes modelos no se encontraron en '{args.data_root}': "
                  f"{missing}\n    Modelos disponibles: {all_models}", file=sys.stderr)
            sys.exit(1)
    else:
        target_models = autodetect_target_models(all_models, TARGET_MODEL_KEYWORDS)
        if len(target_models) != 2:
            print(f"[!] Autodetección encontró {len(target_models)} modelo(s) "
                  f"({target_models}) en vez de los 2 esperados (phi-8b y qwen-7b).\n"
                  f"    Modelos disponibles: {all_models}\n"
                  f"    Usa --models 'sufijo1,sufijo2' para especificarlos manualmente.",
                  file=sys.stderr)
            sys.exit(1)

    print(f"Modelos incluidos en el análisis: {target_models}")

    scenarios = parse_scenario_range(args.scenarios)

    counts_by_model = {}
    totals_by_model = {}
    for model in target_models:
        counts, total_files = count_char_occurrences_per_model(
            args.data_root, model, scenarios, SPECIAL_CHARS
        )
        counts_by_model[model] = counts
        totals_by_model[model] = total_files

    plot_grouped_distribution(counts_by_model, totals_by_model, args.output, SPECIAL_CHARS)


if __name__ == "__main__":
    main()