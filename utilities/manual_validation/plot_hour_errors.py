#!/usr/bin/env python3
"""
plot_hour_errors.py

Genera un gráfico de barras agrupadas a partir de 'hour_error_review_log.json'
(el log producido por sample_and_label.py), mostrando por cada modelo dos
barras: el número de errores etiquetados como 'generation_error' y el número
etiquetados como 'classification_error' (validación por embeddings).

Los casos etiquetados como 'skipped' se ignoran en el conteo, pero se
reportan aparte en consola.

Requiere matplotlib:
    pip install matplotlib --break-system-packages

Ejemplo de uso:
    python3 plot_hour_errors.py --log-file hour_error_review_log.json
    python3 plot_hour_errors.py --log-file hour_error_review_log.json --output errores_hour.png
"""

import argparse
import json
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


def load_log(log_file: Path) -> dict:
    if not log_file.exists():
        print(f"[!] No se encontró el archivo de log: {log_file}", file=sys.stderr)
        sys.exit(1)
    with open(log_file, encoding="utf-8") as f:
        return json.load(f)


def count_errors_by_model(log: dict):
    """
    Devuelve un dict: modelo -> {"generation_error": N, "classification_error": N, "skipped": N}
    a partir de las claves 'modelo|escenario|task_id' del log.
    """
    counts = defaultdict(lambda: defaultdict(int))
    for key, entry in log.get("reviewed", {}).items():
        model = key.split("|", 1)[0]
        label = entry.get("label", "unknown")
        counts[model][label] += 1
    return counts


def plot_grouped_bars(counts: dict, output_path: Path, title: str):
    models = sorted(counts.keys())
    if not models:
        print("[!] El log no contiene revisiones etiquetadas. No hay nada que graficar.")
        sys.exit(0)

    generation_vals = [counts[m].get("generation_error", 0) for m in models]
    classification_vals = [counts[m].get("classification_error", 0) for m in models]
    skipped_vals = [counts[m].get("skipped", 0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(6, len(models) * 1.6), 6))
    bars_gen = ax.bar(x - width / 2, generation_vals, width, label="Error de generación",
                       color="#4C72B0")
    bars_cls = ax.bar(x + width / 2, classification_vals, width, label="Error de clasificación/validación",
                       color="#DD8452")

    ax.set_xlabel("Modelo")
    ax.set_ylabel("Número de errores etiquetados")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.4)
    ax.set_axisbelow(True)

    for bars in (bars_gen, bars_cls):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f"{int(height)}", xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"[SUCCESS] Gráfico guardado en: {output_path.resolve()}")

    print("\nResumen por modelo:")
    for m in models:
        g = counts[m].get("generation_error", 0)
        c = counts[m].get("classification_error", 0)
        s = counts[m].get("skipped", 0)
        print(f"  {m:20s}  generación={g:4d}  clasificación={c:4d}  skip={s:4d}")


def main():
    parser = argparse.ArgumentParser(
        description="Grafica errores de generación vs clasificación por modelo, a partir "
                    "de hour_error_review_log.json."
    )
    parser.add_argument("--log-file", type=Path, default=Path("hour_error_review_log.json"),
                         help="Ruta al log de revisiones (default: hour_error_review_log.json)")
    parser.add_argument("--output", type=Path, default=Path("hour_errors_by_model.png"),
                         help="Ruta del archivo PNG de salida (default: hour_errors_by_model.png)")
    parser.add_argument("--title", type=str,
                         default="Errores en categoría 'Hour' por modelo: generación vs clasificación",
                         help="Título del gráfico")
    args = parser.parse_args()

    log = load_log(args.log_file)
    counts = count_errors_by_model(log)
    plot_grouped_bars(counts, args.output, args.title)


if __name__ == "__main__":
    main()