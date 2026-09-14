import argparse
import json
import pathlib
import re
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np

__all__ = ["generate_metrics_chart", "list_available_dates"]

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif", "Liberation Serif"] + plt.rcParams["font.serif"]

DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _resolve_root(data_root_str: str) -> pathlib.Path:
    root_path = pathlib.Path(data_root_str)
    if not root_path.exists():
        fallback_path = (pathlib.Path(__file__).resolve().parent.parent / "data").resolve()
        print(f"[DEBUG] Path '{data_root_str}' not found. Trying fallback path: {fallback_path}")
        root_path = fallback_path
    return root_path


def list_available_dates(data_root_str: str = "data") -> List[str]:
    """Carpetas YYYY-MM-DD presentes bajo data/, de mas antigua a mas reciente."""
    root_path = _resolve_root(data_root_str)
    if not root_path.exists():
        return []
    return sorted(d.name for d in root_path.iterdir()
                  if d.is_dir() and DATE_PATTERN.match(d.name))


def _pretty_label(rel_path: pathlib.Path) -> str:
    """
    Convierte la ruta relativa de una carpeta de modelo en una etiqueta legible.

        constellation_dataset_llama3_1_8b              -> llama3_1_8b
        few_shot/constellation_dataset_llama3_1_8b     -> few_shot | llama3_1_8b
    """
    parts = [p for p in rel_path.parts]
    model = parts[-1].replace("constellation_dataset_", "")
    prefix = [p for p in parts[:-1]]
    return " | ".join(prefix + [model]) if prefix else model


def _discover_and_parse_metrics(data_root_str: str = "data",
                                date: Optional[str] = None) -> Dict[str, Dict[str, List[float]]]:
    root_path = _resolve_root(data_root_str)
    print(f"[DEBUG] Checking data folder at: {root_path.resolve()}")

    scan_root = root_path
    if date:
        scan_root = root_path / date
        if not scan_root.exists():
            available = list_available_dates(data_root_str)
            print(f"[ERROR] No existe la carpeta de fecha '{date}' en {root_path.resolve()}.")
            print(f"[ERROR] Fechas disponibles: {available if available else 'ninguna'}")
            return {}

    print(f"[DEBUG] Final path used for scanning: {scan_root.resolve()} (Exists: {scan_root.exists()})")

    model_metrics: Dict[str, Dict[str, List[float]]] = {}

    # rglob en vez de glob: encuentra las carpetas de modelo a cualquier
    # profundidad, asi funciona igual con data/<fecha>/<modelo>/ que con
    # data/<fecha>/<estrategia>/<modelo>/.
    for const_dir in sorted(scan_root.rglob("constellation_*")):
        if not const_dir.is_dir():
            continue

        label = _pretty_label(const_dir.relative_to(scan_root))
        print(f"[DEBUG] Processing constellation folder: {label}")

        model_metrics.setdefault(label, {"sensor": [], "priority": [], "day": [], "hour": []})

        scenario_count = 0
        for scenario_dir in const_dir.glob("scenario_*"):
            if not scenario_dir.is_dir():
                continue

            target_json = scenario_dir / "ollama_prompts_combined.json"
            if not target_json.exists():
                continue

            try:
                with target_json.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                metrics = data.get("scenario_metrics", {})

                s_acc = metrics.get("global_sensor_accuracy")
                p_acc = metrics.get("global_priority_accuracy")
                d_acc = metrics.get("global_day_accuracy")
                h_acc = metrics.get("global_hour_accuracy")

                if all(v is not None for v in [s_acc, p_acc, d_acc, h_acc]):
                    model_metrics[label]["sensor"].append(float(s_acc))
                    model_metrics[label]["priority"].append(float(p_acc))
                    model_metrics[label]["day"].append(float(d_acc))
                    model_metrics[label]["hour"].append(float(h_acc))
                    scenario_count += 1
            except Exception:
                continue

        print(f"[DEBUG] Total scenarios successfully evaluated for {label}: {scenario_count}")

        if scenario_count == 0:
            model_metrics.pop(label, None)

    return model_metrics


def generate_metrics_chart(data_dir: str = "data",
                           output_image_path: Optional[str] = None,
                           date: Optional[str] = None) -> None:
    raw_data = _discover_and_parse_metrics(data_dir, date=date)

    if not raw_data:
        print("[ERROR] No valid data found in constellation folders.")
        return

    if output_image_path is None:
        suffix = f"_{date}" if date else ""
        output_image_path = f"utilities/output/model_accuracy_comparison{suffix}.png"

    models = sorted(raw_data.keys())
    categories = ["Day", "Hour", "Sensor", "Priority"]
    keys = ["day", "hour", "sensor", "priority"]

    processed_averages = {m: [np.mean(raw_data[m][k]) if raw_data[m][k] else 0.0 for k in keys]
                          for m in models}
    n_scenarios = {m: len(raw_data[m]["hour"]) for m in models}

    x = np.arange(len(categories))
    n_models = len(models)
    width = (0.8 / n_models) if n_models > 1 else 0.4

    fig, ax = plt.subplots(figsize=(max(11, 2.2 * n_models), 6), dpi=150)

    base_colors = ["#2e86c1", "#27ae60", "#e67e22", "#9b59b6", "#e74c3c"]
    if n_models > len(base_colors):
        cmap = plt.get_cmap("tab20")
        base_colors = [cmap(i / max(n_models - 1, 1)) for i in range(n_models)]

    show_values = n_models <= 6

    for i, model in enumerate(models):
        offset = (i - (n_models - 1) / 2.0) * width
        averages_pct = [val * 100.0 for val in processed_averages[model]]

        rects = ax.bar(
            x + offset,
            averages_pct,
            width,
            label=f"{model}  (n={n_scenarios[model]})",
            color=base_colors[i % len(base_colors)],
            edgecolor="black",
            linewidth=0.8,
        )
        if show_values:
            ax.bar_label(rects, padding=4, fmt="%.1f%%", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Semantic Matching Accuracy (%)", fontsize=11, fontweight="bold")
    title = "KEPLER Dataset Factory - LLM Semantic Generation Accuracy by Category"
    if date:
        title += f"\nRun date: {date}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=15)

    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")

    ax.set_ylim(0, 110)
    ax.grid(axis="y", linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", frameon=True, shadow=False,
              facecolor="#f8f9f9", edgecolor="#d5dbdb", fontsize=8.5)

    out_path = pathlib.Path(output_image_path)
    if not out_path.is_absolute() and not pathlib.Path("utilities").exists():
        out_path = (pathlib.Path(__file__).resolve().parent / "output" / out_path.name).resolve()

    out_path.parent.mkdir(parents=True, exist_ok=True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[SUCCESS] Metrics chart generated at: {out_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Grafica las metricas de precision semantica de una corrida de KDF."
    )
    parser.add_argument("--date", type=str, default=None,
                        help="Carpeta de fecha YYYY-MM-DD bajo data/. Si se omite, se usa "
                             "la mas reciente disponible.")
    parser.add_argument("--data-dir", type=str, default="data",
                        help="Raiz de datos (default: data)")
    parser.add_argument("--output", type=str, default=None,
                        help="Ruta del PNG de salida. Por defecto incluye la fecha en el nombre.")
    parser.add_argument("--list-dates", action="store_true",
                        help="Solo listar las fechas disponibles y salir.")
    args = parser.parse_args()

    if args.list_dates:
        dates = list_available_dates(args.data_dir)
        print("Fechas disponibles:" if dates else "No hay carpetas de fecha bajo data/.")
        for d in dates:
            print(f"  {d}")
        raise SystemExit(0)

    selected = args.date
    if selected is None:
        dates = list_available_dates(args.data_dir)
        if dates:
            selected = dates[-1]
            print(f"[INFO] No se especifico --date; usando la mas reciente: {selected}")
        else:
            print("[INFO] No hay carpetas de fecha; escaneando data/ directamente.")

    generate_metrics_chart(data_dir=args.data_dir, output_image_path=args.output, date=selected)