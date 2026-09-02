#!/usr/bin/env python3
"""
sample_and_label.py

Herramienta de muestreo y etiquetado manual para el diagnóstico de errores
en la categoría 'Hour' del Request Generator de KDF (Fase 1 del proyecto:
Diagnóstico del pipeline actual).

Estructura de datos esperada:
    <data-root>/constellation_dataset_<modelo>/scenario_<n>/ollama_prompts_combined.json
    (se asumen escenarios 1 a 25 para cada modelo, configurable con --scenarios)

Cada tarea con validation.hour_match == false es candidata a revisión manual.
Para cada tarea muestreada se le muestra al usuario la hora esperada
(ground truth), la hora clasificada por el validador y el texto en lenguaje
natural generado, y se le pide clasificar la causa raíz del error como:

    [g] error de GENERACIÓN     -> el LLM no expresó el concepto temporal
                                    correcto en el texto generado.
    [c] error de CLASIFICACIÓN  -> el LLM sí expresó el concepto correcto,
                                    pero el validador (embeddings + similitud
                                    coseno) lo clasificó incorrectamente.
    [s] SKIP                    -> caso ambiguo, se revisará más adelante.
    [q] guardar y salir.

El progreso se persiste en un archivo JSON (--log-file) para que ejecuciones
futuras:
  (a) no vuelvan a muestrear tareas ya etiquetadas, y
  (b) excluyan del muestreo los escenarios que ya fueron completamente
      revisados (es decir, todas sus tareas con hour_match=false ya tienen
      etiqueta).

Ejemplos de uso:
    # Sesión de etiquetado: 5 tareas por modelo, ruta de datos por defecto ./data
    python3 sample_and_label.py --data-root data --samples-per-model 5

    # Ver solo el resumen acumulado de lo ya etiquetado, sin muestrear
    python3 sample_and_label.py --data-root data --summary-only

    # Sesión reproducible (misma muestra si no ha cambiado el log)
    python3 sample_and_label.py --data-root data --samples-per-model 10 --seed 42
"""

import argparse
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

VALID_LABELS = {"g": "generation_error", "c": "classification_error", "s": "skipped"}


# ------------------------------------------------------------------ #
# Descubrimiento de datos
# ------------------------------------------------------------------ #

def discover_models(data_root: Path):
    """Encuentra las carpetas constellation_dataset_<modelo> dentro de data_root."""
    models = []
    for p in sorted(data_root.glob("constellation_dataset_*")):
        if p.is_dir():
            models.append(p.name.replace("constellation_dataset_", "", 1))
    return models


def scenario_json_path(data_root: Path, model: str, scenario: int) -> Path:
    return (data_root / f"constellation_dataset_{model}"
            / f"scenario_{scenario}" / "ollama_prompts_combined.json")


def load_scenario(data_root: Path, model: str, scenario: int):
    path = scenario_json_path(data_root, model, scenario)
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [!] No se pudo leer {path}: {e}", file=sys.stderr)
        return None


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


# ------------------------------------------------------------------ #
# Persistencia del progreso
# ------------------------------------------------------------------ #

def load_log(log_file: Path):
    if log_file.exists():
        with open(log_file, encoding="utf-8") as f:
            return json.load(f)
    return {"reviewed": {}}


def save_log(log_file: Path, log: dict):
    tmp = log_file.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    tmp.replace(log_file)  # escritura atómica


def record_key(model, scenario, task_id):
    return f"{model}|{scenario}|{task_id}"


# ------------------------------------------------------------------ #
# Recolección y muestreo de candidatos
# ------------------------------------------------------------------ #

def collect_failing_tasks(data_root: Path, models, scenarios, log: dict):
    """
    Recorre todos los (modelo, escenario) disponibles y devuelve:
      - candidates: dict modelo -> lista de tareas con hour_match=false que
        AÚN NO han sido revisadas (no están en el log).
      - scenario_totals: dict (modelo, escenario) -> (total_fallas, revisadas)
        usado para reportar cobertura y omitir escenarios ya completos.
    """
    candidates = defaultdict(list)
    scenario_totals = {}

    for model in models:
        for scenario in scenarios:
            data = load_scenario(data_root, model, scenario)
            if data is None:
                continue

            failing = [
                t for t in data.get("tasks", [])
                if not t.get("validation", {}).get("hour_match", True)
            ]

            reviewed_count = sum(
                1 for t in failing
                if record_key(model, scenario, t["task_id"]) in log["reviewed"]
            )
            scenario_totals[(model, scenario)] = (len(failing), reviewed_count)

            if failing and reviewed_count == len(failing):
                continue  # escenario ya completamente revisado: se omite del muestreo

            for t in failing:
                key = record_key(model, scenario, t["task_id"])
                if key in log["reviewed"]:
                    continue
                validation = t.get("validation", {})
                candidates[model].append({
                    "model": model,
                    "scenario": scenario,
                    "task_id": t.get("task_id"),
                    "expected_hour": t.get("ground_truth", {}).get("expected_hour"),
                    "predicted_hour": validation.get("predicted_hour"),
                    "hour_similarity": validation.get("hour_similarity"),
                    "generated_output": t.get("generated_output", ""),
                })

    return candidates, scenario_totals


def sample_per_model(candidates: dict, samples_per_model: int, rng: random.Random):
    """Muestrea uniformemente hasta `samples_per_model` tareas por modelo,
    escogidas al azar entre todos los escenarios pendientes de ese modelo."""
    sampled = []
    for model, tasks in candidates.items():
        rng.shuffle(tasks)
        chosen = tasks[:samples_per_model]
        if len(chosen) < samples_per_model:
            print(f"  [!] Modelo '{model}': solo quedan {len(chosen)} tareas pendientes "
                  f"(se pidieron {samples_per_model}).")
        sampled.extend(chosen)
    rng.shuffle(sampled)
    return sampled


# ------------------------------------------------------------------ #
# Interfaz de terminal
# ------------------------------------------------------------------ #

def print_task(task: dict, idx: int, total: int):
    print("\n" + "=" * 78)
    print(f"[{idx}/{total}]  Modelo: {task['model']}   Escenario: {task['scenario']}   "
          f"Task ID: {task['task_id']}")
    print("-" * 78)
    sim = task["hour_similarity"]
    sim_txt = f"{sim:.4f}" if isinstance(sim, (int, float)) else "N/A"
    print(f"  Ground truth (hour) : {task['expected_hour']}")
    print(f"  Clasificado como    : {task['predicted_hour']}  (similitud coseno = {sim_txt})")
    print("-" * 78)
    print("  Texto generado (generated_output):")
    print(f"    {task['generated_output']}")
    print("=" * 78)


def ask_label():
    while True:
        resp = input("  ¿Causa del error? [g]eneración / [c]lasificación / [s]kip / [q]uit: ").strip().lower()
        if resp in ("g", "c", "s", "q"):
            return resp
        print("  Entrada no válida. Use g, c, s o q.")


def print_summary(log: dict, scenario_totals: dict):
    counts = defaultdict(lambda: defaultdict(int))
    for key, entry in log["reviewed"].items():
        model = key.split("|", 1)[0]
        counts[model][entry["label"]] += 1

    print("\n" + "#" * 78)
    print("RESUMEN DE ETIQUETADO ACUMULADO")
    print("#" * 78)
    total_g = total_c = total_s = 0
    for model in sorted(counts):
        g = counts[model].get("generation_error", 0)
        c = counts[model].get("classification_error", 0)
        s = counts[model].get("skipped", 0)
        total_g += g
        total_c += c
        total_s += s
        print(f"  {model:20s}  generación={g:4d}  clasificación={c:4d}  skip={s:4d}")
    print("-" * 78)
    print(f"  {'TOTAL':20s}  generación={total_g:4d}  clasificación={total_c:4d}  skip={total_s:4d}")

    with_failures = [(m, s) for (m, s), (tot, _) in scenario_totals.items() if tot > 0]
    completed = [(m, s) for (m, s) in with_failures if scenario_totals[(m, s)][0] == scenario_totals[(m, s)][1]]
    print(f"\n  Escenarios con fallas en 'Hour' completamente revisados: "
          f"{len(completed)}/{len(with_failures)}")
    print("#" * 78 + "\n")


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(
        description="Muestreo y etiquetado manual de errores en la categoría Hour del RG de KDF."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                         help="Carpeta que contiene constellation_dataset_<modelo>/ (default: ./data)")
    parser.add_argument("--samples-per-model", type=int, default=5,
                         help="Número de tareas a muestrear por modelo en esta sesión (default: 5)")
    parser.add_argument("--scenarios", type=str, default="1-25",
                         help="Rango de escenarios a considerar, ej. '1-25' o '1,3,5' (default: 1-25)")
    parser.add_argument("--log-file", type=Path, default=Path("hour_error_review_log.json"),
                         help="Archivo JSON donde se guarda el progreso "
                              "(default: hour_error_review_log.json)")
    parser.add_argument("--seed", type=int, default=None,
                         help="Semilla para el muestreo aleatorio (opcional, reproducibilidad)")
    parser.add_argument("--summary-only", action="store_true",
                         help="Solo imprime el resumen acumulado de lo ya etiquetado, sin muestrear")
    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"[!] La carpeta de datos '{args.data_root}' no existe.", file=sys.stderr)
        sys.exit(1)

    scenarios = parse_scenario_range(args.scenarios)
    rng = random.Random(args.seed)

    models = discover_models(args.data_root)
    if not models:
        print(f"[!] No se encontraron carpetas 'constellation_dataset_*' en "
              f"'{args.data_root}'.", file=sys.stderr)
        sys.exit(1)
    print(f"Modelos encontrados ({len(models)}): {', '.join(models)}")

    log = load_log(args.log_file)

    print("Escaneando escenarios y calculando tareas pendientes de revisión...")
    candidates, scenario_totals = collect_failing_tasks(args.data_root, models, scenarios, log)

    if args.summary_only:
        print_summary(log, scenario_totals)
        return

    total_candidates = sum(len(v) for v in candidates.values())
    print(f"Tareas con hour_match=false aún sin revisar: {total_candidates}")
    if total_candidates == 0:
        print("No hay tareas pendientes de revisión. Nada que muestrear.")
        print_summary(log, scenario_totals)
        return

    sampled = sample_per_model(candidates, args.samples_per_model, rng)
    print(f"\nSe muestrearon {len(sampled)} tareas para esta sesión "
          f"(hasta {args.samples_per_model} por modelo).\n")

    for i, task in enumerate(sampled, start=1):
        print_task(task, i, len(sampled))
        resp = ask_label()
        if resp == "q":
            print("\nGuardando progreso y saliendo...")
            break
        label = VALID_LABELS[resp]
        key = record_key(task["model"], task["scenario"], task["task_id"])
        log["reviewed"][key] = {
            "label": label,
            "expected_hour": task["expected_hour"],
            "predicted_hour": task["predicted_hour"],
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
        }
        save_log(args.log_file, log)  # se guarda inmediatamente tras cada etiqueta

    print_summary(log, scenario_totals)


if __name__ == "__main__":
    main()