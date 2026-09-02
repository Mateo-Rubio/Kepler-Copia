#!/usr/bin/env python3
"""
run_all_models.py

Corre el pipeline completo de KDF (Data Collector -> Request Generator ->
Physics Engine) para VARIOS modelos de Ollama, generando todos los
escenarios configurados (por defecto 25) por cada uno, y al final invoca
utilities/plot_metrics.py para producir el gráfico comparativo final.

Diferencias clave respecto a 'python src/main.py':

  1. Corre TODOS los modelos en una sola ejecución (no hay que editar
     config.yaml y volver a lanzar el script 5 veces).
  2. Si un escenario puntual falla (timeout de Ollama, error de red con
     Nominatim, error del modelo de embeddings, etc.), el error se registra
     y el script CONTINÚA con el siguiente escenario, en vez de abortar
     todo el proceso como hace 'src/main.py' (que envuelve el loop completo
     en un único try/except con sys.exit(1)).
  3. Es REANUDABLE: si vuelves a correr el script, los escenarios que ya
     tienen su 'ollama_prompts_combined.json' completo (con el número
     esperado de tareas) se OMITEN automáticamente, así que no se
     regeneran ni se vuelve a gastar tiempo/llamadas a Ollama en ellos.

Debe ejecutarse desde la raíz del repositorio de KDF (mismo nivel que
'src/', 'config.yaml' y 'utilities/').

Ejemplo de uso:
    # Corre los 5 modelos del paper, 25 escenarios cada uno (usa config.yaml
    # como base para todo excepto dataset_name/ollama_model, que se
    # sobreescriben por modelo).
    python3 run_all_models.py

    # Solo 2 modelos, 5 escenarios cada uno (para probar rápido)
    python3 run_all_models.py --models "llama3.1:8b,phi4:14b" --num-scenarios 5

    # Reintentar SOLO los escenarios que fallaron en una corrida anterior
    python3 run_all_models.py --retry-failed-only
"""

import argparse
import json
import os
import pathlib
import sys
import traceback
from datetime import datetime, timezone, timedelta

try:
    import requests
except ImportError:
    print("[!] Este script requiere 'requests'. Instálalo con: pip install requests", file=sys.stderr)
    sys.exit(1)

try:
    import yaml
except ImportError:
    print("[!] Este script requiere PyYAML. Instálalo con: pip install pyyaml", file=sys.stderr)
    sys.exit(1)

from src.modules.data_collector.main import data_collector_main
from src.modules.physics_engine.main import physics_engine_main
from src.modules.prompt_factory.main import prompt_factory_main

# Modelos evaluados en el paper original de KDF. Se pueden sobreescribir
# con --models.
DEFAULT_MODELS = [
    "gemma2:27b",
    "llama3.1:8b",
    "phi3.5:3.8b",
    "phi4:14b",
    "qwen2:7b",
]

FAILURE_LOG_PATH = pathlib.Path("run_all_models_failures.json")


def model_to_dataset_name(model: str) -> str:
    """Convierte 'phi3.5:8b' -> 'constellation_dataset_phi3_5_8b' (mismo
    patrón de nombres usado en el resto del repositorio)."""
    clean = model.replace(":", "_").replace(".", "_")
    return f"constellation_dataset_{clean}"


def get_ollama_url() -> str:
    """Misma variable de entorno que usa src/modules/prompt_factory/generator.py,
    para que el chequeo previo apunte exactamente al mismo servidor Ollama que
    se usará durante la generación real."""
    return os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")


def check_ollama_models_available(models: list, embedding_model: str = "mxbai-embed-large",
                                   timeout: int = 10):
    """
    Verifica, ANTES de correr nada, que:
      1. El servicio de Ollama esté accesible.
      2. Todos los modelos de generación solicitados estén descargados.
      3. El modelo de embeddings usado por el validador también lo esté.

    Devuelve (available, missing, embedding_missing):
      - available: lista de modelos solicitados que SÍ están descargados
      - missing: lista de modelos solicitados que NO están descargados
      - embedding_missing: True si falta el modelo de embeddings

    Lanza ConnectionError si Ollama no responde en absoluto (nada que
    verificar si el servicio ni siquiera está corriendo).
    """
    base_url = get_ollama_url().rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ConnectionError(
            f"No se pudo conectar con Ollama en '{base_url}'. "
            f"¿Está corriendo el servicio ('ollama serve')? Detalle: {e}"
        )

    installed_raw = [m.get("name", "") for m in response.json().get("models", [])]
    # Ollama a veces añade ':latest' implícito; comparamos también sin ese sufijo
    # para no dar falsos negativos (ej. usuario pidió 'phi4' y Ollama tiene
    # 'phi4:latest').
    installed_normalized = set(installed_raw)
    for name in installed_raw:
        if name.endswith(":latest"):
            installed_normalized.add(name[: -len(":latest")])

    available, missing = [], []
    for model in models:
        if model in installed_normalized:
            available.append(model)
        else:
            missing.append(model)

    embedding_missing = (embedding_model not in installed_normalized)

    return available, missing, embedding_missing


def load_config(config_path: str) -> dict:
    p = pathlib.Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"No se encontró el archivo de configuración: {config_path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not cfg:
        raise ValueError("El archivo de configuración está vacío.")
    return cfg


def load_semantic_categories(categories_path: str) -> dict:
    p = pathlib.Path(categories_path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def scenario_already_complete(scenario_dir: pathlib.Path, expected_tasks: int) -> bool:
    """
    Un escenario se considera COMPLETO si su ollama_prompts_combined.json
    existe, es JSON válido, y tiene exactamente el número esperado de tareas
    procesadas. Esto es lo que permite que el script sea reanudable.
    """
    combined_path = scenario_dir / "ollama_prompts_combined.json"
    if not combined_path.exists():
        return False
    try:
        with combined_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        total = data.get("scenario_metrics", {}).get("total_tasks_evaluated", 0)
        return total >= expected_tasks
    except (json.JSONDecodeError, OSError):
        return False


def load_failure_log() -> list:
    if FAILURE_LOG_PATH.exists():
        with FAILURE_LOG_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_failure_log(failures: list):
    with FAILURE_LOG_PATH.open("w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2, ensure_ascii=False)


def run_single_scenario(model: str, dataset_name: str, idx: int, cfg: dict,
                         sem_categories: dict, current_seed: int):
    """Ejecuta el pipeline completo (Data Collector -> RG -> Physics Engine)
    para UN escenario. Lanza excepción si algo falla; el llamador decide
    qué hacer con ese error (ver main())."""
    sim_cfg = cfg.get("simulation", {})
    pay_cfg = cfg.get("payload", {})
    task_cfg = cfg.get("task_generation", {})
    path_cfg = cfg.get("paths", {})

    max_release_delay = task_cfg["max_release_delay"]
    max_lifetime = task_cfg["max_lifetime"]
    total_required_duration_s = max_release_delay + max_lifetime

    scenario_dir = pathlib.Path("data") / dataset_name / f"scenario_{idx}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    scenario_report_path = scenario_dir / "scenario_report.json"

    collector_kwargs = {
        "sat_k": sim_cfg.get("sat_k"),
        "gs_k": sim_cfg.get("gs_k"),
        "tasks_k": sim_cfg.get("tasks_k", 10),
        "bounding_boxes": task_cfg.get("bounding_boxes"),
        "polygon_ratio": task_cfg.get("polygon_ratio", 0.5),
        "min_area_deg": task_cfg.get("min_area_deg", 0.05),
        "max_area_deg": task_cfg.get("max_area_deg", 0.20),
        "min_release_delay": task_cfg.get("min_release_delay", 0),
        "max_release_delay": max_release_delay,
        "min_lifetime": task_cfg.get("min_lifetime", 1800),
        "max_lifetime": max_lifetime,
        "min_duration": task_cfg.get("min_duration", 5),
        "max_duration": task_cfg.get("max_duration", 30),
        "gs_file_path": path_cfg.get("gs_file_path", "data/ground_station.csv"),
        "available_sensors": pay_cfg.get("sensors_pool"),
        "sensor_weights": pay_cfg.get("sensor_weights"),
        "band_weights_map": pay_cfg.get("bands_config", {}),
        "storage_capacity_pool_mb": pay_cfg.get("storage_capacity_pool_mb"),
        "sensor_generation_rates": pay_cfg.get("sensor_generation_rates"),
        "min_sensors_per_sat": pay_cfg.get("min_sensors_per_sat", 1),
        "max_sensors_per_sat": pay_cfg.get("max_sensors_per_sat", 2),
        "priority_weights": task_cfg.get("priority_weights"),
        "seed": current_seed,
        "output_path": str(scenario_report_path),
    }
    if sim_cfg.get("sat_group_name"):
        collector_kwargs["sat_group_name"] = sim_cfg.get("sat_group_name")
    else:
        collector_kwargs["sat_file_path"] = path_cfg.get("sat_file_path")

    context = data_collector_main(**collector_kwargs)

    t0 = context.tle_epoch_utc if getattr(context, "tle_epoch_utc", None) else datetime.now(timezone.utc)
    tf = t0 + timedelta(seconds=total_required_duration_s)

    # Reloj propio para la generación de lenguaje natural, independiente de
    # la epoca del TLE (ver el fix aplicado a src/main.py). Se captura una
    # sola vez para este escenario y se comparte entre generación y
    # ground_truth.
    generation_now_utc = datetime.now(timezone.utc)

    if sim_cfg.get("semantic_enabled", True):
        prompt_cfg = task_cfg.get("prompt_generation", {})
        ollama_temp = sim_cfg.get("ollama_temperature", 0.3)

        prompt_factory_main(
            targets=context.targets,
            prompt_config=prompt_cfg,
            output_dir=str(scenario_dir),
            model_name=model,
            temperature=ollama_temp,
            sensor_categories=sem_categories.get("sensor_categories"),
            priority_categories=sem_categories.get("priority_categories"),
            days_categories=sem_categories.get("days_categories"),
            hours_categories=sem_categories.get("hours_categories"),
            simulation_t0=generation_now_utc,
        )

    physics_report_path = scenario_dir / "physics_passes_report.json"
    physics_engine_main(
        context=context,
        bands_config=pay_cfg.get("bands_config", {}),
        sensor_constraints=pay_cfg.get("sensor_constraints", {}),
        simulation_start_utc=t0,
        simulation_end_utc=tf,
        output_path=str(physics_report_path),
        step_seconds=20,
        min_duration=collector_kwargs["min_duration"],
        max_duration=collector_kwargs["max_duration"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Corre el pipeline de KDF para varios modelos y todos sus escenarios, "
                    "continuando ante fallos puntuales y siendo reanudable."
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                         help="Ruta al config.yaml BASE (default: config.yaml). Se reutiliza "
                              "para todos los modelos, excepto dataset_name/ollama_model, que "
                              "se sobreescriben automáticamente por modelo.")
    parser.add_argument("--categories", type=str, default="semantic_categories.json",
                         help="Ruta al JSON de categorías semánticas")
    parser.add_argument("--models", type=str, default=None,
                         help="Lista de modelos separados por coma (ej. 'llama3.1:8b,phi4:14b'). "
                              "Default: los 5 modelos del paper original.")
    parser.add_argument("--num-scenarios", type=int, default=None,
                         help="Número de escenarios por modelo (default: el de config.yaml, "
                              "usualmente 25)")
    parser.add_argument("--retry-failed-only", action="store_true",
                         help="Solo reintenta los (modelo, escenario) que quedaron registrados "
                              "como fallidos en la corrida anterior (run_all_models_failures.json)")
    parser.add_argument("--skip-plot", action="store_true",
                         help="No generar el gráfico final automáticamente al terminar")
    parser.add_argument("--skip-model-check", action="store_true",
                         help="Omitir la verificación previa de modelos disponibles en Ollama "
                              "(no recomendado, salvo que ya la hayas hecho manualmente)")
    parser.add_argument("--continue-without-missing", action="store_true",
                         help="Si algún modelo no está descargado, continuar solo con los "
                              "disponibles en vez de detener la ejecución por completo")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sem_categories = load_semantic_categories(args.categories)
    sim_cfg = cfg.get("simulation", {})
    task_cfg = cfg.get("task_generation", {})

    models = ([m.strip() for m in args.models.split(",") if m.strip()]
              if args.models else DEFAULT_MODELS)
    num_scenarios = args.num_scenarios or sim_cfg.get("num_scenarios", 25)
    base_seed = sim_cfg.get("seed", 42)
    tasks_k = sim_cfg.get("tasks_k", 10)

    # --------------------------------------------------------------- #
    # Verificación previa: ¿están todos los modelos realmente
    # descargados en Ollama ANTES de gastar tiempo generando escenarios?
    # --------------------------------------------------------------- #
    if not args.skip_model_check:
        print("Verificando disponibilidad de modelos en Ollama...")
        try:
            available, missing, embedding_missing = check_ollama_models_available(models)
        except ConnectionError as e:
            print(f"\n[ERROR] {e}", file=sys.stderr)
            sys.exit(1)

        for m in available:
            print(f"  [OK]      {m}")
        for m in missing:
            print(f"  [FALTA]   {m}")
        if embedding_missing:
            print(f"  [FALTA]   mxbai-embed-large  (modelo de embeddings usado por el validador)")

        if missing or embedding_missing:
            print()
            print("[!] Faltan modelos por descargar. Ejecuta antes de continuar:")
            for m in missing:
                print(f"      ollama pull {m}")
            if embedding_missing:
                print(f"      ollama pull mxbai-embed-large")

            if missing and not args.continue_without_missing:
                print("\nAbortando (usa --continue-without-missing para seguir solo con los "
                      "modelos disponibles, o --skip-model-check para omitir esta verificación).")
                sys.exit(1)
            elif missing:
                print(f"\n[INFO] Continuando SOLO con los modelos disponibles: {available}")
                models = available
                if not models:
                    print("[ERROR] Ningún modelo solicitado está disponible. Nada que correr.",
                          file=sys.stderr)
                    sys.exit(1)
        else:
            print("Todos los modelos requeridos están disponibles.\n")

    previous_failures = load_failure_log()
    retry_set = {(f["model"], f["scenario"]) for f in previous_failures}

    if args.retry_failed_only and not retry_set:
        print("[INFO] No hay fallos previos registrados en "
              f"'{FAILURE_LOG_PATH}'. Nada que reintentar.")
        return

    print("=" * 70)
    print(f" Modelos a procesar: {models}")
    print(f" Escenarios por modelo: {num_scenarios}")
    if args.retry_failed_only:
        print(f" Modo: SOLO reintentar {len(retry_set)} fallo(s) previo(s)")
    print("=" * 70)

    new_failures = []
    total_run = total_skipped = total_ok = total_failed = 0

    for model in models:
        dataset_name = model_to_dataset_name(model)
        print(f"\n{'#' * 70}\n MODELO: {model}  ->  data/{dataset_name}/\n{'#' * 70}")

        for idx in range(1, num_scenarios + 1):
            if args.retry_failed_only and (model, idx) not in retry_set:
                continue

            scenario_dir = pathlib.Path("data") / dataset_name / f"scenario_{idx}"

            if not args.retry_failed_only and scenario_already_complete(scenario_dir, tasks_k):
                total_skipped += 1
                print(f"  [SKIP] {model} escenario {idx}: ya completo, se omite.")
                continue

            current_seed = base_seed + idx if base_seed is not None else None
            total_run += 1
            print(f"\n  [RUN] {model} escenario {idx}/{num_scenarios} (seed={current_seed})...")

            try:
                run_single_scenario(model, dataset_name, idx, cfg, sem_categories, current_seed)
                total_ok += 1
                print(f"  [OK]  {model} escenario {idx} completado.")
            except Exception as e:
                total_failed += 1
                print(f"  [FAIL] {model} escenario {idx}: {e}", file=sys.stderr)
                traceback.print_exc(file=sys.stderr)
                new_failures.append({
                    "model": model,
                    "scenario": idx,
                    "error": str(e),
                    "failed_at": datetime.now(timezone.utc).isoformat(),
                })
                # NO se aborta: se continúa con el siguiente escenario.
                continue

    # El log de fallos se reescribe con los fallos de ESTA corrida. Si
    # --retry-failed-only tenía éxito en todos los reintentos, new_failures
    # queda vacío y el archivo de fallos se elimina más abajo.
    save_failure_log(new_failures)

    print("\n" + "=" * 70)
    print(" RESUMEN")
    print("=" * 70)
    print(f"  Ejecutados en esta corrida : {total_run}")
    print(f"  Omitidos (ya completos)    : {total_skipped}")
    print(f"  Exitosos                   : {total_ok}")
    print(f"  Fallidos                   : {total_failed}")
    if new_failures:
        print(f"\n  [!] Hay {len(new_failures)} escenario(s) fallido(s), registrados en "
              f"'{FAILURE_LOG_PATH}'.")
        print("      Corrige la causa (ej. reinicia Ollama) y reintenta solo esos con:")
        print("      python3 run_all_models.py --retry-failed-only")
    else:
        if FAILURE_LOG_PATH.exists():
            FAILURE_LOG_PATH.unlink()

    if not args.skip_plot:
        print("\nGenerando gráfico comparativo final...")
        try:
            from utilities.plot_metrics import generate_metrics_chart
            generate_metrics_chart()
        except Exception as e:
            print(f"[!] No se pudo generar el gráfico final: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()