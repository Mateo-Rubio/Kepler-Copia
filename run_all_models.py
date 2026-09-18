import argparse
import json
import os
import pathlib
import yaml
import requests
from datetime import datetime, timezone, timedelta
from src.modules.data_collector.main import data_collector_main
from src.modules.physics_engine.main import physics_engine_main
from src.modules.prompt_factory.main import prompt_factory_main

RUN_DATE = datetime.now(timezone.utc).strftime("%Y-%m-%d")
DEFAULT_MODELS = [
    "gemma2:27b",
    "llama3.1:8b",
    "phi3.5:3.8b",
    "phi4:14b",
    "qwen2:7b",
]
GENERATION_NOW_UTC = datetime(2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
CATEGORIES_FILE = "semantic_categories.json"
    
def build_scenario_dir(strategy: str, dataset_name: str, idx: int) -> pathlib.Path:
    return (pathlib.Path("data") / RUN_DATE / strategy / dataset_name / f"scenario_{idx}")

def model_to_dataset_name(model: str) -> str:
    clean = model.replace(":", "_").replace(".", "_")
    return f"constellation_dataset_{clean}"

def check_ollama_models_available(models: list, embedding_model: str = "mxbai-embed-large",
                                   timeout: int = 10):
    ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    base_url = ollama_url.rstrip("/")
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=timeout)
        response.raise_for_status()
    except requests.RequestException as e:
        raise ConnectionError(
            f"KEPLER-DF could not establish a connection to Ollama with url: '{base_url}': {e} "

        )

    installed_raw = [m.get("name", "") for m in response.json().get("models", [])]
    installed_normalized = set(installed_raw)
    for name in installed_raw:
        if name.endswith(":latest"):
            installed_normalized.add(name[: -len(":latest")])
    for model in models:
        if model not in installed_normalized:
            raise ConnectionError(
                f"The model '{model}' is not installed in ollama"
            )
    if embedding_model not in installed_normalized:
        raise ConnectionError(
            f"The embedding model '{embedding_model}' is not installed in ollama"
        )


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
    """
    Upload the JSON file containing the semantic categories and anchor text.
    """
    p = pathlib.Path(categories_path)
    if not p.exists():
        print(f"[WARNING] Semantic categories file not found at: {categories_path}. Using default values.")
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)

def scenario_already_complete(scenario_dir: pathlib.Path, expected_tasks: int) -> bool:
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
    
def validate_config_bounds_sanity(task_cfg: dict) -> None:
    """
    Verify that the configuration limits and time frames are logically consistent.
    """
    min_release = task_cfg.get("min_release_delay")
    max_release = task_cfg.get("max_release_delay")
    min_lifetime = task_cfg.get("min_lifetime")
    max_lifetime = task_cfg.get("max_lifetime")

    for name, val in [("min_release_delay", min_release), ("max_release_delay", max_release), 
                      ("min_lifetime", min_lifetime), ("max_lifetime", max_lifetime)]:
        if val is None:
            raise ValueError(f"CRITICAL CONFIG ERROR: Parameter '{name}' is missing in task_generation block.")
        if not isinstance(val, (int, float)) or val < 0:
            raise ValueError(f"CRITICAL CONFIG ERROR: Parameter '{name}' must be a non-negative number. Got: {val}")

    if min_release > max_release:
        raise ValueError(f"CRITICAL CONFIG ERROR: 'min_release_delay' ({min_release}s) cannot be greater than 'max_release_delay' ({max_release}s).")

    if min_lifetime > max_lifetime:
        raise ValueError(f"CRITICAL CONFIG ERROR: 'min_lifetime' ({min_lifetime}s) cannot be greater than 'max_lifetime' ({max_lifetime}s).")

    if max_lifetime == 0:
        raise ValueError("CRITICAL CONFIG ERROR: 'max_lifetime' cannot be zero. Tasks would expire instantly.")

def run_single_scenario(model: str, dataset_name: str, idx: int, cfg: dict,
                         sem_categories: dict, current_seed: int,
                         strategy: str = "zero_shot"):
    sim_cfg = cfg.get("simulation", {})
    pay_cfg = cfg.get("payload", {})
    task_cfg = cfg.get("task_generation", {})
    path_cfg = cfg.get("paths", {})
    max_release_delay = task_cfg["max_release_delay"]
    max_lifetime = task_cfg["max_lifetime"]
    total_required_duration_s = max_release_delay + max_lifetime
    
    scenario_dir = build_scenario_dir(strategy, dataset_name, idx)
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
    generation_now_utc = GENERATION_NOW_UTC

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
            strategy=strategy,
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
    ## TODO Translate pls
    parser = argparse.ArgumentParser(
        description="Corre el pipeline de KDF para varios modelos y todos sus escenarios, "
                    "continuando ante fallos puntuales y siendo reanudable."
    )
    parser.add_argument("--models", type=str, default=None,
                         help="Lista de modelos separados por coma (ej. 'llama3.1:8b,phi4:14b'). "
                              "Default: los 5 modelos del paper original.")
    parser.add_argument("--strategy", type=str, default="zero_shot",
                        choices=["zero_shot", "few_shot", "chain_of_thought", "chaining"])
    args = parser.parse_args()

    cfg = load_config("config.yaml")
    sem_categories = load_semantic_categories(CATEGORIES_FILE)
    sim_cfg = cfg.get("simulation", {})
    models = ([m.strip() for m in args.models.split(",") if m.strip()]
          if args.models else DEFAULT_MODELS)
    num_scenarios = sim_cfg.get("num_scenarios", 25)
    base_seed = sim_cfg.get("seed", 42)
    tasks_k = sim_cfg.get("tasks_k", 10)
    check_ollama_models_available(models)
    validate_config_bounds_sanity(cfg.get("task_generation", {}))
    total_run = total_skipped = total_ok = total_failed = 0

    for model in models:
        dataset_name = model_to_dataset_name(model)
        print(f"\n{'#' * 70}\n MODEL: {model}  ->  data/{dataset_name}/\n{'#' * 70}")

        for idx in range(1, num_scenarios + 1):
            scenario_dir = build_scenario_dir(args.strategy, dataset_name, idx)
            if scenario_already_complete(scenario_dir, tasks_k):
                total_skipped += 1
                print(f"  [SKIP] {model} scenario {idx}: already complete, skipping.")
                continue

            current_seed = base_seed + idx if base_seed is not None else None
            total_run += 1
            print(f"\n  [RUN] {model} escenario {idx}/{num_scenarios} (seed={current_seed})...")
            try:
                run_single_scenario(model, dataset_name, idx, cfg, sem_categories,
                                    current_seed, strategy=args.strategy)
                total_ok += 1
                print(f"  [OK]  {model} escenario {idx} completado.")
            except Exception as e:
                total_failed += 1
                print(f"  [FAIL] {model} escenario {idx}: {e}")
                continue
    ## TODO: Translate
    print("\n" + "=" * 70)
    print(" RESUMEN")
    print("=" * 70)
    print(f"  Ejecutados en esta corrida : {total_run}")
    print(f"  Omitidos (ya completos)    : {total_skipped}")
    print(f"  Exitosos                   : {total_ok}")
    print(f"  Fallidos                   : {total_failed}")

if __name__ == "__main__":
    main()