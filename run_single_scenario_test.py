#!/usr/bin/env python3
"""
run_single_scenario_test.py

Prueba puntual del Request Generator (RG) de KDF: regenera de forma
DETERMINISTA las mismas tareas (targets) de un escenario ya existente
(usando la misma semilla que 'main.py' calcula internamente: seed = base_seed
+ numero_de_escenario) y ejecuta SOLO la fase del Request Generator
(llamadas a Ollama + validación semántica) con el modelo indicado, guardando
el resultado en una carpeta de escenario nueva (ej. scenario_26).

No vuelve a correr el Physics Engine: prompt_factory_main no consume nada de
su salida, así que esta prueba es autosuficiente y mucho más rápida que
correr el pipeline completo.

IMPORTANTE: debe ejecutarse desde la raíz del repositorio de KDF (mismo
nivel que 'src/' y 'config.yaml'), ya que importa los módulos internos del
proyecto.

Advertencia sobre determinismo: si 'data_collector_main' obtiene datos
orbitales en vivo (ej. TLEs desde Celestrak por red) en cada llamada, los
targets regenerados podrían NO ser idénticos byte a byte a los del escenario
original, aunque la semilla sea la misma. Si tu Data Collector usa TLEs ya
cacheados/guardados localmente, el determinismo está garantizado.

Ejemplo de uso:
    python3 run_single_scenario_test.py \\
        --config config.yaml \\
        --reuse-scenario 1 \\
        --output-scenario 26 \\
        --model llama3.1:8b
"""

import argparse
import pathlib
import json
from datetime import datetime, timezone

try:
    import yaml
except ImportError:
    raise ImportError("Este script requiere 'pyyaml'. Instálalo con: pip install pyyaml")

from src.modules.data_collector.main import data_collector_main
from src.modules.prompt_factory.main import prompt_factory_main


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
        print(f"[WARNING] No se encontró '{categories_path}'. Se usarán las categorías por defecto.")
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Reejecuta el Request Generator sobre las mismas tareas de un "
                    "escenario existente (misma semilla), con un modelo dado, guardando "
                    "el resultado en una carpeta de escenario nueva."
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                         help="Ruta al config.yaml del dataset (default: config.yaml)")
    parser.add_argument("--categories", type=str, default="semantic_categories.json",
                         help="Ruta al JSON de categorías semánticas "
                              "(default: semantic_categories.json)")
    parser.add_argument("--reuse-scenario", type=int, required=True,
                         help="Número del escenario existente cuya semilla (y por lo tanto "
                              "tareas) se quiere reutilizar, ej. 1")
    parser.add_argument("--output-scenario", type=int, required=True,
                         help="Número del nuevo escenario donde se guardará el resultado, "
                              "ej. 26")
    parser.add_argument("--model", type=str, default=None,
                         help="Modelo de Ollama a usar (default: el definido en config.yaml, "
                              "ej. llama3.1:8b)")
    parser.add_argument("--output-root", type=str, default=None,
                         help="Carpeta raíz del dataset de salida "
                              "(default: data/<dataset_name> según config.yaml)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    sem_categories = load_semantic_categories(args.categories)

    sim_cfg = cfg.get("simulation", {})
    pay_cfg = cfg.get("payload", {})
    task_cfg = cfg.get("task_generation", {})
    path_cfg = cfg.get("paths", {})

    dataset_name = sim_cfg.get("dataset_name", "dataset_output")
    base_seed = sim_cfg.get("seed", 42)
    reused_seed = base_seed + args.reuse_scenario  # misma fórmula usada en main.py

    model_name = args.model or sim_cfg.get("ollama_model", "llama3.1:8b")
    ollama_temp = sim_cfg.get("ollama_temperature", 0.3)

    output_root = (pathlib.Path(args.output_root) if args.output_root
                   else pathlib.Path("data") / dataset_name)
    scenario_dir = output_root / f"scenario_{args.output_scenario}"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    max_release_delay = task_cfg["max_release_delay"]
    max_lifetime = task_cfg["max_lifetime"]

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
        "seed": reused_seed,
        # Se guarda en la carpeta del ESCENARIO NUEVO; no sobreescribe el original.
        "output_path": str(scenario_dir / "scenario_report.json"),
    }
    if sim_cfg.get("sat_group_name"):
        collector_kwargs["sat_group_name"] = sim_cfg.get("sat_group_name")
    else:
        collector_kwargs["sat_file_path"] = path_cfg.get("sat_file_path")

    print("=" * 60)
    print(f" Reutilizando semilla del escenario {args.reuse_scenario} (seed={reused_seed})")
    print(f" Guardando resultado como escenario {args.output_scenario}")
    print(f" Modelo: {model_name}   Temperatura: {ollama_temp}")
    print(f" Carpeta destino: {scenario_dir.resolve()}")
    print("=" * 60)

    print("\n[1/2] Regenerando tareas (Data Collector, misma semilla)...")
    context = data_collector_main(**collector_kwargs)
    print(f"      -> {len(context.targets)} tareas regeneradas de forma determinista.")

    t0 = (context.tle_epoch_utc if getattr(context, "tle_epoch_utc", None)
          else datetime.now(timezone.utc))

    prompt_cfg = task_cfg.get("prompt_generation", {})

    print(f"\n[2/2] Ejecutando Request Generator con modelo '{model_name}'...")
    prompt_factory_main(
        targets=context.targets,
        prompt_config=prompt_cfg,
        output_dir=str(scenario_dir),
        model_name=model_name,
        temperature=ollama_temp,
        sensor_categories=sem_categories.get("sensor_categories"),
        priority_categories=sem_categories.get("priority_categories"),
        days_categories=sem_categories.get("days_categories"),
        hours_categories=sem_categories.get("hours_categories"),
        simulation_t0=t0,
    )

    print(f"\n[SUCCESS] Resultado guardado en: {scenario_dir.resolve()}")


if __name__ == "__main__":
    main()