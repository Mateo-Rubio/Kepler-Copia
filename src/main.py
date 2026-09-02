import sys
import pathlib
import json
from datetime import datetime, timezone, timedelta

yaml_available = True
try:
    import yaml
except ImportError:
    yaml_available = False

from src.modules.data_collector.main import data_collector_main
from src.modules.physics_engine.main import physics_engine_main
from src.modules.prompt_factory.main import prompt_factory_main


def load_config(config_path: str) -> dict:
    """
    Load the project's YAML configuration file.
    """
    if not yaml_available:
        raise ImportError("The 'pyyaml' package is required to parse YAML configurations. Run 'pip install pyyaml'.")
    p = pathlib.Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")
    with p.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not config:
        raise ValueError("The configuration file is empty.")
    return config


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


def main():
    print("===================================")
    print("Launching Kepler: DatasetFactory...")
    print("===================================")
    
    CONFIG_FILE = "config.yaml"
    CATEGORIES_FILE = "semantic_categories.json"
    
    try:
        cfg = load_config(CONFIG_FILE)
        sem_categories = load_semantic_categories(CATEGORIES_FILE)
        
        sim_cfg = cfg.get("simulation", {})
        pay_cfg = cfg.get("payload", {})
        task_cfg = cfg.get("task_generation", {})
        path_cfg = cfg.get("paths", {})
        
        validate_config_bounds_sanity(task_cfg)
        
        dataset_name = sim_cfg.get("dataset_name", "dataset_output")
        num_scenarios = sim_cfg.get("num_scenarios", 1)
        base_seed = sim_cfg.get("seed", 42)
        semantic_enabled = sim_cfg.get("semantic_enabled", True)
        
        bands_config = pay_cfg.get("bands_config", {})
        sensor_constraints = pay_cfg.get("sensor_constraints", {})
        
        max_release_delay = task_cfg["max_release_delay"]
        max_lifetime = task_cfg["max_lifetime"]
        total_required_duration_s = max_release_delay + max_lifetime
        
        print(f"[DATASET] Parent Dataset Directory: 'data/{dataset_name}'")
        print(f"[DATASET] Total iterations to generate: {num_scenarios}")
        print(f"[CONFIG] Semantic Prompt Phase Enabled: {semantic_enabled}\n")

        for idx in range(1, num_scenarios + 1):
            scenario_folder_name = f"scenario_{idx}"
            scenario_dir = pathlib.Path("data") / dataset_name / scenario_folder_name
            scenario_dir.mkdir(parents=True, exist_ok=True)
            
            current_seed = base_seed + idx if base_seed is not None else None
            
            print("\n" + "#" * 60)
            print(f" GENERATING ITERATION {idx}/{num_scenarios}: '{scenario_folder_name}'")
            print(f" Target Folder: {scenario_dir.resolve()}")
            print(f" Active Seed: {current_seed}")
            print("#" * 60)
            
            print(f"\nExecuting Phase 1: Data Collection & Asset Enrichment...")
            
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
                "band_weights_map": bands_config,  
                "storage_capacity_pool_mb": pay_cfg.get("storage_capacity_pool_mb"),
                "sensor_generation_rates": pay_cfg.get("sensor_generation_rates"),
                "min_sensors_per_sat": pay_cfg.get("min_sensors_per_sat", 1),
                "max_sensors_per_sat": pay_cfg.get("max_sensors_per_sat", 2),
                "priority_weights": task_cfg.get("priority_weights"),  
                "seed": current_seed,
                "output_path": str(scenario_report_path)
            }
            
            if sim_cfg.get("sat_group_name"):
                collector_kwargs["sat_group_name"] = sim_cfg.get("sat_group_name")
            else:
                collector_kwargs["sat_file_path"] = path_cfg.get("sat_file_path")
                
            context = data_collector_main(**collector_kwargs)
            
            t0 = context.tle_epoch_utc if getattr(context, "tle_epoch_utc", None) else datetime.now(timezone.utc)
            tf = t0 + timedelta(seconds=total_required_duration_s)
            generation_now_utc = datetime.now(timezone.utc)
            
            print(f"Phase 1 SUCCESSFUL. Generated {len(context.targets)} Targets for current iteration.")
            print(f"[TIME] Dynamic SGP4 Alignment: SUCCESS (t0 anchored to TLE Epoch)")
            print(f"[TIME] Simulation Start (t0): {t0.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"[TIME] Calculated Simulation End (tf): {tf.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"[TIME] Active Planning Horizon Window: {total_required_duration_s / 3600:.2f} hours")
            print(f"[TIME] Request Generation Reference (wall-clock): " + f"{generation_now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
            
            if semantic_enabled:
                print("\nExecuting Semantic Prompt Generation Phase (Ollama Inferences & Semantic Embedding Validation)...")
                
                prompt_cfg = task_cfg.get("prompt_generation", {})
                ollama_model = sim_cfg.get("ollama_model", "llama3.1:8b")
                ollama_temp = sim_cfg.get("ollama_temperature", 0.3)
                
                prompt_factory_main(
                    targets=context.targets,
                    prompt_config=prompt_cfg,
                    output_dir=str(scenario_dir),  
                    model_name=ollama_model,
                    temperature=ollama_temp,
                    sensor_categories=sem_categories.get("sensor_categories"),
                    priority_categories=sem_categories.get("priority_categories"),
                    days_categories=sem_categories.get("days_categories"),
                    hours_categories=sem_categories.get("hours_categories"),
                    simulation_t0=generation_now_utc
                )
            else:
                print("\n[SKIP] Semantic Prompt Generation Phase disabled via configuration.")
            
            print("\nExecuting Phase 2: Physics Matrix Propagation & Target Intersection...")
            
            physics_report_path = scenario_dir / "physics_passes_report.json"
            
            physics_engine_main(
                context=context,
                bands_config=bands_config,
                sensor_constraints=sensor_constraints,
                simulation_start_utc=t0,
                simulation_end_utc=tf,
                output_path=str(physics_report_path), 
                step_seconds=20,
                min_duration=collector_kwargs["min_duration"],
                max_duration=collector_kwargs["max_duration"]
            )
            
            print(f"Iteration '{scenario_folder_name}' processed and isolated.")
            
        print("\n==================================================")
        print(f"Global Bulk Generation SUCCESSFUL. Total Scenarios Built: {num_scenarios}")
        print("==================================================")
            
    except Exception as e:
        print(f"\nPipeline CRASHED during setup or execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()