#!/usr/bin/env python3
"""
compare_prompt_configs.py

Verifica que el 'system_instruction_template' (el prompt base usado por el
Request Generator) sea idéntico entre los archivos config.yaml de todos los
modelos evaluados en KDF. Esto es un control necesario antes de atribuir
diferencias de accuracy en la categoría 'Hour' al modelo LLM en sí: si la
plantilla de prompting variara sin documentarlo entre modelos, cualquier
diferencia de desempeño quedaría confundida (no sería atribuible solo al
modelo).

Estructura de datos esperada:
    <data-root>/constellation_dataset_<modelo>/config.yaml

Requiere PyYAML:
    pip install pyyaml --break-system-packages

Ejemplos de uso:
    # Compara solo el prompt base (system_instruction_template)
    python3 compare_prompt_configs.py --data-root data

    # Compara además categorías de sensores, pesos de prioridad y temperatura
    python3 compare_prompt_configs.py --data-root data --full
"""

import argparse
import difflib
import hashlib
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    print("[!] Este script requiere PyYAML. Instálalo con:\n"
          "    pip install pyyaml --break-system-packages", file=sys.stderr)
    sys.exit(1)


def discover_configs(data_root: Path):
    """Encuentra los config.yaml dentro de cada carpeta constellation_dataset_<modelo>."""
    configs = {}
    for model_dir in sorted(data_root.glob("constellation_dataset_*")):
        if not model_dir.is_dir():
            continue
        cfg_path = model_dir / "config.yaml"
        model_name = model_dir.name.replace("constellation_dataset_", "", 1)
        if cfg_path.exists():
            configs[model_name] = cfg_path
        else:
            print(f"  [!] No se encontró config.yaml en {model_dir}", file=sys.stderr)
    return configs


def extract_field(config: dict, dotted_path: str):
    """Extrae un campo anidado de un dict usando notación 'a.b.c'."""
    value = config
    for key in dotted_path.split("."):
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def compare_field(configs: dict, dotted_path: str, label: str):
    print(f"\n{'=' * 78}\nComparando campo: {label}  ('{dotted_path}')\n{'=' * 78}")

    values = {}
    for model, path in configs.items():
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        value = extract_field(cfg, dotted_path)
        if value is None:
            print(f"  [!] Modelo '{model}': el campo '{dotted_path}' no existe en {path}")
            continue
        values[model] = value if isinstance(value, str) else str(value)

    if not values:
        print("  No se encontraron valores para comparar.")
        return

    hashes = {model: sha256(v) for model, v in values.items()}
    groups = {}
    for model, h in hashes.items():
        groups.setdefault(h, []).append(model)

    if len(groups) == 1:
        print(f"  [OK] IDENTICO en los {len(values)} modelo(s): {', '.join(sorted(values.keys()))}")
        return

    print(f"  [!!] DIFERENCIAS encontradas: {len(groups)} versiones distintas del campo.\n")
    for i, (h, models_in_group) in enumerate(groups.items(), start=1):
        print(f"  Grupo {i} ({len(models_in_group)} modelo(s)): {', '.join(sorted(models_in_group))}")

    ref_model = sorted(values.keys())[0]
    ref_text = values[ref_model]
    print(f"\n  --- Diferencias respecto al modelo de referencia '{ref_model}' ---")
    for model in sorted(values.keys()):
        if model == ref_model or hashes[model] == hashes[ref_model]:
            continue
        print(f"\n  >>> Diff: '{ref_model}' vs '{model}'")
        diff = difflib.unified_diff(
            ref_text.splitlines(), values[model].splitlines(),
            fromfile=ref_model, tofile=model, lineterm="",
        )
        for line in diff:
            print(f"    {line}")


def main():
    parser = argparse.ArgumentParser(
        description="Compara el prompt base (system_instruction_template) entre los "
                    "config.yaml de distintos modelos de KDF."
    )
    parser.add_argument("--data-root", type=Path, default=Path("data"),
                         help="Carpeta que contiene constellation_dataset_<modelo>/ (default: ./data)")
    parser.add_argument("--full", action="store_true",
                         help="También compara sensor_categories, priority_weights y "
                              "ollama_temperature, además del system_instruction_template")
    args = parser.parse_args()

    if not args.data_root.exists():
        print(f"[!] La carpeta de datos '{args.data_root}' no existe.", file=sys.stderr)
        sys.exit(1)

    configs = discover_configs(args.data_root)
    if not configs:
        print(f"[!] No se encontraron config.yaml en '{args.data_root}'.", file=sys.stderr)
        sys.exit(1)

    print(f"Modelos encontrados ({len(configs)}): {', '.join(sorted(configs.keys()))}")

    compare_field(
        configs,
        "task_generation.prompt_generation.system_instruction_template",
        "Plantilla de instrucción del sistema (prompt base)",
    )

    if args.full:
        compare_field(configs, "task_generation.prompt_generation.sensor_categories",
                      "Categorías de sensores")
        compare_field(configs, "task_generation.priority_weights", "Pesos de prioridad")
        compare_field(configs, "simulation.ollama_temperature", "Temperatura del modelo")


if __name__ == "__main__":
    main()