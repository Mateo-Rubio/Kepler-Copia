import json
import pathlib
import subprocess
import time
import yaml

INTERVALO_S = 60
tasks_k = yaml.safe_load(open("config.yaml", encoding="utf-8"))["simulation"]["tasks_k"]

def completo(d: pathlib.Path) -> bool:
    try:
        with (d / "ollama_prompts_combined.json").open("r", encoding="utf-8") as f:
            total = json.load(f).get("scenario_metrics", {}).get("total_tasks_evaluated", 0)
        with (d / "physics_passes_report.json").open("r", encoding="utf-8") as f:
            json.load(f)
        return total >= tasks_k
    except (json.JSONDecodeError, OSError):
        return False

hechos = set()
while True:
    nuevos = [d for d in pathlib.Path("data").glob("*/*/temp_*/rep_*/scenario_*")
              if d not in hechos and completo(d)]
    for i in range(0, len(nuevos), 100):
        subprocess.run(["git", "add", *map(str, nuevos[i:i + 100])])
    if subprocess.run(["git", "diff", "--cached", "--quiet"]).returncode:
        subprocess.run(["git", "commit", "-m", f"Datos: {len(nuevos)} escenarios completos"])
        subprocess.run(["git", "push", "origin", "HEAD"])
    hechos.update(nuevos)
    time.sleep(INTERVALO_S)