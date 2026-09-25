import csv
import json
import pathlib

ROOT = pathlib.Path("data/2026-09-16")
OUT = pathlib.Path("analysis/aciertos.csv")

FUGA = ("stage 1 - reasoning", "stage 2 - final", "which exact string",
        "copy it verbatim")

COLUMNAS = ["estrategia", "modelo", "escenario", "task_id",
            "expected_day", "expected_hour",
            "day_match", "hour_match", "sensor_match", "priority_match",
            "fuga_razonamiento"]

filas = []
for f in sorted(ROOT.rglob("ollama_prompts_combined.json")):
    estrategia = f.relative_to(ROOT).parts[0]
    modelo = f.parent.parent.name.replace("constellation_dataset_", "")
    escenario = int(f.parent.name.replace("scenario_", ""))

    for t in json.loads(f.read_text(encoding="utf-8"))["tasks"]:
        gt, val = t["ground_truth"], t["validation"]
        texto = t["generated_output"].lower()
        filas.append({
            "estrategia": estrategia,
            "modelo": modelo,
            "escenario": escenario,
            "task_id": t["task_id"],
            "expected_day": gt["expected_day"],
            "expected_hour": gt["expected_hour"],
            "day_match": val["day_match"],
            "hour_match": val["hour_match"],
            "sensor_match": val["sensor_match"],
            "priority_match": val["priority_match"],
            "fuga_razonamiento": any(k in texto for k in FUGA),
        })

OUT.parent.mkdir(parents=True, exist_ok=True)
with OUT.open("w", encoding="utf-8", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=COLUMNAS)
    w.writeheader()
    w.writerows(filas)

print(f"{len(filas)} filas -> {OUT}")