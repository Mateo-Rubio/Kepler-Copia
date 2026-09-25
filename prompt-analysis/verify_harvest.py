#!/usr/bin/env python3
"""
verify_harvest.py

Verificaciones previas a cualquier analisis de la Fase 2.

PASO 1 - Integridad de la cosecha:
  - Todos los escenarios presentes y completos (15 tareas).
  - Un unico timestamp_utc en TODAS las celdas (reloj congelado).
  - El campo 'strategy' coincide con la carpeta.
  - 'prompt_sent' contiene la marca esperada de cada estrategia.

PASO 2 - Pareo:
  - Para cada (modelo, escenario, task_id), el ground_truth debe ser
    IDENTICO en las tres estrategias. Es lo que habilita McNemar y
    Cochran's Q. Sin esto solo se pueden comparar promedios.

Uso:
    python verify_harvest.py --root data/2026-09-16
    python verify_harvest.py --root data/2026-09-16 --tasks 15
"""

import argparse
import json
import pathlib
import sys
from collections import defaultdict

# Marca que debe aparecer en prompt_sent para confirmar que la plantilla
# correcta llego al modelo. zero_shot no tiene marca propia: se verifica
# por ausencia de las otras dos.
MARCAS = {
    "few_shot": "--- EXAMPLE 1 ---",
    "chaining": "resolved by an upstream module",
    "chain_of_thought": "STAGE 1 - REASONING",
}

CAMPOS_GT = ["expected_sensor", "expected_priority", "expected_day", "expected_hour"]


ESTRATEGIAS_VALIDAS = {"zero_shot", "few_shot", "chain_of_thought", "chaining"}


def cargar_celdas(root: pathlib.Path):
    """
    Devuelve ({(estrategia, modelo, escenario): datos_json}, errores)
    recorriendo root/<estrategia>/.../<modelo>/scenario_<n>/...

    Dos protecciones que antes faltaban:
      - El primer nivel bajo root DEBE ser un nombre de estrategia conocido.
        Si se apunta la raiz un nivel de mas (ej. --root data en vez de
        data/<fecha>), esto lo detecta en vez de tomar la fecha por
        estrategia.
      - Las claves duplicadas se reportan. Antes se sobrescribian en
        silencio, perdiendo celdas sin dejar rastro (pasa si hay carpetas
        rep_*, o con la raiz mal apuntada).
    """
    celdas = {}
    origen = {}
    errores = []

    for f in sorted(root.rglob("ollama_prompts_combined.json")):
        partes = f.relative_to(root).parts
        estrategia = partes[0]

        if estrategia not in ESTRATEGIAS_VALIDAS:
            errores.append(
                f"'{estrategia}' no es una estrategia conocida (en {f}). "
                f"Comprueba --root: debe apuntar a la carpeta que contiene "
                f"directamente {sorted(ESTRATEGIAS_VALIDAS)}."
            )
            continue

        try:
            escenario = int(f.parent.name.replace("scenario_", ""))
        except ValueError:
            errores.append(f"Nombre de escenario no numerico: {f.parent}")
            continue

        modelo = f.parent.parent.name.replace("constellation_dataset_", "")
        clave = (estrategia, modelo, escenario)

        if clave in celdas:
            errores.append(
                f"Clave duplicada {clave}:\n"
                f"      ya cargada desde {origen[clave]}\n"
                f"      y ahora desde   {f}\n"
                f"      (una de las dos se perderia en silencio)"
            )
            continue

        try:
            celdas[clave] = json.loads(f.read_text(encoding="utf-8"))
            origen[clave] = f
        except (json.JSONDecodeError, OSError) as e:
            errores.append(f"Ilegible {f}: {e}")

    return celdas, errores


def paso1(celdas, tareas_esperadas):
    print("=" * 72)
    print(" PASO 1 - INTEGRIDAD DE LA COSECHA")
    print("=" * 72)

    estrategias = sorted({k[0] for k in celdas})
    modelos = sorted({k[1] for k in celdas})
    escenarios = sorted({k[2] for k in celdas})

    print(f"\nEstrategias : {estrategias}")
    print(f"Modelos     : {modelos}")
    print(f"Escenarios  : {min(escenarios)}-{max(escenarios)} ({len(escenarios)} distintos)")
    print(f"Celdas      : {len(celdas)} "
          f"(esperadas {len(estrategias) * len(modelos) * len(escenarios)})")

    problemas = []

    # --- Cobertura: que no falte ninguna combinacion -----------------
    faltantes = [(e, m, s)
                 for e in estrategias for m in modelos for s in escenarios
                 if (e, m, s) not in celdas]
    if faltantes:
        problemas.append(f"{len(faltantes)} celda(s) ausente(s)")
        print(f"\n[FALTAN] {len(faltantes)} celdas. Primeras 10:")
        for c in faltantes[:10]:
            print(f"    {c}")
    else:
        print("\n[OK] No falta ninguna combinacion estrategia x modelo x escenario.")

    # --- Completitud: 15 tareas por celda ---------------------------
    incompletas = []
    for k, j in celdas.items():
        n = j.get("scenario_metrics", {}).get("total_tasks_evaluated", 0)
        if n < tareas_esperadas or len(j.get("tasks", [])) < tareas_esperadas:
            incompletas.append((k, n, len(j.get("tasks", []))))
    if incompletas:
        problemas.append(f"{len(incompletas)} celda(s) incompleta(s)")
        print(f"\n[INCOMPLETAS] {len(incompletas)}:")
        for k, n, t in incompletas[:10]:
            print(f"    {k} -> metrics={n}, tasks={t}")
    else:
        print(f"[OK] Las {len(celdas)} celdas tienen {tareas_esperadas} tareas.")

    # --- Reloj congelado --------------------------------------------
    relojes = defaultdict(list)
    for k, j in celdas.items():
        relojes[j.get("timestamp_utc")].append(k)
    print(f"\nRelojes distintos encontrados: {len(relojes)}")
    for ts, ks in sorted(relojes.items(), key=lambda x: -len(x[1])):
        print(f"    {ts}  ->  {len(ks)} celdas")
    if len(relojes) > 1:
        problemas.append("mas de un timestamp_utc (reloj NO congelado)")
        print("\n[CRITICO] Hay mas de un instante de referencia. Las celdas que no")
        print("          comparten reloj NO son comparables entre si.")
        minoritario = min(relojes.items(), key=lambda x: len(x[1]))
        print(f"          Celdas con '{minoritario[0]}': {minoritario[1][:10]}")
    else:
        print("[OK] Un unico instante de referencia en todas las celdas.")

    # --- Coherencia de estrategia -----------------------------------
    desajuste_campo, desajuste_marca = [], []
    for (e, m, s), j in celdas.items():
        if j.get("strategy") != e:
            desajuste_campo.append(((e, m, s), j.get("strategy")))

        # Se revisan TODAS las tareas de la celda, no solo la primera.
        marca = MARCAS.get(e)
        sin_marca, con_intrusa = 0, set()
        for t in j.get("tasks", []):
            prompt = t.get("prompt_sent", "")
            if marca and marca not in prompt:
                sin_marca += 1
            if e == "zero_shot":
                con_intrusa.update(n for n, mk in MARCAS.items() if mk in prompt)
        if sin_marca:
            desajuste_marca.append(((e, m, s), f"{sin_marca} tarea(s) sin '{marca}'"))
        if con_intrusa:
            desajuste_marca.append(((e, m, s), f"contiene marca de {sorted(con_intrusa)}"))

    if desajuste_campo:
        problemas.append(f"{len(desajuste_campo)} celda(s) con campo 'strategy' erroneo")
        print(f"\n[STRATEGY] {len(desajuste_campo)} celdas cuyo campo no coincide con la carpeta:")
        for k, v in desajuste_campo[:10]:
            print(f"    {k} -> campo dice '{v}'")
    else:
        print("[OK] El campo 'strategy' coincide con la carpeta en todas las celdas.")

    if desajuste_marca:
        problemas.append(f"{len(desajuste_marca)} celda(s) con plantilla equivocada")
        print(f"\n[CRITICO] {len(desajuste_marca)} celdas cuyo prompt_sent no corresponde:")
        for k, v in desajuste_marca[:10]:
            print(f"    {k} -> {v}")
    else:
        print("[OK] El prompt enviado corresponde a la plantilla de cada estrategia.")

    return problemas


def paso2(celdas):
    print("\n" + "=" * 72)
    print(" PASO 2 - PAREO (mismo ground_truth en todas las estrategias)")
    print("=" * 72)

    estrategias = sorted({k[0] for k in celdas})
    if len(estrategias) < 2:
        print("Solo hay una estrategia; el pareo no aplica.")
        return []

    # gt[(modelo, escenario, task_id)][estrategia] = tupla de ground truth
    gt = defaultdict(dict)
    for (e, m, s), j in celdas.items():
        for t in j.get("tasks", []):
            clave = (m, s, t["task_id"])
            gt[clave][e] = tuple(t["ground_truth"].get(c) for c in CAMPOS_GT)

    total = parejas_ok = 0
    discrepantes = []
    incompletas = 0
    huerfanas = []
    campos_rotos = defaultdict(int)

    for clave, por_estrategia in gt.items():
        if len(por_estrategia) < len(estrategias):
            incompletas += 1
            huerfanas.append(clave)
            continue
        total += 1
        valores = list(por_estrategia.values())
        if all(v == valores[0] for v in valores):
            parejas_ok += 1
        else:
            for i, campo in enumerate(CAMPOS_GT):
                if len({v[i] for v in valores}) > 1:
                    campos_rotos[campo] += 1
            if len(discrepantes) < 8:
                discrepantes.append((clave, por_estrategia))

    problemas = []

    print(f"\nTareas comparables      : {total}")
    if total:
        print(f"Ground truth identico   : {parejas_ok} ({parejas_ok / total * 100:.2f}%)")
    else:
        print("Ground truth identico   : n/a")

    # Una tarea que no aparece en las tres estrategias NO puede compararse.
    # Antes se excluia en silencio y el veredicto salia OK igualmente; en el
    # extremo (task_ids distintos entre estrategias) se validaban 0 tareas y
    # el script daba luz verde.
    if incompletas:
        problemas.append(
            f"{incompletas} tarea(s) no presentes en las {len(estrategias)} estrategias")
        print(f"\n[CRITICO] {incompletas} tareas no estan en las {len(estrategias)} "
              f"estrategias y NO pueden parearse.")
        for clave in sorted(huerfanas)[:8]:
            print(f"            {clave} -> solo en {sorted(gt[clave].keys())}")

    if total == 0:
        problemas.append("ninguna tarea comparable entre estrategias")
        print("\n[CRITICO] No hay una sola tarea comparable. El pareo no existe.")
        return problemas
    if total and parejas_ok < total:
        problemas.append(f"{total - parejas_ok} tarea(s) sin pareo")
        print(f"\n[CRITICO] {total - parejas_ok} tareas NO estan pareadas.")
        print("          Campos que difieren:")
        for campo, n in sorted(campos_rotos.items(), key=lambda x: -x[1]):
            print(f"            {campo}: {n}")
        print("\n          Ejemplos:")
        for clave, por_e in discrepantes:
            print(f"            {clave}")
            for e, v in sorted(por_e.items()):
                print(f"               {e:18s} {v}")
    elif total:
        print("\n[OK] El pareo se sostiene: cada tarea plantea exactamente el mismo")
        print("     problema en las tres estrategias. Se pueden usar pruebas pareadas")
        print("     (McNemar por pares, Cochran's Q para las tres a la vez).")

    return problemas


def main():
    ap = argparse.ArgumentParser(description="Verifica integridad y pareo de la cosecha de la Fase 2.")
    ap.add_argument("--root", type=str, required=True,
                    help="Carpeta raiz de la corrida, ej. data/2026-09-16")
    ap.add_argument("--tasks", type=int, default=15,
                    help="Tareas esperadas por escenario (default: 15)")
    args = ap.parse_args()

    root = pathlib.Path(args.root)
    if not root.exists():
        print(f"[ERROR] No existe: {root}", file=sys.stderr)
        sys.exit(1)

    celdas, errores_carga = cargar_celdas(root)

    if errores_carga:
        print("=" * 72)
        print(" ERRORES AL CARGAR LA COSECHA")
        print("=" * 72)
        for e in errores_carga:
            print(f"  - {e}")
        print("\n VEREDICTO: NO proceder al analisis todavia")
        sys.exit(1)

    if not celdas:
        print(f"[ERROR] No se encontro ningun ollama_prompts_combined.json bajo {root}",
              file=sys.stderr)
        sys.exit(1)

    problemas = paso1(celdas, args.tasks)
    problemas += paso2(celdas)

    print("\n" + "=" * 72)
    if problemas:
        print(" VEREDICTO: NO proceder al analisis todavia")
        print("=" * 72)
        for p in problemas:
            print(f"  - {p}")
        sys.exit(1)
    else:
        print(" VEREDICTO: cosecha integra y pareada. Adelante con el extractor.")
        print("=" * 72)


if __name__ == "__main__":
    main()