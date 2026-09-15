import os
import json
import re
import requests
import time
import urllib3 
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any, Tuple
from src.core.datatypes import TargetTask

__all__ = ["generate_ollama_semantic_prompt", "build_single_task_string", "build_task_fields"]


def _get_geocoded_info(lat: Optional[float], lon: Optional[float]) -> Dict[str, str]:
    if lat is None or lon is None:
        return {"country": "N/A", "city": "N/A", "landmark": "N/A"}
    
    time.sleep(1.2)  
    
    try:
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        
        url = "https://nominatim.openstreetmap.org/reverse"
        params = {
            "lat": float(lat),
            "lon": float(lon),
            "format": "json",
            "addressdetails": 1
        }
        headers = {
            "User-Agent": f"satellite_constellation_research_{int(time.time())}"
        }
        
        response = requests.get(url, params=params, headers=headers, verify=False, timeout=15)
        response.raise_for_status()
        raw_data = response.json()
        
        if raw_data and "address" in raw_data:
            address = raw_data["address"]
            country = address.get("country", "N/A")
            
            city = address.get(
                "city", 
                address.get(
                    "town", 
                    address.get(
                        "village", 
                        address.get("county", "N/A")
                    )
                )
            )
            
            landmark = address.get(
                "tourism", 
                address.get(
                    "historic", 
                    address.get(
                        "amenity", 
                        address.get("building", "N/A")
                    )
                )
            )
            
            return {
                "country": country,
                "city": city,
                "landmark": landmark
            }
        else:
            print(f"[DEBUG GEOLOC] Invalid structure returned by Nominatim: {lat}, {lon}")
            
    except Exception as e:
        print(f"[DEBUG GEOLOC ERROR] Caught exception: {e}")
        
    return {"country": "N/A", "city": "N/A", "landmark": "N/A"}


def build_task_fields(task: TargetTask, now_utc: datetime) -> Dict[str, Any]:
    primary_sensor = task.required_sensors[0] if task.required_sensors else "VISUAL"
    
    raw_deadline = getattr(task, "deadline", getattr(task, "deadline_s", 0))
    deadline_epoch = now_utc.timestamp() + raw_deadline
    task_deadline_utc = datetime.fromtimestamp(deadline_epoch, tz=timezone.utc)
    
    release_time = getattr(task, 'release_time', 0)
    remaining_hours = (raw_deadline - release_time) / 3600.0

    if remaining_hours < 12:
        day_tag = "today"
    elif remaining_hours < 36:
        day_tag = "tomorrow"
    elif remaining_hours < 60:
        day_tag = "the day after tomorrow"
    elif remaining_hours < 84:
        day_tag = "in three days"
    else:
        day_tag = "in four days"

    utc_hour = task_deadline_utc.hour
    if 6 <= utc_hour < 11:
        hour_tag = "in the morning"
    elif 11 <= utc_hour < 14:
        hour_tag = "around mid-day"
    elif 14 <= utc_hour < 18:
        hour_tag = "during the afternoon"
    elif 18 <= utc_hour < 23:
        hour_tag = "in the evening"
    else:
        hour_tag = "overnight"

    lat, lon = None, None
    coords = getattr(task, "coordinates", None)
    if coords and isinstance(coords, (list, tuple)) and len(coords) > 0:
        valid_coords = [c for c in coords if isinstance(c, (list, tuple)) and len(c) >= 2]
        if valid_coords:
            lat = sum(float(c[0]) for c in valid_coords) / len(valid_coords)
            lon = sum(float(c[1]) for c in valid_coords) / len(valid_coords)
            
    if lat is None or lon is None:
        lat = getattr(task, "latitude", getattr(task, "lat", None))
        lon = getattr(task, "longitude", getattr(task, "lon", None))
    
    if lat is None or lon is None:
        for attr in ["location", "position", "geometry"]:
            loc_obj = getattr(task, attr, None)
            if loc_obj:
                if isinstance(loc_obj, dict):
                    lat = loc_obj.get("latitude") or loc_obj.get("lat")
                    lon = loc_obj.get("longitude") or loc_obj.get("lon")
                else:
                    lat = getattr(loc_obj, "latitude", getattr(loc_obj, "lat", None))
                    lon = getattr(loc_obj, "longitude", getattr(loc_obj, "lon", None))
                break
                
    if lat is None or lon is None:
        lat_env = getattr(task, "lat_envelope", None)
        lon_env = getattr(task, "lon_envelope", None)
        if lat_env and lon_env and isinstance(lat_env, (list, tuple)) and isinstance(lon_env, (list, tuple)):
            if len(lat_env) >= 2 and len(lon_env) >= 2:
                lat = sum(lat_env) / len(lat_env)
                lon = sum(lon_env) / len(lon_env)
            
    geo_info = _get_geocoded_info(lat, lon)
    priority = getattr(task, "priority", getattr(task, "priority_level", 1))
    
    task_json_data = {
        "primary_sensor": primary_sensor,
        "location_details": {
            "country": geo_info['country'],
            "city": geo_info['city'] if geo_info['city'] != "N/A" else task.region_tag.split('_')[-1].capitalize()
        },
        "priority_level": priority,
        "target_day": day_tag,
        "target_diurnal_period": hour_tag
    }
    
    return task_json_data


def build_single_task_string(task: TargetTask, now_utc: datetime) -> str:
    return json.dumps(build_task_fields(task, now_utc), indent=2, ensure_ascii=False)



def _ollama_endpoint() -> str:
    raw_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    clean_url = raw_url.replace("[", "").replace("]", "").split("(")[0].strip()
    return f"{clean_url}/api/generate"


def _call_ollama(
    prompt: str,
    model_name: str,
    temperature: float,
    num_predict: int = 1200,
    repeat_penalty: float = 1.05,
    echo: bool = True,
) -> str:
    """Una sola llamada a Ollama. Devuelve el texto crudo, sin parsear."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            "repeat_penalty": repeat_penalty,
        },
    }

    response = requests.post(_ollama_endpoint(), json=payload, stream=True, timeout=180)
    response.raise_for_status()

    output_chunks = []
    for line in response.iter_lines():
        if not line:
            continue
        line_object = json.loads(line)
        if "response" not in line_object:
            continue

        chunk = line_object["response"]
        if echo:
            print(chunk, end="", flush=True)
        output_chunks.append(chunk)

        if line_object.get("done"):
            break

    if echo:
        print()
    return "".join(output_chunks)


def _extract_prompt_block(raw_text: str) -> str:
    """Extrae el primer bloque ```prompt ... ```; si no hay, devuelve el texto."""
    pattern = r"\x60{3}(?:prompt|text)?\s*(.*?)\s*\x60{3}"
    match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else raw_text.strip()


def generate_ollama_semantic_prompt(
    targets: List[TargetTask],
    strategy_cfg: Dict[str, Any],
    shared_header: str = "",
    now_utc: Optional[datetime] = None,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.4,
    num_predict: int = 1200,
    repeat_penalty: float = 1.05,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Devuelve (prompts_map, stage1_map).

    prompts_map : task_id -> solicitud final en lenguaje natural
    stage1_map  : task_id -> frase temporal resuelta en la etapa 1
                  (vacio para las estrategias de una sola llamada)
    """
    generated_prompts_map: Dict[str, str] = {}
    stage1_map: Dict[str, str] = {}

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    two_stage = int(strategy_cfg.get("n_calls", 1)) == 2

    for task in targets:
        task_string = build_single_task_string(task, now_utc)

        print(f"\n[OLLAMA] Generating prompt for {task.task_id}...")

        if two_stage:
            fields = build_task_fields(task, now_utc)

            stage1_prompt = strategy_cfg["template_stage1"].format(
                target_day=fields["target_day"],
                target_diurnal_period=fields["target_diurnal_period"],
            )
            raw_phrase = _call_ollama(
                stage1_prompt, model_name, temperature,
                num_predict=40, repeat_penalty=repeat_penalty, echo=False,
            )

            phrase = raw_phrase.strip().strip('"').strip()
            phrase = phrase.splitlines()[0].strip() if phrase else ""
            stage1_map[task.task_id] = phrase
            print(f"[STAGE 1] {task.task_id} -> {phrase!r}")

            full_prompt = strategy_cfg["template_stage2"].format(
                shared_header=shared_header,
                verified_temporal_phrase=phrase,
                tasks_dataset=task_string,
            )
        else:
            stage1_map[task.task_id] = ""
            full_prompt = strategy_cfg["template"].format(
                shared_header=shared_header,
                tasks_dataset=task_string,
            )

        raw_text = _call_ollama(
            full_prompt, model_name, temperature,
            num_predict=num_predict, repeat_penalty=repeat_penalty,
        )
        generated_prompts_map[task.task_id] = _extract_prompt_block(raw_text)

    return generated_prompts_map, stage1_map