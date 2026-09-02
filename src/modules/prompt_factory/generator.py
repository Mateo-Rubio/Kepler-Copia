import os
import json
import re
import requests
import time
import urllib3 
from datetime import datetime, timezone
from typing import List, Dict, Optional
from src.core.datatypes import TargetTask

__all__ = ["generate_ollama_semantic_prompt", "build_single_task_string"]


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


def build_single_task_string(task: TargetTask, now_utc: datetime) -> str:
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
    
    return json.dumps(task_json_data, indent=2, ensure_ascii=False)


def generate_ollama_semantic_prompt(
    targets: List[TargetTask],
    system_instruction_template: str,
    now_utc: Optional[datetime] = None,
    model_name: str = "llama3.1:8b",
    temperature: float = 0.4,
    num_predict: int = 700,
    repeat_penalty: float = 1.05,
) -> Dict[str, str]:
    raw_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
    clean_url = raw_url.replace("[", "").replace("]", "").split("(")[0].strip()
    target_endpoint = f"{clean_url}/api/generate"
    
    generated_prompts_map = {}

    if now_utc is None:
        now_utc = datetime.now(timezone.utc)

    for task in targets:
        task_string = build_single_task_string(task, now_utc)
        full_prompt = system_instruction_template.format(tasks_dataset=task_string)
        
        payload = {
            "model": model_name,
            "prompt": full_prompt,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
                "repeat_penalty": repeat_penalty
            }
        }

        print(f"\n[OLLAMA] Generating prompt for {task.task_id}...")
        response = requests.post(
            target_endpoint,
            json=payload,
            stream=True,
            timeout=180
        )
        response.raise_for_status()

        output_chunks = []
        for line in response.iter_lines():
            if not line:
                continue
            line_object = json.loads(line)
            if "response" not in line_object:
                continue
            
            chunk = line_object["response"]
            print(chunk, end="", flush=True)
            output_chunks.append(chunk)

            if line_object.get("done"):
                break

        print()
        raw_text = "".join(output_chunks)
        
        pattern = r"\x60{3}(?:prompt|text)?\s*(.*?)\s*\x60{3}"
        match = re.search(pattern, raw_text, re.DOTALL | re.IGNORECASE)
        parsed_text = match.group(1).strip() if match else raw_text.strip()
            
        generated_prompts_map[task.task_id] = parsed_text
        
    return generated_prompts_map