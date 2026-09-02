import json
import pathlib
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from src.core.datatypes import TargetTask
from .generator import generate_ollama_semantic_prompt, build_single_task_string
from .validation import SemanticValidator

__all__ = ["prompt_factory_main"]


def prompt_factory_main(
    targets: List[TargetTask],
    prompt_config: Optional[Dict[str, Any]],
    output_dir: str = "data",
    model_name: str = "llama3.1:8b",
    temperature: float = 0.3,
    sensor_categories: Optional[Dict[str, str]] = None,
    priority_categories: Optional[List[str]] = None,
    days_categories: Optional[List[str]] = None,
    hours_categories: Optional[List[str]] = None,
    simulation_t0: Optional[datetime] = None
) -> Dict[str, str]:
    if prompt_config is None:
        return {}

    system_instruction = prompt_config.get("system_instruction_template", "")
    if not system_instruction:
        raise ValueError("The YAML configuration is missing the 'system_instruction_template' key.")

    if not sensor_categories:
        sensor_categories = {
            "TIR": "thermal mapping heat signature surface temperature profile thermal anomaly detection",
            "VNIR": "multispectral vegetation analysis chlorophyll absorption level NDVI index",
            "SAR": "active radar scan microwave surface imaging cloud-penetrating capture",
            "NIR": "near-infrared reflection water body boundary mapping soil moisture assessment",
            "VISUAL": "high-resolution optical snapshot daylight photography RGB true-color imagery",
            "VIS": "high-resolution optical snapshot daylight photography RGB true-color imagery"
        }

    if not priority_categories:
        priority_categories = [
            "low priority background survey filler task flexible schedule",
            "medium priority standard operational tasking routine monitoring monitoring",
            "high priority absolute institutional priority binding contract requirement urgent execution mandatory"
        ]

    if not days_categories:
        days_categories = [
            "today",
            "tomorrow",
            "the day after tomorrow",
            "in three days",
            "in four days"
        ]

    if not hours_categories:
        hours_categories = [
            "morning",
            "mid-day",
            "afternoon",
            "evening",
            "overnight"
        ]

    priority_map = {1: priority_categories[0], 2: priority_categories[1], 3: priority_categories[2]}

    day_mapping_to_category = {
        "today": days_categories[0],
        "tomorrow": days_categories[1],
        "the day after tomorrow": days_categories[2],
        "in three days": days_categories[3],
        "in four days": days_categories[4]
    }

    hour_mapping_to_category = {
        "in the morning": hours_categories[0],
        "around mid-day": hours_categories[1],
        "during the afternoon": hours_categories[2],
        "in the evening": hours_categories[3],
        "overnight": hours_categories[4]
    }

    validator = SemanticValidator(
        sensor_categories=sensor_categories,
        priority_categories=priority_categories,
        days_categories=days_categories,
        hours_categories=hours_categories,
        embedding_model="mxbai-embed-large"
    )

    fixed_now_utc = simulation_t0 if simulation_t0 is not None else datetime.now(timezone.utc)

    prompts_map = generate_ollama_semantic_prompt(
        targets=targets,
        system_instruction_template=system_instruction,
        model_name=model_name,
        temperature=temperature,
        now_utc=fixed_now_utc
    )

    dir_path = pathlib.Path(output_dir)
    dir_path.mkdir(parents=True, exist_ok=True)

    task_lookup = {task.task_id: task for task in targets}
    
    processed_tasks_list = []
    successful_sensor_matches = 0
    successful_priority_matches = 0
    successful_day_matches = 0
    successful_hour_matches = 0

    for task_id, clean_text in prompts_map.items():
        txt_file_path = dir_path / f"ollama_prompt_{task_id}.txt"
        txt_file_path.write_text(clean_text, encoding="utf-8")

        task = task_lookup.get(task_id)
        if task:
            task_string = build_single_task_string(task, fixed_now_utc)
            full_prompt_sent = system_instruction.format(tasks_dataset=task_string)

            raw_sensor = task.required_sensors[0] if task.required_sensors else "VISUAL"
            expected_sensor = "VISUAL" if raw_sensor == "VIS" else raw_sensor
            expected_priority = priority_map.get(task.priority, priority_categories[0])

            raw_deadline = getattr(task, "deadline", getattr(task, "deadline_s", 0))
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

            deadline_epoch = fixed_now_utc.timestamp() + raw_deadline
            task_deadline_utc = datetime.fromtimestamp(deadline_epoch, tz=timezone.utc)
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

            expected_day = day_mapping_to_category[day_tag]
            expected_hour = hour_mapping_to_category[hour_tag]

            metrics = validator.validate_generated_request(clean_text)
            
            sensor_match = metrics["predicted_sensor"] == expected_sensor
            priority_match = metrics["predicted_priority"] == expected_priority

            day_match = (metrics["predicted_day"] == expected_day) and (metrics["day_similarity"] >= 0.85)
            hour_match = (metrics["predicted_hour"] == expected_hour) and (metrics["hour_similarity"] >= 0.85)

            if sensor_match:
                successful_sensor_matches += 1
            if priority_match:
                successful_priority_matches += 1
            if day_match:
                successful_day_matches += 1
            if hour_match:
                successful_hour_matches += 1

            processed_tasks_list.append({
                "task_id": task_id,
                "prompt_sent": full_prompt_sent,
                "generated_output": clean_text,
                "ground_truth": {
                    "expected_sensor": expected_sensor,
                    "expected_priority": expected_priority,
                    "expected_day": expected_day,
                    "expected_hour": expected_hour
                },
                "validation": {
                    "predicted_sensor": metrics["predicted_sensor"],
                    "sensor_similarity": metrics["sensor_similarity"],
                    "sensor_match": sensor_match,
                    
                    "predicted_priority": metrics["predicted_priority"],
                    "priority_similarity": metrics["priority_similarity"],
                    "priority_match": priority_match,
                    
                    "predicted_day": metrics["predicted_day"],
                    "day_similarity": metrics["day_similarity"],
                    "day_match": day_match,
                    
                    "predicted_hour": metrics["predicted_hour"],
                    "hour_similarity": metrics["hour_similarity"],
                    "hour_match": hour_match
                }
            })

    if processed_tasks_list:
        total_tasks_count = len(processed_tasks_list)
        
        scenario_summary = {
            "total_tasks_evaluated": total_tasks_count,
            "global_sensor_accuracy": float(successful_sensor_matches / total_tasks_count),
            "global_priority_accuracy": float(successful_priority_matches / total_tasks_count),
            "global_day_accuracy": float(successful_day_matches / total_tasks_count),
            "global_hour_accuracy": float(successful_hour_matches / total_tasks_count),
            "successful_sensor_matches": successful_sensor_matches,
            "successful_priority_matches": successful_priority_matches,
            "successful_day_matches": successful_day_matches,
            "successful_hour_matches": successful_hour_matches
        }

        combined_payload = {
            "timestamp_utc": fixed_now_utc.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "scenario_metrics": scenario_summary,
            "tasks": processed_tasks_list
        }
        
        json_file_path = dir_path / "ollama_prompts_combined.json"
        json_file_path.write_text(
            json.dumps(combined_payload, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    print(f"\n[SUCCESS] Saved {len(prompts_map)} independent human request TXT files and 1 unified JSON catalog with structural validation metrics to directory: {dir_path.resolve()}")
    return prompts_map