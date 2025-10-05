# newMongo.py
from pymongo import MongoClient
from datetime import datetime
from typing import Dict, Any, Optional

client = MongoClient("mongodb://localhost:27017/")
db = client["exoplanet_DB"]
collection = db["predictions"]

def normalize_target(target: str) -> str:
    """Normalize target names (case-insensitive and strip spaces)."""
    return target.strip().lower()

def get_cached_result(target: str) -> Optional[Dict[str, Any]]:
    """Return cached result if the target already exists."""
    normalized = normalize_target(target)
    return collection.find_one({"target": normalized}, {"_id": 0})

def save_result(result_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save the full model output JSON to MongoDB.
    Expects the 'results' array with at least one target entry.
    """
    if not result_json or "results" not in result_json:
        raise ValueError("Invalid JSON format: missing 'results' field")

    first_target = result_json["results"][0].get("target", "unknown")
    normalized_target = normalize_target(first_target)

    record = {
        "target": normalized_target,
        "original_target": first_target,
        "checkpoint": result_json.get("checkpoint"),
        "device": result_json.get("device"),
        "elapsed_seconds": result_json.get("elapsed_seconds"),
        "results": result_json["results"],
        "timestamp": datetime.utcnow(),
    }

    collection.replace_one({"target": normalized_target}, record, upsert=True)
    return record
