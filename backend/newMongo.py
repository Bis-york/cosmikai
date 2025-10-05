# newMongo.py
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from datetime import datetime
from typing import Any, Dict, List, Optional

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


def list_cached_targets() -> List[str]:
    """Return a sorted list of the original target names stored in the cache."""
    cursor = collection.find({}, {"_id": 0, "original_target": 1})
    names = [str(doc.get("original_target", "")).strip() for doc in cursor]
    return sorted({name for name in names if name})


def database_status() -> Dict[str, Any]:
    """Return basic health information about the MongoDB connection."""
    try:
        client.admin.command("ping")
        count = collection.estimated_document_count()
        return {"ok": True, "document_count": count}
    except PyMongoError as exc:
        return {"ok": False, "error": str(exc)}

