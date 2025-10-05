# newMongo.py
import os
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from datetime import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

# Mongo connection settings are configurable via environment variables.
MONGO_URI = os.getenv("COSMIKAI_MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB = os.getenv("COSMIKAI_MONGO_DB", "exoplanet_DB")
MONGO_COLLECTION = os.getenv("COSMIKAI_MONGO_COLLECTION", "predictions")

client = MongoClient(MONGO_URI)
db = client[MONGO_DB]
collection = db[MONGO_COLLECTION]


def mongo_config() -> Tuple[str, str, str]:
    """Return the Mongo connection configuration being used."""
    return MONGO_URI, MONGO_DB, MONGO_COLLECTION

def _legacy_normalize(target: str) -> str:
    return target.strip().lower()


def normalize_target(target: str) -> str:
    """Normalize target names in a consistent, punctuation-free manner."""
    legacy = _legacy_normalize(target)
    cleaned = re.sub(r"[^a-z0-9]", "", legacy)
    return cleaned or legacy


def _candidate_keys(target: str) -> List[str]:
    normalized = normalize_target(target)
    legacy = _legacy_normalize(target)
    if normalized == legacy:
        return [normalized]
    return [normalized, legacy]

def get_cached_result(target: str) -> Optional[Dict[str, Any]]:
    """Return cached result if the target already exists."""
    keys = _candidate_keys(target)
    return collection.find_one({"target": {"$in": keys}}, {"_id": 0})

def save_result(result_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Save the full model output JSON to MongoDB.
    Expects the 'results' array with at least one target entry.
    """
    if not result_json or "results" not in result_json:
        raise ValueError("Invalid JSON format: missing 'results' field")

    first_target = result_json["results"][0].get("target", "unknown")
    key_candidates = _candidate_keys(first_target)
    normalized_target = key_candidates[0]

    record = {
        "target": normalized_target,
        "original_target": first_target,
        "target_aliases": key_candidates,
        "checkpoint": result_json.get("checkpoint"),
        "device": result_json.get("device"),
        "elapsed_seconds": result_json.get("elapsed_seconds"),
        "results": result_json["results"],
        "timestamp": datetime.utcnow(),
    }

    collection.replace_one({"target": {"$in": key_candidates}}, record, upsert=True)
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
