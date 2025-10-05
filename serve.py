"""Serve the CosmiKai API with a readiness check for MongoDB."""
from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict

import uvicorn

from backend.newMongo import database_status, mongo_config


def wait_for_mongo(timeout: float, interval: float) -> Dict[str, Any]:
    """Poll the MongoDB instance until it responds or the timeout expires."""
    deadline = time.monotonic() + timeout
    last_status: Dict[str, Any] | None = None

    while time.monotonic() < deadline:
        status = database_status()
        last_status = status
        if status.get("ok"):
            return status
        time.sleep(interval)

    error_message = last_status.get("error") if last_status else "unknown error"
    raise TimeoutError(f"MongoDB did not respond within {timeout:.1f}s: {error_message}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the CosmiKai API and ensure MongoDB is reachable.")
    parser.add_argument("--host", default="0.0.0.0", help="API bind address (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="API port (default: 8000)")
    parser.add_argument(
        "--mongo-timeout",
        type=float,
        default=20.0,
        help="Seconds to wait for MongoDB before failing (default: 20)",
    )
    parser.add_argument(
        "--mongo-interval",
        type=float,
        default=0.5,
        help="Polling interval when waiting for MongoDB (default: 0.5)",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn auto-reload (development only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    uri, db_name, collection = mongo_config()

    print(f"Checking MongoDB at {uri} (database={db_name}, collection={collection})...", flush=True)
    try:
        status = wait_for_mongo(timeout=args.mongo_timeout, interval=args.mongo_interval)
    except TimeoutError as exc:  # pragma: no cover - startup fast fail
        print(f"MongoDB check failed: {exc}", file=sys.stderr)
        return 2

    doc_count = status.get("document_count", "unknown")
    print(f"MongoDB ready (documents cached: {doc_count}). Starting API server...", flush=True)

    config = uvicorn.Config(
        "backend.newmain:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )
    server = uvicorn.Server(config)
    return 0 if server.run() else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
