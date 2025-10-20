"""Serve the CosmiKai API with a readiness check for MongoDB."""
from __future__ import annotations

import argparse
import sys
import time
from importlib import util as importlib_util
from types import ModuleType
from pathlib import Path
from typing import Any, Dict

import uvicorn

PROJECT_ROOT = Path(__file__).resolve().parent
BACKEND_DIR = PROJECT_ROOT / "backend"
NEWMONGO_PATH = BACKEND_DIR / "newMongo.py"

if not NEWMONGO_PATH.exists():
    raise FileNotFoundError(f"Expected backend module at {NEWMONGO_PATH} but it was not found.")

# Ensure project directories are on sys.path for downstream imports (e.g. backend.newmain in uvicorn).
for candidate in (PROJECT_ROOT,):
    path_str = str(candidate)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

# Also ensure backend dir is NOT in sys.path (should be accessed via backend.module)
backend_str = str(BACKEND_DIR)
while backend_str in sys.path:
    sys.path.remove(backend_str)

def _load_backend_module(module_file: str) -> ModuleType:
    """
    Load a backend module directly from its file path.

    This bypasses Python's normal import resolution, so the module is available
    even if PYTHONPATH is misconfigured inside the container.
    """
    package_name = "backend"
    module_name = f"{package_name}.{module_file}"
    module_path = BACKEND_DIR / f"{module_file}.py"

    if not module_path.exists():
        raise FileNotFoundError(f"Expected module at {module_path}")

    # Ensure the package placeholder exists so attribute resolution works.
    if package_name not in sys.modules:
        package = ModuleType(package_name)
        package.__path__ = [str(BACKEND_DIR)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package

    spec = importlib_util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load spec for {module_name} from {module_path}")

    module = importlib_util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


# Pre-load all backend modules in dependency order to support relative imports
_load_backend_module("data_analyzer")
_load_backend_module("predict")
newmongo_module = _load_backend_module("newMongo")
_load_backend_module("newmain")
database_status = newmongo_module.database_status  # type: ignore[attr-defined]
mongo_config = newmongo_module.mongo_config  # type: ignore[attr-defined]


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
