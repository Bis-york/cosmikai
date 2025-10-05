# main.py — Part 1

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, subprocess, sys, os

from backend.connection import get_result_by_star_id, insert_result

app = FastAPI(title="Exoplanet Service", version="1.0.0")

# Allow your frontend to call this API (adjust origins as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# main.py — Part 2

class AnalyzeRequest(BaseModel):
    star_id: str | None = None     # Provide this OR csv_path
    csv_path: str | None = None    # Provide this OR star_id

@app.get("/")
async def root():
    return {"message": "Exoplanet backend is running 🚀"}


# main.py — Part 3

def run_predict_with_star_id(star_id: str) -> dict:
    """
    Calls modelcode/predict.py with a star_id and expects JSON on stdout.
    Adjust args if your script uses different flags.
    """
    cmd = [sys.executable, os.path.join(os.getcwd(), "modelcode", "predict.py"), "--star-id", star_id]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"predict.py failed: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"predict.py did not return valid JSON: {e}\nOutput was:\n{proc.stdout[:500]}")


def run_base_with_csv(csv_path: str) -> dict:
    """
    Calls modelcode/base.py with a CSV path and expects JSON on stdout.
    Adjust args if your script uses different flags.
    """
    cmd = [sys.executable, os.path.join("modelcode", "base.py"), "--csv", csv_path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"base.py failed: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"base.py did not return valid JSON: {e}\nOutput was:\n{proc.stdout[:500]}")


# main.py — Part 4

@app.post("/analyze")
async def analyze(req: AnalyzeRequest):
    print("📩 Incoming request:", req.dict())

    try:
        has_star = bool(req.star_id and req.star_id.strip())
        has_csv  = bool(req.csv_path and req.csv_path.strip())
        if has_star == has_csv:
            raise HTTPException(status_code=400, detail="Provide exactly one of: 'star_id' OR 'csv_path'.")

        # --- STAR ID path ---
        if has_star:
            star_id = req.star_id.strip() if req.star_id is not None else ""
            cached = get_result_by_star_id(star_id)
            if cached and cached.get("result_json"):
                print("✅ Found cached result for:", star_id)
                try:
                    return json.loads(cached["result_json"])
                except Exception:
                    return {"cached_raw": cached["result_json"]}

            print("⚙️ Running simulate_predict.py for:", star_id)
            result = run_predict_with_star_id(star_id)
            insert_result(star_id, json.dumps(result))
            print("💾 Saved result for:", star_id)
            return result

        # --- CSV path (optional for later) ---
        if has_csv:
            csv_path = req.csv_path.strip() if req.csv_path is not None else ""
            raise HTTPException(status_code=400, detail="CSV path handling not implemented yet.")

    except RuntimeError as e:
        print("❌ Runtime error:", e)
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        print("❌ Unexpected error:", e)
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")


@app.get("/health")
async def health():
    return {"status": "ok"}
