from fastapi import FastAPI
from newMongo import get_cached_result, save_result
import subprocess, json, os

app = FastAPI()

@app.post("/predict")
def predict_star(star_id: str):
    # Check cache first
    cached = get_cached_result(star_id)
    if cached:
        print(f"✅ Cache hit for {star_id}")
        return cached["results"]

    # Run model if not cached
    print(f"⚙️ Running model for {star_id}")
    subprocess.run(
    ["python", "predict.py", star_id, "--mission", "TESS", "--save-json", "results/output.json"],
    check=True
)


    # Read model output
    with open("results/output.json", "r") as f:
        result_json = json.load(f)

    # Save result to MongoDB
    save_result(result_json)
    print(f"💾 Saved {star_id} result to MongoDB")

    return result_json
