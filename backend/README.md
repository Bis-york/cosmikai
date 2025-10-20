# CosmiKai Backend

This is the standalone backend API for CosmiKai exoplanet detection.

## Running with Docker

### Option 1: Using Docker Compose (Recommended)

From the `backend/` directory:

```bash
docker-compose up --build
```

This will:
- Start a MongoDB container
- Build and start the backend API container
- Expose the API on `http://localhost:8000`

### Option 2: Using Docker only

Build the image:
```bash
docker build -t cosmikai-backend .
```

Run with a MongoDB connection:
```bash
docker run -p 8000:8000 \
  -e COSMIKAI_MONGO_URI="mongodb://your-mongo-host:27017/" \
  cosmikai-backend
```

## Running Locally (Development)

### Prerequisites
- Python 3.11+
- MongoDB running on `localhost:27017` (or set `COSMIKAI_MONGO_URI`)

### Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Unix/macOS:
source .venv/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set environment variables (optional):
```bash
# Windows PowerShell:
$env:COSMIKAI_MONGO_URI = "mongodb://localhost:27017/"
$env:COSMIKAI_MONGO_DB = "exoplanet_DB"
$env:COSMIKAI_MONGO_COLLECTION = "predictions"

# Unix/macOS:
export COSMIKAI_MONGO_URI="mongodb://localhost:27017/"
export COSMIKAI_MONGO_DB="exoplanet_DB"
export COSMIKAI_MONGO_COLLECTION="predictions"
```

4. Run the server:
```bash
python server_setup.py
```

The API will be available at `http://localhost:8000`

## API Documentation

Once running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COSMIKAI_MONGO_URI` | `mongodb://mongo:27017/` | MongoDB connection URI |
| `COSMIKAI_MONGO_DB` | `exoplanet_DB` | MongoDB database name |
| `COSMIKAI_MONGO_COLLECTION` | `predictions` | MongoDB collection name |

## Project Structure

```
backend/
├── Dockerfile              # Container definition
├── docker-compose.yml      # Docker Compose setup with MongoDB
├── requirements.txt        # Python dependencies
├── server_setup.py         # Server entrypoint with MongoDB check
├── newmain.py             # FastAPI application
├── newMongo.py            # MongoDB utilities
├── predict.py             # Lightkurve-based prediction
├── data_analyzer.py       # Core ML model and analysis
├── api.py                 # Alternative API endpoints
└── models/                # Trained model checkpoints
```

## Development

Enable auto-reload for development:
```bash
python server_setup.py --reload
```

Or run directly with uvicorn:
```bash
uvicorn backend.newmain:app --reload --host 0.0.0.0 --port 8000
```
