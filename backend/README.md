# CosmiKai Backend

This is the standalone backend API for CosmiKai exoplanet detection.

## Running with Docker

**Prerequisites**: You need MongoDB already running on your machine (default: `localhost:27017`)

### Option 1: Using Docker Compose (Recommended)

From the `backend/` directory:

```bash
docker-compose up --build
```

This will:
- Build and start the backend API container
- Connect to your existing MongoDB on the host machine
- Expose the API on `http://localhost:8000`

**Note**: The default configuration connects to `mongodb://host.docker.internal:27017/`. If your MongoDB is elsewhere, edit `docker-compose.yml` or set environment variables.

### Option 2: Using Docker only

Build the image:
```bash
docker build -t cosmikai-backend .
```

Run and connect to your MongoDB:
```bash
docker run -p 8000:8000 \
  -e COSMIKAI_MONGO_URI="mongodb://host.docker.internal:27017/" \
  cosmikai-backend
```

**MongoDB Connection Notes**:
- `host.docker.internal` - Connects to services on your host machine (Windows/Mac)
- For Linux, use `--network host` or `mongodb://172.17.0.1:27017/`
- For remote MongoDB, use the full connection string

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
| `COSMIKAI_MONGO_URI` | `mongodb://host.docker.internal:27017/` | MongoDB connection URI (points to host machine MongoDB) |
| `COSMIKAI_MONGO_DB` | `exoplanet_DB` | MongoDB database name |
| `COSMIKAI_MONGO_COLLECTION` | `predictions` | MongoDB collection name |

## Project Structure

```
backend/
├── Dockerfile              # Container definition (backend only)
├── docker-compose.yml      # Docker Compose setup (connects to existing MongoDB)
├── requirements.txt        # Python dependencies
├── server_setup.py         # Server entrypoint with MongoDB check
├── newmain.py             # FastAPI application
├── newMongo.py            # MongoDB utilities
├── predict.py             # Lightkurve-based prediction
├── data_analyzer.py       # Core ML model and analysis
├── api.py                 # Alternative API endpoints
└── models/                # Trained model checkpoints
```

## Notes

- This backend expects MongoDB to already be running
- No MongoDB container is created (you manage your own MongoDB)
- Frontend folders are not included in this Docker setup

## Development

Enable auto-reload for development:
```bash
python server_setup.py --reload
```

Or run directly with uvicorn:
```bash
uvicorn backend.newmain:app --reload --host 0.0.0.0 --port 8000
```
