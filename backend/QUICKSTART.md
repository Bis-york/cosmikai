# 🚀 Quick Start - Backend Only

## Build and Run (3 commands)

```bash
cd backend
docker-compose up --build
```

Done! API is at: http://localhost:8000/docs

---

## Alternative: Docker Only

```bash
cd backend

# Build
docker build -t cosmikai-backend .

# Run (needs MongoDB running elsewhere)
docker run -p 8000:8000 \
  -e COSMIKAI_MONGO_URI="mongodb://host.docker.internal:27017/" \
  cosmikai-backend
```

---

## Local Development (No Docker)

```powershell
cd backend

# Setup (first time only)
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Run (every time)
python server_setup.py
```

API at: http://localhost:8000

---

## Test the API

Open in browser:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

Or use curl:
```bash
curl http://localhost:8000/health
```

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'backend'"**
- ✅ Fixed! All imports now use relative syntax
- Rebuild: `docker-compose up --build`

**"MongoDB connection failed"**
- Make sure MongoDB is running (docker-compose starts it automatically)
- Or set correct URI: `-e COSMIKAI_MONGO_URI="mongodb://your-host:27017/"`

**Port 8000 already in use**
- Change port: `docker run -p 8001:8000 ...`
- Or in docker-compose.yml: `"8001:8000"`

---

## File Checklist

Required files in `backend/`:
- ✅ `Dockerfile` - Container definition
- ✅ `docker-compose.yml` - With MongoDB
- ✅ `requirements.txt` - Python packages
- ✅ `server_setup.py` - Server entrypoint
- ✅ `newmain.py` - FastAPI app
- ✅ `newMongo.py` - MongoDB utils
- ✅ `predict.py` - Predictions
- ✅ `data_analyzer.py` - ML model
- ✅ `__init__.py` - Package marker

All present ✓

---

**See DOCKER_SETUP.md for detailed documentation**
