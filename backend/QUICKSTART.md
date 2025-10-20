# 🚀 Quick Start - Backend Only

## Prerequisites
✅ **MongoDB running on your machine** (default: `localhost:27017`)

## Simplest Way - One Command! (Windows)

```powershell
cd backend
.\start.ps1
```

This will:
1. ✅ Test your MongoDB connection
2. 📦 Build the Docker image
3. 🚀 Start the backend API

Done! API is at: http://localhost:8000/docs

---

## Manual Method (2 commands)

```bash
cd backend

# Test MongoDB first (optional but recommended)
python test_mongodb.py

# Build and run
docker-compose up --build
```

**Note**: Connects to your existing MongoDB via `host.docker.internal:27017`

---

## Alternative: Docker Only

```bash
cd backend

# Build
docker build -t cosmikai-backend .

# Run (connects to your MongoDB on host machine)
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
- ✅ Make sure YOUR MongoDB is running on `localhost:27017`
- Check with: `mongosh` or `mongo` in terminal
- If MongoDB is on different host/port, edit `docker-compose.yml`
- For Linux: Change URI to `mongodb://172.17.0.1:27017/`

**Port 8000 already in use**
- Change port: `docker run -p 8001:8000 ...`
- Or in docker-compose.yml: `"8001:8000"`

**"host.docker.internal" not resolving (Linux)**
- Use `mongodb://172.17.0.1:27017/` instead
- Or add `--network host` to docker run

---

## File Checklist

Required files in `backend/`:
- ✅ `Dockerfile` - Container definition (backend only, no MongoDB)
- ✅ `docker-compose.yml` - Connects to your MongoDB
- ✅ `start.ps1` / `start.sh` - One-command starter
- ✅ `test_mongodb.py` - MongoDB connection test
- ✅ `requirements.txt` - Python packages
- ✅ `server_setup.py` - Server entrypoint
- ✅ `newmain.py` - FastAPI app
- ✅ `newMongo.py` - MongoDB utils
- ✅ `predict.py` - Predictions
- ✅ `data_analyzer.py` - ML model
- ✅ `__init__.py` - Package marker

All present ✓

**Note**: No MongoDB container - uses your existing MongoDB!

---

**See DOCKER_SETUP.md for detailed documentation**
