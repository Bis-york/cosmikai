# Backend Standalone Setup - Changes Summary

## Overview
The backend directory has been configured to run independently without requiring files from the parent repository.

## Files Created/Modified

### ✅ Created Files
1. **`server_setup.py`** - Standalone server entrypoint
   - Simpler version that works from the backend directory
   - Handles MongoDB readiness checks
   - Launches uvicorn with the FastAPI app

2. **`README.md`** - Complete documentation
   - Docker and Docker Compose instructions
   - Local development setup
   - API documentation links
   - Environment variable reference

3. **`.dockerignore`** - Docker build optimization
   - Excludes unnecessary files from image
   - Keeps image size small

4. **`test_docker.sh`** - Unix/Mac test script
   - Quick validation of Docker build

5. **`test_docker.ps1`** - Windows PowerShell test script
   - Quick validation of Docker build for Windows

### 📝 Modified Files
1. **`Dockerfile`**
   - Restructured to work from backend directory only
   - Creates proper package structure in container
   - Sets correct PYTHONPATH
   - Uses slim base image for smaller size

2. **`setup.cfg`**
   - Updated package configuration
   - Added install_requires with all dependencies
   - Better metadata

3. **Core Python files** (previously fixed):
   - `newmain.py` - Changed to relative imports
   - `api.py` - Changed to relative imports  
   - `predict.py` - Changed to relative imports

## How to Use

### Prerequisites
✅ **MongoDB must already be running** on your machine (default: `localhost:27017`)

You can verify with:
```powershell
# Windows/PowerShell
mongosh --eval "db.version()"

# Or check if service is running
Get-Service -Name MongoDB
```

### From the `backend/` directory:

#### Quick Test (PowerShell on Windows)
```powershell
.\test_docker.ps1
```

#### Quick Test (Bash on Unix/Mac)
```bash
bash test_docker.sh
```

#### Production Use with Docker Compose
```bash
# Start API only (connects to your existing MongoDB)
docker-compose up --build

# Stop
docker-compose down

# View logs
docker-compose logs -f backend
```

**Note**: No MongoDB container is created. The backend connects to your existing MongoDB via `host.docker.internal:27017`

#### Standalone Docker
```bash
# Build
docker build -t cosmikai-backend .

# Run (connects to your MongoDB on host)
docker run -p 8000:8000 \
  -e COSMIKAI_MONGO_URI="mongodb://host.docker.internal:27017/" \
  cosmikai-backend
```

## What Changed to Fix Import Errors

### Problem
The original setup had:
- Absolute imports (`from backend.X import Y`)
- Both PROJECT_ROOT and BACKEND_DIR in sys.path
- server_setup.py in parent directory

This caused `ModuleNotFoundError` in Docker because Python couldn't resolve the package structure correctly.

### Solution
1. **Relative imports** - All internal imports use `.module` syntax
2. **Proper package structure** - Only PROJECT_ROOT in sys.path
3. **Standalone setup** - Each directory can work independently

## Directory Structure in Container
```
/app/
├── __init__.py (empty, created by Dockerfile)
└── backend/
    ├── __init__.py
    ├── server_setup.py
    ├── newmain.py
    ├── newMongo.py
    ├── predict.py
    ├── data_analyzer.py
    └── models/
```

## Testing Checklist

- [ ] Build succeeds: `docker build -t cosmikai-backend .`
- [ ] Image runs: `docker run -p 8000:8000 cosmikai-backend`
- [ ] No import errors in logs
- [ ] API responds: `curl http://localhost:8000/docs`
- [ ] MongoDB connection works (if MongoDB is running)
- [ ] Docker Compose works: `docker-compose up`

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `COSMIKAI_MONGO_URI` | `mongodb://host.docker.internal:27017/` | MongoDB connection string (points to host machine) |
| `COSMIKAI_MONGO_DB` | `exoplanet_DB` | Database name |
| `COSMIKAI_MONGO_COLLECTION` | `predictions` | Collection name |

### MongoDB Connection Strings by Platform

| Platform | Connection String |
|----------|------------------|
| **Windows (Docker Desktop)** | `mongodb://host.docker.internal:27017/` |
| **Mac (Docker Desktop)** | `mongodb://host.docker.internal:27017/` |
| **Linux** | `mongodb://172.17.0.1:27017/` (or use `--network host`) |
| **Remote MongoDB** | `mongodb://your-server-ip:27017/` |
| **MongoDB Atlas** | `mongodb+srv://user:pass@cluster.mongodb.net/` |

## Next Steps

1. Build the image: `docker build -t cosmikai-backend .`
2. Run with docker-compose: `docker-compose up`
3. Test the API: Visit `http://localhost:8000/docs`
4. Check MongoDB connection in logs

## Troubleshooting

**Import errors?**
- Verify all imports in Python files use relative syntax (`.module`)
- Check PYTHONPATH is set to `/app` in container
- Rebuild: `docker-compose up --build`

**MongoDB connection fails?**
- ✅ **Verify YOUR MongoDB is running** on `localhost:27017`
  - Windows: `Get-Service MongoDB` or `mongosh`
  - Mac/Linux: `brew services list` or `systemctl status mongod`
- Check if `host.docker.internal` resolves (Windows/Mac should work, Linux needs `172.17.0.1`)
- Verify connection string in `docker-compose.yml`
- Check MongoDB is accessible (not behind firewall)

**"host.docker.internal" not found (Linux)?**
- Edit `docker-compose.yml` and change to `mongodb://172.17.0.1:27017/`
- Or run with `--network host`: `docker run --network host cosmikai-backend`

**Build fails?**
- Check Docker is running
- Verify all files are present in backend/
- Try `docker system prune` to clean up

**Port already in use?**
- Change port mapping: `-p 8001:8000`
- Or stop other services using port 8000

**Backend starts but can't reach MongoDB?**
- Test connection from host: `mongosh mongodb://localhost:27017`
- If MongoDB requires authentication, update URI: `mongodb://user:pass@host.docker.internal:27017/`
- Check MongoDB is not bound to 127.0.0.1 only (should bind to 0.0.0.0 or allow docker subnet)
