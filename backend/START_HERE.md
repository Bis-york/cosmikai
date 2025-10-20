# ⚡ SUPER SIMPLE START

## What You Need
1. ✅ MongoDB running on your computer
2. ✅ Docker installed

## Start the Backend (Windows)

```powershell
cd backend
.\start.ps1
```

**That's it!** The script will:
- Check if MongoDB is accessible ✓
- Build the Docker image ✓
- Start the API on port 8000 ✓

## Start the Backend (Mac/Linux)

```bash
cd backend
bash start.sh
```

---

## Access the API

Once started, open your browser:

**Swagger UI (interactive)**: http://localhost:8000/docs

**API Base**: http://localhost:8000

---

## If MongoDB Test Fails

The script will tell you if MongoDB isn't running. To start MongoDB:

### Windows
```powershell
net start MongoDB
# or
Get-Service MongoDB | Start-Service
```

### Mac
```bash
brew services start mongodb-community
```

### Linux
```bash
sudo systemctl start mongod
```

Then run `.\start.ps1` again!

---

## Configuration

**Default MongoDB connection**: `localhost:27017`

**To use a different MongoDB**, edit `docker-compose.yml`:

```yaml
environment:
  COSMIKAI_MONGO_URI: mongodb://your-server:27017/
```

---

## Stop the Backend

Press `Ctrl+C` in the terminal, or:

```bash
docker-compose down
```

---

## Troubleshooting

**"MongoDB test failed"**
- Start MongoDB (see commands above)
- Verify with: `mongosh` or `mongo`

**"Port 8000 already in use"**
- Change in `docker-compose.yml`: `"8001:8000"`

**"module backend not found"**
- Rebuild: `docker-compose up --build`

---

**See README.md for full documentation**
