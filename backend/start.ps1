# One-command start script for backend
Write-Host "🚀 Starting CosmiKai Backend..." -ForegroundColor Cyan
Write-Host ""

# Check MongoDB first
python test_mongodb.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ Cannot start: MongoDB is not accessible" -ForegroundColor Red
    Write-Host "   Please start MongoDB first, then run this script again" -ForegroundColor Yellow
    exit 1
}

Write-Host ""
Write-Host "📦 Building and starting backend..." -ForegroundColor Green
docker-compose up --build
