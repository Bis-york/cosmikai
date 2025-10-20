# Quick test script for the backend Docker setup
Write-Host "🔍 Testing CosmiKai Backend Docker Setup" -ForegroundColor Cyan
Write-Host "=========================================" -ForegroundColor Cyan
Write-Host ""

# Check if we're in the backend directory
if (!(Test-Path "Dockerfile") -or !(Test-Path "server_setup.py")) {
    Write-Host "❌ Error: Please run this script from the backend/ directory" -ForegroundColor Red
    exit 1
}

Write-Host "✓ Found required files (Dockerfile, server_setup.py)" -ForegroundColor Green
Write-Host ""

# Test MongoDB connection
Write-Host "🔌 Testing MongoDB connection..." -ForegroundColor Yellow
python test_mongodb.py
if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "⚠️  MongoDB test failed. Docker will still build, but may not start properly." -ForegroundColor Yellow
    Write-Host "   Make sure MongoDB is running before using docker-compose up" -ForegroundColor Yellow
    Write-Host ""
}
Write-Host ""

# Test: Build the Docker image
Write-Host "📦 Building Docker image..." -ForegroundColor Yellow
try {
    docker build -t cosmikai-backend-test . 2>&1 | Select-Object -Last 10
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Docker image built successfully" -ForegroundColor Green
    } else {
        Write-Host "❌ Docker build failed" -ForegroundColor Red
        exit 1
    }
} catch {
    Write-Host "❌ Docker build error: $_" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "🎉 Build successful! You can now run the backend with:" -ForegroundColor Green
Write-Host ""
Write-Host "   # Using docker-compose (includes MongoDB):" -ForegroundColor Cyan
Write-Host "   docker-compose up" -ForegroundColor White
Write-Host ""
Write-Host "   # Or using docker directly:" -ForegroundColor Cyan
Write-Host "   docker run -p 8000:8000 \" -ForegroundColor White
Write-Host "     -e COSMIKAI_MONGO_URI=mongodb://your-mongo:27017/ \" -ForegroundColor White
Write-Host "     cosmikai-backend-test" -ForegroundColor White
Write-Host ""
Write-Host "   # Test the API:" -ForegroundColor Cyan
Write-Host "   Invoke-WebRequest http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
