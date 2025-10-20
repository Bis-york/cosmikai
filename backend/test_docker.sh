#!/usr/bin/env bash
# Quick test script for the backend Docker setup

echo "🔍 Testing CosmiKai Backend Docker Setup"
echo "========================================="
echo ""

# Check if we're in the backend directory
if [ ! -f "Dockerfile" ] || [ ! -f "server_setup.py" ]; then
    echo "❌ Error: Please run this script from the backend/ directory"
    exit 1
fi

echo "✓ Found required files (Dockerfile, server_setup.py)"
echo ""

# Test 1: Build the Docker image
echo "📦 Building Docker image..."
docker build -t cosmikai-backend-test . 2>&1 | tail -10

if [ $? -eq 0 ]; then
    echo "✓ Docker image built successfully"
else
    echo "❌ Docker build failed"
    exit 1
fi

echo ""
echo "🎉 Build successful! You can now run the backend with:"
echo ""
echo "   # Using docker-compose (includes MongoDB):"
echo "   docker-compose up"
echo ""
echo "   # Or using docker directly:"
echo "   docker run -p 8000:8000 \\"
echo "     -e COSMIKAI_MONGO_URI=mongodb://your-mongo:27017/ \\"
echo "     cosmikai-backend-test"
echo ""
