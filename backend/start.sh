#!/bin/bash
# One-command start script for backend

echo "🚀 Starting CosmiKai Backend..."
echo ""

# Check MongoDB first
python3 test_mongodb.py
if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Cannot start: MongoDB is not accessible"
    echo "   Please start MongoDB first, then run this script again"
    exit 1
fi

echo ""
echo "📦 Building and starting backend..."
docker-compose up --build
