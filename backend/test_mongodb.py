#!/usr/bin/env python3
"""Test MongoDB connection before running the backend."""
import sys
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, ServerSelectionTimeoutError
import os

def test_connection(uri: Optional[str] = None) -> int:
    """Test if MongoDB is accessible."""
    if uri is None:
        uri = os.getenv("COSMIKAI_MONGO_URI", "mongodb://localhost:27017/")
    
    print(f"Testing MongoDB connection to: {uri}")
    print("-" * 60)
    
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        # Force connection
        client.admin.command('ping')
        
        print("✅ SUCCESS: MongoDB is accessible!")
        print(f"   Server version: {client.server_info()['version']}")
        
        # Check database
        db_name = os.getenv("COSMIKAI_MONGO_DB", "exoplanet_DB")
        collection_name = os.getenv("COSMIKAI_MONGO_COLLECTION", "predictions")
        
        db = client[db_name]
        collection = db[collection_name]
        count = collection.count_documents({})
        
        print(f"   Database: {db_name}")
        print(f"   Collection: {collection_name}")
        print(f"   Documents: {count}")
        print()
        print("🚀 Ready to run docker-compose up!")
        return 0
        
    except (ConnectionFailure, ServerSelectionTimeoutError) as e:
        print(f"❌ FAILED: Could not connect to MongoDB")
        print(f"   Error: {e}")
        print()
        print("💡 Solutions:")
        print("   1. Make sure MongoDB is running:")
        print("      - Windows: Get-Service MongoDB")
        print("      - Mac: brew services list")
        print("      - Linux: systemctl status mongod")
        print()
        print("   2. Start MongoDB if not running:")
        print("      - Windows: net start MongoDB")
        print("      - Mac: brew services start mongodb-community")
        print("      - Linux: sudo systemctl start mongod")
        print()
        print(f"   3. Verify connection: mongosh {uri}")
        return 1
        
    except Exception as e:
        print(f"❌ FAILED: Unexpected error")
        print(f"   Error: {e}")
        return 1

if __name__ == "__main__":
    uri = sys.argv[1] if len(sys.argv) > 1 else None
    sys.exit(test_connection(uri))
