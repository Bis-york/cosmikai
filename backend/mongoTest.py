from pymongo import MongoClient

# Connect to your local MongoDB server
client = MongoClient("mongodb://localhost:27017/")
print("✅ Connected to MongoDB!")

# List all databases
print("Databases:", client.list_database_names())

# Use your existing database
db = client["exoplanet_DB"]
collection = db["predictions"]

# Insert a test document
test_doc = {"hello": "world"}
collection.insert_one(test_doc)
print("✅ Inserted test document into 'predictions' collection")

# Retrieve it
found = collection.find_one({"hello": "world"})
print("Retrieved:", found)
