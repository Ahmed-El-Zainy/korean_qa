import os
from qdrant_client import QdrantClient

print("Environment variables:")
print(f"QDRANT_URL: {os.getenv('QDRANT_URL')}")
print(f"QDRANT_API_KEY: {os.getenv('QDRANT_API_KEY')}")

# Test direct connection
try:
    client = QdrantClient(url="http://localhost:6333", api_key="")
    collections = client.get_collections()
    print("✅ Direct connection successful")
    print(f"Collections: {collections}")
except Exception as e:
    print(f"❌ Direct connection failed: {e}")
    print(f"Error type: {type(e).__name__}")

# Test with environment variables
try:
    client = QdrantClient(
        url=os.getenv('QDRANT_URL'), 
        api_key=os.getenv('QDRANT_API_KEY')
    )
    collections = client.get_collections()
    print("✅ Environment variable connection successful")
except Exception as e:
    print(f"❌ Environment variable connection failed: {e}")
    print(f"Error type: {type(e).__name__}")
