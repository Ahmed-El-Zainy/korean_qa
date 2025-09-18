#!/usr/bin/env python3
"""
Fix Qdrant collection dimensions for Manufacturing RAG Agent
"""

import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models

load_dotenv()



# QDRANT_API_KEY= os.getenv('QDRANT_API_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DHeUsIY234NwS-6cYDJec807Vdzbs1PHmBBU3_Jz9oo') 
# QDRANT_URL=os.getenv('QDRANT_URL', 'https://50f53cc8-bbb0-4939-8254-8f025a577222.us-west-2-0.aws.cloud.qdrant.io:6333')

# QDRANT_URL= os.getenv('QDRANT_URL', 'http://localhost:6333')

def fix_qdrant_collection():
    """Fix the Qdrant collection dimensions."""
    
    print("🔧 Fixing Qdrant Collection Dimensions")
    print("=" * 50)
    
    # Get connection details
    qdrant_api_key = os.environ["QDRANT_API_KEY"]
    qdrant_url = os.environ["QDRANT_URL"]
    collection_name = 'manufacturing_docs'
    
    if not qdrant_url:
        print("❌ QDRANT_URL not found in environment variables")
        return False
    
    try:
        # Connect to Qdrant
        print(f"🔗 Connecting to Qdrant: {qdrant_url}")
        client = QdrantClient(
            url="https://50f53cc8-bbb0-4939-8254-8f025a577222.us-west-2-0.aws.cloud.qdrant.io:6333", 
            api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.gHOXbfqPucRwhczrW8s3VSZbconqQ6Rk49Uaz9ZChdE",)
        
        # Check if collection exists
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name in collection_names:
            print(f"📋 Collection '{collection_name}' exists")
            
            # Get collection info
            collection_info = client.get_collection(collection_name)
            current_dim = collection_info.config.params.vectors.size
            print(f"📏 Current vector dimensions: {current_dim}")
            
            if current_dim != 1024:
                print(f"⚠️  Need to recreate collection with correct dimensions (1024)")
                
                # Ask for confirmation
                response = input("🗑️  Delete existing collection and recreate? (y/N): ").strip().lower()
                if response != 'y':
                    print("❌ Aborted by user")
                    return False
                
                # Delete existing collection
                print(f"🗑️  Deleting collection '{collection_name}'...")
                client.delete_collection(collection_name)
                print("✅ Collection deleted")
            else:
                print("✅ Collection already has correct dimensions")
                return True
        
        # Create new collection with correct dimensions
        print(f"🆕 Creating collection '{collection_name}' with 4096 dimensions...")
        
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=4096,  # Correct size for Qwen/Qwen3-Embedding-8B
                distance=models.Distance.COSINE
            )
        )
        
        # Create payload indexes
        print("🔍 Creating payload indexes...")
        
        indexes_to_create = [
            ("document_id", models.PayloadFieldSchema(
                data_type=models.PayloadSchemaType.KEYWORD
            )),
            ("document_type", models.PayloadFieldSchema(
                data_type=models.PayloadSchemaType.KEYWORD
            )),
            ("page_number", models.PayloadFieldSchema(
                data_type=models.PayloadSchemaType.INTEGER
            )),
            ("worksheet_name", models.PayloadFieldSchema(
                data_type=models.PayloadSchemaType.KEYWORD
            )),
        ]
        
        for field_name, field_schema in indexes_to_create:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )
                print(f"✅ Created index for '{field_name}'")
            except Exception as e:
                print(f"⚠️  Failed to create index for '{field_name}': {e}")
        
        print("✅ Collection recreated successfully with correct dimensions!")
        return True
        
    except Exception as e:
        
        print(f"❌ Error: {e}")
        return False

def update_config_file():
    """Update config.yaml with correct vector dimensions."""
    
    print("\n🔧 Updating Configuration")
    print("=" * 30)
    
    config_path = "src/config.yaml"
    
    if not os.path.exists(config_path):
        print(f"❌ Config file not found: {config_path}")
        return False
    
    try:
        # Read current config
        with open(config_path, 'r') as f:
            content = f.read()
        
        # Update vector_size if it exists
        import re
        
        # Look for vector_size configuration
        if 'vector_size:' in content:
            # Replace vector_size value
            content = re.sub(r'vector_size:\s*\d+', 'vector_size: 4096', content)
            print("✅ Updated vector_size to 4096")
        else:
            # Add vector_size to vector_store section
            if 'vector_store:' in content:
                content = re.sub(
                    r'(vector_store:\s*\n)',
                    r'\1  vector_size: 4096\n',
                    content
                )
                print("✅ Added vector_size: 4096 to vector_store section")
            else:
                print("⚠️  No vector_store section found, please add manually:")
                print("vector_store:")
                print("  vector_size: 4096")
        
        # Write updated config
        with open(config_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {config_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error updating config: {e}")
        return False

def test_embedding_dimensions():
    """Test the actual embedding dimensions from SiliconFlow."""
    
    print("\n🧪 Testing Embedding Dimensions")
    print("=" * 35)
    
    try:
        import requests
        
        api_key = os.getenv('SILICONFLOW_API_KEY')
        if not api_key:
            print("❌ SILICONFLOW_API_KEY not found")
            return None
        
        # Test embedding generation
        payload = {
            "model": "Qwen/Qwen3-Embedding-8B",
            "input": ["test embedding dimension"],
            "encoding_format": "float"
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        response = requests.post(
            "https://api.siliconflow.com/v1/embeddings",
            json=payload,
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and len(data['data']) > 0:
                embedding = data['data'][0]['embedding']
                dim = len(embedding)
                print(f"✅ Actual embedding dimensions: {dim}")
                return dim
            else:
                print("❌ No embedding data returned")
        else:
            print(f"❌ API error: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing embeddings: {e}")
    
    return None

def main():
    """Main function."""
    
    print("🏭 Manufacturing RAG Agent - Dimension Fix")
    print("=" * 60)
    
    # Test actual embedding dimensions
    actual_dim = test_embedding_dimensions()
    
    if actual_dim and actual_dim != 4096:
        print(f"⚠️  Warning: Expected 4096 dimensions, but got {actual_dim}")
        print("You may need to update the vector_size in your config")
    
    # Fix Qdrant collection
    if fix_qdrant_collection():
        print("\n✅ Qdrant collection fixed successfully!")
    else:
        print("\n❌ Failed to fix Qdrant collection")
        return
    
    # Update config file
    if update_config_file():
        print("✅ Configuration updated successfully!")
    else:
        print("⚠️  Please update config manually")
    
    print("\n🎉 Fix Complete!")
    print("\n📋 Next Steps:")
    print("1. Restart your Gradio demo")
    print("2. Re-upload your documents")
    print("3. Test question answering")
    
    print("\n🚀 To restart the demo:")
    print("python fixed_gradio_demo.py")

if __name__ == "__main__":
    main()