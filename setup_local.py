# #!/usr/bin/env python3
# """
# Local setup script for Manufacturing RAG Agent without Docker.

# This script helps set up the RAG system using Qdrant Cloud or local alternatives.
# """

# import os
# import sys
# import subprocess
# from pathlib import Path
# import requests
# import time

# def check_python_version():
#     """Check if Python version is compatible."""
#     if sys.version_info < (3, 8):
#         print("❌ Python 3.8+ is required")
#         return False
#     print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
#     return True

# def check_dependencies():
#     """Check if required dependencies are installed."""
#     print("\n🔍 Checking dependencies...")
    
#     # Check pip packages
#     required_packages = [
#         'streamlit', 'qdrant-client', 'groq', 'requests',
#         'pandas', 'plotly', 'PyMuPDF', 'openpyxl', 'pytesseract', 'Pillow'
#     ]
    
#     missing_packages = []
#     for package in required_packages:
#         try:
#             __import__(package.replace('-', '_').lower())
#             print(f"✅ {package}")
#         except ImportError:
#             print(f"❌ {package}")
#             missing_packages.append(package)
    
#     if missing_packages:
#         print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
#         try:
#             subprocess.check_call([
#                 sys.executable, '-m', 'pip', 'install'
#             ] + missing_packages)
#             print("✅ All packages installed successfully")
#         except subprocess.CalledProcessError:
#             print("❌ Failed to install packages")
#             return False
    
#     return True

# def check_tesseract():
#     """Check if Tesseract OCR is installed."""
#     print("\n🔍 Checking Tesseract OCR...")
#     try:
#         result = subprocess.run(['tesseract', '--version'], 
#                               capture_output=True, text=True)
#         if result.returncode == 0:
#             version = result.stdout.split('\n')[0]
#             print(f"✅ {version}")
#             return True
#     except FileNotFoundError:
#         pass
    
#     print("❌ Tesseract OCR not found")
#     print("📥 Install with: brew install tesseract")
#     return False

# def setup_environment():
#     """Set up environment variables."""
#     print("\n⚙️ Setting up environment...")
    
#     env_file = Path('.env')
#     env_example = Path('.env.example')
    
#     if not env_file.exists() and env_example.exists():
#         # Copy example file
#         with open(env_example, 'r') as f:
#             content = f.read()
        
#         with open(env_file, 'w') as f:
#             f.write(content)
        
#         print("✅ Created .env file from template")
#         print("📝 Please edit .env file with your API keys:")
#         print("   - GROQ_API_KEY (get from https://console.groq.com/)")
#         print("   - SILICONFLOW_API_KEY (get from https://siliconflow.cn/)")
#         print("   - QDRANT_URL and QDRANT_API_KEY (if using Qdrant Cloud)")
#         return False
#     elif env_file.exists():
#         print("✅ .env file exists")
#         return True
#     else:
#         print("❌ No .env.example file found")
#         return False

# def test_qdrant_connection():
#     """Test Qdrant connection."""
#     print("\n🔍 Testing Qdrant connection...")
    
#     # Try local Qdrant first
#     try:
#         response = requests.get('http://localhost:6333/health', timeout=5)
#         if response.status_code == 200:
#             print("✅ Local Qdrant is running")
#             return True
#     except requests.exceptions.RequestException:
#         pass
    
#     # Try Qdrant Cloud if configured
#     qdrant_url = os.getenv('QDRANT_URL')
#     qdrant_key = os.getenv('QDRANT_API_KEY')
    
#     if qdrant_url and qdrant_key:
#         try:
#             headers = {'api-key': qdrant_key} if qdrant_key else {}
#             response = requests.get(f"{qdrant_url}/health", headers=headers, timeout=10)
#             if response.status_code == 200:
#                 print("✅ Qdrant Cloud connection successful")
#                 return True
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Qdrant Cloud connection failed: {e}")
    
#     print("❌ No Qdrant connection available")
#     print("💡 Options:")
#     print("   1. Start local Qdrant: docker run -p 6333:6333 qdrant/qdrant")
#     print("   2. Use Qdrant Cloud: https://cloud.qdrant.io/")
#     print("   3. Download Qdrant binary: https://github.com/qdrant/qdrant/releases")
#     return False

# def create_data_directories():
#     """Create necessary data directories."""
#     print("\n📁 Creating data directories...")
    
#     directories = [
#         'data',
#         'data/documents',
#         'logs',
#         'results'
#     ]
    
#     for directory in directories:
#         Path(directory).mkdir(parents=True, exist_ok=True)
#         print(f"✅ {directory}/")
    
#     return True

# def test_api_keys():
#     """Test API key configuration."""
#     print("\n🔑 Testing API keys...")
    
#     # Load environment variables
#     from dotenv import load_dotenv
#     load_dotenv()
    
#     # Test Groq API
#     groq_key = os.getenv('GROQ_API_KEY')
#     if groq_key:
#         try:
#             headers = {'Authorization': f'Bearer {groq_key}'}
#             response = requests.get('https://api.groq.com/openai/v1/models', 
#                                   headers=headers, timeout=10)
#             if response.status_code == 200:
#                 print("✅ Groq API key is valid")
#             else:
#                 print(f"❌ Groq API key test failed: {response.status_code}")
#         except requests.exceptions.RequestException as e:
#             print(f"❌ Groq API connection failed: {e}")
#     else:
#         print("❌ GROQ_API_KEY not set")
    
#     # Test Silicon Flow API
#     sf_key = os.getenv('SILICONFLOW_API_KEY')
#     if sf_key:
#         print("✅ Silicon Flow API key is set (cannot test without making API call)")
#     else:
#         print("❌ SILICONFLOW_API_KEY not set")
    
#     return groq_key and sf_key

# def main():
#     """Main setup function."""
#     print("🏭 Manufacturing RAG Agent Setup")
#     print("=" * 40)
    
#     # Check Python version
#     if not check_python_version():
#         sys.exit(1)
    
#     # Check dependencies
#     if not check_dependencies():
#         print("\n❌ Dependency check failed")
#         sys.exit(1)
    
#     # Check Tesseract
#     tesseract_ok = check_tesseract()
    
#     # Setup environment
#     env_setup = setup_environment()
    
#     # Create directories
#     create_data_directories()
    
#     if env_setup:
#         # Test API keys
#         api_keys_ok = test_api_keys()
        
#         # Test Qdrant
#         qdrant_ok = test_qdrant_connection()
        
#         print("\n" + "=" * 40)
#         print("📋 Setup Summary:")
#         print(f"✅ Python: OK")
#         print(f"✅ Dependencies: OK")
#         print(f"{'✅' if tesseract_ok else '❌'} Tesseract OCR: {'OK' if tesseract_ok else 'Missing'}")
#         print(f"{'✅' if api_keys_ok else '❌'} API Keys: {'OK' if api_keys_ok else 'Missing'}")
#         print(f"{'✅' if qdrant_ok else '❌'} Qdrant: {'OK' if qdrant_ok else 'Not available'}")
        
#         if tesseract_ok and api_keys_ok and qdrant_ok:
#             print("\n🎉 Setup complete! You can now run:")
#             print("   python launch_rag_demo.py")
#         else:
#             print("\n⚠️  Setup incomplete. Please address the issues above.")
#             if not tesseract_ok:
#                 print("   Install Tesseract: brew install tesseract")
#             if not api_keys_ok:
#                 print("   Configure API keys in .env file")
#             if not qdrant_ok:
#                 print("   Set up Qdrant (see options above)")
#     else:
#         print("\n📝 Please configure your .env file with API keys, then run this script again.")

# if __name__ == "__main__":
#     main()





#!/usr/bin/env python3
"""
Quick fix script for Manufacturing RAG Agent issues
"""

import os
import sys
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.http import models
import requests

load_dotenv()

def check_environment():
    """Check all environment variables."""
    print("🔍 Checking Environment Variables")
    print("=" * 40)
    
    required_vars = {
        'GROQ_API_KEY': 'Groq LLM API',
        'SILICONFLOW_API_KEY': 'SiliconFlow Embedding API', 
        'QDRANT_URL': 'Qdrant Vector Database URL',
        'QDRANT_API_KEY': 'Qdrant API Key'
    }
    
    issues = []
    
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            # Mask sensitive info
            if 'KEY' in var:
                display_value = f"{value[:8]}...{value[-4:]}" if len(value) > 12 else "***"
            else:
                display_value = value
            print(f"✅ {var}: {display_value}")
        else:
            print(f"❌ {var}: Not set")
            issues.append(f"{var} ({description})")
    
    if issues:
        print(f"\n❌ Missing environment variables:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    
    return True

def test_siliconflow_api():
    """Test SiliconFlow API connection and get actual embedding dimensions."""
    print("\n🧪 Testing SiliconFlow API")
    print("=" * 30)
    
    api_key = os.getenv('SILICONFLOW_API_KEY')
    if not api_key:
        print("❌ No API key found")
        return None
    
    try:
        payload = {
            "model": "Qwen/Qwen3-Embedding-8B",
            "input": ["test embedding to check dimensions"],
            "encoding_format": "float"
        }
        
        headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
        
        print("📡 Testing API connection...")
        response = requests.post(
            "https://api.siliconflow.com/v1/embeddings",
            json=payload,
            headers=headers,
            timeout=15
        )
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                embedding = data['data'][0]['embedding']
                dimensions = len(embedding)
                print(f"✅ API working! Embedding dimensions: {dimensions}")
                return dimensions
            else:
                print("❌ No embedding data returned")
        elif response.status_code == 401:
            print("❌ API Key Invalid - Please check your SILICONFLOW_API_KEY")
            print("   Get a new key from: https://siliconflow.com/")
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Response: {response.text}")
    
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    return None

def test_qdrant_connection():
    """Test Qdrant connection."""
    print("\n🗄️  Testing Qdrant Connection")
    print("=" * 35)
    
    
    qdrant_api_key= os.getenv('QDRANT_API_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.DHeUsIY234NwS-6cYDJec807Vdzbs1PHmBBU3_Jz9oo') 
    # QDRANT_URL=os.getenv('QDRANT_URL', 'https://50f53cc8-bbb0-4939-8254-8f025a577222.us-west-2-0.aws.cloud.qdrant.io:6333')

    # qdrant_api_key = ""
    qdrant_url= os.getenv('QDRANT_URL', 'http://localhost:6333')

    
    if not qdrant_url:
        print("❌ No Qdrant URL found")
        return False
    
    try:
        print(f"🔗 Connecting to: {qdrant_url}")
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Test connection
        collections = client.get_collections()
        print(f"✅ Connected! Found {len(collections.collections)} collections")
        
        # Check manufacturing_docs collection
        collection_names = [col.name for col in collections.collections]
        if 'manufacturing_docs' in collection_names:
            collection_info = client.get_collection('manufacturing_docs')
            current_dim = collection_info.config.params.vectors.size
            points_count = collection_info.points_count
            print(f"📋 Collection 'manufacturing_docs' exists:")
            print(f"   - Vector dimensions: {current_dim}")
            print(f"   - Points count: {points_count}")
            return current_dim
        else:
            print("ℹ️  Collection 'manufacturing_docs' doesn't exist yet")
            return 0
    
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def fix_qdrant_collection(correct_dimensions):
    """Fix the Qdrant collection with correct dimensions."""
    print(f"\n🔧 Fixing Qdrant Collection (Dimensions: {correct_dimensions})")
    print("=" * 60)
    
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_api_key = os.getenv('QDRANT_API_KEY')
    collection_name = 'manufacturing_docs'
    
    try:
        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        
        # Check current collection
        collections = client.get_collections()
        collection_names = [col.name for col in collections.collections]
        
        if collection_name in collection_names:
            collection_info = client.get_collection(collection_name)
            current_dim = collection_info.config.params.vectors.size
            
            if current_dim == correct_dimensions:
                print(f"✅ Collection already has correct dimensions ({correct_dimensions})")
                return True
            
            print(f"🗑️  Deleting existing collection (wrong dimensions: {current_dim})...")
            client.delete_collection(collection_name)
        
        # Create new collection
        print(f"🆕 Creating collection with {correct_dimensions} dimensions...")
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=correct_dimensions,
                distance=models.Distance.COSINE
            )
        )
        
        # Create indexes
        print("🔍 Creating payload indexes...")
        indexes = [
            ("document_id", models.KeywordIndexParams()),
            ("chunk_id", models.KeywordIndexParams()),
            ("page_number", models.IntegerIndexParams()),
            ("worksheet_name", models.KeywordIndexParams()),
        ]
        
        for field_name, field_schema in indexes:
            try:
                client.create_payload_index(
                    collection_name=collection_name,
                    field_name=field_name,
                    field_schema=field_schema
                )
            except Exception as e:
                print(f"⚠️  Index creation warning for {field_name}: {e}")
        
        print("✅ Collection fixed successfully!")
        return True
    
    except Exception as e:
        print(f"❌ Failed to fix collection: {e}")
        return False

def update_gradio_demo():
    """Update the Gradio demo with correct vector dimensions."""
    print("\n📝 Updating Gradio Demo")
    print("=" * 25)
    
    # Check if the demo file exists
    demo_files = [
        'fixed_gradio_demo.py',
        'fixed_gradio_file_handling.py',
        'gradio_demo.py'
    ]
    
    demo_file = None
    for file in demo_files:
        if os.path.exists(file):
            demo_file = file
            break
    
    if not demo_file:
        print("❌ No Gradio demo file found")
        print("Please create fixed_gradio_demo.py with the corrected code")
        return False
    
    try:
        # Read the file
        with open(demo_file, 'r') as f:
            content = f.read()
        
        # Update vector_size
        if "'vector_size': 1024," in content:
            content = content.replace("'vector_size': 1024,", "'vector_size': 4096,")
            print("✅ Updated vector_size from 1024 to 4096")
        elif "'vector_size': 4096," in content:
            print("✅ Vector size already correct (4096)")
        else:
            print("⚠️  Could not find vector_size in demo file")
        
        # Write back
        with open(demo_file, 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {demo_file}")
        return True
    
    except Exception as e:
        print(f"❌ Failed to update demo: {e}")
        return False

def create_quick_demo():
    """Create a quick working demo file."""
    print("\n🚀 Creating Quick Demo")
    print("=" * 22)
    
    demo_content = '''import gradio as gr
import os
from dotenv import load_dotenv
load_dotenv()

# Quick test demo
def test_apis():
    """Test API connections."""
    results = []
    
    # Test Groq
    groq_key = os.getenv('GROQ_API_KEY')
    results.append(f"Groq API Key: {'✅ Set' if groq_key else '❌ Missing'}")
    
    # Test SiliconFlow
    sf_key = os.getenv('SILICONFLOW_API_KEY')
    results.append(f"SiliconFlow API Key: {'✅ Set' if sf_key else '❌ Missing'}")
    
    # Test Qdrant
    qdrant_url = os.getenv('QDRANT_URL')
    results.append(f"Qdrant URL: {'✅ Set' if qdrant_url else '❌ Missing'}")
    
    return "\\n".join(results)

# Create simple interface
with gr.Blocks(title="RAG System Test") as demo:
    gr.Markdown("# 🧪 RAG System API Test")
    
    test_btn = gr.Button("Test APIs")
    output = gr.Textbox(label="Results", lines=10)
    
    test_btn.click(test_apis, outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
'''
    
    with open('quick_test_demo.py', 'w') as f:
        f.write(demo_content)
    
    print("✅ Created quick_test_demo.py")
    return True

def main():
    """Main fix function."""
    print("🏭 Manufacturing RAG Agent - Quick Fix")
    print("=" * 50)
    
    # Step 1: Check environment
    if not check_environment():
        print("\n❌ Environment issues found. Please fix your .env file first.")
        return
    
    # Step 2: Test SiliconFlow API and get dimensions
    dimensions = test_siliconflow_api()
    if not dimensions:
        print("\n❌ SiliconFlow API test failed. Please check your API key.")
        return
    
    # Step 3: Test Qdrant
    current_dim = test_qdrant_connection()
    if current_dim is False:
        print("\n❌ Qdrant connection failed. Please check your Qdrant configuration.")
        return
    
    # Step 4: Fix Qdrant collection if needed
    if current_dim != dimensions:
        print(f"\n⚠️  Collection needs fixing: {current_dim} → {dimensions}")
        if fix_qdrant_collection(dimensions):
            print("✅ Qdrant collection fixed!")
        else:
            print("❌ Failed to fix Qdrant collection")
            return
    
    # Step 5: Update demo file
    if update_gradio_demo():
        print("✅ Demo file updated!")
    
    # Step 6: Create quick test demo
    create_quick_demo()
    
    print("\n🎉 All fixes applied!")
    print("\n📋 Next Steps:")
    print("1. Test APIs: python quick_test_demo.py")
    print("2. Run full demo: python fixed_gradio_demo.py")
    print("3. Upload documents and test questions")

if __name__ == "__main__":
    main()