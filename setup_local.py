#!/usr/bin/env python3
"""
Local setup script for Manufacturing RAG Agent without Docker.

This script helps set up the RAG system using Qdrant Cloud or local alternatives.
"""

import os
import sys
import subprocess
from pathlib import Path
import requests
import time

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    print("\n🔍 Checking dependencies...")
    
    # Check pip packages
    required_packages = [
        'streamlit', 'qdrant-client', 'groq', 'requests',
        'pandas', 'plotly', 'PyMuPDF', 'openpyxl', 'pytesseract', 'Pillow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_').lower())
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n📦 Installing missing packages: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install'
            ] + missing_packages)
            print("✅ All packages installed successfully")
        except subprocess.CalledProcessError:
            print("❌ Failed to install packages")
            return False
    
    return True

def check_tesseract():
    """Check if Tesseract OCR is installed."""
    print("\n🔍 Checking Tesseract OCR...")
    try:
        result = subprocess.run(['tesseract', '--version'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split('\n')[0]
            print(f"✅ {version}")
            return True
    except FileNotFoundError:
        pass
    
    print("❌ Tesseract OCR not found")
    print("📥 Install with: brew install tesseract")
    return False

def setup_environment():
    """Set up environment variables."""
    print("\n⚙️ Setting up environment...")
    
    env_file = Path('.env')
    env_example = Path('.env.example')
    
    if not env_file.exists() and env_example.exists():
        # Copy example file
        with open(env_example, 'r') as f:
            content = f.read()
        
        with open(env_file, 'w') as f:
            f.write(content)
        
        print("✅ Created .env file from template")
        print("📝 Please edit .env file with your API keys:")
        print("   - GROQ_API_KEY (get from https://console.groq.com/)")
        print("   - SILICON_FLOW_API_KEY (get from https://siliconflow.cn/)")
        print("   - QDRANT_URL and QDRANT_API_KEY (if using Qdrant Cloud)")
        return False
    elif env_file.exists():
        print("✅ .env file exists")
        return True
    else:
        print("❌ No .env.example file found")
        return False

def test_qdrant_connection():
    """Test Qdrant connection."""
    print("\n🔍 Testing Qdrant connection...")
    
    # Try local Qdrant first
    try:
        response = requests.get('http://localhost:6333/health', timeout=5)
        if response.status_code == 200:
            print("✅ Local Qdrant is running")
            return True
    except requests.exceptions.RequestException:
        pass
    
    # Try Qdrant Cloud if configured
    qdrant_url = os.getenv('QDRANT_URL')
    qdrant_key = os.getenv('QDRANT_API_KEY')
    
    if qdrant_url and qdrant_key:
        try:
            headers = {'api-key': qdrant_key} if qdrant_key else {}
            response = requests.get(f"{qdrant_url}/health", headers=headers, timeout=10)
            if response.status_code == 200:
                print("✅ Qdrant Cloud connection successful")
                return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Qdrant Cloud connection failed: {e}")
    
    print("❌ No Qdrant connection available")
    print("💡 Options:")
    print("   1. Start local Qdrant: docker run -p 6333:6333 qdrant/qdrant")
    print("   2. Use Qdrant Cloud: https://cloud.qdrant.io/")
    print("   3. Download Qdrant binary: https://github.com/qdrant/qdrant/releases")
    return False

def create_data_directories():
    """Create necessary data directories."""
    print("\n📁 Creating data directories...")
    
    directories = [
        'data',
        'data/documents',
        'logs',
        'results'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ {directory}/")
    
    return True

def test_api_keys():
    """Test API key configuration."""
    print("\n🔑 Testing API keys...")
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test Groq API
    groq_key = os.getenv('GROQ_API_KEY')
    if groq_key:
        try:
            headers = {'Authorization': f'Bearer {groq_key}'}
            response = requests.get('https://api.groq.com/openai/v1/models', 
                                  headers=headers, timeout=10)
            if response.status_code == 200:
                print("✅ Groq API key is valid")
            else:
                print(f"❌ Groq API key test failed: {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"❌ Groq API connection failed: {e}")
    else:
        print("❌ GROQ_API_KEY not set")
    
    # Test Silicon Flow API
    sf_key = os.getenv('SILICON_FLOW_API_KEY')
    if sf_key:
        print("✅ Silicon Flow API key is set (cannot test without making API call)")
    else:
        print("❌ SILICON_FLOW_API_KEY not set")
    
    return groq_key and sf_key

def main():
    """Main setup function."""
    print("🏭 Manufacturing RAG Agent Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    if not check_dependencies():
        print("\n❌ Dependency check failed")
        sys.exit(1)
    
    # Check Tesseract
    tesseract_ok = check_tesseract()
    
    # Setup environment
    env_setup = setup_environment()
    
    # Create directories
    create_data_directories()
    
    if env_setup:
        # Test API keys
        api_keys_ok = test_api_keys()
        
        # Test Qdrant
        qdrant_ok = test_qdrant_connection()
        
        print("\n" + "=" * 40)
        print("📋 Setup Summary:")
        print(f"✅ Python: OK")
        print(f"✅ Dependencies: OK")
        print(f"{'✅' if tesseract_ok else '❌'} Tesseract OCR: {'OK' if tesseract_ok else 'Missing'}")
        print(f"{'✅' if api_keys_ok else '❌'} API Keys: {'OK' if api_keys_ok else 'Missing'}")
        print(f"{'✅' if qdrant_ok else '❌'} Qdrant: {'OK' if qdrant_ok else 'Not available'}")
        
        if tesseract_ok and api_keys_ok and qdrant_ok:
            print("\n🎉 Setup complete! You can now run:")
            print("   python launch_rag_demo.py")
        else:
            print("\n⚠️  Setup incomplete. Please address the issues above.")
            if not tesseract_ok:
                print("   Install Tesseract: brew install tesseract")
            if not api_keys_ok:
                print("   Configure API keys in .env file")
            if not qdrant_ok:
                print("   Set up Qdrant (see options above)")
    else:
        print("\n📝 Please configure your .env file with API keys, then run this script again.")

if __name__ == "__main__":
    main()