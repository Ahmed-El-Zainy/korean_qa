#!/usr/bin/env python3
"""
Simple test script to verify the Manufacturing RAG Agent setup.
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        # Core Python modules
        import json
        import sqlite3
        import hashlib
        import time
        from datetime import datetime
        print("✅ Core Python modules")
        
        # Third-party dependencies
        import pandas as pd
        import numpy as np
        import requests
        from dotenv import load_dotenv
        print("✅ Basic dependencies")
        
        # Document processing
        import fitz  # PyMuPDF
        import openpyxl
        from PIL import Image
        import pytesseract
        print("✅ Document processing libraries")
        
        # RAG system components
        from src.rag.document_processor import DocumentProcessor, DocumentChunk
        from src.rag.embedding_system import EmbeddingSystem
        from src.rag.vector_store import QdrantVectorStore
        from src.rag.groq_client import GroqClient
        from src.rag.rag_engine import RAGEngine
        from src.rag.metadata_manager import MetadataManager
        from src.rag.ingestion_pipeline import DocumentIngestionPipeline
        print("✅ RAG system components")
        
        # UI components
        import streamlit as st
        import plotly.express as px
        print("✅ UI components")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\n🧪 Testing configuration...")
    
    try:
        from src.config import Config
        
        # Test with default config
        config = Config("src/config.yaml")
        print("✅ Configuration loaded successfully")
        
        # Test RAG config access
        rag_config = config.rag_config
        print(f"✅ RAG config: {len(rag_config)} settings")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_environment():
    """Test environment setup."""
    print("\n🧪 Testing environment...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Check for .env file
    env_file = Path('.env')
    if env_file.exists():
        print("✅ .env file exists")
    else:
        print("❌ .env file missing")
        return False
    
    # Check critical environment variables
    required_vars = ['GROQ_API_KEY', 'SILICON_FLOW_API_KEY']
    optional_vars = ['QDRANT_URL', 'QDRANT_API_KEY']
    
    missing_required = []
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"❌ {var} is missing")
            missing_required.append(var)
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var} is set")
        else:
            print(f"⚠️  {var} is not set (optional)")
    
    return len(missing_required) == 0

def test_data_directories():
    """Test data directory structure."""
    print("\n🧪 Testing data directories...")
    
    directories = ['data', 'data/documents', 'logs', 'results']
    
    for directory in directories:
        dir_path = Path(directory)
        if dir_path.exists():
            print(f"✅ {directory}/ exists")
        else:
            print(f"⚠️  {directory}/ missing, creating...")
            dir_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ {directory}/ created")
    
    return True

def test_basic_functionality():
    """Test basic RAG system functionality."""
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Test document processor factory
        from src.rag.document_processor import DocumentProcessorFactory, DocumentType
        from src.rag.pdf_processor import PDFProcessor
        from src.rag.excel_processor import ExcelProcessor
        from src.rag.image_processor import ImageProcessor
        
        # Register processors
        DocumentProcessorFactory.register_processor(DocumentType.PDF, PDFProcessor)
        DocumentProcessorFactory.register_processor(DocumentType.EXCEL, ExcelProcessor)
        DocumentProcessorFactory.register_processor(DocumentType.IMAGE, ImageProcessor)
        
        supported_types = DocumentProcessorFactory.get_supported_types()
        print(f"✅ Document processors: {len(supported_types)} types supported")
        
        # Test configuration classes
        config_dict = {
            'silicon_flow_api_key': 'test_key',
            'groq_api_key': 'test_key',
            'qdrant_url': 'http://localhost:6333',
            'qdrant_collection': 'test_collection',
            'vector_size': 1024
        }
        
        # Test component initialization (without actual API calls)
        from src.rag.metadata_manager import MetadataManager
        
        # Use temporary database for testing
        import tempfile
        temp_dir = tempfile.mkdtemp()
        test_config = config_dict.copy()
        test_config['metadata_db_path'] = str(Path(temp_dir) / 'test.db')
        
        metadata_manager = MetadataManager(test_config)
        print("✅ Metadata manager initialized")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🏭 Manufacturing RAG Agent - Setup Test")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_config),
        ("Environment Test", test_environment),
        ("Data Directories Test", test_data_directories),
        ("Basic Functionality Test", test_basic_functionality),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with error: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📋 Test Results Summary:")
    
    passed = 0
    total = len(tests)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your setup is ready.")
        print("Next steps:")
        print("   1. Configure your API keys in .env")
        print("   2. Set up Qdrant (local or cloud)")
        print("   3. Run: python quick_start.py")
        print("   4. Run: python launch_rag_demo.py")
    else:
        print("\n⚠️  Some tests failed. Please address the issues above.")
        print("Common solutions:")
        print("   - Install missing dependencies: pip install -r requirements.txt")
        print("   - Create .env file: cp .env.example .env")
        print("   - Configure API keys in .env file")

if __name__ == "__main__":
    main()