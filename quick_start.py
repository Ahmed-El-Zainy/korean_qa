#!/usr/bin/env python3
"""
Quick start script for Manufacturing RAG Agent.

This script provides a simple way to test the RAG system without the full Streamlit interface.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_setup():
    """Check if the system is properly set up."""
    required_keys = ['GROQ_API_KEY', 'SILICONFLOW_API_KEY']
    missing_keys = []
    
    for key in required_keys:
        if not os.getenv(key):
            missing_keys.append(key)
    
    if missing_keys:
        print(f"❌ Missing required environment variables: {', '.join(missing_keys)}")
        print("📝 Please set them in your .env file")
        return False
    
    return True

def test_document_processing():
    """Test document processing with a simple example."""
    print("🧪 Testing document processing...")
    
    try:
        from src.document_processor import DocumentProcessorFactory
        from src.pdf_processor import PDFProcessor
        from src.excel_processor import ExcelProcessor
        from src.image_processor import ImageProcessor
        
        # Test processor factory
        config = {
            'max_file_size_mb': 10,
            'image_processing': True,
            'table_extraction': True
        }
        
        # Register processors
        from src.document_processor import DocumentType
        DocumentProcessorFactory.register_processor(DocumentType.PDF, PDFProcessor)
        DocumentProcessorFactory.register_processor(DocumentType.EXCEL, ExcelProcessor)
        DocumentProcessorFactory.register_processor(DocumentType.IMAGE, ImageProcessor)
        
        print("✅ Document processors initialized")
        return True
        
    except Exception as e:
        print(f"❌ Document processing test failed: {e}")
        return False

def test_embedding_system():
    """Test embedding system."""
    print("🧪 Testing embedding system...")
    
    try:
        from src.embedding_system import EmbeddingSystem
        
        config = {
            'siliconflow_api_key': os.getenv('SILICONFLOW_API_KEY'),
            'embedding_model': 'BAAI/bge-large-zh-v1.5',
            'reranker_model': 'BAAI/bge-reranker-large',
            'batch_size': 2,
            'max_retries': 2,
            'enable_embedding_cache': True
        }
        
        embedding_system = EmbeddingSystem(config)
        print("✅ Embedding system initialized")
        
        # Test with simple text
        test_texts = ["Manufacturing process efficiency", "Quality control metrics"]
        print("🔄 Generating test embeddings...")
        
        # Note: This will make an actual API call
        embeddings = embedding_system.generate_embeddings(test_texts)
        
        if embeddings and len(embeddings) == 2:
            print(f"✅ Generated embeddings: {len(embeddings)} vectors of size {len(embeddings[0])}")
            return True
        else:
            print("❌ Embedding generation failed")
            return False
            
    except Exception as e:
        print(f"❌ Embedding system test failed: {e}")
        return False

def test_llm_system():
    """Test LLM system."""
    print("🧪 Testing LLM system...")
    
    try:
        from src.groq_client import LLMSystem
        
        config = {
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            'llm_model': 'llama-3.1-8b-instant',
            'max_retries': 2
        }
        
        llm_system = LLMSystem(config)
        print("✅ LLM system initialized")
        
        # Test with simple question
        print("🔄 Testing question answering...")
        context = "The manufacturing yield for Q3 was 95.2% with a total production of 10,000 units."
        question = "What was the manufacturing yield for Q3?"
        
        # Note: This will make an actual API call
        answer = llm_system.answer_question(question, context)
        
        if answer and "95.2%" in answer:
            print(f"✅ LLM response: {answer[:100]}...")
            return True
        else:
            print(f"❌ Unexpected LLM response: {answer}")
            return False
            
    except Exception as e:
        print(f"❌ LLM system test failed: {e}")
        return False

def test_qdrant_connection():
    """Test Qdrant connection."""
    print("🧪 Testing Qdrant connection...")
    
    try:
        from src.vector_store import QdrantVectorStore
        
        config = {
            'qdrant_url': os.getenv('QDRANT_URL', 'http://localhost:6333'),
            'qdrant_api_key': os.getenv('QDRANT_API_KEY'),
            'qdrant_collection': 'test_collection',
            'vector_size': 1024
        }
        
        vector_store = QdrantVectorStore(config)
        
        # Test health check
        if vector_store.health_check():
            print("✅ Qdrant connection successful")
            
            # Get collection info
            info = vector_store.get_collection_info()
            if info:
                print(f"✅ Collection info: {info.total_points} points")
            
            return True
        else:
            print("❌ Qdrant health check failed")
            return False
            
    except Exception as e:
        print(f"❌ Qdrant connection test failed: {e}")
        return False

def run_simple_demo():
    """Run a simple demo of the RAG system."""
    print("\n🚀 Running Simple RAG Demo")
    print("=" * 40)
    
    try:
        # Initialize components
        config = {
            'siliconflow_api_key': os.getenv('SILICONFLOW_API_KEY'),
            'groq_api_key': os.getenv('GROQ_API_KEY'),
            'qdrant_url': os.getenv('QDRANT_URL', 'http://localhost:6333'),
            'qdrant_api_key': os.getenv('QDRANT_API_KEY'),
            'qdrant_collection': 'demo_collection',
            'embedding_model': 'BAAI/bge-large-zh-v1.5',
            'reranker_model': 'BAAI/bge-reranker-large',
            'llm_model': 'llama-3.1-8b-instant',
            'max_context_chunks': 3,
            'similarity_threshold': 0.7,
            'rerank_top_k': 10,
            'final_top_k': 3,
            'vector_size': 1024
        }
        
        print("🔄 Initializing RAG engine...")
        from src.rag_engine import RAGEngine
        rag_engine = RAGEngine(config)
        
        print("✅ RAG engine initialized successfully!")
        print("\n💡 The system is ready. You can now:")
        print("   1. Run the full demo: python launch_rag_demo.py")
        print("   2. Upload documents and ask questions")
        print("   3. View analytics and system status")
        
        return True
        
    except Exception as e:
        print(f"❌ RAG demo failed: {e}")
        return False

def main():
    """Main function."""
    print("🏭 Manufacturing RAG Agent - Quick Start")
    print("=" * 50)
    
    # Check setup
    if not check_setup():
        print("\n📝 Setup Instructions:")
        print("1. Copy .env.example to .env")
        print("2. Add your API keys:")
        print("   - GROQ_API_KEY from https://console.groq.com/")
        print("   - SILICONFLOW_API_KEY from https://siliconflow.cn/")
        print("3. Set up Qdrant:")
        print("   - Local: docker run -p 6333:6333 qdrant/qdrant")
        print("   - Cloud: https://cloud.qdrant.io/")
        return
    
    print("✅ Environment variables configured")
    
    # Run tests
    tests = [
        ("Document Processing", test_document_processing),
        ("Qdrant Connection", test_qdrant_connection),
        ("Embedding System", test_embedding_system),
        ("LLM System", test_llm_system),
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
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"   {test_name}: {status}")
    
    # Run demo if all critical tests pass
    critical_tests = ["Qdrant Connection", "Embedding System", "LLM System"]
    if all(results.get(test, False) for test in critical_tests):
        run_simple_demo()
    else:
        print("\n⚠️  Some critical tests failed. Please fix the issues above.")
        print("💡 Common solutions:")
        print("   - Check your API keys in .env file")
        print("   - Ensure Qdrant is running (local or cloud)")
        print("   - Install missing dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()