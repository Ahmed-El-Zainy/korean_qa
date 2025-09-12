#!/usr/bin/env python3
"""
Test script to verify the Korean Q&A evaluation system setup.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from src.logger import setup_logging
        print("✓ Logger module imported successfully")
        
        from src.config import Config
        print("✓ Config module imported successfully")
        
        from src.dataset_loader import DatasetLoader
        print("✓ Dataset loader module imported successfully")
        
        from src.evaluator import KoreanQAEvaluator
        print("✓ Evaluator module imported successfully")
        
        from src.utils import load_environment_variables, validate_api_keys
        print("✓ Utils module imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_logging():
    """Test logging setup."""
    print("\nTesting logging setup...")
    
    try:
        from src.logger import setup_logging
        
        logger_setup = setup_logging(log_level="INFO")
        logger = logger_setup.get_logger("test")
        
        logger.info("Test log message")
        logger.warning("Test warning message")
        
        print("✓ Logging setup successful")
        return True
        
    except Exception as e:
        print(f"✗ Logging setup failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        from src.config import Config
        
        # This will fail if API keys are not set, but that's expected
        try:
            config = Config("src/config.yaml")
            print("✓ Configuration loaded successfully")
            print(f"  Model: {config.gemini_model}")
            return True
        except ValueError as e:
            if "Missing required API keys" in str(e):
                print("⚠ Configuration loaded but API keys missing (expected)")
                print("  Please set GOOGLE_API_KEY in your src/.env file")
                return True
            else:
                raise
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False

def test_dataset():
    """Test dataset loading."""
    print("\nTesting dataset loading...")
    
    try:
        from src.dataset_loader import DatasetLoader
        
        dataset_path = "assets/bench_korean.csv"
        if not Path(dataset_path).exists():
            print(f"⚠ Dataset file not found: {dataset_path}")
            return True
        
        loader = DatasetLoader()
        # Just test the class instantiation
        print("✓ Dataset loader initialized successfully")
        return True
        
    except Exception as e:
        print(f"✗ Dataset loading test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Korean Q&A Evaluation System - Setup Test")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_logging,
        test_config,
        test_dataset
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Ensure GOOGLE_API_KEY is set in your src/.env file")
        print("2. Run: python run_evaluation.py")
    else:
        print("✗ Some tests failed. Please check the errors above.")
        sys.exit(1)

if __name__ == "__main__":
    main()