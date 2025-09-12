#!/usr/bin/env python3
"""
Diagnostic script to check the environment and file structure.
"""

import sys
import os
from pathlib import Path

def main():
    print("DIAGNOSTIC INFORMATION")
    print("="*50)
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Script location: {Path(__file__).parent}")
    
    print("\nPYTHON PATH:")
    for path in sys.path:
        print(f"  {path}")
    
    print("\nFILE STRUCTURE:")
    script_dir = Path(__file__).parent
    
    # Check main files
    files_to_check = [
        "src/config.yaml",
        "src/.env", 
        "assets/bench_korean.csv",
        "main.py",
        "deep_eval.py",
        "test_setup.py"
    ]
    
    for file_path in files_to_check:
        full_path = script_dir / file_path
        exists = "✓" if full_path.exists() else "✗"
        print(f"  {exists} {file_path}")
    
    print("\nSRC DIRECTORY CONTENTS:")
    src_dir = script_dir / "src"
    if src_dir.exists():
        for item in src_dir.iterdir():
            print(f"  {item.name}")
    else:
        print("  src directory not found")
    
    print("\nENVIRONMENT VARIABLES:")
    env_vars = ["GOOGLE_API_KEY", "OPENAI_API_KEY", "PATH", "PYTHONPATH"]
    for var in env_vars:
        value = os.getenv(var, "Not set")
        if var in ["GOOGLE_API_KEY", "OPENAI_API_KEY"] and value != "Not set":
            # Hide API keys for security
            value = f"{value[:10]}...{value[-4:]}" if len(value) > 14 else "***"
        print(f"  {var}: {value}")
    
    print("\nTRYING IMPORTS:")
    try:
        sys.path.append(str(script_dir / "src"))
        
        modules_to_test = [
            "src.logger",
            "src.config", 
            "src.utils",
            "src.dataset_loader",
            "src.evaluator"
        ]
        
        for module in modules_to_test:
            try:
                __import__(module)
                print(f"  ✓ {module}")
            except ImportError as e:
                print(f"  ✗ {module}: {e}")
                
    except Exception as e:
        print(f"  Error setting up imports: {e}")
    
    print("\nTRYING EXTERNAL DEPENDENCIES:")
    external_deps = [
        "deepeval",
        "pandas", 
        "yaml",
        "dotenv",
        "pathlib"
    ]
    
    for dep in external_deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError as e:
            print(f"  ✗ {dep}: {e}")

if __name__ == "__main__":
    main()