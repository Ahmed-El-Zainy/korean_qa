#!/usr/bin/env python3
"""
Simple test script to run the evaluation with the correct Python interpreter.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    # Get the Python interpreter that was used to run this script
    python_exe = sys.executable
    print(f"Using Python interpreter: {python_exe}")
    
    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Working directory: {script_dir}")
    
    # Test the setup first
    print("\n" + "="*50)
    print("TESTING SETUP")
    print("="*50)
    
    try:
        result = subprocess.run([python_exe, "test_setup.py"], 
                              capture_output=True, text=True, timeout=30)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
        
        if result.returncode == 0:
            print("\n✓ Setup test passed!")
        else:
            print("\n✗ Setup test failed!")
            return
            
    except subprocess.TimeoutExpired:
        print("Setup test timed out")
        return
    except Exception as e:
        print(f"Error running setup test: {e}")
        return
    
    # If setup test passed, try running the evaluation
    print("\n" + "="*50)
    print("RUNNING EVALUATION")
    print("="*50)
    
    try:
        result = subprocess.run([python_exe, "deep_eval.py"], 
                              capture_output=True, text=True, timeout=300)
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        print(f"Return code: {result.returncode}")
        
    except subprocess.TimeoutExpired:
        print("Evaluation timed out")
    except Exception as e:
        print(f"Error running evaluation: {e}")

if __name__ == "__main__":
    main()