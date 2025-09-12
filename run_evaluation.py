import subprocess
import sys
from pathlib import Path

def run_evaluation():
    """Run the evaluation with default settings."""
    try:
        # Run the main evaluation script
        cmd = [
            sys.executable, 
            "main.py",
            "--dataset", "assets/bench_korean.csv",
            "--log-level", "INFO",
            "--verbose"
        ]
        
        print("Starting Korean Q&A Evaluation...")
        print(f"Command: {' '.join(cmd)}")
        print("-" * 60)
        
        result = subprocess.run(cmd, check=True)
        
        print("-" * 60)
        print("Evaluation completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"Evaluation failed with exit code {e.returncode}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_evaluation()