#!/usr/bin/env python3
"""Launch script for Streamlit demo."""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit demo."""
    try:
        print("🚀 Launching Korean Q&A Evaluation System (Streamlit)")
        print("=" * 60)
        print("📱 The demo will be available at:")
        print("   - Local: http://localhost:8501")
        print("=" * 60)
        
        # Run the Streamlit demo
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_demo.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ], check=True)
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Demo failed with exit code {e.returncode}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()