#!/usr/bin/env python3
"""Launch script for Gradio demo."""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Launch the Gradio demo."""
    try:
        # Set environment variables for better performance
        os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
        
        print("🚀 Launching Korean Q&A Evaluation System (Gradio)")
        print("=" * 60)
        print("📱 The demo will be available at:")
        print("   - Local: http://localhost:7860")
        print("   - Public: A shareable link will be generated")
        print("=" * 60)
        
        # Run the Gradio demo
        subprocess.run([sys.executable, "gradio_demo.py"], check=True)
        
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