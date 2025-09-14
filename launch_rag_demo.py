import subprocess
import sys
from pathlib import Path

def main():
    """Launch the Streamlit demo application."""
    # Get the directory containing this script
    script_dir = Path(__file__).parent
    demo_path = script_dir / "src" / "rag_demo.py"
    
    if not demo_path.exists():
        print(f"Error: Demo file not found at {demo_path}")
        sys.exit(1)
    
    # Launch Streamlit
    try:
        print("🏭 Launching Manufacturing RAG Agent Demo...")
        print(f"📁 Demo path: {demo_path}")
        print("🌐 The demo will open in your default web browser")
        print("🛑 Press Ctrl+C to stop the demo")
        print("-" * 50)
        
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(demo_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
        
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user")
    except Exception as e:
        print(f"❌ Failed to launch demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()