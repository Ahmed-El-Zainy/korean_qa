#!/usr/bin/env python3
"""
Demo launcher for Korean Q&A Evaluation System.
Choose between Gradio and Streamlit interfaces.
"""

import sys
import subprocess
import os
from pathlib import Path

def print_banner():
    """Print welcome banner."""
    print("=" * 70)
    print("🇰🇷 Korean Q&A Evaluation System - Demo Launcher")
    print("=" * 70)
    print()

def print_options():
    """Print available demo options."""
    print("📱 Available Demo Interfaces:")
    print()
    print("1. 🎯 Gradio Demo (Recommended)")
    print("   - Interactive web interface")
    print("   - Real-time evaluation")
    print("   - Shareable public links")
    print("   - Best for: Quick testing and sharing")
    print()
    print("2. 📊 Streamlit Dashboard")
    print("   - Professional analytics interface")
    print("   - Advanced visualizations")
    print("   - Detailed metrics analysis")
    print("   - Best for: In-depth analysis")
    print()
    print("3. 🖥️  Command Line Interface")
    print("   - Traditional CLI evaluation")
    print("   - Batch processing")
    print("   - Automated workflows")
    print("   - Best for: Production use")
    print()
    print("4. 🧪 Test Setup")
    print("   - Verify system configuration")
    print("   - Check dependencies")
    print("   - Validate API keys")
    print()
    print("0. ❌ Exit")
    print()

def launch_gradio():
    """Launch Gradio demo."""
    print("🚀 Launching Gradio Demo...")
    print("📱 Will be available at: http://localhost:7860")
    print("🌐 Public link will be generated for sharing")
    print()
    try:
        subprocess.run([sys.executable, "gradio_demo.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Gradio demo stopped")
    except Exception as e:
        print(f"❌ Error launching Gradio: {e}")

def launch_streamlit():
    """Launch Streamlit demo."""
    print("🚀 Launching Streamlit Dashboard...")
    print("📱 Will be available at: http://localhost:8501")
    print()
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", "streamlit_demo.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0",
            "--browser.gatherUsageStats", "false"
        ], check=True)
    except KeyboardInterrupt:
        print("\n👋 Streamlit demo stopped")
    except Exception as e:
        print(f"❌ Error launching Streamlit: {e}")

def launch_cli():
    """Launch CLI interface."""
    print("🖥️  Command Line Interface Options:")
    print()
    print("1. Quick evaluation with default settings")
    print("2. Custom evaluation with parameters")
    print("3. Back to main menu")
    print()
    
    choice = input("Select option (1-3): ").strip()
    
    if choice == "1":
        print("🚀 Running quick evaluation...")
        try:
            subprocess.run([sys.executable, "run_evaluation.py"], check=True)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == "2":
        print("📝 Custom Evaluation Parameters:")
        dataset = input("Dataset path (default: assets/bench_korean.csv): ").strip()
        if not dataset:
            dataset = "assets/bench_korean.csv"
        
        threshold = input("Threshold (default: 0.8): ").strip()
        if not threshold:
            threshold = "0.8"
        
        log_level = input("Log level (DEBUG/INFO/WARNING/ERROR, default: INFO): ").strip()
        if not log_level:
            log_level = "INFO"
        
        verbose = input("Verbose mode? (y/n, default: y): ").strip().lower()
        verbose_flag = "--verbose" if verbose != "n" else ""
        
        cmd = [
            sys.executable, "main.py",
            "--dataset", dataset,
            "--threshold", threshold,
            "--log-level", log_level
        ]
        if verbose_flag:
            cmd.append(verbose_flag)
        
        print(f"🚀 Running: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, check=True)
        except Exception as e:
            print(f"❌ Error: {e}")
    
    elif choice == "3":
        return
    else:
        print("❌ Invalid option")

def test_setup():
    """Run setup test."""
    print("🧪 Testing system setup...")
    try:
        subprocess.run([sys.executable, "test_setup.py"], check=True)
    except Exception as e:
        print(f"❌ Setup test failed: {e}")

def check_dependencies():
    """Check if required packages are installed."""
    required_packages = [
        "gradio", "streamlit", "plotly", "deepeval", 
        "pandas"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("⚠️  Missing required packages:")
        for package in missing_packages:
            print(f"   - {package}")
        print()
        print("📦 Install missing packages with:")
        print("   pip install -r requirements.txt")
        print()
        return False
    
    return True

def main():
    """Main demo launcher."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    while True:
        print_options()
        choice = input("Select an option (0-4): ").strip()
        
        if choice == "1":
            launch_gradio()
        elif choice == "2":
            launch_streamlit()
        elif choice == "3":
            launch_cli()
        elif choice == "4":
            test_setup()
        elif choice == "0":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid option. Please try again.")
        
        print("\n" + "=" * 50 + "\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n👋 Demo launcher stopped")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)