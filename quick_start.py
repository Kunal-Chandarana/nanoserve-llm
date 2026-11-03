#!/usr/bin/env python3
"""
Quick Start Script for vLLM Learning Project

This script provides a guided introduction to the vLLM learning project.
"""

import os
import sys
import subprocess
from pathlib import Path


def print_banner():
    """Print welcome banner."""
    print("🚀 Welcome to the vLLM Learning Project!")
    print("=" * 60)
    print("This project will help you learn vLLM through hands-on examples.")
    print()


def check_installation():
    """Check if vLLM is installed."""
    try:
        import vllm
        print("✅ vLLM is installed")
        return True
    except ImportError:
        print("❌ vLLM is not installed")
        return False


def show_menu():
    """Show the main menu."""
    print("\n📋 What would you like to do?")
    print("1. 🧪 Test installation")
    print("2. 📦 Download a test model")
    print("3. 🎯 Run basic inference example")
    print("4. 🌐 Start API server")
    print("5. 🌊 Try streaming example")
    print("6. 🔤 Token decoding example")
    print("7. 🔬 Enhanced API server (with token decoding)")
    print("8. 📊 Run benchmarks")
    print("9. 📚 View all examples")
    print("10. 🔧 Setup environment")
    print("11. ❓ Help")
    print("0. 🚪 Exit")
    print()


def run_test_installation():
    """Run installation test."""
    print("\n🧪 Running installation test...")
    try:
        result = subprocess.run([sys.executable, "scripts/test_installation.py"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running test: {e}")
        return False


def download_test_model():
    """Download a test model."""
    print("\n📦 Downloading test model...")
    try:
        result = subprocess.run([sys.executable, "scripts/download_models.py", "--model", "gpt2"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False


def run_example(example_file):
    """Run an example script."""
    print(f"\n🎯 Running {example_file}...")
    try:
        result = subprocess.run([sys.executable, f"examples/{example_file}"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error running example: {e}")
        return False


def show_examples():
    """Show all available examples."""
    print("\n📚 Available Examples:")
    print("=" * 40)
    
    examples = [
        ("01_basic_inference.py", "Basic vLLM usage and text generation"),
        ("02_api_server.py", "OpenAI-compatible API server"),
        ("03_streaming_example.py", "Real-time streaming responses"),
        ("04_batch_inference.py", "Efficient batch processing"),
        ("05_custom_sampling.py", "Custom sampling parameters"),
        ("06_token_decoding.py", "Token-to-text conversion techniques"),
        ("07_benchmarking.py", "Performance benchmarking"),
        ("08_enhanced_api_server.py", "API server with token decoding features"),
    ]
    
    for i, (filename, description) in enumerate(examples, 1):
        print(f"{i}. {filename}")
        print(f"   {description}")
        print()
    
    print("💡 To run an example:")
    print("   python examples/01_basic_inference.py")


def setup_environment():
    """Setup the environment."""
    print("\n🔧 Setting up environment...")
    try:
        result = subprocess.run(["bash", "scripts/setup_environment.sh"], 
                              capture_output=False, text=True)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Error setting up environment: {e}")
        return False


def show_help():
    """Show help information."""
    print("\n❓ Help & Tips")
    print("=" * 40)
    print("🎯 Getting Started:")
    print("   1. First, test your installation (option 1)")
    print("   2. Download a small model like GPT-2 (option 2)")
    print("   3. Try the basic inference example (option 3)")
    print("   4. Explore token decoding (option 6)")
    print("   5. Try the enhanced API server (option 7)")
    print()
    print("📁 Project Structure:")
    print("   examples/     - Learning examples")
    print("   scripts/      - Utility scripts")
    print("   configs/      - Configuration files")
    print("   utils/        - Helper utilities")
    print()
    print("🔧 Troubleshooting:")
    print("   - Ensure you have Python 3.9+")
    print("   - Install CUDA for GPU support")
    print("   - Use smaller models if you have limited GPU memory")
    print("   - Check README.md for detailed instructions")
    print()
    print("📚 Resources:")
    print("   - vLLM Documentation: https://docs.vllm.ai/")
    print("   - GitHub: https://github.com/vllm-project/vllm")


def main():
    """Main function."""
    print_banner()
    
    # Check if we're in the right directory
    if not os.path.exists("README.md") or not os.path.exists("examples"):
        print("❌ Please run this script from the repository root")
        sys.exit(1)
    
    # Check installation
    if not check_installation():
        print("💡 Please install vLLM first:")
        print("   pip install -r requirements.txt")
        print("   or run: bash scripts/setup_environment.sh")
        print()
    
    while True:
        show_menu()
        
        try:
            choice = input("Enter your choice (0-11): ").strip()
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        
        if choice == "0":
            print("👋 Goodbye!")
            break
        elif choice == "1":
            run_test_installation()
        elif choice == "2":
            download_test_model()
        elif choice == "3":
            run_example("01_basic_inference.py")
        elif choice == "4":
            print("\n🌐 Starting API server...")
            print("💡 Press Ctrl+C to stop the server")
            run_example("02_api_server.py")
        elif choice == "5":
            run_example("03_streaming_example.py")
        elif choice == "6":
            run_example("06_token_decoding.py")
        elif choice == "7":
            print("\n🔬 Starting Enhanced API server with token decoding...")
            print("💡 This server includes token decoding features")
            print("💡 Press Ctrl+C to stop the server")
            print("💡 After starting, try: python demo_enhanced_api.py")
            run_example("08_enhanced_api_server.py")
        elif choice == "8":
            run_example("07_benchmarking.py")
        elif choice == "9":
            show_examples()
        elif choice == "10":
            setup_environment()
        elif choice == "11":
            show_help()
        else:
            print("❌ Invalid choice. Please enter 0-11.")
        
        if choice != "0":
            input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()

