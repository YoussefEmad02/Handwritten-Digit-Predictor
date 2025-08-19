#!/usr/bin/env python3
"""
Quick start script for the Handwritten Digit Predictor.

This script provides an interactive setup guide for new users.
"""

import sys
import os
from pathlib import Path


def print_banner():
    """Print the application banner."""
    print("=" * 70)
    print("🤖 Handwritten Digit Predictor - Quick Start Guide")
    print("=" * 70)
    print()
    print("This application recognizes handwritten digits (0-9) using")
    print("a deep learning model trained on the MNIST dataset.")
    print()


def check_requirements():
    """Check if basic requirements are met."""
    print("🔍 Checking system requirements...")
    
    # Check Python version
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
    
    # Check if we're in the right directory
    if not Path("requirements.txt").exists():
        print("❌ requirements.txt not found")
        print("   Please run this script from the project root directory")
        return False
    
    print("✅ Project files found")
    return True


def get_user_choice(prompt, options):
    """Get user choice from a list of options."""
    while True:
        print(f"\n{prompt}")
        for i, option in enumerate(options, 1):
            print(f"  {i}. {option}")
        
        try:
            choice = int(input("\nEnter your choice (1-{}): ".format(len(options))))
            if 1 <= choice <= len(options):
                return choice
            else:
                print("❌ Invalid choice. Please try again.")
        except ValueError:
            print("❌ Please enter a number.")


def setup_environment():
    """Guide user through environment setup."""
    print("\n🔧 Setting up the environment...")
    
    # Check if virtual environment exists
    venv_path = Path("venv")
    if venv_path.exists():
        print("✅ Virtual environment already exists")
        return True
    
    print("Creating virtual environment...")
    
    # Create virtual environment
    try:
        import subprocess
        result = subprocess.run(["python", "-m", "venv", "venv"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Virtual environment created successfully")
            return True
        else:
            print("❌ Failed to create virtual environment")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error creating virtual environment: {e}")
        return False


def install_dependencies():
    """Guide user through dependency installation."""
    print("\n📦 Installing dependencies...")
    
    # Determine the correct pip command
    if os.name == 'nt':  # Windows
        pip_cmd = "venv\\Scripts\\pip"
        activate_cmd = "venv\\Scripts\\activate"
    else:  # Unix/Linux/macOS
        pip_cmd = "venv/bin/pip"
        activate_cmd = "source venv/bin/activate"
    
    print(f"Using pip: {pip_cmd}")
    
    # Install dependencies
    try:
        import subprocess
        result = subprocess.run([pip_cmd, "install", "-r", "requirements.txt"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Dependencies installed successfully")
            return True
        else:
            print("❌ Failed to install dependencies")
            print(f"   Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Error installing dependencies: {e}")
        return False


def train_model():
    """Guide user through model training."""
    print("\n🚀 Training the model...")
    
    print("This step will:")
    print("  - Download the MNIST dataset (~11MB)")
    print("  - Train a CNN model (may take 10-30 minutes)")
    print("  - Save the trained model")
    
    choice = get_user_choice(
        "Would you like to train the model now?",
        ["Yes, train now", "Skip for now", "Show me how to train later"]
    )
    
    if choice == 1:
        print("\n🔄 Starting training...")
        print("This may take a while. You can monitor progress in the terminal.")
        
        try:
            import subprocess
            # Determine the correct python command
            if os.name == 'nt':  # Windows
                python_cmd = "venv\\Scripts\\python"
            else:  # Unix/Linux/macOS
                python_cmd = "venv/bin/python"
            
            result = subprocess.run([python_cmd, "-m", "src.train"], 
                                  capture_output=False)
            
            if result.returncode == 0:
                print("✅ Training completed successfully!")
                return True
            else:
                print("❌ Training failed")
                return False
                
        except Exception as e:
            print(f"❌ Error during training: {e}")
            return False
    
    elif choice == 2:
        print("⏭️  Skipping training for now")
        return True
    
    else:  # choice == 3
        print("\n📚 To train the model later:")
        print("1. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print("   venv\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print("   source venv/bin/activate")
        
        print("2. Run the training script:")
        print("   python -m src.train")
        return True


def run_gui():
    """Guide user through running the GUI."""
    print("\n🖥️  Running the GUI...")
    
    # Check if model exists
    model_path = Path("models/best_model.pt")
    if not model_path.exists():
        print("❌ No trained model found")
        print("   Please train the model first using: python -m src.train")
        return False
    
    print("✅ Trained model found")
    
    choice = get_user_choice(
        "Would you like to run the GUI now?",
        ["Yes, run GUI", "Show me how to run it later"]
    )
    
    if choice == 1:
        print("\n🔄 Launching GUI...")
        
        try:
            import subprocess
            # Determine the correct python command
            if os.name == 'nt':  # Windows
                python_cmd = "venv\\Scripts\\python"
            else:  # Unix/Linux/macOS
                python_cmd = "venv/bin/python"
            
            result = subprocess.run([python_cmd, "-m", "src.gui"], 
                                  capture_output=False)
            
            if result.returncode == 0:
                print("✅ GUI closed successfully")
            else:
                print("❌ GUI encountered an error")
                
        except Exception as e:
            print(f"❌ Error launching GUI: {e}")
            return False
    
    else:  # choice == 2
        print("\n📚 To run the GUI later:")
        print("1. Activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print("   venv\\Scripts\\activate")
        else:  # Unix/Linux/macOS
            print("   source venv/bin/activate")
        
        print("2. Run the GUI:")
        print("   python -m src.gui")
    
    return True


def main():
    """Main quick start function."""
    print_banner()
    
    # Check requirements
    if not check_requirements():
        print("\n❌ System requirements not met. Please fix the issues above.")
        return 1
    
    print("\n✅ All requirements met!")
    
    # Setup environment
    if not setup_environment():
        print("\n❌ Failed to setup environment.")
        return 1
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies.")
        return 1
    
    # Train model
    if not train_model():
        print("\n❌ Failed to train model.")
        return 1
    
    # Run GUI
    if not run_gui():
        print("\n❌ Failed to run GUI.")
        return 1
    
    # Success
    print("\n" + "=" * 70)
    print("🎉 Quick start completed successfully!")
    print("=" * 70)
    print("\nYour Handwritten Digit Predictor is ready to use!")
    print("\nQuick reference:")
    print("  - Activate environment: venv\\Scripts\\activate (Windows) or source venv/bin/activate (Unix)")
    print("  - Train model: python -m src.train")
    print("  - Run GUI: python -m src.gui")
    print("  - Run demo: python demo.py")
    print("  - Run tests: python run_tests.py")
    print("\nFor more information, see README.md")
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⏹️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)
