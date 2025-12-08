"""
EUVlitho Windows Setup Script
Automates Python environment setup for CNN components
"""

import subprocess
import sys
import os
from pathlib import Path

def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major != 3 or version.minor < 9:
        print("⚠️  Warning: Python 3.9+ recommended. You have Python {}.{}.{}".format(
            version.major, version.minor, version.micro))
        return False
    elif version.minor > 11:
        print("⚠️  Warning: Python 3.10-3.11 recommended for best compatibility")
        return True
    else:
        print("✓ Python version is compatible")
        return True

def create_virtual_environment():
    """Create a virtual environment"""
    venv_path = Path("venv_euvlitho")
    
    if venv_path.exists():
        print(f"✓ Virtual environment already exists at {venv_path}")
        return True
    
    print("Creating virtual environment...")
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv_path)], check=True)
        print(f"✓ Virtual environment created at {venv_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to create virtual environment: {e}")
        return False

def install_dependencies():
    """Install required packages"""
    print("\n" + "="*60)
    print("Installing dependencies...")
    print("="*60 + "\n")
    
    # Determine pip path based on OS
    if os.name == 'nt':  # Windows
        pip_path = Path("venv_euvlitho/Scripts/pip.exe")
    else:
        pip_path = Path("venv_euvlitho/bin/pip")
    
    if not pip_path.exists():
        print(f"❌ pip not found at {pip_path}")
        return False
    
    # Upgrade pip first
    print("Upgrading pip...")
    subprocess.run([str(pip_path), "install", "--upgrade", "pip"], check=True)
    
    # Install PyTorch with CUDA support
    print("\nInstalling PyTorch with CUDA 12.1 support...")
    subprocess.run([
        str(pip_path), "install", "torch", "torchvision", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ], check=True)
    
    # Install other requirements
    if Path("requirements.txt").exists():
        print("\nInstalling other dependencies from requirements.txt...")
        subprocess.run([str(pip_path), "install", "-r", "requirements.txt"], check=True)
    
    print("\n✓ All dependencies installed successfully!")
    return True

def verify_installation():
    """Verify that key packages are installed correctly"""
    print("\n" + "="*60)
    print("Verifying installation...")
    print("="*60 + "\n")
    
    if os.name == 'nt':
        python_path = Path("venv_euvlitho/Scripts/python.exe")
    else:
        python_path = Path("venv_euvlitho/bin/python")
    
    test_script = """
import sys
print(f"Python: {sys.version}")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError as e:
    print(f"❌ PyTorch: {e}")

try:
    import pytorch_lightning as pl
    print(f"✓ PyTorch Lightning: {pl.__version__}")
except ImportError as e:
    print(f"❌ PyTorch Lightning: {e}")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError as e:
    print(f"❌ NumPy: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib: {matplotlib.__version__}")
except ImportError as e:
    print(f"❌ Matplotlib: {e}")

try:
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")
except ImportError as e:
    print(f"❌ Pandas: {e}")
"""
    
    result = subprocess.run(
        [str(python_path), "-c", test_script],
        capture_output=True,
        text=True
    )
    print(result.stdout)
    if result.stderr:
        print("Errors:", result.stderr)
    
    return result.returncode == 0

def create_activation_script():
    """Create convenience activation scripts"""
    if os.name == 'nt':
        # Windows batch file
        with open("activate.bat", "w") as f:
            f.write("@echo off\n")
            f.write("call venv_euvlitho\\Scripts\\activate.bat\n")
            f.write("echo ✓ EUVlitho environment activated!\n")
        
        # Windows PowerShell script
        with open("activate.ps1", "w") as f:
            f.write(".\\venv_euvlitho\\Scripts\\Activate.ps1\n")
            f.write("Write-Host '✓ EUVlitho environment activated!' -ForegroundColor Green\n")
        
        print("\n✓ Created activation scripts: activate.bat and activate.ps1")
    else:
        # Unix shell script
        with open("activate.sh", "w") as f:
            f.write("#!/bin/bash\n")
            f.write("source venv_euvlitho/bin/activate\n")
            f.write("echo '✓ EUVlitho environment activated!'\n")
        os.chmod("activate.sh", 0o755)
        print("\n✓ Created activation script: activate.sh")

def main():
    print("\n" + "="*60)
    print("EUVlitho Windows Setup")
    print("="*60 + "\n")
    
    # Check Python version
    if not check_python_version():
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # Create virtual environment
    if not create_virtual_environment():
        print("❌ Setup failed: Could not create virtual environment")
        return
    
    # Install dependencies
    try:
        if not install_dependencies():
            print("❌ Setup failed: Could not install dependencies")
            return
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return
    
    # Verify installation
    verify_installation()
    
    # Create activation scripts
    create_activation_script()
    
    print("\n" + "="*60)
    print("✓ Setup Complete!")
    print("="*60)
    print("\nTo activate the environment:")
    if os.name == 'nt':
        print("  Command Prompt: activate.bat")
        print("  PowerShell:     .\\activate.ps1")
    else:
        print("  ./activate.sh")
    print("\nNext steps:")
    print("  1. Activate the environment")
    print("  2. Run: python analyze_data.py")
    print("  3. Explore the CNN training code in cnn/model/")
    print()

if __name__ == "__main__":
    main()
