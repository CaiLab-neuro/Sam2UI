#!/usr/bin/env python3
"""
SAM2 Video UI Setup Script
=========================

Installs dependencies and downloads models for SAM2 Video UI.

Usage:
    python setup.py
"""

import os
import sys
import subprocess
import platform
import urllib.request
from pathlib import Path

def print_header():
    print("=" * 50)
    print("SAM2 Video UI Setup")
    print("=" * 50)
    print()

def check_python():
    """Check Python version"""
    print("Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"ERROR: Python {version.major}.{version.minor} detected.")
        print("   SAM2 requires Python 3.8 or higher.")
        return False
    print(f"OK: Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_packages():
    """Install required packages"""
    print("\nInstalling Python packages...")
    
    packages = [
        "torch>=1.9.0",
        "torchvision>=0.10.0", 
        "opencv-python>=4.5.0",
        "numpy>=1.21.0",
        "Pillow>=8.3.0",
        "omegaconf>=2.1.0",
        "hydra-core>=1.1.0",
        "timm>=0.6.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
        "tqdm>=4.62.0"
    ]
    
    # Add Windows-specific package
    if platform.system() == "Windows":
        packages.append("pycocotools>=2.0.2")
    else:
        packages.append("pycocotools>=2.0.2")
    
    for package in packages:
        print(f"Installing {package}...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", package], 
                          check=True, capture_output=True)
            print(f"OK: {package}")
        except subprocess.CalledProcessError:
            print(f"FAILED: {package}")
            return False
    
    return True

def create_directories():
    """Create required directories"""
    print("\nCreating directories...")
    
    dirs = ["checkpoints", "configs", "demo/data", "notebooks/images", "notebooks/videos"]
    
    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"OK: {directory}")

def download_models():
    """Download SAM2 model checkpoints"""
    print("\nDownloading SAM2 models...")
    
    models = {
        "sam2_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
        "sam2_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
        "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    }
    
    for filename, url in models.items():
        filepath = Path("checkpoints") / filename
        if filepath.exists():
            print(f"OK: {filename} (already exists)")
            continue
        
        print(f"Downloading {filename}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"OK: {filename}")
        except Exception as e:
            print(f"FAILED: {filename}: {e}")
            return False
    
    return True

def create_launcher():
    """Create launcher script"""
    print("\nCreating launcher...")
    
    if platform.system() == "Windows":
        launcher_content = """@echo off
echo Starting SAM2 Video UI...
python sam2_ui.py
pause
"""
        with open("run.bat", "w") as f:
            f.write(launcher_content)
        print("OK: run.bat created")
    else:
        launcher_content = """#!/bin/bash
echo "Starting SAM2 Video UI..."
python3 sam2_ui.py
"""
        with open("run.sh", "w") as f:
            f.write(launcher_content)
        os.chmod("run.sh", 0o755)
        print("OK: run.sh created")

def verify_setup():
    """Verify installation"""
    print("\nVerifying setup...")
    
    try:
        import torch
        import cv2
        import numpy as np
        from PIL import Image
        import tkinter as tk
        print("OK: All packages imported successfully")
        
        if torch.cuda.is_available():
            print(f"OK: CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA not available - will use CPU")
        
        # Check files
        required_files = ["sam2_ui.py", "checkpoints/sam2_hiera_base_plus.pt"]
        for file_path in required_files:
            if Path(file_path).exists():
                print(f"OK: {file_path}")
            else:
                print(f"MISSING: {file_path}")
                return False
        
        return True
        
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False

def main():
    """Main setup function"""
    print_header()
    
    if not check_python():
        return False
    
    if not install_packages():
        print("\nERROR: Package installation failed")
        return False
    
    create_directories()
    
    if not download_models():
        print("\nERROR: Model download failed")
        return False
    
    create_launcher()
    
    if not verify_setup():
        print("\nERROR: Setup verification failed")
        return False
    
    print("\n" + "=" * 50)
    print("SETUP COMPLETE!")
    print("=" * 50)
    print("\nTo run the SAM2 Video UI:")
    if platform.system() == "Windows":
        print("  Double-click run.bat")
        print("  OR run: python sam2_ui.py")
    else:
        print("  Run: ./run.sh")
        print("  OR run: python3 sam2_ui.py")
    print()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\nSetup interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\nSetup failed: {e}")
        sys.exit(1)
