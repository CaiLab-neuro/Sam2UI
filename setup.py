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
import shutil
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
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"ERROR: Python {version.major}.{version.minor} detected.")
        print("   SAM2 requires Python 3.10 or higher.")
        return False
    print(f"OK: Python {version.major}.{version.minor}.{version.micro}")
    return True

def install_packages():
    """Install required packages"""
    print("\nInstalling Python packages...")

    # Base packages for Sam2UI (compatible with SAM2 requirements)
    packages = [
        "torch>=2.5.1",
        "torchvision>=0.20.1",
        "opencv-python>=4.5.0",
        "numpy>=1.24.4",
        "Pillow>=9.4.0",
        "omegaconf>=2.1.0",
        "hydra-core>=1.3.2",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "scikit-image>=0.18.0",
        "tqdm>=4.66.1",
        "pycocotools>=2.0.2"
    ]

    
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

    dirs = ["demo/data"]

    for directory in dirs:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"OK: {directory}")

def clone_sam2_repository():
    """Clone SAM2 repository into Sam2UI/sam_models/sam2/"""
    print("\nCloning SAM2 repository...")

    sam2_dir = Path("sam_models/sam2")

    # Check if already exists with valid structure
    if sam2_dir.exists():
        # Check if it looks like a valid SAM2 installation
        sam2_package_dir = sam2_dir / "sam2"
        if sam2_package_dir.exists() and (sam2_package_dir / "__init__.py").exists():
            print("OK: SAM2 already installed (skipping clone)")
            print("   To reinstall SAM2, delete the 'sam_models/sam2' directory and re-run setup.py")
            return True
        else:
            # Directory exists but doesn't look like valid SAM2
            print("WARNING: 'sam2' directory exists but appears incomplete")
            print("\nCurrent contents of 'sam2' directory:")
            try:
                contents = list(sam2_dir.iterdir())
                if contents:
                    for item in sorted(contents)[:10]:  # Show first 10 items
                        print(f"  - {item.name}")
                    if len(contents) > 10:
                        print(f"  ... and {len(contents) - 10} more items")
                else:
                    print("  (empty directory)")
            except Exception as e:
                print(f"  (could not list contents: {e})")

            user_input = input("\nRemove and re-clone SAM2? (y/n): ").strip().lower()
            if user_input != 'y':
                print("Keeping existing directory (setup may fail)")
                return True
            print("Removing directory...")
            shutil.rmtree(sam2_dir)

    # Clone repository
    try:
        # Create sam_models directory if it doesn't exist
        Path("sam_models").mkdir(parents=True, exist_ok=True)

        subprocess.run([
            "git", "clone",
            "https://github.com/facebookresearch/sam2.git",
            "sam_models/sam2"
        ], check=True, capture_output=True, text=True)
        print("OK: SAM2 repository cloned")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: Could not clone SAM2 repository")
        print(f"Error: {e.stderr}")
        return False
    except FileNotFoundError:
        print("ERROR: git not found. Please install git first.")
        print("  Download from: https://git-scm.com/downloads")
        return False

def install_sam2_package():
    """Install SAM2 as editable package"""
    print("\nInstalling SAM2 package...")

    sam2_dir = Path("sam_models/sam2")
    if not sam2_dir.exists():
        print("ERROR: SAM2 directory not found")
        return False

    try:
        # Install SAM2 in editable mode
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "./sam_models/sam2"
        ], check=True, capture_output=True)
        print("OK: SAM2 package installed")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FAILED: Could not install SAM2 package")
        return False

def select_models_to_download():
    """Interactive model selection for download"""
    print("\n" + "=" * 50)
    print("SAM2 Model Selection")
    print("=" * 50)

    # Define all available models with sizes
    models_sam21 = [
        ("sam2.1_hiera_tiny.pt", 156, "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"),
        ("sam2.1_hiera_small.pt", 184, "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"),
        ("sam2.1_hiera_base_plus.pt", 324, "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"),
        ("sam2.1_hiera_large.pt", 898, "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"),
    ]

    models_sam2 = [
        ("sam2_hiera_tiny.pt", 156, "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"),
        ("sam2_hiera_small.pt", 184, "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"),
        ("sam2_hiera_base_plus.pt", 324, "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"),
        ("sam2_hiera_large.pt", 898, "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"),
    ]

    print("\nAvailable models:")
    print("\nSAM2.1 Models (Recommended - Released Sept 2024):")
    print("  1. Tiny   (156 MB) - Fastest, lowest quality")
    print("  2. Small  (184 MB) - Good balance")
    print("  3. Base+  (324 MB) - High quality")
    print("  4. Large  (898 MB) - Best quality, slowest")
    print("\nSAM2 Models (Legacy - Released July 2024):")
    print("  5. Tiny   (156 MB)")
    print("  6. Small  (184 MB)")
    print("  7. Base+  (324 MB)")
    print("  8. Large  (898 MB)")
    print("\nOptions:")
    print("  A. Download all SAM2.1 models (~1.5 GB)")
    print("  B. Download recommended (SAM2.1 Small + Base+) (~500 MB)")
    print("  C. Custom selection")
    print("\nYou can also specify:")
    print("  - Single model: 2")
    print("  - Multiple models: 2,3,4")
    print("  - Range: 2-4")

    choice = input("\nEnter your choice (1-8, A, B, C, or range) [default: B]: ").strip().upper()
    if not choice:
        choice = 'B'

    selected = []
    all_models = models_sam21 + models_sam2

    if choice == 'A':
        selected = models_sam21
        print(f"Selected: All SAM2.1 models")
    elif choice == 'B':
        selected = [models_sam21[1], models_sam21[2]]  # Small and Base+
        print(f"Selected: SAM2.1 Small and Base+")
    elif choice == 'C':
        print("\nEnter model numbers to download (comma-separated, e.g., 2,3):")
        selections = input("Models: ").strip().split(',')
        for sel in selections:
            try:
                idx = int(sel.strip()) - 1
                if 0 <= idx < len(all_models):
                    selected.append(all_models[idx])
            except ValueError:
                continue
    elif choice.isdigit() and 1 <= int(choice) <= 8:
        idx = int(choice) - 1
        selected = [all_models[idx]]
        model_name = all_models[idx][0]
        print(f"Selected: {model_name}")
    elif '-' in choice:
        # Handle range syntax (e.g., "2-4")
        try:
            parts = choice.split('-')
            if len(parts) == 2:
                start = int(parts[0].strip())
                end = int(parts[1].strip())
                if 1 <= start <= end <= 8:
                    for i in range(start, end + 1):
                        selected.append(all_models[i - 1])
                    model_names = [m[0] for m in selected]
                    print(f"Selected: {', '.join(model_names)}")
                else:
                    print("Invalid range. Downloading recommended models (Small + Base+)")
                    selected = [models_sam21[1], models_sam21[2]]
            else:
                print("Invalid range format. Downloading recommended models (Small + Base+)")
                selected = [models_sam21[1], models_sam21[2]]
        except ValueError:
            print("Invalid range. Downloading recommended models (Small + Base+)")
            selected = [models_sam21[1], models_sam21[2]]
    elif ',' in choice:
        # Handle comma-separated list (e.g., "2,3,4")
        try:
            selections = choice.split(',')
            for sel in selections:
                idx = int(sel.strip()) - 1
                if 0 <= idx < len(all_models):
                    selected.append(all_models[idx])
            if selected:
                model_names = [m[0] for m in selected]
                print(f"Selected: {', '.join(model_names)}")
            else:
                print("No valid models selected. Downloading recommended models (Small + Base+)")
                selected = [models_sam21[1], models_sam21[2]]
        except ValueError:
            print("Invalid input. Downloading recommended models (Small + Base+)")
            selected = [models_sam21[1], models_sam21[2]]
    else:
        print("Invalid choice. Downloading recommended models (Small + Base+)")
        selected = [models_sam21[1], models_sam21[2]]

    return selected

def download_checkpoints(selected_models):
    """Download selected model checkpoints with progress tracking"""
    print("\nDownloading model checkpoints...")

    checkpoint_dir = Path("sam_models/sam2/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    total_size = sum(size for _, size, _ in selected_models)
    print(f"\nTotal download size: {total_size} MB")

    for filename, size, url in selected_models:
        filepath = checkpoint_dir / filename

        if filepath.exists():
            print(f"OK: {filename} (already exists)")
            continue

        print(f"\nDownloading {filename} ({size} MB)...")
        try:
            # Download with progress tracking
            def progress_hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100) if total_size > 0 else 0
                print(f"\rProgress: {percent:.1f}%", end='', flush=True)

            urllib.request.urlretrieve(url, filepath, reporthook=progress_hook)
            print(f"\nOK: {filename}")
        except Exception as e:
            print(f"\nFAILED: {filename}: {e}")
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

def prompt_sam3_installation():
    """Ask user if they want to install SAM3 (optional)"""
    print("\n" + "=" * 50)
    print("SAM3 Installation (Optional)")
    print("=" * 50)
    print("\nSAM3 provides text-based prompting capabilities.")
    print("Requirements:")
    print("  - Python 3.12+")
    print("  - PyTorch 2.7+")
    print("  - HuggingFace account with SAM3 access")
    print("  - ~848M parameter model")

    choice = input("\nInstall SAM3? (y/n) [default: n]: ").strip().lower()
    if choice != 'y':
        print("Skipping SAM3 installation")
        return True

    return install_sam3()

def _display_sam3_checkpoint_instructions():
    """Display SAM3 checkpoint download instructions"""
    print("\n" + "=" * 50)
    print("SAM3 Checkpoint Download Instructions")
    print("=" * 50)
    print("\n1. REQUEST ACCESS TO CHECKPOINTS:")
    print("   Visit: https://huggingface.co/facebook/sam3")
    print("   Click 'Request Access' and wait for approval")
    print("\n2. AUTHENTICATE WITH HUGGINGFACE:")
    print("   a. Generate token at: https://huggingface.co/settings/tokens")
    print("   b. Run: huggingface-cli login")
    print("   c. Paste your token when prompted")
    print("\n3. DOWNLOAD CHECKPOINT:")
    print("   After authentication, download the checkpoint:")
    print("   ")
    print("   Method 1 - Using Python:")
    print("   python -c \"from huggingface_hub import snapshot_download; \\")
    print("       snapshot_download(repo_id='facebook/sam3', local_dir='sam_models/sam3/checkpoints')\"")
    print("   ")
    print("   Method 2 - Manual download:")
    print("   - Download sam3.pt (3.45 GB) from https://huggingface.co/facebook/sam3")
    print("   - Place in: sam_models/sam3/checkpoints/sam3.pt")
    print("\n4. VERIFY INSTALLATION:")
    print("   Run: python -c 'from sam3.model_builder import build_sam3_video_predictor; print(\"SAM3 OK\")'")
    print("\nNote: SAM3 will not work until checkpoint is downloaded")

def install_sam3():
    """Clone and install SAM3 repository"""
    print("\nCloning SAM3 repository...")

    sam3_dir = Path("sam_models/sam3")
    sam3_checkpoint = sam3_dir / "checkpoints" / "sam3.pt"

    # Check if already exists
    if sam3_dir.exists():
        if sam3_checkpoint.exists():
            print("SAM3 already installed (package and checkpoint found)")
            return True
        else:
            print("SAM3 package exists, but checkpoint missing")
            print("Skipping package installation...")
            # Skip to checkpoint instructions
            _display_sam3_checkpoint_instructions()
            return True

    try:
        # Check Python version
        if sys.version_info < (3, 12):
            print("ERROR: SAM3 requires Python 3.12+")
            print(f"Current version: {sys.version_info.major}.{sys.version_info.minor}")
            print("Please create a Python 3.12+ environment for SAM3")
            return False

        # Check PyTorch version
        try:
            import torch
            torch_version = torch.__version__.split('+')[0]
            major, minor = map(int, torch_version.split('.')[:2])
            if major < 2 or (major == 2 and minor < 7):
                print(f"ERROR: SAM3 requires PyTorch 2.7+")
                print(f"Current version: {torch_version}")
                print("Please upgrade PyTorch:")
                print("  pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
                return False

            # Check CUDA version
            if torch.cuda.is_available():
                cuda_version = torch.version.cuda
                if cuda_version:
                    cuda_major = int(cuda_version.split('.')[0])
                    cuda_minor = int(cuda_version.split('.')[1])
                    if cuda_major < 12 or (cuda_major == 12 and cuda_minor < 6):
                        print(f"WARNING: SAM3 requires CUDA 12.6+")
                        print(f"Current version: {cuda_version}")
                        print("SAM3 may not work correctly with older CUDA versions")
                        print("Consider upgrading CUDA or using CPU mode")
            else:
                print("WARNING: CUDA not available. SAM3 requires CUDA 12.6+ for optimal performance")
        except ImportError:
            print("WARNING: PyTorch not installed. Please install PyTorch 2.7+ first")
            return False

        # Create sam_models directory if it doesn't exist
        Path("sam_models").mkdir(parents=True, exist_ok=True)

        # Clone SAM3
        subprocess.run([
            "git", "clone",
            "https://github.com/facebookresearch/sam3.git",
            "sam_models/sam3"
        ], check=True, capture_output=True)
        print("OK: SAM3 repository cloned")

        # Install SAM3
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-e", "./sam_models/sam3"
        ], check=True, capture_output=True)
        print("OK: SAM3 package installed")

        # Install additional dependencies needed for SAM3
        print("Installing additional SAM3 dependencies...")
        additional_deps = [
            "einops",            # Required for model loading
            "huggingface-hub",   # Required for checkpoint download
            "decord"             # Required for video loading
        ]
        for dep in additional_deps:
            try:
                subprocess.run([
                    sys.executable, "-m", "pip", "install", dep
                ], check=True, capture_output=True)
                print(f"OK: {dep} installed")
            except subprocess.CalledProcessError:
                print(f"WARNING: Failed to install {dep}")
                print(f"You may need to manually install it: pip install {dep}")

        # Create checkpoint directory
        checkpoint_dir = Path("sam_models/sam3/checkpoints")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        print(f"OK: Created checkpoint directory: {checkpoint_dir}")

        # Display checkpoint download instructions
        _display_sam3_checkpoint_instructions()

        return True

    except subprocess.CalledProcessError:
        print(f"FAILED: Could not install SAM3")
        return False

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

        # Check SAM2 installation
        try:
            from sam2.build_sam import build_sam2_video_predictor
            print("OK: SAM2 package imported successfully")
        except ImportError as e:
            print(f"ERROR: Could not import SAM2: {e}")
            return False

        if torch.cuda.is_available():
            print(f"OK: CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("WARNING: CUDA not available - will use CPU")

        # Check files and directories
        required_paths = [
            "sam2_ui.py",
            "sam_models/sam2/",
            "sam_models/sam2/sam2/",
            "sam_models/sam2/checkpoints/"
        ]

        for path in required_paths:
            if Path(path).exists():
                print(f"OK: {path}")
            else:
                print(f"MISSING: {path}")
                return False

        # Check at least one checkpoint exists
        checkpoint_dir = Path("sam_models/sam2/checkpoints")
        checkpoints = list(checkpoint_dir.glob("*.pt"))
        if checkpoints:
            print(f"OK: Found {len(checkpoints)} checkpoint(s)")
        else:
            print("WARNING: No checkpoints found")
            return False

        # Optional: Check SAM3
        if Path("sam_models/sam3").exists():
            print("OK: SAM3 installed (optional)")
            # Check SAM3 dependencies
            try:
                import einops
                print("OK: SAM3 dependencies (einops) available")
            except ImportError:
                print("WARNING: einops not found - SAM3 may not work")
                print("  Install with: pip install einops")

        return True

    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False

def main():
    """Main setup function"""
    print_header()

    # Step 1: Check Python version
    if not check_python():
        return False

    # Step 2: Install base packages
    if not install_packages():
        print("\nERROR: Package installation failed")
        return False

    # Step 3: Create directories
    create_directories()

    # Step 4: Clone SAM2 repository
    if not clone_sam2_repository():
        print("\nERROR: SAM2 repository cloning failed")
        return False

    # Step 5: Install SAM2 package
    if not install_sam2_package():
        print("\nERROR: SAM2 package installation failed")
        return False

    # Step 6: Select and download model checkpoints
    selected_models = select_models_to_download()
    if not download_checkpoints(selected_models):
        print("\nERROR: Model download failed")
        return False

    # Step 7: Optional SAM3 installation
    if not prompt_sam3_installation():
        print("\nWARNING: SAM3 installation failed (continuing anyway)")

    # Step 8: Create launcher
    create_launcher()

    # Step 9: Verify setup
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