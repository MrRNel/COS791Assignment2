#!/usr/bin/env python
"""
Installation script for COS791 Assignment 2 - Cheetah Detection Project

This script:
1. Installs PyTorch with CUDA 12.1 support from the PyTorch index
2. Installs all remaining dependencies from requirements.txt

Usage:
    python install_dependencies.py
"""

import subprocess
import sys
import os
import platform


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {command}")
    print(f"{'='*60}\n")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=False,
            text=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: {description} failed!")
        print(f"Exit code: {e.returncode}")
        sys.exit(1)


def main():
    """Main installation process."""
    print("\n" + "="*60)
    print("COS791 Assignment 2 - Dependency Installation")
    print("="*60)
    
    # Detect OS
    is_windows = platform.system() == "Windows"
    is_linux = platform.system() == "Linux"
    is_mac = platform.system() == "Darwin"
    
    print(f"\nDetected OS: {platform.system()} ({platform.release()})")
    
    # Check if we're in the right directory
    if not os.path.exists("requirements.txt"):
        print("\nERROR: requirements.txt not found!")
        print("Please run this script from the project root directory.")
        sys.exit(1)
    
    # Step 1: Install pip if not available, then upgrade
    print("\nStep 1: Ensuring pip is available...")
    try:
        # Check if pip is available
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True, capture_output=True)
        print("pip is available, upgrading...")
        run_command(
            f"{sys.executable} -m pip install --upgrade pip",
            "Upgrading pip"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("pip is not available, installing...")
        # Install pip using ensurepip
        run_command(
            f"{sys.executable} -m ensurepip --upgrade --default-pip",
            "Installing pip"
        )
    
    # Step 2: Install PyTorch with CUDA 12.1 from PyTorch index
    print("\nStep 2: Installing PyTorch with CUDA 12.1 support...")
    
    # Use appropriate PyTorch installation based on OS
    if is_windows or is_linux:
        # Windows and Linux: use --index-url for CUDA support (no version suffix needed)
        pytorch_command = f"{sys.executable} -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121"
    elif is_mac:
        # Mac typically uses MPS backend, or CPU-only
        pytorch_command = f"{sys.executable} -m pip install torch==2.5.1 torchvision==0.20.1"
    else:
        # Default fallback
        pytorch_command = f"{sys.executable} -m pip install torch==2.5.1 torchvision==0.20.1"
    
    run_command(pytorch_command, "Installing PyTorch and TorchVision with CUDA support")
    
    # Step 3: Install remaining requirements from requirements.txt
    print("\nStep 3: Installing remaining dependencies from requirements.txt...")
    
    # On Linux/Windows, skip torch/torchvision in requirements.txt since we install them with --index-url
    if is_windows or is_linux:
        # Create a temporary requirements file without torch/torchvision
        import tempfile
        temp_requirements = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        with open('requirements.txt', 'r') as f:
            for line in f:
                if not line.strip().startswith('torch==') and not line.strip().startswith('torchvision=='):
                    temp_requirements.write(line)
        temp_requirements.close()
        
        run_command(
            f"{sys.executable} -m pip install -r {temp_requirements.name}",
            "Installing dependencies from requirements.txt"
        )
        
        # Clean up temp file
        import os
        os.unlink(temp_requirements.name)
    else:
        # Mac or other: use requirements.txt as-is
        run_command(
            f"{sys.executable} -m pip install -r requirements.txt",
            "Installing dependencies from requirements.txt"
        )
    
    # Step 4: Verify installation
    print("\nStep 4: Verifying installation...")
    try:
        import torch
        import ultralytics
        print(f"\nSUCCESS: PyTorch version: {torch.__version__}")
        print(f"SUCCESS: CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            try:
                cuda_version = torch.version.cuda  # type: ignore
                print(f"SUCCESS: CUDA version: {cuda_version}")
            except AttributeError:
                print("SUCCESS: CUDA is available")
            print(f"SUCCESS: CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"SUCCESS: Ultralytics version: {ultralytics.__version__}")
        print("\nAll dependencies installed successfully!")
    except ImportError as e:
        print(f"\nWARNING: Could not verify installation - {e}")
    
    print("\n" + "="*60)
    print("Installation Complete!")
    print("="*60)
    print("\nYou can now run your cheetah detection scripts.")
    print("\n")


if __name__ == "__main__":
    main()
