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
    
    # Step 1: Upgrade pip
    print("\nStep 1: Upgrading pip...")
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "Upgrading pip"
    )
    
    # Step 2: Install PyTorch with CUDA 12.1 from PyTorch index
    print("\nStep 2: Installing PyTorch with CUDA 12.1 support...")
    
    # Use appropriate PyTorch installation based on OS
    if is_windows:
        pytorch_command = f"{sys.executable} -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121"
    elif is_linux:
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
