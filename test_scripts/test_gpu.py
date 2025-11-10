#!/usr/bin/env python
"""
GPU availability test script for JHTDB-ML project.

Run this after restarting the container with GPU access to verify everything works.
"""

import torch
import sys

def test_gpu():
    """Test GPU availability and CUDA setup."""

    print("=" * 70)
    print("GPU DIAGNOSTICS FOR JHTDB-ML")
    print("=" * 70)

    # PyTorch version
    print(f"\n✓ PyTorch version: {torch.__version__}")

    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"\n{'✓' if cuda_available else '✗'} CUDA available: {cuda_available}")

    if not cuda_available:
        print("\n❌ GPU NOT ACCESSIBLE!")
        print("\nPossible causes:")
        print("  1. Container not started with --gpus all flag")
        print("  2. NVIDIA Container Toolkit not installed on host")
        print("  3. PyTorch installed without CUDA support")
        print("\nPlease rebuild/restart the container with GPU support.")
        sys.exit(1)

    # CUDA version
    print(f"✓ CUDA version: {torch.version.cuda}")

    # cuDNN
    print(f"✓ cuDNN enabled: {torch.backends.cudnn.enabled}")
    if torch.backends.cudnn.enabled:
        print(f"  cuDNN version: {torch.backends.cudnn.version()}")

    # Number of GPUs
    num_gpus = torch.cuda.device_count()
    print(f"\n✓ Number of GPUs detected: {num_gpus}")

    # GPU details
    for i in range(num_gpus):
        print(f"\n  GPU {i}: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"    - Compute Capability: {props.major}.{props.minor}")
        print(f"    - Total Memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    - Multi-Processors: {props.multi_processor_count}")

    # Current device
    current_device = torch.cuda.current_device()
    print(f"\n✓ Current CUDA device: {current_device}")
    print(f"  Device name: {torch.cuda.get_device_name(current_device)}")

    # Test tensor operations
    print("\n" + "=" * 70)
    print("TESTING TENSOR OPERATIONS ON GPU")
    print("=" * 70)

    try:
        # Create tensors on GPU
        x = torch.randn(1000, 1000, device='cuda')
        y = torch.randn(1000, 1000, device='cuda')

        # Matrix multiplication
        z = torch.matmul(x, y)

        print("\n✓ Successfully created tensors on GPU")
        print(f"  Tensor shape: {x.shape}")
        print(f"  Tensor device: {x.device}")
        print(f"  Matrix multiplication result shape: {z.shape}")

        # Memory info
        print(f"\n✓ GPU Memory Usage:")
        print(f"  Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"  Cached: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    except Exception as e:
        print(f"\n✗ GPU tensor operation failed: {e}")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("✅ ALL GPU TESTS PASSED!")
    print("=" * 70)
    print("\nYour environment is ready for GPU-accelerated training.")
    print("You can now run the ML pipeline with CUDA support.\n")

if __name__ == "__main__":
    test_gpu()
