"""Check the local training environment and GPU availability."""

import platform
import sys

import accelerate
import bitsandbytes
import datasets
import peft
import torch
import transformers
import trl


def main() -> None:
    """Print package, Python, and GPU information."""
    print("=" * 50)
    print("Polymer Tutor Mini - Environment Check")
    print("=" * 50)

    print(f"Operating system: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Transformers: {transformers.__version__}")
    print(f"Datasets: {datasets.__version__}")
    print(f"Accelerate: {accelerate.__version__}")
    print(f"PEFT: {peft.__version__}")
    print(f"TRL: {trl.__version__}")
    print(f"BitsAndBytes: {bitsandbytes.__version__}")

    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if not cuda_available:
        raise RuntimeError("CUDA GPU was not detected by PyTorch.")

    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    total_memory = torch.cuda.get_device_properties(0).total_memory
    print(f"GPU memory: {total_memory / 1024**3:.2f} GB")

    test_tensor = torch.randn(1024, 1024, device="cuda")
    result = test_tensor @ test_tensor

    print(f"Test tensor device: {result.device}")
    print("Environment check passed successfully.")


if __name__ == "__main__":
    main()
