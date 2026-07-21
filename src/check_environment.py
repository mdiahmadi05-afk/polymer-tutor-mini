"""Check Python, AI packages, and CUDA GPU availability."""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

import torch


PACKAGES = [
    "numpy",
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "peft",
    "trl",
    "bitsandbytes",
    "sentencepiece",
    "safetensors",
]


def package_version(package_name: str) -> str:
    """Return the installed version of a package."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "not installed"


def main() -> None:
    print("=" * 55)
    print("Polymer Tutor Mini - Environment Check")
    print("=" * 55)

    print(f"Operating system: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]}")

    print("\nInstalled packages:")
    for package in PACKAGES:
        print(f"  {package}: {package_version(package)}")

    print("\nGPU information:")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    print(f"  PyTorch CUDA runtime: {torch.version.cuda}")

    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access an NVIDIA GPU.")

    gpu_index = torch.cuda.current_device()
    gpu_properties = torch.cuda.get_device_properties(gpu_index)

    print(f"  GPU: {torch.cuda.get_device_name(gpu_index)}")
    print(f"  GPU memory: {gpu_properties.total_memory / 1024**3:.2f} GB")

    x = torch.randn(1024, 1024, device="cuda")
    result = x @ x

    print(f"  Test tensor device: {result.device}")
    print("\nEnvironment check passed successfully.")


if __name__ == "__main__":
    main()
