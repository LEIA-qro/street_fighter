import torch
import sys

def verify_gpu():
    print("-" * 50)
    print("      PYTORCH GPU DIAGNOSTIC TOOL")
    print("-" * 50)
    
    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_available}")
    
    if not cuda_available:
        print("Result: PyTorch cannot see any NVIDIA GPUs. Training will run on CPU.")
        return

    # Get number of GPUs
    device_count = torch.cuda.device_count()
    print(f"Number of GPUs detected: {device_count}")
    
    # Current Default Device
    try:
        current_device = torch.cuda.current_device()
        print(f"Current Default Device Index: {current_device}")
        print(f"Current Default Device Name: {torch.cuda.get_device_name(current_device)}")
    except Exception as e:
        print(f"Error accessing current device: {e}")
    
    print("\nListing all detected GPUs:")
    for i in range(device_count):
        try:
            props = torch.cuda.get_device_properties(i)
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
            print(f"      Total Memory: {props.total_memory / 1024**2:.0f} MB")
            print(f"      Compute Capability: {props.major}.{props.minor}")
        except Exception as e:
            print(f"  [{i}] Error reading properties: {e}")

    print("\n" + "-" * 50)
    print("HOW TO SELECT A SPECIFIC GPU:")
    print("To use a different GPU, run your command with the environment variable:")
    print("  Windows (PowerShell): $env:CUDA_VISIBLE_DEVICES=\"1\"; python src/scripts/train.py --algo ppo")
    print("  Windows (CMD): set CUDA_VISIBLE_DEVICES=1 && python src/scripts/train.py --algo ppo")
    print("-" * 50)

if __name__ == "__main__":
    verify_gpu()
