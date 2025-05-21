# Notebook 0: Setup and Configuration Check
# This script is intended to verify the development environment,
# check GPU accessibility, and test basic library imports and configurations.

import sys
print(f"Python Version: {sys.version}")

# Example: Check for GPU (if using PyTorch - uncomment to test)
# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"GPU count: {torch.cuda.device_count()}")
#     print(f"Current GPU: {torch.cuda.current_device()}")
#     print(f"Device name: {torch.cuda.get_device_name(torch.cuda.current_device())}")

print("\nSetup check script finished.") 