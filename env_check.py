"""
Environment check script - verifies all dependencies are installed correctly
Run this to confirm the environment is set up before running PINN models
"""

import sys
import numpy as np
import matplotlib
import tensorflow as tf
import keras
import pandas as pd

print("=" * 60)
print("Environment Check")
print("=" * 60)

print(f"Python Version: {sys.version.split()[0]}")
print(f"NumPy: {np.__version__}")
print(f"Matplotlib: {matplotlib.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"TensorFlow: {tf.__version__}")
print(f"Keras: {keras.__version__}")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPUs Detected: {len(gpus)}")
for i, gpu in enumerate(gpus):
    print(f"  GPU {i}: {gpu}")

if len(gpus) == 0:
    print("  (No GPUs - CPU mode only)")

print("\n" + "=" * 60)
print("Environment check: OK ✓")
print("=" * 60)
