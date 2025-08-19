"""
Configuration settings for the Handwritten Digit Predictor.

This module contains all the configurable parameters for training,
model architecture, and inference.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_DIR = PROJECT_ROOT / "data"

# Ensure directories exist
MODELS_DIR.mkdir(exist_ok=True)
ASSETS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)

# Model file paths
BEST_MODEL_PATH = MODELS_DIR / "best_model.pt"
TRAINING_CURVE_PATH = ASSETS_DIR / "training_curve.png"
CONFUSION_MATRIX_PATH = ASSETS_DIR / "confusion_matrix.png"

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 64,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "weight_decay": 1e-4,
    "patience": 10,  # Early stopping patience
    "min_delta": 0.001,  # Minimum improvement for early stopping
    "validation_split": 0.1,
    "random_seed": 42,
    "log_interval": 100,  # Log every N batches
    "save_interval": 5,   # Save checkpoint every N epochs
}

# Model configuration
MODEL_CONFIG = {
    "input_channels": 1,  # Grayscale
    "input_size": 28,     # MNIST standard size
    "num_classes": 10,    # Digits 0-9
    "dropout_rate": 0.3,
    "batch_norm_momentum": 0.1,
}

# Data augmentation configuration
AUGMENTATION_CONFIG = {
    "rotation_range": 15,      # Degrees
    "translation_range": 2,    # Pixels
    "scale_range": (0.8, 1.2), # Scale factor range
    "elastic_alpha": 1.0,      # Elastic deformation strength
    "elastic_sigma": 50.0,     # Elastic deformation smoothness
    "noise_factor": 0.05,      # Gaussian noise standard deviation
}

# Inference configuration
INFERENCE_CONFIG = {
    "canvas_size": 280,        # GUI canvas size
    "target_size": 28,         # Target size for model input
    "threshold": 0.5,          # Binarization threshold
    "normalize_mean": 0.1307,  # MNIST normalization mean
    "normalize_std": 0.3081,   # MNIST normalization std
}

# Device configuration
DEVICE_CONFIG = {
    "use_cuda": True,          # Use GPU if available
    "cuda_device": 0,          # GPU device index
}

# Logging configuration
LOGGING_CONFIG = {
    "log_interval": 100,       # Log every N batches
    "save_interval": 5,        # Save checkpoint every N epochs
    "verbose": True,           # Print training progress
}
