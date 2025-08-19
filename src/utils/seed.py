"""
Random seed management utilities.

This module provides functions to set random seeds for reproducible results
across different libraries (PyTorch, NumPy, Python random).
"""

import random
import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducible results.
    
    Args:
        seed: Random seed value to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior for CUDA operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"Random seed set to {seed} for reproducible results")


def get_random_state() -> dict:
    """
    Get the current random state of all libraries.
    
    Returns:
        Dictionary containing random states
    """
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    }


def restore_random_state(state: dict) -> None:
    """
    Restore random state from a previously saved state.
    
    Args:
        state: Dictionary containing random states to restore
    """
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['torch_cuda'])
