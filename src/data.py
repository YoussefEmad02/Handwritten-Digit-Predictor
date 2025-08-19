"""
Data loading and preprocessing for MNIST dataset.

This module handles downloading, loading, and preprocessing of the MNIST dataset
with data augmentation for training.
"""

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Tuple, Optional
import os

from .config import TRAIN_CONFIG, DATA_DIR
from .utils.transforms import get_training_transforms, get_validation_transforms, get_test_transforms


class MNISTDataModule:
    """
    Data module for MNIST dataset with train/validation/test splits.
    """
    
    def __init__(self, batch_size: int = 64, augment: bool = True):
        """
        Initialize MNIST data module.
        
        Args:
            batch_size: Batch size for data loaders
            augment: Whether to apply data augmentation
        """
        self.batch_size = batch_size
        self.augment = augment
        self.data_dir = DATA_DIR
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Initialize transforms
        self.train_transforms = get_training_transforms(augment)
        self.val_transforms = get_validation_transforms()
        self.test_transforms = get_test_transforms()
        
        # Initialize datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Initialize data loaders
        self.train_loader = None
        self.val_loader = None
        self.test_loader = None
    
    def setup(self) -> None:
        """
        Setup datasets and data loaders.
        """
        print("Setting up MNIST datasets...")
        
        # Download and load training dataset
        self.train_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=self.train_transforms
        )
        
        # Download and load test dataset
        self.test_dataset = datasets.MNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=self.test_transforms
        )
        
        # Split training data into train and validation
        train_size = int((1 - TRAIN_CONFIG["validation_split"]) * len(self.train_dataset))
        val_size = len(self.train_dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            self.train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(TRAIN_CONFIG["random_seed"])
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        print(f"Training samples: {len(self.train_dataset)}")
        print(f"Validation samples: {len(self.val_dataset)}")
        print(f"Test samples: {len(self.test_dataset)}")
        print("Data setup complete!")
    
    def get_train_loader(self) -> DataLoader:
        """
        Get training data loader.
        
        Returns:
            Training data loader
        """
        if self.train_loader is None:
            self.setup()
        return self.train_loader
    
    def get_val_loader(self) -> DataLoader:
        """
        Get validation data loader.
        
        Returns:
            Validation data loader
        """
        if self.val_loader is None:
            self.setup()
        return self.val_loader
    
    def get_test_loader(self) -> DataLoader:
        """
        Get test data loader.
        
        Returns:
            Test data loader
        """
        if self.test_loader is None:
            self.setup()
        return self.test_loader
    
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Get all data loaders.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        if self.train_loader is None:
            self.setup()
        return self.train_loader, self.val_loader, self.test_loader
    
    def get_dataset_info(self) -> dict:
        """
        Get information about the datasets.
        
        Returns:
            Dictionary with dataset information
        """
        if self.train_dataset is None:
            self.setup()
        
        return {
            "train_samples": len(self.train_dataset),
            "val_samples": len(self.val_dataset),
            "test_samples": len(self.test_dataset),
            "num_classes": 10,
            "input_shape": (1, 28, 28),
            "batch_size": self.batch_size,
            "augmentation": self.augment
        }


def create_data_loaders(batch_size: int = 64, augment: bool = True) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Convenience function to create data loaders.
    
    Args:
        batch_size: Batch size for data loaders
        augment: Whether to apply data augmentation
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    data_module = MNISTDataModule(batch_size=batch_size, augment=augment)
    return data_module.get_data_loaders()


def get_sample_batch(data_loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get a sample batch from a data loader.
    
    Args:
        data_loader: Data loader to get sample from
        
    Returns:
        Tuple of (data, targets)
    """
    for batch in data_loader:
        return batch
    raise RuntimeError("Data loader is empty")


def visualize_batch(data_loader: DataLoader, num_samples: int = 8) -> None:
    """
    Visualize a batch of samples from the data loader.
    
    Args:
        data_loader: Data loader to visualize
        num_samples: Number of samples to show
    """
    import matplotlib.pyplot as plt
    
    data, targets = get_sample_batch(data_loader)
    
    # Denormalize data
    mean = torch.tensor([0.1307])
    std = torch.tensor([0.3081])
    data = data * std + mean
    
    # Create subplot grid
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(data))):
        img = data[i].squeeze().numpy()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Label: {targets[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Test the data module
    print("Testing MNIST data module...")
    
    # Create data loaders
    train_loader, val_loader, test_loader = create_data_loaders(batch_size=32)
    
    # Get dataset info
    data_module = MNISTDataModule()
    info = data_module.get_dataset_info()
    print(f"Dataset info: {info}")
    
    # Visualize a sample batch
    print("Visualizing sample batch...")
    visualize_batch(train_loader)
    
    print("Data module test complete!")
