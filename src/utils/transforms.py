"""
Data augmentation transforms for training.

This module provides various data augmentation techniques to improve
model generalization and robustness.
"""

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from typing import Tuple, Optional
import random


class ElasticTransform:
    """
    Elastic deformation transform for data augmentation.
    
    This transform applies elastic deformation to images, which is particularly
    effective for handwritten digits as it simulates natural variations in
    handwriting style.
    """
    
    def __init__(self, alpha: float = 1.0, sigma: float = 50.0, 
                 alpha_affine: float = 50.0):
        """
        Initialize elastic transform.
        
        Args:
            alpha: Elastic deformation strength
            sigma: Elastic deformation smoothness
            alpha_affine: Affine transformation strength
        """
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply elastic transform to image.
        
        Args:
            img: Input image tensor (C, H, W)
            
        Returns:
            Transformed image tensor
        """
        if img.dim() != 3:
            raise ValueError("Input must be 3D tensor (C, H, W)")
        
        # Get image dimensions
        _, h, w = img.shape
        
        # Generate random affine transformation
        angle = random.uniform(-10, 10)
        translate_x = random.uniform(-2, 2)
        translate_y = random.uniform(-2, 2)
        scale = random.uniform(0.9, 1.1)
        
        # Create affine transformation matrix
        theta = torch.tensor([
            [scale * np.cos(np.radians(angle)), -scale * np.sin(np.radians(angle)), translate_x],
            [scale * np.sin(np.radians(angle)), scale * np.cos(np.radians(angle)), translate_y]
        ], dtype=torch.float32)
        
        # Apply affine transformation
        grid = F.affine_grid(theta.unsqueeze(0), img.unsqueeze(0).size(), 
                           align_corners=False)
        img = F.grid_sample(img.unsqueeze(0), grid, align_corners=False).squeeze(0)
        
        # Generate random displacement field for elastic deformation
        dx = torch.randn(h, w) * self.alpha
        dy = torch.randn(h, w) * self.alpha
        
        # Smooth the displacement field
        dx = F.avg_pool2d(dx.unsqueeze(0).unsqueeze(0), 
                          kernel_size=3, stride=1, padding=1).squeeze()
        dy = F.avg_pool2d(dy.unsqueeze(0).unsqueeze(0), 
                          kernel_size=3, stride=1, padding=1).squeeze()
        
        # Normalize displacement field
        dx = dx / dx.max() * self.alpha
        dy = dy / dy.max() * self.alpha
        
        # Create coordinate grid
        x, y = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        x = x.float() + dx
        y = y.float() + dy
        
        # Normalize coordinates to [-1, 1]
        x = 2 * x / (h - 1) - 1
        y = 2 * y / (w - 1) - 1
        
        # Stack coordinates
        grid = torch.stack([y, x], dim=-1).unsqueeze(0)
        
        # Apply elastic transformation
        img = F.grid_sample(img.unsqueeze(0), grid, align_corners=False, 
                           mode='bilinear', padding_mode='zeros').squeeze(0)
        
        return img


class RandomNoise:
    """
    Add random noise to images for data augmentation.
    """
    
    def __init__(self, std: float = 0.05):
        """
        Initialize random noise transform.
        
        Args:
            std: Standard deviation of Gaussian noise
        """
        self.std = std
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Add random noise to image.
        
        Args:
            img: Input image tensor
            
        Returns:
            Image with added noise
        """
        noise = torch.randn_like(img) * self.std
        img = img + noise
        return torch.clamp(img, 0, 1)


def get_training_transforms(augment: bool = True) -> transforms.Compose:
    """
    Get training transforms with optional augmentation.
    
    Args:
        augment: Whether to apply data augmentation
        
    Returns:
        Composition of transforms
    """
    if augment:
        return transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=(-10, 10)
            ),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ElasticTransform(alpha=1.0, sigma=50.0),
            RandomNoise(std=0.05)
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])


def get_validation_transforms() -> transforms.Compose:
    """
    Get validation transforms (no augmentation).
    
    Returns:
        Composition of transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


def get_test_transforms() -> transforms.Compose:
    """
    Get test transforms (no augmentation).
    
    Returns:
        Composition of transforms
    """
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


class CustomRandomCrop:
    """
    Custom random crop that maintains aspect ratio and centers the digit.
    """
    
    def __init__(self, size: int, padding: int = 4):
        """
        Initialize custom random crop.
        
        Args:
            size: Target size
            padding: Padding to add before cropping
        """
        self.size = size
        self.padding = padding
    
    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        """
        Apply custom random crop.
        
        Args:
            img: Input image tensor
            
        Returns:
            Cropped image tensor
        """
        # Add padding
        if self.padding > 0:
            img = F.pad(img, (self.padding, self.padding, self.padding, self.padding), 
                       mode='constant', value=0)
        
        # Get dimensions
        _, h, w = img.shape
        
        # Calculate crop boundaries
        th, tw = self.size, self.size
        
        if h == th and w == tw:
            return img
        
        # Center crop with small random offset
        i = (h - th) // 2 + random.randint(-2, 2)
        j = (w - tw) // 2 + random.randint(-2, 2)
        
        # Ensure boundaries are valid
        i = max(0, min(i, h - th))
        j = max(0, min(j, w - tw))
        
        return img[:, i:i+th, j:j+tw]
