"""
CNN model architecture for handwritten digit recognition.

This module defines a convolutional neural network architecture specifically
designed to achieve ≥99% accuracy on the MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .config import MODEL_CONFIG


class DigitCNN(nn.Module):
    """
    Convolutional Neural Network for digit recognition.
    
    Architecture designed to achieve ≥99% accuracy on MNIST:
    - 3 convolutional layers with batch normalization and ReLU
    - Dropout for regularization
    - Global average pooling + fully connected layers
    - Optimized for 28x28 grayscale input
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.3):
        """
        Initialize the CNN model.
        
        Args:
            num_classes: Number of output classes (10 for digits 0-9)
            dropout_rate: Dropout rate for regularization
        """
        super(DigitCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # First convolutional block: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32, momentum=MODEL_CONFIG["batch_norm_momentum"])
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32, momentum=MODEL_CONFIG["batch_norm_momentum"])
        
        # Second convolutional block: 32 -> 64 channels
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64, momentum=MODEL_CONFIG["batch_norm_momentum"])
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64, momentum=MODEL_CONFIG["batch_norm_momentum"])
        
        # Third convolutional block: 64 -> 128 channels
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128, momentum=MODEL_CONFIG["batch_norm_momentum"])
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(128, momentum=MODEL_CONFIG["batch_norm_momentum"])
        
        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128, 256)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """
        Initialize model weights using Xavier/Glorot initialization.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First convolutional block with residual connection
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)  # 28x28 -> 14x14
        
        # Second convolutional block
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)  # 14x14 -> 7x7
        
        # Third convolutional block
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        
        # Global average pooling
        x = self.global_pool(x)  # 7x7 -> 1x1
        x = x.view(x.size(0), -1)  # Flatten
        
        # Fully connected layers with dropout
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
    
    def get_features(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract features from the last convolutional layer and final output.
        
        Args:
            x: Input tensor
            
        Returns:
            Tuple of (conv_features, final_output)
        """
        # Forward pass through convolutional layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn5(self.conv5(x)))
        conv_features = F.relu(self.bn6(self.conv6(x)))
        
        # Continue to final output
        x = self.global_pool(conv_features)
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        final_output = self.fc3(x)
        
        return conv_features, final_output


class LightweightDigitCNN(nn.Module):
    """
    Lightweight version of the CNN for faster inference.
    
    This is a smaller model that trades some accuracy for speed,
    useful for real-time applications.
    """
    
    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.2):
        """
        Initialize the lightweight CNN model.
        
        Args:
            num_classes: Number of output classes
            dropout_rate: Dropout rate for regularization
        """
        super(LightweightDigitCNN, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        
        # Simplified architecture: 2 conv blocks + 2 FC layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(64, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, num_classes)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the lightweight network.
        
        Args:
            x: Input tensor of shape (batch_size, 1, 28, 28)
            
        Returns:
            Output logits of shape (batch_size, num_classes)
        """
        # First conv block
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)  # 28x28 -> 14x14
        
        # Second conv block
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 14x14 -> 7x7
        
        # Global average pooling and flatten
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_model(model_type: str = "full", num_classes: int = 10) -> nn.Module:
    """
    Factory function to create a model.
    
    Args:
        model_type: Type of model ("full" or "lightweight")
        num_classes: Number of output classes
        
    Returns:
        Initialized model
    """
    if model_type == "full":
        return DigitCNN(num_classes=num_classes, dropout_rate=MODEL_CONFIG["dropout_rate"])
    elif model_type == "lightweight":
        return LightweightDigitCNN(num_classes=num_classes, dropout_rate=0.2)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def count_parameters(model: nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing DigitCNN model...")
    
    # Create model
    model = create_model("full")
    print(f"Model created with {count_parameters(model):,} parameters")
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 1, 28, 28)
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output logits: {output}")
    
    # Test lightweight model
    print("\nTesting LightweightDigitCNN model...")
    light_model = create_model("lightweight")
    print(f"Lightweight model created with {count_parameters(light_model):,} parameters")
    
    light_output = light_model(x)
    print(f"Lightweight output shape: {light_output.shape}")
    
    print("Model test complete!")
