"""
Evaluation metrics and visualization utilities.

This module provides functions for calculating accuracy, confusion matrix,
and generating visualizations for model evaluation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from typing import List, Tuple, Optional
import torch


def calculate_accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate accuracy for predictions.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        
    Returns:
        Accuracy as a float between 0 and 1
    """
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)
    
    correct = (predictions == targets).sum().item()
    total = targets.size(0)
    return correct / total


def calculate_confusion_matrix(predictions: torch.Tensor, targets: torch.Tensor) -> np.ndarray:
    """
    Calculate confusion matrix for predictions.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        
    Returns:
        Confusion matrix as numpy array
    """
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)
    
    return confusion_matrix(targets.cpu().numpy(), predictions.cpu().numpy())


def plot_confusion_matrix(conf_matrix: np.ndarray, 
                         class_names: List[str] = None,
                         save_path: Optional[str] = None,
                         title: str = "Confusion Matrix") -> None:
    """
    Plot and optionally save confusion matrix.
    
    Args:
        conf_matrix: Confusion matrix array
        class_names: List of class names (digits 0-9)
        save_path: Path to save the plot (optional)
        title: Plot title
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()


def plot_training_curves(train_losses: List[float], 
                        train_accuracies: List[float],
                        val_losses: List[float] = None,
                        val_accuracies: List[float] = None,
                        save_path: Optional[str] = None) -> None:
    """
    Plot training curves for loss and accuracy.
    
    Args:
        train_losses: List of training losses per epoch
        train_accuracies: List of training accuracies per epoch
        val_losses: List of validation losses per epoch (optional)
        val_accuracies: List of validation accuracies per epoch (optional)
        save_path: Path to save the plot (optional)
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    if val_losses:
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracies
    ax2.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    if val_accuracies:
        ax2.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def print_classification_report(predictions: torch.Tensor, 
                              targets: torch.Tensor,
                              class_names: List[str] = None) -> None:
    """
    Print detailed classification report.
    
    Args:
        predictions: Model predictions (logits or probabilities)
        targets: Ground truth labels
        class_names: List of class names (digits 0-9)
    """
    if predictions.dim() > 1:
        predictions = torch.argmax(predictions, dim=1)
    
    if class_names is None:
        class_names = [str(i) for i in range(10)]
    
    report = classification_report(
        targets.cpu().numpy(), 
        predictions.cpu().numpy(),
        target_names=class_names,
        digits=4
    )
    print("Classification Report:")
    print(report)


def evaluate_model(model: torch.nn.Module, 
                  test_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> Tuple[float, np.ndarray]:
    """
    Evaluate model on test dataset.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run evaluation on
        
    Returns:
        Tuple of (test_accuracy, confusion_matrix)
    """
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = torch.argmax(output, dim=1)
            
            all_predictions.append(predictions)
            all_targets.append(target)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Calculate metrics
    accuracy = calculate_accuracy(all_predictions, all_targets)
    conf_matrix = calculate_confusion_matrix(all_predictions, all_targets)
    
    return accuracy, conf_matrix
