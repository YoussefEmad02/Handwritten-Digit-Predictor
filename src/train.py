"""
Training script for the handwritten digit recognition model.

This script trains a CNN model on the MNIST dataset with data augmentation,
early stopping, and checkpoint saving to achieve ≥99% accuracy.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from pathlib import Path
from typing import Dict, List, Tuple
import argparse

from .config import TRAIN_CONFIG, MODEL_CONFIG, BEST_MODEL_PATH, TRAINING_CURVE_PATH
from .data import create_data_loaders
from .model import create_model, count_parameters
from .utils.seed import set_seed
from .utils.metrics import plot_training_curves


class EarlyStopping:
    """
    Early stopping implementation to prevent overfitting.
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop early.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            
        return self.early_stop


class Trainer:
    """
    Training class that handles the complete training loop.
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the trainer.
        
        Args:
            model: Model to train
            device: Device to train on
        """
        self.model = model
        self.device = device
        self.model.to(device)
        
        # Initialize optimizer and scheduler
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=TRAIN_CONFIG["learning_rate"],
            weight_decay=TRAIN_CONFIG["weight_decay"]
        )
        
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Early stopping
        self.early_stopping = EarlyStopping(
            patience=TRAIN_CONFIG["patience"],
            min_delta=TRAIN_CONFIG["min_delta"]
        )
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        print(f"Training on device: {device}")
        print(f"Model parameters: {count_parameters(model):,}")
    
    def train_epoch(self, train_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            output = self.model(data)
            loss = self.criterion(output, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
            # Log progress
            if batch_idx % TRAIN_CONFIG["log_interval"] == 0:
                print(f'Batch {batch_idx}/{len(train_loader)}, '
                      f'Loss: {loss.item():.4f}')
        
        avg_loss = running_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Validate for one epoch.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Tuple of (average_loss, accuracy)
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                
                # Forward pass
                output = self.model(data)
                loss = self.criterion(output, target)
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
        
        avg_loss = running_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """
        Save model checkpoint.
        
        Args:
            epoch: Current epoch number
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': min(self.val_losses) if self.val_losses else float('inf')
        }
        
        # Save regular checkpoint
        checkpoint_path = BEST_MODEL_PATH.parent / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            torch.save(checkpoint, BEST_MODEL_PATH)
            print(f"New best model saved with validation loss: {checkpoint['best_val_loss']:.4f}")
    
    def train(self, train_loader: torch.utils.data.DataLoader, 
              val_loader: torch.utils.data.DataLoader) -> Dict[str, List[float]]:
        """
        Complete training loop.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            
        Returns:
            Dictionary containing training history
        """
        print("Starting training...")
        start_time = time.time()
        
        best_val_loss = float('inf')
        
        for epoch in range(1, TRAIN_CONFIG["num_epochs"] + 1):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validation phase
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            # Update learning rate
            self.scheduler.step(val_loss)
            
            # Record history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # Check if this is the best model
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
            
            # Save checkpoint
            if epoch % TRAIN_CONFIG["save_interval"] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start_time
            print(f'Epoch {epoch}/{TRAIN_CONFIG["num_epochs"]} '
                  f'({epoch_time:.2f}s):')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            print(f'  Learning Rate: {self.optimizer.param_groups[0]["lr"]:.6f}')
            
            # Check early stopping
            if self.early_stopping(val_loss):
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Best validation accuracy: {max(self.val_accuracies):.4f}")
        
        # Plot training curves
        plot_training_curves(
            self.train_losses,
            self.train_accuracies,
            self.val_losses,
            self.val_accuracies,
            save_path=TRAINING_CURVE_PATH
        )
        
        return {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies
        }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train digit recognition model')
    parser.add_argument('--model-type', type=str, default='full',
                       choices=['full', 'lightweight'],
                       help='Type of model to train')
    parser.add_argument('--batch-size', type=int, default=TRAIN_CONFIG['batch_size'],
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['num_epochs'],
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['learning_rate'],
                       help='Learning rate')
    parser.add_argument('--no-augment', action='store_true',
                       help='Disable data augmentation')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    set_seed(TRAIN_CONFIG["random_seed"])
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader = create_data_loaders(
        batch_size=args.batch_size,
        augment=not args.no_augment
    )
    
    # Create model
    print("Creating model...")
    model = create_model(model_type=args.model_type)
    
    # Create trainer
    trainer = Trainer(model, device)
    
    # Train model
    history = trainer.train(train_loader, val_loader)
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    test_loss, test_acc = trainer.validate_epoch(test_loader)
    print(f"Final test accuracy: {test_acc:.4f}")
    
    if test_acc >= 0.99:
        print("🎉 Target accuracy of 99% achieved!")
    else:
        print(f"⚠️  Target accuracy not reached. Current: {test_acc:.2%}")
    
    print(f"Training completed. Best model saved to: {BEST_MODEL_PATH}")


if __name__ == "__main__":
    main()
