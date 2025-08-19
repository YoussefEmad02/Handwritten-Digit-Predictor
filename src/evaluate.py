"""
Evaluation script for the trained handwritten digit recognition model.

This script loads the best trained model and evaluates it on the test set,
generating accuracy metrics, confusion matrix, and classification report.
"""

import torch
import torch.nn as nn
import argparse
from pathlib import Path
import time

from .config import BEST_MODEL_PATH, CONFUSION_MATRIX_PATH
from .data import create_data_loaders
from .model import create_model
from .utils.metrics import (
    evaluate_model, 
    plot_confusion_matrix, 
    print_classification_report
)


def load_trained_model(model_path: Path, device: torch.device) -> nn.Module:
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {model_path}")
    
    print(f"Loading model from {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model (we need to know the model type)
    # For now, assume it's the full model
    model = create_model("full")
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    # Print model info
    print(f"Model loaded successfully!")
    print(f"Training completed at epoch: {checkpoint.get('epoch', 'Unknown')}")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'Unknown'):.4f}")
    
    return model


def evaluate_trained_model(model: nn.Module, test_loader: torch.utils.data.DataLoader,
                          device: torch.device) -> None:
    """
    Evaluate the trained model on the test set.
    
    Args:
        model: Trained model to evaluate
        test_loader: Test data loader
        device: Device to run evaluation on
    """
    print("\n" + "="*50)
    print("EVALUATING MODEL ON TEST SET")
    print("="*50)
    
    # Set model to evaluation mode
    model.eval()
    
    # Evaluate model
    start_time = time.time()
    test_accuracy, confusion_matrix = evaluate_model(model, test_loader, device)
    evaluation_time = time.time() - start_time
    
    # Print results
    print(f"\nTest Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    
    # Check if target accuracy is achieved
    if test_accuracy >= 0.99:
        print("🎉 TARGET ACCURACY OF 99% ACHIEVED! 🎉")
    else:
        print(f"⚠️  Target accuracy not reached. Current: {test_accuracy:.2%}")
    
    # Plot confusion matrix
    print("\nGenerating confusion matrix...")
    plot_confusion_matrix(
        confusion_matrix,
        save_path=CONFUSION_MATRIX_PATH,
        title="MNIST Test Set Confusion Matrix"
    )
    
    # Get detailed predictions for classification report
    print("\nGenerating classification report...")
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predictions = torch.argmax(output, dim=1)
            
            all_predictions.append(predictions)
            all_targets.append(target)
    
    # Concatenate all predictions and targets
    all_predictions = torch.cat(all_predictions)
    all_targets = torch.cat(all_targets)
    
    # Print classification report
    print_classification_report(all_predictions, all_targets)
    
    # Print per-class accuracy
    print("\nPer-Class Accuracy:")
    for i in range(10):
        class_mask = (all_targets == i)
        if class_mask.sum() > 0:
            class_accuracy = (all_predictions[class_mask] == all_targets[class_mask]).float().mean()
            print(f"  Digit {i}: {class_accuracy:.4f} ({class_accuracy:.2%})")
    
    return test_accuracy, confusion_matrix


def analyze_model_predictions(model: nn.Module, test_loader: torch.utils.data.DataLoader,
                             device: torch.device, num_samples: int = 10) -> None:
    """
    Analyze model predictions on individual samples.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        device: Device to run on
        num_samples: Number of samples to analyze
    """
    print(f"\n" + "="*50)
    print(f"ANALYZING {num_samples} SAMPLE PREDICTIONS")
    print("="*50)
    
    model.eval()
    samples_analyzed = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            if samples_analyzed >= num_samples:
                break
                
            data, target = data.to(device), target.to(device)
            output = model(data)
            probabilities = torch.softmax(output, dim=1)
            predictions = torch.argmax(output, dim=1)
            
            for i in range(min(len(data), num_samples - samples_analyzed)):
                true_label = target[i].item()
                pred_label = predictions[i].item()
                confidence = probabilities[i][pred_label].item()
                
                print(f"Sample {samples_analyzed + 1}:")
                print(f"  True Label: {true_label}")
                print(f"  Predicted: {pred_label}")
                print(f"  Confidence: {confidence:.4f}")
                
                if true_label == pred_label:
                    print(f"  ✅ CORRECT")
                else:
                    print(f"  ❌ INCORRECT")
                
                # Show top-3 predictions
                top3_probs, top3_indices = torch.topk(probabilities[i], 3)
                print(f"  Top-3 predictions:")
                for j in range(3):
                    print(f"    {top3_indices[j].item()}: {top3_probs[j].item():.4f}")
                
                print()
                samples_analyzed += 1


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description='Evaluate trained digit recognition model')
    parser.add_argument('--model-path', type=str, default=str(BEST_MODEL_PATH),
                       help='Path to the trained model checkpoint')
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for evaluation')
    parser.add_argument('--analyze-samples', type=int, default=10,
                       help='Number of sample predictions to analyze')
    parser.add_argument('--no-confusion-matrix', action='store_true',
                       help='Skip confusion matrix generation')
    
    args = parser.parse_args()
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"❌ Model checkpoint not found at {model_path}")
        print("Please train the model first using: python -m src.train")
        return
    
    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Load trained model
        model = load_trained_model(model_path, device)
        
        # Create test data loader
        print("Creating test data loader...")
        _, _, test_loader = create_data_loaders(
            batch_size=args.batch_size,
            augment=False  # No augmentation for evaluation
        )
        
        # Evaluate model
        test_accuracy, confusion_matrix = evaluate_trained_model(
            model, test_loader, device
        )
        
        # Analyze sample predictions
        if args.analyze_samples > 0:
            analyze_model_predictions(
                model, test_loader, device, args.analyze_samples
            )
        
        # Summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {model_path}")
        print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy:.2%})")
        print(f"Confusion Matrix: {CONFUSION_MATRIX_PATH}")
        
        if test_accuracy >= 0.99:
            print("🎉 SUCCESS: Target accuracy of 99% achieved!")
        else:
            print(f"⚠️  Target accuracy not reached. Consider:")
            print("  - Training for more epochs")
            print("  - Adjusting learning rate")
            print("  - Using more data augmentation")
            print("  - Trying different model architecture")
        
    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
