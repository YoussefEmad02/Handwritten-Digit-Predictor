#!/usr/bin/env python3
"""
Demo script for the Handwritten Digit Predictor.

This script demonstrates the model's capabilities by creating synthetic
digits and showing predictions.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocess import create_synthetic_digit
from inference import DigitPredictor
from config import BEST_MODEL_PATH
import matplotlib.pyplot as plt
import numpy as np


def create_demo_digits():
    """Create synthetic digits for demonstration."""
    print("Creating synthetic digits for demonstration...")
    
    digits = []
    for i in range(10):
        digit_tensor = create_synthetic_digit(i)
        digits.append(digit_tensor)
    
    return digits


def visualize_digits(digits, predictions=None, probabilities=None):
    """Visualize the synthetic digits and predictions."""
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, digit_tensor in enumerate(digits):
        # Convert tensor to image for display
        digit_img = digit_tensor.squeeze().numpy()
        
        # Denormalize for display
        mean = 0.1307
        std = 0.3081
        digit_img = (digit_img * std + mean) * 255
        digit_img = np.clip(digit_img, 0, 255).astype(np.uint8)
        
        # Display digit
        axes[i].imshow(digit_img, cmap='gray')
        
        if predictions is not None and probabilities is not None:
            pred = predictions[i]
            prob = probabilities[i]
            title = f"True: {i}\nPred: {pred}\nConf: {prob:.2%}"
        else:
            title = f"Digit {i}"
        
        axes[i].set_title(title, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def run_demo():
    """Run the complete demonstration."""
    print("=" * 60)
    print("Handwritten Digit Predictor Demo")
    print("=" * 60)
    
    # Check if model exists
    if not BEST_MODEL_PATH.exists():
        print("❌ No trained model found!")
        print("Please train the model first using: python -m src.train")
        return
    
    try:
        # Create synthetic digits
        digits = create_demo_digits()
        print(f"✅ Created {len(digits)} synthetic digits")
        
        # Load model
        print("Loading trained model...")
        predictor = DigitPredictor()
        print("✅ Model loaded successfully!")
        
        # Make predictions
        print("Making predictions...")
        predictions = []
        probabilities = []
        
        for i, digit_tensor in enumerate(digits):
            pred, probs = predictor.predict(digit_tensor)
            predictions.append(pred)
            probabilities.append(probs[pred])  # Confidence for predicted class
            
            print(f"  Digit {i}: Predicted {pred} (confidence: {probs[pred]:.2%})")
        
        # Calculate accuracy
        correct = sum(1 for i, pred in enumerate(predictions) if pred == i)
        accuracy = correct / len(predictions)
        print(f"\n✅ Overall Accuracy: {accuracy:.2%} ({correct}/{len(predictions)})")
        
        # Visualize results
        print("\nDisplaying results...")
        visualize_digits(digits, predictions, probabilities)
        
        # Show detailed results
        print("\nDetailed Results:")
        print("-" * 40)
        for i in range(10):
            true_digit = i
            pred_digit = predictions[i]
            confidence = probabilities[i]
            status = "✅" if true_digit == pred_digit else "❌"
            
            print(f"{status} True: {true_digit}, Predicted: {pred_digit}, "
                  f"Confidence: {confidence:.2%}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_demo()
