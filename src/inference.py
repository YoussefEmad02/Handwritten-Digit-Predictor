"""
Inference module for handwritten digit recognition.

This module provides functions to load the trained model and perform
recognition on preprocessed images.
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Tuple, List, Optional
import time

from .config import BEST_MODEL_PATH
from .model import create_model
from .preprocess import preprocess_canvas_image


class DigitPredictor:
    """
    Class for digit recognition using a trained model.
    """
    
    def __init__(self, model_path: Optional[Path] = None, device: Optional[str] = None):
        """
        Initialize the digit predictor.
        
        Args:
            model_path: Path to the trained model checkpoint
            device: Device to run inference on ('cpu', 'cuda', or None for auto)
        """
        self.model_path = model_path or BEST_MODEL_PATH
        self.device = self._setup_device(device)
        self.model = None
        self.is_loaded = False
        
        # Load model automatically
        self.load_model()
    
    def _setup_device(self, device: Optional[str]) -> torch.device:
        """
        Setup the device for inference.
        
        Args:
            device: Device specification
            
        Returns:
            PyTorch device
        """
        if device is None:
            # Auto-detect
            if torch.cuda.is_available():
                device = "cuda"
                print("CUDA available, using GPU for inference")
            else:
                device = "cpu"
                print("CUDA not available, using CPU for inference")
        elif device == "cuda" and not torch.cuda.is_available():
            print("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        return torch.device(device)
    
    def load_model(self) -> None:
        """
        Load the trained model from checkpoint.
        """
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found at {self.model_path}. "
                "Please train the model first using: python -m src.train"
            )
        
        print(f"Loading model from {self.model_path}")
        
        try:
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Create model (assume full model for now)
            self.model = create_model("full")
            
            # Load state dict
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            
            self.is_loaded = True
            
            # Print model info
            print(f"Model loaded successfully!")
            print(f"Training completed at epoch: {checkpoint.get('epoch', 'Unknown')}")
            print(f"Best validation loss: {checkpoint.get('best_val_loss', 'Unknown'):.4f}")
            print(f"Device: {self.device}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, image) -> Tuple[int, List[float]]:
        """
        Recognize a digit from a preprocessed image.
        
        Args:
            image: Preprocessed image tensor or PIL Image
            
        Returns:
            Tuple of (predicted_digit, probabilities)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess image if it's a PIL Image
        if hasattr(image, 'convert'):  # PIL Image
            image = preprocess_canvas_image(image)
        
        # Ensure image is on correct device
        image = image.to(self.device)
        
        # Add batch dimension if needed
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            start_time = time.time()
            output = self.model(image)
            inference_time = time.time() - start_time
            
            # Get probabilities
            probabilities = torch.softmax(output, dim=1)
            
            # Get predicted digit
            predicted_digit = torch.argmax(probabilities, dim=1).item()
            
            # Convert probabilities to list
            prob_list = probabilities[0].cpu().numpy().tolist()
            
            print(f"Inference completed in {inference_time*1000:.1f}ms")
            
            return predicted_digit, prob_list
    
    def predict_with_confidence(self, image, confidence_threshold: float = 0.5) -> Tuple[int, float, bool]:
        """
        Recognize a digit with a confidence threshold.
        
        Args:
            image: Preprocessed image
            confidence_threshold: Minimum confidence for recognition
            
        Returns:
            Tuple of (predicted_digit, confidence, is_confident)
        """
        predicted_digit, probabilities = self.predict(image)
        confidence = probabilities[predicted_digit]
        
        is_confident = confidence >= confidence_threshold
        
        return predicted_digit, confidence, is_confident
    
    def get_top_predictions(self, image, top_k: int = 3) -> List[Tuple[int, float]]:
        """
        Get top-k recognition candidates with probabilities.
        
        Args:
            image: Preprocessed image
            top_k: Number of top candidates to return
            
        Returns:
            List of (digit, probability) tuples sorted by probability
        """
        predicted_digit, probabilities = self.predict(image)
        
        # Get top-k indices
        top_indices = torch.topk(torch.tensor(probabilities), top_k).indices
        
        # Create list of (digit, probability) tuples
        top_predictions = [(int(idx), probabilities[idx]) for idx in top_indices]
        
        return top_predictions
    
    def batch_predict(self, images: List) -> List[Tuple[int, List[float]]]:
        """
        Perform recognition on a batch of images.
        
        Args:
            images: List of preprocessed images
            
        Returns:
            List of (predicted_digit, probabilities) tuples
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Preprocess all images
        processed_images = []
        for image in images:
            if hasattr(image, 'convert'):  # PIL Image
                processed = preprocess_canvas_image(image)
            else:
                processed = image
            
            processed_images.append(processed)
        
        # Stack into batch
        batch = torch.stack(processed_images).to(self.device)
        
        # Make recognitions
        with torch.no_grad():
            output = self.model(batch)
            probabilities = torch.softmax(output, dim=1)
            predicted_digits = torch.argmax(probabilities, dim=1)
            
            # Convert to list format
            results = []
            for i in range(len(images)):
                digit = predicted_digits[i].item()
                probs = probabilities[i].cpu().numpy().tolist()
                results.append((digit, probs))
            
            return results


def load_model(model_path: Optional[Path] = None, device: Optional[str] = None) -> DigitPredictor:
    """
    Convenience function to load a trained model.
    
    Args:
        model_path: Path to the trained model checkpoint
        device: Device to run inference on
        
    Returns:
        Initialized DigitPredictor instance
    """
    return DigitPredictor(model_path=model_path, device=device)


def predict_digit(image, model_path: Optional[Path] = None) -> Tuple[int, List[float]]:
    """
    Convenience function to predict a digit from an image.
    
    Args:
        image: Input image (PIL Image or tensor)
        model_path: Path to the trained model checkpoint
        
    Returns:
        Tuple of (predicted_digit, probabilities)
    """
    predictor = DigitPredictor(model_path=model_path)
    return predictor.predict(image)


def test_inference():
    """Test the inference module."""
    print("Testing inference module...")
    
    try:
        # Create predictor
        predictor = DigitPredictor()
        
        # Test with synthetic data
        from .preprocess import create_synthetic_digit
        
        # Create test digit
        test_tensor = create_synthetic_digit(3)
        
        # Make recognition
        predicted_digit, probabilities = predictor.predict(test_tensor)
        
        print(f"Test recognition successful!")
        print(f"Input: synthetic digit '3'")
        print(f"Recognized: {predicted_digit}")
        print(f"Top-3 recognition probabilities:")
        
        top_predictions = predictor.get_top_predictions(test_tensor, top_k=3)
        for digit, prob in top_predictions:
            print(f"  {digit}: {prob:.4f}")
        
        print("Inference test complete!")
        
    except Exception as e:
        print(f"Inference test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_inference()
