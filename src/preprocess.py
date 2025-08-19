"""
Image preprocessing for handwritten digit recognition.

This module converts drawn canvas images to the format expected by the
trained MNIST model (28x28 grayscale, normalized, centered).
"""

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter
from typing import Tuple, Optional
import cv2

from .config import INFERENCE_CONFIG


def detect_digit_region(image: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Detect the bounding box of the digit in the image.
    
    This function finds the smallest rectangle that contains all non-zero
    pixels (the drawn digit) and adds padding around it.
    
    Args:
        image: Input image as numpy array (grayscale)
        
    Returns:
        Tuple of (left, top, right, bottom) coordinates
    """
    # Find non-zero pixels (the digit)
    coords = np.column_stack(np.where(image > 0))
    
    if len(coords) == 0:
        # No digit found, return center region
        h, w = image.shape
        center_h, center_w = h // 2, w // 2
        size = min(h, w) // 4
        return (center_w - size, center_h - size, 
                center_w + size, center_h + size)
    
    # Get bounding box
    min_y, min_x = coords.min(axis=0)
    max_y, max_x = coords.max(axis=0)
    
    # Add padding (20% of digit size)
    digit_height = max_y - min_y
    digit_width = max_x - min_x
    padding_h = int(digit_height * 0.2)
    padding_w = int(digit_width * 0.2)
    
    # Ensure padding doesn't exceed image boundaries
    min_x = max(0, min_x - padding_w)
    min_y = max(0, min_y - padding_h)
    max_x = min(image.shape[1], max_x + padding_w)
    max_y = min(image.shape[0], max_y + padding_h)
    
    return (min_x, min_y, max_x, max_y)


def center_digit(image: np.ndarray) -> np.ndarray:
    """
    Center the digit in the image by finding its bounding box and
    centering it within the available space.
    
    Args:
        image: Input image as numpy array
        
    Returns:
        Centered image as numpy array
    """
    # Detect digit region
    left, top, right, bottom = detect_digit_region(image)
    
    # Extract digit with padding
    digit_region = image[top:bottom, left:right]
    
    # Calculate target dimensions to maintain aspect ratio
    target_size = INFERENCE_CONFIG["target_size"]
    digit_height, digit_width = digit_region.shape
    
    # Calculate scaling factor to fit digit in target size
    scale = min(target_size / digit_height, target_size / digit_width)
    
    # Scale digit
    new_height = int(digit_height * scale)
    new_width = int(digit_width * scale)
    
    # Resize digit
    digit_resized = cv2.resize(digit_region, (new_width, new_height), 
                              interpolation=cv2.INTER_AREA)
    
    # Create new image with digit centered
    centered_image = np.zeros((target_size, target_size), dtype=np.uint8)
    
    # Calculate position to center the digit
    start_y = (target_size - new_height) // 2
    start_x = (target_size - new_width) // 2
    
    # Place digit in center
    centered_image[start_y:start_y + new_height, 
                  start_x:start_x + new_width] = digit_resized
    
    return centered_image


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to MNIST format (0-1 range, MNIST mean/std).
    
    Args:
        image: Input image as numpy array (0-255)
        
    Returns:
        Normalized image as numpy array (0-1)
    """
    # Convert to float and normalize to 0-1
    image_float = image.astype(np.float32) / 255.0
    
    # Apply MNIST normalization
    mean = INFERENCE_CONFIG["normalize_mean"]
    std = INFERENCE_CONFIG["normalize_std"]
    
    normalized = (image_float - mean) / std
    
    return normalized


def preprocess_canvas_image(image: Image.Image) -> torch.Tensor:
    """
    Preprocess a canvas image for digit recognition.
    
    This function performs the complete preprocessing pipeline:
    1. Convert to grayscale
    2. Invert if needed (white digit on black background)
    3. Detect and center the digit
    4. Resize to target size
    5. Apply thresholding and denoising
    6. Normalize to MNIST format
    7. Convert to PyTorch tensor
    
    Args:
        image: PIL Image from canvas
        
    Returns:
        Preprocessed image as PyTorch tensor (1, 28, 28)
    """
    # Convert to numpy array
    if image.mode != 'L':
        image = image.convert('L')
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Check if we need to invert (white digit on black background)
    # MNIST has black digits on white background
    if np.mean(img_array) > 128:
        # Image is mostly white, invert it
        img_array = 255 - img_array
    
    # Apply Gaussian blur to smooth the strokes
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    
    # Apply thresholding to create binary image
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Center the digit
    centered = center_digit(binary)
    
    # Apply morphological operations to clean up the image
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(centered, cv2.MORPH_CLOSE, kernel)
    
    # Normalize to MNIST format
    normalized = normalize_image(cleaned)
    
    # Convert to PyTorch tensor
    tensor = torch.from_numpy(normalized).float()
    
    # Add channel dimension (C, H, W)
    tensor = tensor.unsqueeze(0)
    
    return tensor


def preprocess_for_display(image: Image.Image) -> Image.Image:
    """
    Preprocess image for display purposes (shows the 28x28 version).
    
    Args:
        image: PIL Image from canvas
        
    Returns:
        Preprocessed image ready for display
    """
    # Convert to numpy array
    if image.mode != 'L':
        image = image.convert('L')
    
    img_array = np.array(image)
    
    # Check if we need to invert
    if np.mean(img_array) > 128:
        img_array = 255 - img_array
    
    # Apply preprocessing
    img_array = cv2.GaussianBlur(img_array, (3, 3), 0)
    _, binary = cv2.threshold(img_array, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Center the digit
    centered = center_digit(binary)
    
    # Convert back to PIL Image
    return Image.fromarray(centered)


def create_synthetic_digit(digit: int, size: int = 28) -> torch.Tensor:
    """
    Create a synthetic digit for testing purposes.
    
    Args:
        digit: Digit to create (0-9)
        size: Size of the output image
        
    Returns:
        Synthetic digit as tensor
    """
    # Create a simple synthetic digit
    image = np.zeros((size, size), dtype=np.uint8)
    
    # Simple patterns for each digit
    if digit == 0:
        # Draw a circle
        cv2.circle(image, (size//2, size//2), size//3, 255, 2)
    elif digit == 1:
        # Draw a vertical line
        cv2.line(image, (size//2, 0), (size//2, size), 255, 2)
    elif digit == 2:
        # Draw a 2 shape
        cv2.line(image, (size//4, 0), (3*size//4, 0), 255, 2)
        cv2.line(image, (3*size//4, 0), (3*size//4, size//2), 255, 2)
        cv2.line(image, (3*size//4, size//2), (size//4, size//2), 255, 2)
        cv2.line(image, (size//4, size//2), (size//4, size), 255, 2)
        cv2.line(image, (size//4, size), (3*size//4, size), 255, 2)
    elif digit == 3:
        # Draw a 3 shape
        cv2.line(image, (size//4, 0), (3*size//4, 0), 255, 2)
        cv2.line(image, (3*size//4, 0), (3*size//4, size), 255, 2)
        cv2.line(image, (size//4, size//2), (3*size//4, size//2), 255, 2)
        cv2.line(image, (size//4, size), (3*size//4, size), 255, 2)
    else:
        # For other digits, just draw a simple pattern
        cv2.putText(image, str(digit), (size//3, 2*size//3), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
    
    # Normalize
    normalized = normalize_image(image)
    
    # Convert to tensor
    tensor = torch.from_numpy(normalized).float().unsqueeze(0)
    
    return tensor


if __name__ == "__main__":
    # Test preprocessing functions
    print("Testing preprocessing functions...")
    
    # Create a test image
    test_image = Image.new('L', (280, 280), 'white')
    
    # Test preprocessing
    try:
        processed = preprocess_canvas_image(test_image)
        print(f"Preprocessing successful! Output shape: {processed.shape}")
        print(f"Output range: [{processed.min():.3f}, {processed.max():.3f}]")
    except Exception as e:
        print(f"Preprocessing failed: {e}")
    
    # Test synthetic digit creation
    try:
        synthetic = create_synthetic_digit(3)
        print(f"Synthetic digit created! Shape: {synthetic.shape}")
    except Exception as e:
        print(f"Synthetic digit creation failed: {e}")
    
    print("Preprocessing test complete!")
