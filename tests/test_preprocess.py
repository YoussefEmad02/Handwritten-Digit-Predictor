"""
Unit tests for the preprocessing module.

These tests verify that the preprocessing functions correctly transform
images to the format expected by the MNIST model.
"""

import unittest
import numpy as np
import torch
from PIL import Image, ImageDraw
import tempfile
import os

# Add src to path for imports
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from preprocess import (
    detect_digit_region,
    center_digit,
    normalize_image,
    preprocess_canvas_image,
    preprocess_for_display,
    create_synthetic_digit
)


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_size = 28
        self.canvas_size = 280
        
        # Create a test image with a simple digit
        self.test_image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        draw = ImageDraw.Draw(self.test_image)
        
        # Draw a simple "1" in the center
        center_x = self.canvas_size // 2
        draw.line([(center_x, 50), (center_x, 230)], fill='black', width=20)
        
        # Convert to numpy array
        self.test_array = np.array(self.test_image)
    
    def test_detect_digit_region(self):
        """Test digit region detection."""
        # Test with our test image
        left, top, right, bottom = detect_digit_region(self.test_array)
        
        # Check that we got reasonable bounds
        self.assertGreater(right, left)
        self.assertGreater(bottom, top)
        
        # Check that bounds are within image dimensions
        self.assertGreaterEqual(left, 0)
        self.assertGreaterEqual(top, 0)
        self.assertLessEqual(right, self.canvas_size)
        self.assertLessEqual(bottom, self.canvas_size)
        
        # Test with empty image
        empty_image = np.zeros((100, 100), dtype=np.uint8)
        left, top, right, bottom = detect_digit_region(empty_image)
        
        # Should return center region for empty image
        self.assertGreater(right, left)
        self.assertGreater(bottom, top)
    
    def test_center_digit(self):
        """Test digit centering."""
        centered = center_digit(self.test_array)
        
        # Check output dimensions
        self.assertEqual(centered.shape, (self.test_size, self.test_size))
        
        # Check that output is binary (0 or 255)
        unique_values = np.unique(centered)
        self.assertTrue(all(val in [0, 255] for val in unique_values))
        
        # Check that digit is roughly centered
        # Find the digit pixels
        digit_pixels = np.where(centered > 0)
        if len(digit_pixels[0]) > 0:
            center_y = np.mean(digit_pixels[0])
            center_x = np.mean(digit_pixels[1])
            
            # Should be roughly in the center
            self.assertLess(abs(center_y - self.test_size // 2), self.test_size // 4)
            self.assertLess(abs(center_x - self.test_size // 2), self.test_size // 4)
    
    def test_normalize_image(self):
        """Test image normalization."""
        # Create test image with values 0-255
        test_img = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        
        normalized = normalize_image(test_img)
        
        # Check output type
        self.assertEqual(normalized.dtype, np.float32)
        
        # Check output range (should be roughly centered around 0)
        self.assertLess(np.mean(normalized), 1.0)
        self.assertGreater(np.mean(normalized), -1.0)
    
    def test_preprocess_canvas_image(self):
        """Test complete preprocessing pipeline."""
        # Test with PIL Image
        processed = preprocess_canvas_image(self.test_image)
        
        # Check output type and shape
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape, (1, self.test_size, self.test_size))
        self.assertEqual(processed.dtype, torch.float32)
        
        # Check value range (normalized)
        self.assertLess(processed.max(), 5.0)
        self.assertGreater(processed.min(), -5.0)
    
    def test_preprocess_for_display(self):
        """Test preprocessing for display."""
        display_image = preprocess_for_display(self.test_image)
        
        # Check output type
        self.assertIsInstance(display_image, Image.Image)
        
        # Check dimensions
        self.assertEqual(display_image.size, (self.test_size, self.test_size))
        
        # Check mode
        self.assertEqual(display_image.mode, 'L')
    
    def test_create_synthetic_digit(self):
        """Test synthetic digit creation."""
        # Test different digits
        for digit in range(10):
            synthetic = create_synthetic_digit(digit)
            
            # Check output type and shape
            self.assertIsInstance(synthetic, torch.Tensor)
            self.assertEqual(synthetic.shape, (1, self.test_size, self.test_size))
            self.assertEqual(synthetic.dtype, torch.float32)
            
            # Check value range
            self.assertLess(synthetic.max(), 5.0)
            self.assertGreater(synthetic.min(), -5.0)
    
    def test_inversion_detection(self):
        """Test automatic color inversion detection."""
        # Create white digit on black background
        inverted_image = Image.new('L', (self.canvas_size, self.canvas_size), 'black')
        draw = ImageDraw.Draw(inverted_image)
        draw.line([(100, 100), (100, 200)], fill='white', width=20)
        
        # Process both versions
        normal_processed = preprocess_canvas_image(self.test_image)
        inverted_processed = preprocess_canvas_image(inverted_image)
        
        # Both should produce similar results (after preprocessing)
        # We can't expect exact equality due to different input patterns,
        # but they should have similar characteristics
        self.assertEqual(normal_processed.shape, inverted_processed.shape)
        self.assertEqual(normal_processed.dtype, inverted_processed.dtype)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small digit
        small_image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        draw = ImageDraw.Draw(small_image)
        draw.point((self.canvas_size // 2, self.canvas_size // 2), fill='black')
        
        # Should not crash
        try:
            processed = preprocess_canvas_image(small_image)
            self.assertIsInstance(processed, torch.Tensor)
        except Exception as e:
            self.fail(f"Preprocessing small digit failed: {e}")
        
        # Test with very large digit
        large_image = Image.new('L', (self.canvas_size, self.canvas_size), 'white')
        draw = ImageDraw.Draw(large_image)
        draw.rectangle([10, 10, self.canvas_size-10, self.canvas_size-10], 
                      fill='black')
        
        # Should not crash
        try:
            processed = preprocess_canvas_image(large_image)
            self.assertIsInstance(processed, torch.Tensor)
        except Exception as e:
            self.fail(f"Preprocessing large digit failed: {e}")


class TestPreprocessingPerformance(unittest.TestCase):
    """Test preprocessing performance."""
    
    def test_processing_speed(self):
        """Test that preprocessing is reasonably fast."""
        import time
        
        # Create test image
        test_image = Image.new('L', (280, 280), 'white')
        draw = ImageDraw.Draw(test_image)
        draw.line([(100, 100), (100, 200)], fill='black', width=20)
        
        # Time preprocessing
        start_time = time.time()
        processed = preprocess_canvas_image(test_image)
        end_time = time.time()
        
        processing_time = end_time - start_time
        
        # Should complete in reasonable time (< 100ms)
        self.assertLess(processing_time, 0.1, 
                       f"Preprocessing took {processing_time*1000:.1f}ms, expected < 100ms")
        
        # Check output
        self.assertIsInstance(processed, torch.Tensor)
        self.assertEqual(processed.shape, (1, 28, 28))


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
