"""
Tkinter GUI for handwritten digit recognition.

This module provides a user-friendly interface where users can draw digits
on a canvas and get real-time predictions from the trained model.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from PIL import Image, ImageDraw, ImageTk
import numpy as np
from pathlib import Path
import threading
import time
from typing import Optional, Tuple

from .config import INFERENCE_CONFIG, BEST_MODEL_PATH
from .inference import DigitPredictor
from .preprocess import preprocess_for_display


class DrawingCanvas:
    """
    Custom canvas widget for drawing digits.
    """
    
    def __init__(self, parent, width: int = 280, height: int = 280, **kwargs):
        """
        Initialize the drawing canvas.
        
        Args:
            parent: Parent widget
            width: Canvas width
            height: Canvas height
            **kwargs: Additional canvas arguments
        """
        self.width = width
        self.height = height
        
        # Create canvas
        self.canvas = tk.Canvas(parent, width=width, height=height, 
                               bg='white', **kwargs)
        
        # Create PIL image for drawing
        self.image = Image.new('L', (width, height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
        # Drawing state
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.brush_size = 15
        
        # Bind events
        self.canvas.bind('<Button-1>', self.start_draw)
        self.canvas.bind('<B1-Motion>', self.draw_line)
        self.canvas.bind('<ButtonRelease-1>', self.stop_draw)
        self.canvas.bind('<Button-3>', self.clear_canvas)  # Right click to clear
        
        # Draw grid (optional)
        self.draw_grid()
    
    def draw_grid(self):
        """Draw a subtle grid on the canvas."""
        grid_color = '#f0f0f0'
        
        # Vertical lines
        for x in range(0, self.width, 40):
            self.canvas.create_line(x, 0, x, self.height, fill=grid_color, width=1)
        
        # Horizontal lines
        for y in range(0, self.height, 40):
            self.canvas.create_line(0, y, self.width, y, fill=grid_color, width=1)
    
    def start_draw(self, event):
        """Start drawing."""
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
    
    def draw_line(self, event):
        """Draw a line segment."""
        if self.drawing and self.last_x is not None and self.last_y is not None:
            # Draw on canvas
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  fill='black', width=self.brush_size, capstyle=tk.ROUND)
            
            # Draw on PIL image
            self.draw.line([self.last_x, self.last_y, event.x, event.y],
                          fill='black', width=self.brush_size)
            
            self.last_x = event.x
            self.last_y = event.y
    
    def stop_draw(self, event):
        """Stop drawing."""
        self.drawing = False
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self, event=None):
        """Clear the canvas."""
        self.canvas.delete("all")
        self.image = Image.new('L', (self.width, self.height), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.draw_grid()
    
    def get_image(self) -> Image.Image:
        """Get the current canvas image."""
        return self.image.copy()
    
    def set_brush_size(self, size: int):
        """Set the brush size."""
        self.brush_size = size


class DigitRecognitionGUI:
    """
    Main GUI application for digit recognition.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI application.
        
        Args:
            root: Root Tkinter window
        """
        self.root = root
        self.root.title("Handwritten Digit Recognition")
        self.root.geometry("800x600")
        self.root.resizable(False, False)
        
        # Initialize predictor
        self.predictor = None
        self.model_loaded = False
        
        # Create GUI elements
        self.create_widgets()
        
        # Load model in background
        self.load_model_async()
    
    def create_widgets(self):
        """Create and arrange GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="Handwritten Digit Recognition", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel - Drawing area
        left_panel = ttk.Frame(main_frame)
        left_panel.grid(row=1, column=0, padx=(0, 20), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Canvas label
        canvas_label = ttk.Label(left_panel, text="Draw a digit (0-9):")
        canvas_label.grid(row=0, column=0, pady=(0, 10))
        
        # Drawing canvas
        self.canvas = DrawingCanvas(left_panel, width=280, height=280)
        self.canvas.canvas.grid(row=1, column=0)
        
        # Brush size control
        brush_frame = ttk.Frame(left_panel)
        brush_frame.grid(row=2, column=0, pady=(10, 0))
        
        ttk.Label(brush_frame, text="Brush size:").pack(side=tk.LEFT)
        self.brush_var = tk.IntVar(value=15)
        brush_scale = ttk.Scale(brush_frame, from_=5, to=25, variable=self.brush_var,
                               orient=tk.HORIZONTAL, length=150,
                               command=self.update_brush_size)
        brush_scale.pack(side=tk.LEFT, padx=(10, 0))
        
        # Right panel - Controls and results
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Control buttons
        controls_frame = ttk.LabelFrame(right_panel, text="Controls", padding="10")
        controls_frame.grid(row=0, column=0, pady=(0, 20), sticky=(tk.W, tk.E))
        
        # Predict button
        self.predict_button = ttk.Button(controls_frame, text="Predict", 
                                       command=self.predict_digit, state='disabled')
        self.predict_button.pack(fill=tk.X, pady=(0, 5))
        
        # Clear button
        clear_button = ttk.Button(controls_frame, text="Clear Canvas", 
                                command=self.canvas.clear_canvas)
        clear_button.pack(fill=tk.X, pady=(0, 5))
        
        # Save button
        save_button = ttk.Button(controls_frame, text="Save Image", 
                               command=self.save_image)
        save_button.pack(fill=tk.X)
        
        # Results frame
        results_frame = ttk.LabelFrame(right_panel, text="Prediction Results", padding="10")
        results_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Prediction display
        self.prediction_label = ttk.Label(results_frame, text="Draw a digit and click Predict", 
                                        font=('Arial', 14))
        self.prediction_label.pack(pady=(0, 10))
        
        # Confidence display
        self.confidence_label = ttk.Label(results_frame, text="", font=('Arial', 10))
        self.confidence_label.pack()
        
        # Top-3 predictions
        self.top3_frame = ttk.Frame(results_frame)
        self.top3_frame.pack(fill=tk.X, pady=(10, 0))
        
        # Preprocessed image preview
        preview_frame = ttk.LabelFrame(right_panel, text="Preprocessed Image (28x28)", padding="10")
        preview_frame.grid(row=2, column=0, pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.preview_label = ttk.Label(preview_frame, text="No image to preview")
        self.preview_label.pack()
        
        # Status bar
        status_frame = ttk.Frame(main_frame)
        status_frame.grid(row=2, column=0, columnspan=3, pady=(20, 0), sticky=(tk.W, tk.E))
        
        self.status_label = ttk.Label(status_frame, text="Loading model...", 
                                    font=('Arial', 9))
        self.status_label.pack(side=tk.LEFT)
        
        # Model info
        self.model_info_label = ttk.Label(status_frame, text="", font=('Arial', 9))
        self.model_info_label.pack(side=tk.RIGHT)
    
    def update_brush_size(self, value):
        """Update brush size when scale changes."""
        self.canvas.set_brush_size(int(float(value)))
    
    def load_model_async(self):
        """Load the model in a background thread."""
        def load_model():
            try:
                self.predictor = DigitPredictor()
                self.model_loaded = True
                
                # Update GUI in main thread
                self.root.after(0, self.model_loaded_success)
                
            except Exception as e:
                error_msg = f"Failed to load model: {e}"
                self.root.after(0, lambda: self.model_loaded_error(error_msg))
        
        # Start loading thread
        thread = threading.Thread(target=load_model, daemon=True)
        thread.start()
    
    def model_loaded_success(self):
        """Called when model loads successfully."""
        self.status_label.config(text="Model loaded successfully!")
        self.predict_button.config(state='normal')
        
        # Get model info
        if self.predictor and self.predictor.is_loaded:
            model_path = self.predictor.model_path.name
            device = str(self.predictor.device)
            self.model_info_label.config(text=f"Model: {model_path} | Device: {device}")
    
    def model_loaded_error(self, error_msg: str):
        """Called when model loading fails."""
        self.status_label.config(text=error_msg)
        messagebox.showerror("Model Loading Error", 
                           f"Failed to load the trained model.\n\n{error_msg}\n\n"
                           "Please ensure you have trained the model first using:\n"
                           "python -m src.train")
    
    def predict_digit(self):
        """Predict the digit drawn on the canvas."""
        if not self.model_loaded:
            messagebox.showerror("Error", "Model not loaded yet. Please wait.")
            return
        
        try:
            # Get canvas image
            canvas_image = self.canvas.get_image()
            
            # Check if image is empty (all pixels are white/255)
            canvas_array = np.array(canvas_image)
            if canvas_array.min() == 255:  # All white (no black pixels)
                messagebox.showwarning("Warning", "Please draw a digit first.")
                return
            
            # Update status
            self.status_label.config(text="Making prediction...")
            self.root.update()
            
            # Make prediction
            predicted_digit, probabilities = self.predictor.predict(canvas_image)
            
            # Update prediction display
            self.prediction_label.config(text=f"Predicted: {predicted_digit}")
            
            # Update confidence display
            confidence = probabilities[predicted_digit]
            self.confidence_label.config(
                text=f"Confidence: {confidence:.2%}",
                foreground='green' if confidence > 0.8 else 'orange' if confidence > 0.6 else 'red'
            )
            
            # Update top-3 predictions
            self.update_top3_display(probabilities)
            
            # Update preprocessed image preview
            self.update_preview(canvas_image)
            
            # Update status
            self.status_label.config(text="Prediction completed successfully!")
            
        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            self.status_label.config(text=error_msg)
            messagebox.showerror("Prediction Error", error_msg)
    
    def update_top3_display(self, probabilities: list):
        """Update the top-3 predictions display."""
        # Clear existing widgets
        for widget in self.top3_frame.winfo_children():
            widget.destroy()
        
        # Get top-3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        
        # Create labels for top-3
        for i, idx in enumerate(top3_indices):
            prob = probabilities[idx]
            color = 'green' if i == 0 else 'black'
            
            label = ttk.Label(self.top3_frame, 
                            text=f"{idx}: {prob:.2%}",
                            font=('Arial', 10),
                            foreground=color)
            label.pack(anchor=tk.W)
    
    def update_preview(self, canvas_image: Image.Image):
        """Update the preprocessed image preview."""
        try:
            # Preprocess image for display
            preprocessed = preprocess_for_display(canvas_image)
            
            # Resize for display (make it larger for visibility)
            display_size = 112  # 4x larger than 28x28
            preprocessed_display = preprocessed.resize((display_size, display_size), Image.NEAREST)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(preprocessed_display)
            
            # Update preview label
            self.preview_label.config(image=photo, text="")
            self.preview_label.image = photo  # Keep reference
            
        except Exception as e:
            self.preview_label.config(text=f"Preview error: {e}")
    
    def save_image(self):
        """Save the current canvas image."""
        try:
            # Get file path from user
            file_path = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                title="Save Canvas Image"
            )
            
            if file_path:
                # Get canvas image and save
                canvas_image = self.canvas.get_image()
                canvas_image.save(file_path)
                
                messagebox.showinfo("Success", f"Image saved to:\n{file_path}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save image: {e}")


def main():
    """Main function to launch the GUI."""
    try:
        # Check if model exists
        if not BEST_MODEL_PATH.exists():
            result = messagebox.askyesno(
                "Model Not Found",
                "No trained model found. Would you like to train a new model now?\n\n"
                "This will take several minutes but only needs to be done once."
            )
            
            if result:
                # Launch training
                import subprocess
                subprocess.run(["python", "-m", "src.train"])
            else:
                messagebox.showinfo(
                    "Information",
                    "Please train the model first using:\n"
                    "python -m src.train\n\n"
                    "Then run the GUI again."
                )
                return
        
        # Create and run GUI
        root = tk.Tk()
        app = DigitRecognitionGUI(root)
        root.mainloop()
        
    except Exception as e:
        messagebox.showerror("Error", f"Failed to launch GUI: {e}")


if __name__ == "__main__":
    main()
