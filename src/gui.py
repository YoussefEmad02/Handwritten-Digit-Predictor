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


class ModernStyle:
    """Modern color scheme and styling constants."""
    
    # Color palette
    PRIMARY = "#6366f1"      # Indigo
    PRIMARY_DARK = "#4f46e5" # Darker indigo
    SECONDARY = "#ec4899"    # Pink
    SUCCESS = "#10b981"      # Emerald
    WARNING = "#f59e0b"      # Amber
    ERROR = "#ef4444"        # Red
    INFO = "#3b82f6"         # Blue
    
    # Background colors
    BG_PRIMARY = "#ffffff"   # White
    BG_SECONDARY = "#f8fafc" # Light gray
    BG_DARK = "#1e293b"      # Dark slate
    
    # Text colors
    TEXT_PRIMARY = "#1e293b"   # Dark slate
    TEXT_SECONDARY = "#64748b" # Slate
    TEXT_LIGHT = "#ffffff"     # White
    
    # Border colors
    BORDER = "#e2e8f0"      # Light gray
    BORDER_FOCUS = "#6366f1" # Primary
    
    # Canvas colors
    CANVAS_BG = "#ffffff"     # White
    CANVAS_GRID = "#f1f5f9"   # Very light gray
    CANVAS_DRAW = "#1e293b"   # Dark slate
    
    # Button styles
    BUTTON_BG = "#6366f1"
    BUTTON_FG = "#ffffff"
    BUTTON_HOVER = "#4f46e5"
    BUTTON_ACTIVE = "#4338ca"


class ModernCanvas(tk.Canvas):
    """Modern styled canvas with enhanced visual effects."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(
            bg=ModernStyle.CANVAS_BG,
            relief="flat",
            bd=0,
            highlightthickness=2,
            highlightbackground=ModernStyle.BORDER,
            highlightcolor=ModernStyle.BORDER_FOCUS
        )
        
        # Add hover effect
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        
    def on_enter(self, event):
        self.configure(highlightbackground=ModernStyle.PRIMARY)
        
    def on_leave(self, event):
        self.configure(highlightbackground=ModernStyle.BORDER)


class DrawingCanvas:
    """
    Custom canvas widget for drawing digits with modern styling.
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
        
        # Create modern canvas
        self.canvas = ModernCanvas(parent, width=width, height=height, **kwargs)
        
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
        
        # Draw grid
        self.draw_grid()
    
    def draw_grid(self):
        """Draw a subtle modern grid on the canvas."""
        grid_color = ModernStyle.CANVAS_GRID
        
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
        """Draw a line segment with smooth curves."""
        if self.drawing and self.last_x is not None and self.last_y is not None:
            # Draw on canvas with smooth curves
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y,
                                  fill=ModernStyle.CANVAS_DRAW, width=self.brush_size, 
                                  capstyle=tk.ROUND, smooth=True)
            
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


class ModernButton(tk.Button):
    """Modern styled button with hover effects."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(
            bg=ModernStyle.BUTTON_BG,
            fg=ModernStyle.BUTTON_FG,
            font=('Segoe UI', 9, 'bold'),
            relief="flat",
            bd=0,
            padx=15,
            pady=6,
            cursor="hand2"
        )
        
        # Bind hover effects
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        
    def on_enter(self, event):
        self.configure(bg=ModernStyle.BUTTON_HOVER)
        
    def on_leave(self, event):
        self.configure(bg=ModernStyle.BUTTON_BG)


class ModernLabelFrame(tk.LabelFrame):
    """Modern styled label frame."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.configure(
            bg=ModernStyle.BG_SECONDARY,
            fg=ModernStyle.TEXT_PRIMARY,
            font=('Segoe UI', 9, 'bold'),
            relief="flat",
            bd=0,
            padx=10,
            pady=10
        )


class DigitRecognitionGUI:
    """
    Main GUI application for digit recognition with modern design.
    """
    
    def __init__(self, root):
        """
        Initialize the GUI application.
        
        Args:
            root: Root Tkinter window
        """
        self.root = root
        self.root.title("✨ AI Digit Recognition")
        
        # Make window maximized (not fullscreen)
        self.root.state('zoomed')  # For Windows - maximizes the window
        
        # Configure modern styling
        self.setup_styling()
        
        # Initialize predictor
        self.predictor = None
        self.model_loaded = False
        
        # Create GUI elements
        self.create_widgets()
        
        # Load model in background
        self.load_model_async()
        
        # Bind keyboard shortcuts
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
        self.root.bind('<Escape>', lambda e: self.root.state('normal'))
        
        # Bind window resize event for responsive layout
        self.root.bind('<Configure>', self.on_window_resize)
    
    def setup_styling(self):
        """Setup modern styling for the application."""
        # Configure root window
        self.root.configure(bg=ModernStyle.BG_PRIMARY)
        
        # Configure ttk styles
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure modern colors
        style.configure('Modern.TFrame', background=ModernStyle.BG_PRIMARY)
        style.configure('Modern.TLabel', background=ModernStyle.BG_PRIMARY, foreground=ModernStyle.TEXT_PRIMARY)
        style.configure('Modern.TButton', background=ModernStyle.BUTTON_BG, foreground=ModernStyle.BUTTON_FG)
        style.configure('Title.TLabel', font=('Segoe UI', 24, 'bold'), foreground=ModernStyle.PRIMARY)
        style.configure('Subtitle.TLabel', font=('Segoe UI', 12), foreground=ModernStyle.TEXT_SECONDARY)
        style.configure('Result.TLabel', font=('Segoe UI', 18, 'bold'), foreground=ModernStyle.TEXT_PRIMARY)
        style.configure('Confidence.TLabel', font=('Segoe UI', 14), foreground=ModernStyle.TEXT_SECONDARY)
        
        # Configure scale
        style.configure('Modern.Horizontal.TScale', 
                       background=ModernStyle.BG_PRIMARY,
                       troughcolor=ModernStyle.BORDER,
                       slidercolor=ModernStyle.PRIMARY)
    
    def create_widgets(self):
        """Create and arrange GUI widgets with modern design."""
        # Main container with gradient-like effect - smaller padding
        main_container = tk.Frame(self.root, bg=ModernStyle.BG_PRIMARY, relief="flat", bd=0)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Header section with gradient background - much smaller height
        header_frame = tk.Frame(main_container, bg=ModernStyle.PRIMARY, relief="flat", bd=0)
        header_frame.pack(fill=tk.X, pady=10)
        
        # Title with modern typography - smaller font
        title_label = tk.Label(header_frame, 
                              text="✨ AI Digit Recognition",
                              font=('Segoe UI', 16, 'bold'),
                              fg=ModernStyle.TEXT_LIGHT,
                              bg=ModernStyle.PRIMARY,
                              pady=8)
        title_label.pack()
        
        # Subtitle - smaller font
        subtitle_label = tk.Label(header_frame,
                                 text="Draw a digit and watch the AI predict it in real-time!",
                                 font=('Segoe UI', 9),
                                 fg=ModernStyle.TEXT_LIGHT,
                                 bg=ModernStyle.PRIMARY,
                                 pady=8)
        subtitle_label.pack()
        
        # Maximize toggle button - smaller
        maximize_button = tk.Button(header_frame,
                                   text="⛶ Toggle Maximize",
                                   font=('Segoe UI', 8),
                                   fg=ModernStyle.TEXT_LIGHT,
                                   bg=ModernStyle.PRIMARY_DARK,
                                   relief="flat",
                                   bd=0,
                                   cursor="hand2",
                                   command=self.toggle_fullscreen)
        maximize_button.pack(pady=8)
        
        # Main content area
        content_frame = tk.Frame(main_container, bg=ModernStyle.BG_PRIMARY, relief="flat", bd=0)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Drawing area and controls - smaller padding
        left_panel = tk.Frame(content_frame, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=0)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)
        
        # Canvas section - smaller padding
        canvas_section = ModernLabelFrame(left_panel, text="🎨 Drawing Canvas")
        canvas_section.pack(fill=tk.X, padx=10, pady=10)
        
        # Canvas label - smaller font and padding
        canvas_label = tk.Label(canvas_section, 
                               text="Draw a digit (0-9) below:",
                               font=('Segoe UI', 10, 'bold'),
                               fg=ModernStyle.TEXT_PRIMARY,
                               bg=ModernStyle.BG_SECONDARY)
        canvas_label.pack(pady=8)
        
        # Drawing canvas with responsive sizing - smaller size
        canvas_container = tk.Frame(canvas_section, bg=ModernStyle.BG_SECONDARY)
        canvas_container.pack()
        
        # Responsive canvas size - smaller default
        canvas_size = 200
        self.canvas = DrawingCanvas(canvas_container, width=canvas_size, height=canvas_size)
        self.canvas.canvas.pack(padx=10, pady=10)
        
        # Brush size control with modern styling - smaller padding
        brush_frame = tk.Frame(canvas_section, bg=ModernStyle.BG_SECONDARY)
        brush_frame.pack(pady=10)
        
        brush_label = tk.Label(brush_frame, 
                               text="🖌️ Brush Size:",
                               font=('Segoe UI', 9, 'bold'),
                               fg=ModernStyle.TEXT_PRIMARY,
                               bg=ModernStyle.BG_SECONDARY)
        brush_label.pack(side=tk.LEFT)
        
        self.brush_var = tk.IntVar(value=15)
        brush_scale = ttk.Scale(brush_frame, 
                               from_=5, to=25, 
                               variable=self.brush_var,
                               orient=tk.HORIZONTAL, 
                               length=150,
                               style='Modern.Horizontal.TScale',
                               command=self.update_brush_size)
        brush_scale.pack(side=tk.LEFT, padx=8)
        
        # Control buttons section under canvas - smaller padding
        controls_section = ModernLabelFrame(left_panel, text="🎮 Controls")
        controls_section.pack(fill=tk.X, padx=10, pady=10)
        
        # Predict button - smaller padding
        self.predict_button = ModernButton(controls_section, 
                                          text="🚀 Predict Digit",
                                          command=self.predict_digit,
                                          state='disabled')
        self.predict_button.pack(fill=tk.X, pady=6)
        
        # Clear button - smaller padding
        clear_button = ModernButton(controls_section, 
                                   text="🧹 Clear Canvas",
                                   command=self.canvas.clear_canvas,
                                   bg=ModernStyle.SECONDARY)
        clear_button.pack(fill=tk.X, pady=6)
        
        # Save button - smaller padding
        save_button = ModernButton(controls_section, 
                                  text="💾 Save Image",
                                  command=self.save_image,
                                  bg=ModernStyle.INFO)
        save_button.pack(fill=tk.X)
        

        
        # Right panel - Only prediction results - smaller padding
        right_panel = tk.Frame(content_frame, bg=ModernStyle.BG_SECONDARY, relief="flat", bd=0)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Results section - smaller padding
        results_section = ModernLabelFrame(right_panel, text="🎯 Prediction Results")
        results_section.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Prediction display with modern styling - smaller font and padding
        self.prediction_label = tk.Label(results_section, 
                                        text="Draw a digit and click Predict! 🎨",
                                        font=('Segoe UI', 16, 'bold'),
                                        fg=ModernStyle.TEXT_PRIMARY,
                                        bg=ModernStyle.BG_SECONDARY,
                                        pady=10)
        self.prediction_label.pack()
        
        # Confidence display - smaller font and padding
        self.confidence_label = tk.Label(results_section, 
                                        text="",
                                        font=('Segoe UI', 11),
                                        fg=ModernStyle.TEXT_SECONDARY,
                                        bg=ModernStyle.BG_SECONDARY)
        self.confidence_label.pack(pady=8)
        
        # Top-3 predictions - smaller font and padding
        top3_label = tk.Label(results_section,
                             text="Top 3 Predictions:",
                             font=('Segoe UI', 10, 'bold'),
                             fg=ModernStyle.TEXT_PRIMARY,
                             bg=ModernStyle.BG_SECONDARY)
        top3_label.pack(anchor=tk.W, padx=10, pady=8)
        
        # Add a separator line - thinner
        separator = tk.Frame(results_section, height=1, bg=ModernStyle.BORDER)
        separator.pack(fill=tk.X, padx=10, pady=3)
        
        self.top3_frame = tk.Frame(results_section, bg=ModernStyle.BG_SECONDARY)
        self.top3_frame.pack(fill=tk.X, padx=10, pady=6)
        
        # Preprocessed image preview section - moved from left panel
        preview_section = ModernLabelFrame(results_section, text="🔍 Preprocessed Image (28x28)")
        preview_section.pack(fill=tk.X, padx=10, pady=10)
        
        self.preview_label = tk.Label(preview_section, 
                                     text="No image to preview",
                                     font=('Segoe UI', 9),
                                     fg=ModernStyle.TEXT_SECONDARY,
                                     bg=ModernStyle.BG_SECONDARY)
        self.preview_label.pack(pady=15)
    
    def update_brush_size(self, value):
        """Update brush size when scale changes."""
        self.canvas.set_brush_size(int(float(value)))
    
    def toggle_fullscreen(self):
        """Toggle between maximized and normal window mode."""
        if self.root.state() == 'zoomed':
            self.root.state('normal')
        else:
            self.root.state('zoomed')
    
    def on_window_resize(self, event):
        """Handle window resize events for responsive layout."""
        if event.widget == self.root:
            # Update canvas size based on window dimensions
            window_width = event.width
            window_height = event.height
            
            # Responsive canvas sizing
            if hasattr(self, 'canvas'):
                # Calculate responsive canvas size
                if window_width < 800:
                    canvas_size = 200
                elif window_width < 1200:
                    canvas_size = 250
                else:
                    canvas_size = 300
                
                # Update canvas size if it changed significantly
                if abs(canvas_size - self.canvas.width) > 20:
                    self.canvas.width = canvas_size
                    self.canvas.height = canvas_size
                    # Recreate canvas with new size
                    self.canvas.canvas.configure(width=canvas_size, height=canvas_size)
                    self.canvas.clear_canvas()
    
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
        self.predict_button.config(state='normal')
        
        # Update window title to show model is loaded
        if self.predictor and self.predictor.is_loaded:
            model_path = self.predictor.model_path.name
            device = str(self.predictor.device)
            self.root.title(f"✨ AI Digit Recognition - Model: {model_path} | Device: {device}")
    
    def model_loaded_error(self, error_msg: str):
        """Called when model loading fails."""
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
            
            # Make prediction
            predicted_digit, probabilities = self.predictor.predict(canvas_image)
            
            # Update prediction display with emojis and colors
            self.prediction_label.config(text=f"🎯 Predicted: {predicted_digit}")
            
            # Update confidence display with color coding
            confidence = probabilities[predicted_digit]
            if confidence > 0.8:
                confidence_color = ModernStyle.SUCCESS
                confidence_emoji = "🟢"
            elif confidence > 0.6:
                confidence_color = ModernStyle.WARNING
                confidence_emoji = "🟡"
            else:
                confidence_color = ModernStyle.ERROR
                confidence_emoji = "🔴"
            
            self.confidence_label.config(
                text=f"{confidence_emoji} Confidence: {confidence:.2%}",
                fg=confidence_color
            )
            
            # Update top-3 predictions
            self.update_top3_display(probabilities)
            
            # Update preprocessed image preview
            self.update_preview(canvas_image)
            
        except Exception as e:
            error_msg = f"Prediction failed: {e}"
            messagebox.showerror("Prediction Error", error_msg)
    
    def update_top3_display(self, probabilities: list):
        """Update the top-3 predictions display with modern styling."""
        # Clear existing widgets
        for widget in self.top3_frame.winfo_children():
            widget.destroy()
        
        # Get top-3 predictions
        top3_indices = np.argsort(probabilities)[-3:][::-1]
        
        # Create labels for top-3 with modern styling
        for i, idx in enumerate(top3_indices):
            prob = probabilities[idx]
            
            # Color coding for top predictions
            if i == 0:
                color = ModernStyle.SUCCESS
                emoji = "🥇"
                font_weight = "bold"
            elif i == 1:
                color = ModernStyle.PRIMARY
                emoji = "🥈"
                font_weight = "normal"
            else:
                color = ModernStyle.TEXT_SECONDARY
                emoji = "🥉"
                font_weight = "normal"
            
            # Create prediction item frame
            item_frame = tk.Frame(self.top3_frame, bg=ModernStyle.BG_SECONDARY)
            item_frame.pack(fill=tk.X, pady=1)
            
            # Rank and digit
            rank_label = tk.Label(item_frame,
                                 text=f"{emoji} {idx}",
                                 font=('Segoe UI', 12, font_weight),
                                 fg=color,
                                 bg=ModernStyle.BG_SECONDARY)
            rank_label.pack(side=tk.LEFT)
            
            # Probability bar
            bar_width = 120
            bar_height = 6
            bar_frame = tk.Frame(item_frame, bg=ModernStyle.BORDER, width=bar_width, height=bar_height)
            bar_frame.pack(side=tk.RIGHT, padx=8)
            bar_frame.pack_propagate(False)
            
            # Progress bar
            progress_width = int(bar_width * prob)
            progress_bar = tk.Frame(bar_frame, bg=color, width=progress_width, height=bar_height)
            progress_bar.pack(side=tk.LEFT)
            
            # Percentage label
            percent_label = tk.Label(item_frame,
                                   text=f"{prob:.1%}",
                                   font=('Segoe UI', 9),
                                   fg=ModernStyle.TEXT_SECONDARY,
                                   bg=ModernStyle.BG_SECONDARY)
            percent_label.pack(side=tk.RIGHT, padx=4)
    
    def update_preview(self, canvas_image: Image.Image):
        """Update the preprocessed image preview with modern styling."""
        try:
            # Preprocess image for display
            preprocessed = preprocess_for_display(canvas_image)
            
            # Resize for display (make it larger for visibility)
            display_size = 140  # 5x larger than 28x28
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
                
                messagebox.showinfo("Success", f"💾 Image saved to:\n{file_path}")
                
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
