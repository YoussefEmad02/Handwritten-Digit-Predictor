# Handwritten Digit Predictor

A Python application that predicts handwritten digits (0-9) drawn on a canvas using a deep learning model trained on the MNIST dataset.

## Features

- **High Accuracy**: ≥99% test accuracy on MNIST dataset
- **Interactive GUI**: Draw digits with mouse and get instant predictions
- **Modular Design**: Clean, organized code structure with separate modules
- **Real-time Inference**: Fast prediction (<50ms on CPU)
- **Robust Preprocessing**: Handles various stroke styles and positions

## Quick Start

### Option 1: Interactive Setup (Recommended)
For the easiest setup experience, use our interactive quick start script:

**Windows users:**
```bash
# Double-click quick_start.bat or run:
quick_start.bat
# or run:
python quick_start.py
```

**macOS/Linux users:**
```bash
python quick_start.py
```

This will guide you through the entire setup process automatically!

### Option 2: Manual Setup

#### 1. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 3. Train the Model
```bash
python -m src.train
```
This will:
- Download MNIST dataset automatically (~11MB)
- Train a CNN model with data augmentation (10-30 minutes)
- Save the best model checkpoint to `models/best_model.pt`
- Generate training curves in `assets/training_curve.png`

#### 4. Evaluate the Model
```bash
python -m src.evaluate
```
This will:
- Load the trained model and evaluate on test set
- Print test accuracy and confusion matrix
- Save confusion matrix visualization to `assets/confusion_matrix.png`

#### 5. Launch the GUI
```bash
python -m src.gui
```
This will:
- Open a drawing canvas where you can draw digits
- Click "Predict" to get digit predictions with probabilities
- Use "Clear" to erase the canvas

## Project Structure

```
handwrite_predictor/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── quick_start.py            # Interactive setup guide
├── quick_start.bat           # Windows quick start (double-click)
├── setup.py                  # Automated setup script
├── demo.py                   # Demo script with synthetic data
├── run_tests.py              # Test runner
├── src/                      # Source code
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── data.py               # Data loading and augmentation
│   ├── model.py              # CNN model architecture
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation script
│   ├── inference.py          # Inference pipeline
│   ├── preprocess.py         # Image preprocessing
│   ├── gui.py                # Tkinter GUI
│   └── utils/                # Utility functions
│       ├── __init__.py
│       ├── metrics.py        # Evaluation metrics
│       ├── seed.py           # Random seed management
│       └── transforms.py     # Data augmentation transforms
├── models/                   # Trained model checkpoints
├── assets/                   # Generated visualizations
└── tests/                    # Unit tests
    ├── __init__.py
    ├── test_preprocess.py    # Preprocessing tests
    └── test_inference.py     # Inference tests
```

## Model Architecture

The CNN model consists of:
- 2-3 convolutional layers with ReLU activation and batch normalization
- Dropout layers for regularization
- Fully connected layers for classification
- Optimized for MNIST digit recognition

## Preprocessing Pipeline

The preprocessing converts drawn canvas images to MNIST-compatible format:
1. **Inversion**: Converts white-on-black to black-on-white if needed
2. **Centering**: Automatically centers the digit in the image
3. **Resizing**: Scales to 28×28 pixels while maintaining aspect ratio
4. **Normalization**: Applies MNIST-style normalization
5. **Thresholding**: Converts to binary image with proper contrast

## Troubleshooting

### Common Issues

**White/Black Inverted**: The preprocessing automatically detects and corrects color inversion.

**Model Not Found**: If you get a "Model not found" error, run the training script first:
```bash
python -m src.train
```

**Poor Prediction Accuracy**: 
- Ensure the digit is drawn clearly in the center of the canvas
- Try drawing with consistent stroke thickness
- Make sure the digit fills most of the canvas area

**GUI Not Responding**: 
- Check if all dependencies are installed correctly
- Ensure you have a compatible Python version (3.10+)

### Performance Tips

- For faster inference, the model automatically uses GPU if available
- The GUI is optimized for real-time drawing and prediction
- Model loading happens once at startup for optimal performance

## CLI Commands

### Core Commands
* `python -m src.train` → trains model and saves best checkpoint
* `python -m src.evaluate` → evaluates on test set  
* `python -m src.gui` → launches the GUI

### Utility Commands
* `python demo.py` → runs demo with synthetic data
* `python run_tests.py` → runs all unit tests
* `python setup.py` → automated setup and dependency installation
* `python quick_start.py` → interactive setup guide

### Windows Users
* Double-click `quick_start.bat` for easy setup

## Technical Details

- **Framework**: PyTorch for deep learning
- **GUI**: Tkinter for cross-platform compatibility
- **Image Processing**: Pillow, OpenCV, and NumPy for robust image handling
- **Data Augmentation**: Random shifts, rotations, elastic distortions, and noise
- **Training**: Adam optimizer with learning rate scheduling and early stopping
- **Model**: CNN with batch normalization, dropout, and global average pooling

## License

This project is open source and available under the MIT License.
