# YOLOv12 Cheetah Detection Project

## Important: Settings Used for Results

The results in this project were produced using:
- **Epochs**: 100
- **Batch Size**: 95
- **Model**: YOLOv12n (nano)

Run with these exact settings to reproduce results:
```bash
python training.py --epochs 100 --batch 95
```

## Quick Start

### 1. Setup

**Recommended: Use the automated installation script**
```bash
python install_dependencies.py
```

**Alternative: Manual installation**
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install PyTorch with CUDA support (required for YOLOv12)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -r requirements.txt
```

**Note:** The installation script (`install_dependencies.py`) automatically handles PyTorch installation from the correct index and installs all dependencies. Use it for the easiest setup experience.

#### Updating Requirements

To update the `requirements.txt` file with the current environment:

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Freeze current dependencies
pip freeze > requirements.txt
```

**Note:** The current `requirements.txt` includes PyTorch with CUDA support. When freezing new requirements, remember to install PyTorch separately using the `--index-url` flag as shown in the manual installation section above.

### 2. Train the Model
```bash
# Activate virtual environment first
venv\Scripts\activate  # Windows

# Train with default settings
python training.py

# Or with custom parameters
python training.py --model-size s --epochs 100 --batch 95
```

### 3. Run the Gradio App
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows

# Start the detection app
python cheetah_detector_app.py
```

The app will be available at: `http://127.0.0.1:7860`

## Gradio App Features
- Upload images or videos for cheetah detection
- Adjustable confidence threshold (0.1 - 0.9)
- Image size selection (320 - 1024)
- Visual detection results with bounding boxes
- Video processing with frame-by-frame detection
- Test all images from the test dataset

## Training Options
- `--model-size`: Model size (n, s, m, l, x) - default: s
- `--epochs`: Number of epochs - default: 100
- `--batch`: Batch size - default: 95
- `--seed`: Random seed - default: 42

## Dataset
The dataset images in this project were annotated using **Label Studio**, a full-fledged open source solution for data labeling.

- **Label Studio instance**: [https://label.forgenetics.co.za/](https://label.forgenetics.co.za)
- **Tool**: [Label Studio](https://labelstudio.org/) - brought to you by Human Signal

## Project Structure
- `install_dependencies.py` - Automated dependency installation script
- `training.py` - Main training script
- `cheetah_detector_app.py` - Gradio web interface
- `cheetah_data/` - Dataset (train/val/test splits)
- `cheetah_detection/best_run/` - Best trained model
- Results saved to `cheetah_detection/run_YYYYMMDD_HHMMSS/`

## Student ID: U25748956
