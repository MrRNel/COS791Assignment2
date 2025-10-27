# Dockerfile for YOLOv12 Cheetah Detection Training
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libgthread-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install CPU-only PyTorch
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Copy application code
COPY training.py .
COPY cheetah_detector_app.py .

# Copy data directory (cheetah_data will be large, consider using volume mount instead)
COPY cheetah_data/ cheetah_data/

# Copy pretrained models
COPY yolo12n.pt .
COPY yolo12s.pt .

# Create output directories and best_run structure
RUN mkdir -p cheetah_detection/best_run output

# Copy the trained model
COPY cheetah_detection/best_run/ cheetah_detection/best_run/

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Expose ports (for Gradio app if needed)
EXPOSE 7860

# Default command (can be overridden)
CMD ["python", "cheetah_detector_app.py"]

