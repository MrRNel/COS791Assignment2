FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA support
RUN pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY cheetah_detector_app.py .
COPY cheetah_detection/ ./cheetah_detection/

# Expose port
EXPOSE 7860

# Environment variables
ENV SERVER_HOST=0.0.0.0
ENV PORT=7860

# Run the app
CMD ["python", "cheetah_detector_app.py"]

