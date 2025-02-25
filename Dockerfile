# Build stage
FROM python:3.10-slim as builder

# Build arguments and environment variables
ARG USE_GPU=false
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    swig \
    curl \
    git \
    tesseract-ocr \
    libtesseract-dev \
    python3-opencv \
    libmupdf-dev \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements files
COPY pre_requirements.txt .

# Install dependencies based on GPU/CPU flag
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy==1.24.3 && \
    pip install --no-cache-dir fastapi==0.115.6 && \
    pip install --no-cache-dir uvicorn==0.34.0 && \
    if [ "$USE_GPU" = "true" ] ; then \
        pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cu121 && \
        pip install --no-cache-dir onnxruntime-gpu==1.16.3 && \
        pip install --no-cache-dir tensorflow==2.15.0 && \
        # Explicitly install ONNX Runtime GPU dependencies
        pip install --no-cache-dir nvidia-cublas-cu12 nvidia-cudnn-cu12 && \
        # Set ONNX Runtime configurations
        echo "export ONNXRUNTIME_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider" >> /opt/venv/bin/activate && \
        echo "export ONNXRUNTIME_DEVICE=cuda:0" >> /opt/venv/bin/activate && \
        echo "export ONNXRUNTIME_CUDA_DEVICE_ID=0" >> /opt/venv/bin/activate ; \
    else \
        pip install --no-cache-dir torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu && \
        pip install --no-cache-dir onnxruntime==1.15.1 && \
        pip install --no-cache-dir tensorflow-cpu==2.15.0 ; \
    fi && \
    pip install --no-cache-dir -r pre_requirements.txt

# Install InsightFace and EasyOCR with GPU support
RUN if [ "$USE_GPU" = "true" ] ; then \
        ONNXRUNTIME_PROVIDERS=CUDAExecutionProvider,CPUExecutionProvider \
        pip install --no-cache-dir easyocr insightface==0.7.3 ; \
    else \
        pip install --no-cache-dir easyocr insightface==0.7.3 ; \
    fi

# Add Celery installation
RUN pip install --no-cache-dir "celery[redis]==5.3.6" redis==5.0.1

# Pre-download models during build
COPY app/scripts/download_ocr_models.py /app/scripts/
COPY app/scripts/download_insightface_models.py /app/scripts/
RUN mkdir -p /ml_models/face/models && \
    python /app/scripts/download_ocr_models.py && \
    python /app/scripts/download_insightface_models.py

# Final stage
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Set environment variables for noninteractive installation
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    PYTHONUNBUFFERED=1

# Copy Python from builder
COPY --from=builder /usr/local /usr/local

# Copy virtual environment and models from builder
COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /app/models/ocr /app/models/ocr
COPY --from=builder /ml_models /ml_models
ENV PATH="/opt/venv/bin:$PATH"

# Install runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    python3-opencv \
    curl \
    tesseract-ocr \
    libmupdf-dev \
    swig \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uninstall current PyMuPDF and install specific version
RUN pip uninstall -y PyMuPDF fitz && \
    pip install --no-cache-dir PyMuPDF==1.21.1

# Explicitly install pydantic and pydantic-settings
RUN pip install --no-cache-dir pydantic==2.10.3 pydantic-settings==2.7.0

# Copy application code
COPY ./app /app/app

# GPU support and environment setup
ARG USE_GPU=false
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONMALLOC=malloc \
    MALLOC_TRIM_THRESHOLD_=100000 \
    PORT=8004 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video \
    TORCH_CUDA_ARCH_LIST="7.5" \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_REQUIRE_CUDA="cuda>=12.8" \
    NUM_WORKERS=1 \
    WORKER_TIMEOUT=300 \
    MAX_REQUESTS_JITTER=50 \
    BACKLOG=2048 \
    MAX_BATCH_SIZE=32 \
    MAX_MEMORY_USAGE=0.8 \
    # ONNX Runtime configurations
    ONNXRUNTIME_PROVIDERS="CUDAExecutionProvider,CPUExecutionProvider" \
    ONNXRUNTIME_DEVICE="cuda:0" \
    ONNXRUNTIME_CUDA_DEVICE_ID=0 \
    INSIGHTFACE_USE_CUDA=1 \
    ONNXRUNTIME_PROVIDER_PRIORITIES="CUDAExecutionProvider,CPUExecutionProvider" \
    CUDA_MODULE_LOADING=LAZY \
    # Celery environment variables
    CELERY_BROKER_URL="redis://redis:6379/0" \
    CELERY_RESULT_BACKEND="redis://redis:6379/0"

# Set working directory and copy application
WORKDIR /app
COPY . .

# Create necessary directories
RUN mkdir -p temp_videos && \
    chmod -R 755 temp_videos

# Original healthcheck
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8004/health || exit 1

# Run application with optimized settings for GPU
CMD ["/opt/venv/bin/uvicorn", \
     "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8004", \
     "--workers", "1", \
     "--timeout-keep-alive", "75", \
     "--log-level", "info", \
     "--limit-concurrency", "50", \
     "--backlog", "2048"]