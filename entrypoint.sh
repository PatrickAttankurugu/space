#!/bin/bash
set -e

# Function to log messages with timestamps
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to wait for Redis
wait_for_redis() {
    log_message "Waiting for Redis..."
    until timeout 1 bash -c "echo > /dev/tcp/redis/6379" >/dev/null 2>&1
    do
        log_message "Redis is unavailable - sleeping"
        sleep 1
    done
    log_message "Redis is available"
}

# Function to check GPU availability
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        log_message "GPU detected: $(nvidia-smi --query-gpu=gpu_name --format=csv,noheader)"
        return 0
    else
        log_message "No GPU detected, using CPU mode"
        return 1
    fi
}

# Function to optimize GPU memory and setup CUDA providers
optimize_gpu() {
    log_message "Optimizing GPU settings..."
    
    # CUDA Core Settings
    export CUDA_VISIBLE_DEVICES=0
    export NVIDIA_VISIBLE_DEVICES=all
    export NVIDIA_DRIVER_CAPABILITIES=compute,utility,video
    export CUDA_MODULE_LOADING=LAZY
    
    # Memory Management
    export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
    export CUDA_CACHE_DISABLE=0
    export CUDA_CACHE_PATH=/tmp/cuda-cache
    
    # ONNX Runtime Settings
    export ONNXRUNTIME_PROVIDERS='["CUDAExecutionProvider", "CPUExecutionProvider"]'
    export ONNXRUNTIME_DEVICE='cuda:0'
    export ONNXRUNTIME_CUDA_DEVICE_ID=0
    export ONNXRUNTIME_PROVIDER_PRIORITIES='CUDAExecutionProvider,CPUExecutionProvider'
    
    # InsightFace Settings
    export INSIGHTFACE_USE_CUDA=1
    export INSIGHTFACE_CTX_ID=0
    
    # Thread Settings
    export OMP_NUM_THREADS=1
    export MKL_NUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    
    # Create CUDA cache directory
    mkdir -p /tmp/cuda-cache
    chmod 777 /tmp/cuda-cache
    
    log_message "GPU optimization completed"
}

# Function to setup working directory
setup_workspace() {
    log_message "Setting up workspace..."
    mkdir -p /app/logs
    mkdir -p /app/temp_videos
    chmod -R 777 /app/logs /app/temp_videos
}

# Main entrypoint logic
case "$1" in
    "celery")
        wait_for_redis
        if check_gpu; then
            optimize_gpu
        fi
        log_message "Starting Celery worker..."
        exec celery -A app.celery_app.celery worker \
            --pool=solo \
            --loglevel=INFO \
            --concurrency=1 \
            --max-tasks-per-child=50 \
            -Q high_priority,default,low_priority \
            --prefetch-multiplier=1 \
            --time-limit=3600 \
            --soft-time-limit=3300 \
            --max-memory-per-child=8000000
        ;;
        
    "celery-gpu")
        wait_for_redis
        if ! check_gpu; then
            log_message "ERROR: GPU mode requested but no GPU detected"
            exit 1
        fi
        optimize_gpu
        log_message "Starting Celery worker with GPU optimization..."
        exec celery -A app.celery_app.celery worker \
            --pool=solo \
            --loglevel=INFO \
            --concurrency=1 \
            --max-tasks-per-child=50 \
            -Q high_priority,default,low_priority \
            --prefetch-multiplier=1 \
            --time-limit=3600 \
            --soft-time-limit=3300 \
            --max-memory-per-child=8000000
        ;;
        
    "api")
        setup_workspace
        if [ "$USE_GPU" = "true" ]; then
            if check_gpu; then
                optimize_gpu
            else
                log_message "WARNING: GPU requested but not available, falling back to CPU"
            fi
        fi
        log_message "Starting FastAPI application..."
        exec uvicorn app.main:app \
            --host 0.0.0.0 \
            --port 8004 \
            --workers 1 \
            --timeout-keep-alive 75 \
            --log-level info \
            --limit-concurrency 50 \
            --backlog 2048 \
            --loop uvloop \
            --http httptools
        ;;
        
    *)
        log_message "Starting custom command..."
        if [ "$USE_GPU" = "true" ]; then
            if check_gpu; then
                optimize_gpu
            fi
        fi
        exec "$@"
        ;;
esac