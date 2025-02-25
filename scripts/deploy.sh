#!/bin/bash

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    log_message "GPU detected - using GPU configuration"
    export USE_GPU=true
    export COMPOSE_FILE=docker-compose.gpu.yml
else
    log_message "No GPU detected - using CPU configuration"
    export USE_GPU=false
    export COMPOSE_FILE=docker-compose.yml
fi

# Build and deploy
log_message "Building Docker image..."
docker-compose -f $COMPOSE_FILE build --no-cache

log_message "Starting services..."
docker-compose -f $COMPOSE_FILE up -d

log_message "Deployment complete!"