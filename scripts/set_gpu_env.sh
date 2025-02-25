#!/bin/bash

# Check for NVIDIA GPU
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "GPU detected - enabling GPU support"
    export USE_GPU=true
    export COMPOSE_FILE=docker-compose.gpu.yml
    export INSIGHTFACE_CTX_ID=0
else
    echo "No GPU detected - using CPU configuration"
    export USE_GPU=false
    export COMPOSE_FILE=docker-compose.yml
    export INSIGHTFACE_CTX_ID=-1
fi 