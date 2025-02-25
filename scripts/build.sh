#!/bin/bash

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check command status and exit if failed
check_status() {
    if [ $? -ne 0 ]; then
        log_message "ERROR: $1"
        exit 1
    fi
}

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log_message "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Check for .env.prod file
if [ ! -f "deployment/config/.env.prod" ]; then
    log_message "ERROR: .env.prod file not found!"
    exit 1
fi

# Verify model sizes before building
log_message "Verifying ML model sizes..."
declare -A EXPECTED_SIZES=(
    ["ml_models/document_verification/ghana_card_modelV2.pth"]="1000000000"  # 1GB minimum
)

for model in "${!EXPECTED_SIZES[@]}"; do
    if [ -f "$model" ]; then
        size=$(stat -f%z "$model" 2>/dev/null || stat -c%s "$model" 2>/dev/null)
        if [ "$size" -lt "${EXPECTED_SIZES[$model]}" ]; then
            log_message "WARNING: $model seems too small. It might be corrupted."
            log_message "Expected: ${EXPECTED_SIZES[$model]} bytes, Got: $size bytes"
            log_message "Would you like to redownload? (y/n)"
            read -r response
            if [ "$response" = "y" ]; then
                ./scripts/setup.sh
                check_status "Failed to redownload models"
            fi
        fi
    else
        log_message "ERROR: Required model $model not found!"
        exit 1
    fi
done

# Build the Docker image with fallback options
log_message "Building Docker image..."
if ! docker-compose -f deployment/templates/docker-compose.yml build --no-cache; then
    log_message "WARNING: Full build failed, trying alternative build..."
    # Fallback to CPU-only build if GPU build fails
    export DOCKER_BUILDKIT=1
    export COMPOSE_DOCKER_CLI_BUILD=1
    if ! docker build -t ml-api:latest --build-arg USE_GPU=false .; then
        log_message "ERROR: Both GPU and CPU builds failed!"
        exit 1
    fi
    log_message "✓ CPU-only build completed successfully"
else
    log_message "✓ GPU-enabled build completed successfully"
fi

log_message "Build completed successfully!"

# Create model directories if they don't exist
mkdir -p ml_models/{document_verification,face/models}

# Download models if they don't exist
if [ ! -d "ml_models/face/models/buffalo_l" ]; then
    python app/scripts/download_insightface_models.py
fi