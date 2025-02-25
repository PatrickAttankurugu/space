#!/bin/bash

# Exit on error
set -e

# Function to log messages
log_message() {
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] $1"
}

# Check for NVIDIA GPU and drivers
check_gpu() {
    if ! command -v nvidia-smi &> /dev/null; then
        log_message "ERROR: NVIDIA driver not found. Please install NVIDIA drivers first."
        exit 1
    fi
    
    if ! nvidia-smi &> /dev/null; then
        log_message "ERROR: NVIDIA driver not working properly."
        exit 1
    fi
    
    log_message "GPU detected: $(nvidia-smi -L)"
}

# Install Docker if not present
install_docker() {
    if ! command -v docker &> /dev/null; then
        log_message "Installing Docker..."
        curl -fsSL https://get.docker.com -o get-docker.sh
        sudo sh get-docker.sh
        sudo usermod -aG docker $USER
        rm get-docker.sh
    fi
}

# Install NVIDIA Container Toolkit
install_nvidia_toolkit() {
    if ! command -v nvidia-container-toolkit &> /dev/null; then
        log_message "Installing NVIDIA Container Toolkit..."
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
        curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
        sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
        sudo systemctl restart docker
    fi
}

# Check system requirements
check_requirements() {
    # Check CPU cores
    cpu_cores=$(nproc)
    if [ "$cpu_cores" -lt 8 ]; then
        log_message "WARNING: Less than 8 CPU cores available ($cpu_cores cores)"
    fi

    # Check memory
    total_mem=$(free -g | awk '/^Mem:/{print $2}')
    if [ "$total_mem" -lt 30 ]; then
        log_message "WARNING: Less than 30GB RAM available (${total_mem}GB)"
    fi

    # Check disk space
    free_space=$(df -BG / | awk 'NR==2 {print $4}' | sed 's/G//')
    if [ "$free_space" -lt 50 ]; then
        log_message "WARNING: Less than 50GB free disk space available (${free_space}GB)"
    fi
}

# Deploy application
deploy_application() {
    log_message "Starting deployment..."
    
    # Export GPU environment variables
    export USE_GPU=true
    export NVIDIA_VISIBLE_DEVICES=all
    export CUDA_VISIBLE_DEVICES=0
    
    # Pull latest code if in git repository
    if [ -d .git ]; then
        git pull
    fi
    
    # Build and start services
    log_message "Building and starting services..."
    docker-compose -f docker-compose.yml build --no-cache
    docker-compose -f docker-compose.yml up -d
    
    # Wait for services to be healthy
    log_message "Waiting for services to be healthy..."
    sleep 30
    
    # Check service health
    if curl -f http://localhost:8004/health &> /dev/null; then
        log_message "✓ API service is healthy"
    else
        log_message "ERROR: API service is not healthy"
        exit 1
    fi
    
    # Check Celery workers
    if curl -f http://localhost:5555 &> /dev/null; then
        log_message "✓ Celery monitoring is accessible"
    else
        log_message "WARNING: Celery monitoring is not accessible"
    fi
}

# Main deployment process
main() {
    log_message "Starting deployment process for AWS g4dn.2xlarge..."
    
    check_gpu
    install_docker
    install_nvidia_toolkit
    check_requirements
    deploy_application
    
    log_message "Deployment completed successfully!"
}

# Run main function
main