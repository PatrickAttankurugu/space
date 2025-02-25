#!/bin/bash

# Create directories
mkdir -p logs
mkdir -p ml_models/{document_verification,face/models}
mkdir -p temp_videos
mkdir -p static
mkdir -p deployment/config

# Install gdown if not present
if ! pip show gdown > /dev/null 2>&1; then
    echo "Installing gdown..."
    pip install gdown
fi

# Download ML models
if [ ! -f "ml_models/document_verification/ghana_card_modelV2.pth" ]; then
    echo "Downloading Ghana Card Model..."
    python -m gdown "https://drive.google.com/uc?id=14PycPjUl3V_csDy7Z_1cxPbGoSEW6e3M" -O "ml_models/document_verification/ghana_card_modelV2.pth"
fi

# Download Deepfake Models
if [ ! -f "ml_models/deepfake/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36" ]; then
    echo "Downloading Deepfake Model 1..."
    python -m gdown "https://drive.google.com/uc?id=1Q8EDSx1jOFx4SGv90YkEVeVnksADjHcm" -O "ml_models/deepfake/final_111_DeepFakeClassifier_tf_efficientnet_b7_ns_0_36"
fi

if [ ! -f "ml_models/deepfake/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19" ]; then
    echo "Downloading Deepfake Model 2..."
    python -m gdown "https://drive.google.com/uc?id=1ypnKmX7NvNfo6RYcOWZehEDQEHQScs1O" -O "ml_models/deepfake/final_555_DeepFakeClassifier_tf_efficientnet_b7_ns_0_19"
fi

if [ ! -f "ml_models/deepfake/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31" ]; then
    echo "Downloading Deepfake Model 3..."
    python -m gdown "https://drive.google.com/uc?id=1M_VRMvLjC3WLgMjH9eIszC5x7wbSG1YR" -O "ml_models/deepfake/final_777_DeepFakeClassifier_tf_efficientnet_b7_ns_0_31"
fi

# Copy environment file if it doesn't exist
if [ ! -f "deployment/config/.env.prod" ]; then
    echo "Creating environment file from template..."
    cp "deployment/config/.env.example" "deployment/config/.env.prod"
    echo "Please update deployment/config/.env.prod with your settings"
fi

echo "Setup completed successfully!"