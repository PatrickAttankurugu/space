#!/bin/bash

# Ensure we're on deployment branch
if [[ $(git branch --show-current) != "deployment" ]]; then
    echo "Not on deployment branch!"
    exit 1
fi

# Build and test locally
docker-compose -f docker-compose.prod.yml build
docker-compose -f docker-compose.prod.yml up -d
sleep 10
curl http://localhost:8004/health

# If health check passes, proceed with deployment
if [ $? -eq 0 ]; then
    ./scripts/deploy.sh
else
    echo "Health check failed!"
    exit 1
fi 