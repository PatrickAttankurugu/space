# PowerShell deployment script
param(
    [string]$Environment = "production"
)

# Configuration
$APP_NAME = "ml-api"
$DOCKER_IMAGE = "ml-api:latest"
$BACKUP_DIR = ".\backups"
$LOG_DIR = ".\logs"
$MODEL_DIR = ".\ml_models"
$TEMP_DIR = ".\temp_videos"
$HEALTH_CHECK_URLS = @(
    "http://localhost:8004/health",
    "http://localhost:8004/api/v1/deepfake/health/",
    "http://localhost:8004/api/v1/documents/health",
    "http://localhost:8004/api/v1/kyc/health",
    "http://localhost:8004/api/v1/license/health"
)

# Create directories if they don't exist
New-Item -ItemType Directory -Force -Path $BACKUP_DIR
New-Item -ItemType Directory -Force -Path $LOG_DIR
New-Item -ItemType Directory -Force -Path $MODEL_DIR
New-Item -ItemType Directory -Force -Path $TEMP_DIR

# Logging function
function Write-Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "[$timestamp] $Message"
    Add-Content -Path "$LOG_DIR\deployment.log" -Value "[$timestamp] $Message"
}

# Backup function
function Backup-Current {
    Write-Log "Creating backup..."
    $timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
    $container = docker ps -q -f name=$APP_NAME
    if ($container) {
        docker commit $container "${APP_NAME}:backup_${timestamp}"
        docker save "${APP_NAME}:backup_${timestamp}" -o "$BACKUP_DIR\backup_${timestamp}.tar"
        Write-Log "Backup created: backup_${timestamp}.tar"
    }
}

# Health check function
function Test-Health {
    Write-Log "Performing health checks..."
    $maxAttempts = 30
    $attempt = 1
    
    while ($attempt -le $maxAttempts) {
        $allHealthy = $true
        foreach ($url in $HEALTH_CHECK_URLS) {
            try {
                Write-Log "Checking $url..."
                $response = Invoke-WebRequest -Uri $url -UseBasicParsing
                if ($response.StatusCode -ne 200) {
                    $allHealthy = $false
                    Write-Log "Health check failed for $url: $($response.StatusCode)"
                }
            }
            catch {
                $allHealthy = $false
                Write-Log "Health check attempt $attempt failed for $url"
            }
        }
        
        if ($allHealthy) {
            Write-Log "All health checks passed!"
            return $true
        }
        
        $attempt++
        Start-Sleep -Seconds 2
    }
    return $false
}

# Main deployment process
try {
    Write-Log "Starting deployment process..."
    
    # Backup current deployment
    Backup-Current
    
    # Pull latest changes
    Write-Log "Pulling latest changes..."
    git pull origin main
    
    # Copy ML models if they exist in specified location
    if (Test-Path ".\ml_models_backup") {
        Write-Log "Copying ML models..."
        Copy-Item ".\ml_models_backup\*" ".\ml_models\" -Recurse -Force
    }
    
    # Build new Docker image
    Write-Log "Building Docker image..."
    docker build -t $DOCKER_IMAGE .
    
    # Stop and remove existing container
    $container = docker ps -q -f name=$APP_NAME
    if ($container) {
        Write-Log "Stopping existing container..."
        docker stop $APP_NAME
        docker rm $APP_NAME
    }
    
    # Start new container
    Write-Log "Starting new container..."
    docker run -d `
        --name $APP_NAME `
        --restart unless-stopped `
        -p 8004:8004 `
        --env-file ".\deployment\config\.env.prod" `
        -v "${PWD}\logs:/app/logs" `
        -v "${PWD}\ml_models:/app/ml_models" `
        -v "${PWD}\temp_videos:/app/temp_videos" `
        $DOCKER_IMAGE
    
    # Perform health check
    if (-not (Test-Health)) {
        throw "Health checks failed"
    }
    
    Write-Log "Deployment completed successfully!"
}
catch {
    Write-Log "Error during deployment: $_"
    Write-Log "Starting rollback process..."
    # Implement rollback logic here
    exit 1
}