# PowerShell monitoring script
param(
    [int]$CheckInterval = 300,  # 5 minutes
    [int]$CpuThreshold = 80,
    [int]$MemoryThreshold = 80
)

$APP_NAME = "ml-api"
$LOG_FILE = ".\logs\monitor.log"

function Write-Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $LOG_FILE -Value $logMessage
}

function Get-ContainerStats {
    $stats = docker stats --no-stream --format "{{.CPUPerc}};{{.MemPerc}}" $APP_NAME
    return $stats -split ";"
}

while ($true) {
    try {
        $container = docker ps -q -f name=$APP_NAME
        if (-not $container) {
            Write-Log "⚠️ Container is not running!"
            continue
        }

        $stats = Get-ContainerStats
        $cpuUsage = $stats[0] -replace '%',''
        $memUsage = $stats[1] -replace '%',''

        Write-Log "CPU Usage: ${cpuUsage}% | Memory Usage: ${memUsage}%"

        if ([double]$cpuUsage -gt $CpuThreshold) {
            Write-Log "⚠️ High CPU usage alert: ${cpuUsage}%"
        }

        if ([double]$memUsage -gt $MemoryThreshold) {
            Write-Log "⚠️ High memory usage alert: ${memUsage}%"
        }
    }
    catch {
        Write-Log "Error monitoring container: $_"
    }

    Start-Sleep -Seconds $CheckInterval
}