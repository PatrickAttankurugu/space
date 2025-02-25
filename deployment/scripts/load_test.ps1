# PowerShell load testing script
param(
    [int]$Concurrent = 10,
    [int]$TotalRequests = 100,
    [int]$DurationSeconds = 60,
    [string]$BaseUrl = "http://localhost:8000"
)

$LOG_FILE = ".\logs\load_test.log"

function Write-Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $LOG_FILE -Value $logMessage
}

# Test endpoints configuration
$ENDPOINTS = @(
    @{
        Name = "Health Check"
        Method = "GET"
        Url = "$BaseUrl/health"
        Weight = 20  # 20% of requests
    },
    @{
        Name = "DeepFake Health"
        Method = "GET"
        Url = "$BaseUrl/api/v1/deepfake/health/"
        Weight = 15  # 15% of requests
    },
    @{
        Name = "Documents Health"
        Method = "GET"
        Url = "$BaseUrl/api/v1/documents/health"
        Weight = 15  # 15% of requests
    },
    @{
        Name = "License Health"
        Method = "GET"
        Url = "$BaseUrl/api/v1/license/health"
        Weight = 15  # 15% of requests
    },
    @{
        Name = "KYC Health"
        Method = "GET"
        Url = "$BaseUrl/api/v1/kyc/health"
        Weight = 15  # 15% of requests
    },
    @{
        Name = "Document Verification"
        Method = "POST"
        Url = "$BaseUrl/api/v1/documents/verify"
        Body = '{"image_path": "test_image.jpg"}'
        Weight = 10  # 10% of requests
    },
    @{
        Name = "DeepFake Detection"
        Method = "POST"
        Url = "$BaseUrl/api/v1/deepfake/analyze/"
        Body = '{"video_path": "test_video.mp4"}'
        Weight = 10  # 10% of requests
    }
)

# Statistics tracking
$stats = @{
    TotalRequests = 0
    SuccessfulRequests = 0
    FailedRequests = 0
    ResponseTimes = @()
    StartTime = Get-Date
}

function Get-WeightedEndpoint {
    $random = Get-Random -Minimum 1 -Maximum 101
    $currentWeight = 0
    foreach ($endpoint in $ENDPOINTS) {
        $currentWeight += $endpoint.Weight
        if ($random -le $currentWeight) {
            return $endpoint
        }
    }
    return $ENDPOINTS[0]
}

function Invoke-EndpointTest {
    param($Endpoint)
    
    $startTime = Get-Date
    try {
        $params = @{
            Uri = $Endpoint.Url
            Method = $Endpoint.Method
            UseBasicParsing = $true
        }
        
        if ($Endpoint.Body) {
            $params.Body = $Endpoint.Body
            $params.ContentType = "application/json"
        }
        
        $response = Invoke-WebRequest @params
        $endTime = Get-Date
        $responseTime = ($endTime - $startTime).TotalMilliseconds
        
        return @{
            Success = $true
            ResponseTime = $responseTime
            StatusCode = $response.StatusCode
        }
    }
    catch {
        return @{
            Success = $false
            StatusCode = $_.Exception.Response.StatusCode.value__
            Error = $_.Exception.Message
        }
    }
}

Write-Log "Starting load test..."
Write-Log "Configuration: $Concurrent concurrent users, $TotalRequests total requests over $DurationSeconds seconds"

try {
    1..$TotalRequests | ForEach-Object -ThrottleLimit $Concurrent -Parallel {
        $endpoint = Get-WeightedEndpoint
        $result = Invoke-EndpointTest -Endpoint $endpoint
        
        $stats.TotalRequests++
        if ($result.Success) {
            $stats.SuccessfulRequests++
            $stats.ResponseTimes += $result.ResponseTime
        } else {
            $stats.FailedRequests++
        }
        
        if ($result.Success) {
            Write-Log "✅ ${endpoint.Name}: ${result.StatusCode} (${result.ResponseTime}ms)"
        } else {
            Write-Log "❌ ${endpoint.Name}: ${result.StatusCode} - ${result.Error}"
        }
    }
} finally {
    $endTime = Get-Date
    $duration = ($endTime - $stats.StartTime).TotalSeconds
    
    Write-Log "Load test completed in $duration seconds"
    Write-Log "Total Requests: $($stats.TotalRequests)"
    Write-Log "Successful: $($stats.SuccessfulRequests)"
    Write-Log "Failed: $($stats.FailedRequests)"
    Write-Log "Average Response Time: $($stats.ResponseTimes | Measure-Object -Average).Average ms"
    Write-Log "Requests/Second: $([math]::Round($stats.TotalRequests / $duration, 2))"
} 