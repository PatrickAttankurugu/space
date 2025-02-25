param(
    [string]$Environment = "production",
    [string]$BaseUrl = "http://localhost:8000"
)

$LOG_FILE = ".\logs\endpoint_tests.log"

function Write-Log {
    param($Message)
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] $Message"
    Write-Host $logMessage
    Add-Content -Path $LOG_FILE -Value $logMessage
}

function Test-Endpoint {
    param(
        $Url,
        $Method = "GET",
        $Body = $null,
        $ContentType = "application/json"
    )
    try {
        $params = @{
            Uri = $Url
            Method = $Method
            UseBasicParsing = $true
        }
        
        if ($Body) {
            $params.Body = $Body
            $params.ContentType = $ContentType
        }
        
        $response = Invoke-WebRequest @params
        return @{
            Status = $response.StatusCode
            Success = $response.StatusCode -eq 200
            Content = $response.Content
        }
    } catch {
        return @{
            Status = $_.Exception.Response.StatusCode.value__
            Success = $false
            Error = $_.Exception.Message
        }
    }
}

# Define all endpoints
$ENDPOINTS = @(
    @{
        Name = "Main Health Check"
        Url = "$BaseUrl/health"
        Method = "GET"
    },
    @{
        Name = "DeepFake Health Check"
        Url = "$BaseUrl/api/v1/deepfake/health/"
        Method = "GET"
    },
    @{
        Name = "Documents Health Check"
        Url = "$BaseUrl/api/v1/documents/health"
        Method = "GET"
    },
    @{
        Name = "License Health Check"
        Url = "$BaseUrl/api/v1/license/health"
        Method = "GET"
    },
    @{
        Name = "KYC Health Check"
        Url = "$BaseUrl/api/v1/kyc/health"
        Method = "GET"
    },
    @{
        Name = "Document Verification"
        Url = "$BaseUrl/api/v1/documents/verify"
        Method = "POST"
        Body = '{"image_path": "test_image.jpg"}'
    },
    @{
        Name = "DeepFake Detection"
        Url = "$BaseUrl/api/v1/deepfake/analyze/"
        Method = "POST"
        Body = '{"video_path": "test_video.mp4"}'
    },
    @{
        Name = "Face Comparison"
        Url = "$BaseUrl/api/v1/kyc/compare-faces"
        Method = "POST"
        Body = '{"image1_path": "face1.jpg", "image2_path": "face2.jpg"}'
    }
)

# Run tests
Write-Log "Starting endpoint tests..."
Write-Log "Environment: $Environment"
Write-Log "Base URL: $BaseUrl"
Write-Log "------------------------"

foreach ($endpoint in $ENDPOINTS) {
    Write-Log "Testing endpoint: $($endpoint.Name) - $($endpoint.Method) $($endpoint.Url)"
    $result = Test-Endpoint -Url $endpoint.Url -Method $endpoint.Method -Body $endpoint.Body
    
    if ($result.Success) {
        Write-Log "✅ Success (${result.Status})"
        Write-Log "Response: $($result.Content)"
    } else {
        Write-Log "❌ Failed (${result.Status}): $($result.Error)"
    }
    Write-Log "------------------------"
}

Write-Log "Endpoint tests completed"
