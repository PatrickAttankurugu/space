# API Documentation for ML System API

This documentation provides detailed information about the KYC and Document Verification API endpoints, their usage, and examples.

## Table of Contents
- [API Documentation for ML System API](#api-documentation-for-ml-system-api)
  - [Table of Contents](#table-of-contents)
  - [Base URL](#base-url)
  - [Authentication](#authentication)
  - [Endpoints](#endpoints)
    - [1. Face Comparison](#1-face-comparison)
- [Face Comparison Example](#face-comparison-example)
- [Ghana Card Verification Example](#ghana-card-verification-example)
- [Face Comparison](#face-comparison)
- [Ghana Card Verification](#ghana-card-verification)

## Base URL
http://localhost:8000/api/v1

## Authentication
- API key authentication required for face comparison endpoint
- Include API key in request header: `X-API-Key: your_api_key_here`
- Test environment key: `test_key_123`
- Production keys are provided separately

## Endpoints

### 1. Face Comparison
Compare two facial images and determine if they match.

**Endpoint:** `POST /kyc/compare-faces`

**Authentication Required:** Yes

**Request Headers:**
Content-Type: application/json
X-API-Key: your_api_key_here
**Request Body:**
```
{
    "image1_base64": "base64_encoded_string",
    "image2_base64": "base64_encoded_string"
}
Successful Response (200 OK):
{
    "match": true,
    "confidence": 98.5,
    "face_found": true,
    "similarity_score": 0.985,
    "image1_quality": {
        "overall_quality": 0.92,
        "brightness": 0.88,
        "contrast": 0.95,
        "sharpness": 0.90,
        "resolution": 1.0,
        "width": 800,
        "height": 600
    },
    "image2_quality": {
        "overall_quality": 0.90,
        "brightness": 0.85,
        "contrast": 0.92,
        "sharpness": 0.88,
        "resolution": 1.0,
        "width": 800,
        "height": 600
    },
    "face1_attributes": {
        "age": 25,
        "gender": "M",
        "bbox": [100, 100, 200, 200]
    },
    "face2_attributes": {
        "age": 26,
        "gender": "M",
        "bbox": [120, 120, 220, 220]
    },
    "processing_time": 1.23,
    "error_message": null
}
Error Response (400 Bad Request):
{
    "detail": "Invalid image format"
}
1. Ghana Card Verification
Verify Ghana card authenticity and extract MRZ data.
Endpoint: POST /documents/verify-ghana-card
Authentication Required: No
Request Headers:
Content-Type: application/json
Request Body:
{
    "front_image_base64": "base64_encoded_string",
    "back_image_base64": "base64_encoded_string"
}
Successful Response (200 OK):
{
    "is_valid": true,
    "confidence": 98.87,
    "detected_features": [
        "ECOWAS Logo",
        "Ghana Coat of Arms",
        "Ghana Flag",
        "Ghana Map"
    ],
    "feature_probabilities": {
        "ECOWAS Logo": 0.9997,
        "Ghana Coat of Arms": 0.9749,
        "Ghana Flag": 0.9828,
        "Ghana Map": 0.9928,
        "Valid Ghana Card": 0.9928
    },
    "num_features_detected": 4,
    "mrz_data": {
        "Document Type": "ID",
        "Country Code": "GHA",
        "Document Number": "GHA123456789",
        "Surname": "DOE",
        "Given Names": "JOHN",
        "Nationality": "GHA",
        "Date of Birth": "15th January, 1990",
        "Gender": "M",
        "Expiry Date": "14th January, 2025"
    },
    "error_message": null,
    "error_code": null,
    "processing_time": 1.05
}
Error Codes

| Code                 | Description                                     |
|----------------------|-------------------------------------------------|
| INVALID_IMAGE_FORMAT | Image format not supported or corrupted         |
| LOW_QUALITY_IMAGE    | Image quality below required threshold          |
| INSUFFICIENT_FEATURES| Not enough Ghana card features detected         |
| MRZ_NOT_READABLE     | Unable to read MRZ data from image              |
| INTERNAL_ERROR       | Server-side processing error                    |


Supported Formats: JPEG, JPG, PNG
Maximum Size: 10MB
Minimum Dimensions: 224x224 pixels
Maximum Dimensions: 4096x4096 pixels
Color Space: RGB
Base64 Encoding: Include image data only (exclude data:image/jpeg;base64, prefix)

Rate Limiting

60 requests per minute per API key
Temporary blocking after exceeding limit
Blocking duration: 5 minutes

Response Codes
| Code | Description                           |
|------|---------------------------------------|
| 200  | Success                               |
| 400  | Bad Request (invalid input)           |
| 401  | Unauthorized (missing API key)        |
| 403  | Forbidden (invalid API key)           |
| 429  | Too Many Requests                     |
| 500  | Internal Server Error                 |


Example Usage
Python

import requests
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Face Comparison Example
def compare_faces(image1_path, image2_path, api_key):
    response = requests.post(
        'http://localhost:8000/api/v1/kyc/compare-faces',
        headers={'X-API-Key': api_key},
        json={
            'image1_base64': encode_image(image1_path),
            'image2_base64': encode_image(image2_path)
        }
    )
    return response.json()

# Ghana Card Verification Example
def verify_ghana_card(front_image_path, back_image_path):
    response = requests.post(
        'http://localhost:8000/api/v1/documents/verify-ghana-card',
        json={
            'front_image_base64': encode_image(front_image_path),
            'back_image_base64': encode_image(back_image_path)
        }
    )
    return response.json()

cURL Examples

# Face Comparison
curl -X POST \
  'http://localhost:8000/api/v1/kyc/compare-faces' \
  -H 'X-API-Key: test_key_123' \
  -H 'Content-Type: application/json' \
  -d '{
    "image1_base64": "base64_string",
    "image2_base64": "base64_string"
  }'

# Ghana Card Verification
curl -X POST \
  'http://localhost:8000/api/v1/documents/verify-ghana-card' \
  -H 'Content-Type: application/json' \
  -d '{
    "front_image_base64": "base64_string",
    "back_image_base64": "base64_string"
  }'
Notes

All timestamps are in UTC
Base64 strings should be URL-safe
Processing times may vary based on server load and image size
Keep original image aspect ratio when resizing
Ensure good lighting and focus for optimal MRZ reading
For face comparison, ensure faces are clearly visible and well-lit

Best Practices

Always validate base64 strings before sending
Handle timeouts appropriately (recommended: 30s)
Implement proper error handling
Monitor rate limits
Store API keys securely
Compress images when possible while maintaining quality
Test with various image qualities and conditions