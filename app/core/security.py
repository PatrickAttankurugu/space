from fastapi import Security, HTTPException, Request
from fastapi.security.api_key import APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.status import HTTP_403_FORBIDDEN, HTTP_429_TOO_MANY_REQUESTS
import secrets
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Set
from app.core.config import settings

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize API key security scheme
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=True)

# Rate limiting storage (in-memory for development)
# In production, use Redis or a database
request_counts: Dict[str, list] = {}
blocked_ips: Dict[str, float] = {}
suspicious_activities: Dict[str, list] = {}
failed_attempts: Dict[str, int] = {}

class SecurityViolation(Exception):
    """Custom exception for security violations"""
    def __init__(self, message: str, details: dict = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

def clean_expired_data() -> None:
    """Clean expired data from tracking dictionaries"""
    current_time = time.time()
    
    # Clean blocked IPs
    blocked_ips_to_remove = [
        ip for ip, block_time in blocked_ips.items()
        if current_time - block_time >= settings.BLOCK_DURATION
    ]
    for ip in blocked_ips_to_remove:
        blocked_ips.pop(ip)
    
    # Clean request counts older than 1 minute
    for api_key in request_counts:
        request_counts[api_key] = [
            req_time for req_time in request_counts[api_key]
            if current_time - req_time < 60
        ]
    
    # Clean failed attempts older than 1 hour
    failed_attempts_to_remove = [
        key for key in failed_attempts.keys()
        if current_time - key > 3600
    ]
    for key in failed_attempts_to_remove:
        failed_attempts.pop(key)

def is_rate_limited(api_key: str, request: Request) -> bool:
    """
    Check if the request exceeds rate limit
    
    Args:
        api_key: The API key to check
        request: The incoming request
        
    Returns:
        bool: True if rate limited, False otherwise
        
    Raises:
        HTTPException: If rate limit is exceeded
    """
    try:
        client_ip = request.client.host
        current_time = time.time()
        
        # Clean expired data periodically
        clean_expired_data()
        
        # Check if IP is blocked
        if client_ip in blocked_ips:
            block_time_remaining = settings.BLOCK_DURATION - (current_time - blocked_ips[client_ip])
            if block_time_remaining > 0:
                logger.warning(f"Blocked IP attempted access: {client_ip}")
                raise HTTPException(
                    status_code=HTTP_429_TOO_MANY_REQUESTS,
                    detail={
                        "error": "Too many requests",
                        "message": f"Please try again after {int(block_time_remaining)} seconds",
                        "block_time_remaining": int(block_time_remaining)
                    }
                )
            else:
                blocked_ips.pop(client_ip)
        
        # Initialize or clean old requests
        if api_key not in request_counts:
            request_counts[api_key] = []
        
        # Remove requests older than 1 minute
        request_counts[api_key] = [
            req_time for req_time in request_counts[api_key]
            if current_time - req_time < 60
        ]
        
        # Check rate limit
        if len(request_counts[api_key]) >= settings.RATE_LIMIT_PER_MINUTE:
            # Block IP
            blocked_ips[client_ip] = current_time
            logger.warning(f"Rate limit exceeded for API key: {api_key[:4]}... from IP: {client_ip}")
            
            raise HTTPException(
                status_code=HTTP_429_TOO_MANY_REQUESTS,
                detail={
                    "error": "Rate limit exceeded",
                    "message": f"Maximum {settings.RATE_LIMIT_PER_MINUTE} requests per minute allowed",
                    "retry_after": 60
                }
            )
        
        # Add current request
        request_counts[api_key].append(current_time)
        return False
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in rate limiting: {str(e)}")
        return False

async def get_api_key(
    request: Request,
    api_key: str = Security(api_key_header)
) -> str:
    """
    Validate API key and check rate limits
    
    Args:
        request: The incoming request
        api_key: The API key from the header
        
    Returns:
        str: The validated API key
        
    Raises:
        HTTPException: If validation fails
    """
    try:
        client_ip = request.client.host
        
        # Check if API key exists
        if not api_key:
            logger.warning(f"Missing API key from IP: {client_ip}")
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail={
                    "error": "Missing API key",
                    "message": "Please provide an API key in the X-API-Key header"
                }
            )
        
        # Verify API key format
        if not validate_api_key_format(api_key):
            logger.warning(f"Invalid API key format from IP: {client_ip}")
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail={
                    "error": "Invalid API key format",
                    "message": "The provided API key format is invalid"
                }
            )
        
        # Verify API key
        if not settings.verify_api_key(api_key):
            # Track failed attempts
            failed_attempts[client_ip] = failed_attempts.get(client_ip, 0) + 1
            
            # Block IP if too many failed attempts
            if failed_attempts[client_ip] >= settings.MAX_FAILED_ATTEMPTS:
                blocked_ips[client_ip] = time.time()
                logger.warning(f"IP blocked due to multiple failed attempts: {client_ip}")
                
                raise HTTPException(
                    status_code=HTTP_403_FORBIDDEN,
                    detail={
                        "error": "Too many failed attempts",
                        "message": f"Access blocked for {settings.BLOCK_DURATION} seconds"
                    }
                )
            
            logger.warning(f"Invalid API key attempted from IP: {client_ip}")
            raise HTTPException(
                status_code=HTTP_403_FORBIDDEN,
                detail={
                    "error": "Invalid API key",
                    "message": "The provided API key is not valid"
                }
            )
        
        # Check rate limits
        is_rate_limited(api_key, request)
        
        # Get API key environment
        env = settings.get_api_key_environment(api_key)
        request.state.api_environment = env
        request.state.api_key = api_key
        
        # Log successful request
        logger.info(
            f"Successful API key validation - "
            f"Key: {api_key[:4]}..., "
            f"IP: {client_ip}, "
            f"Environment: {env}"
        )
        
        return api_key
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in API key validation: {str(e)}")
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail={
                "error": "Authentication failed",
                "message": "An error occurred while validating the API key"
            }
        )

def generate_api_key(environment: str = "test") -> str:
    """
    Generate a secure API key
    
    Args:
        environment (str): 'test' or 'production'
    
    Returns:
        str: A secure API key
    
    Raises:
        ValueError: If invalid environment specified
    """
    if environment not in ["test", "production"]:
        raise ValueError("Environment must be either 'test' or 'production'")
    
    # Generate a 32-byte random string
    random_part = secrets.token_urlsafe(32)
    
    # Add environment prefix for easier identification
    prefix = "test_" if environment == "test" else "live_"
    key = f"{prefix}{random_part}"
    
    logger.info(f"Generated new API key for {environment} environment")
    return key

def validate_api_key_format(api_key: str) -> bool:
    """
    Validate API key format
    
    Args:
        api_key (str): The API key to validate
    
    Returns:
        bool: True if the format is valid
    """
    try:
        if not api_key:
            return False
        
        # Check minimum length (prefix + 32 bytes in base64 ~ 43 chars minimum)
        if len(api_key) < 43:
            return False
        
        # Check prefix
        if not (api_key.startswith("test_") or api_key.startswith("live_")):
            return False
        
        # Check character set (base64url)
        valid_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
        key_part = api_key.split("_")[1]
        if not all(c in valid_chars for c in key_part):
            return False
        
        return True
        
    except Exception:
        return False

def revoke_api_key(api_key: str) -> bool:
    """
    Revoke an API key
    
    Args:
        api_key (str): The API key to revoke
    
    Returns:
        bool: True if successfully revoked
    """
    try:
        # Validate key format first
        if not validate_api_key_format(api_key):
            return False
        
        # Clear rate limit counters
        if api_key in request_counts:
            del request_counts[api_key]
        
        # Log revocation
        logger.info(f"API key revoked: {api_key[:4]}...")
        
        # In production, you would also:
        # 1. Mark the key as revoked in the database
        # 2. Add the key to a blacklist
        # 3. Notify administrators
        
        return True
        
    except Exception as e:
        logger.error(f"Error revoking API key: {str(e)}")
        return False