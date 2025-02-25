from urllib.parse import urlparse
import validators
import logging

logger = logging.getLogger(__name__)

def validate_url(url: str) -> bool:
    """Validate if the provided URL is valid and accessible"""
    try:
        if not url:
            return False
        
        # Parse URL to ensure it's well-formed
        parsed = urlparse(str(url))
        if not all([parsed.scheme, parsed.netloc]):
            return False
        
        # Check if it's a valid URL format
        if not validators.url(str(url)):
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"URL validation error: {str(e)}")
        return False 