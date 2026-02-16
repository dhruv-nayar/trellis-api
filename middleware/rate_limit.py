"""
Rate Limiting Middleware using slowapi
"""

import logging
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request
from fastapi.responses import JSONResponse
from api.config import settings

logger = logging.getLogger(__name__)


def get_api_key_or_ip(request: Request) -> str:
    """
    Get rate limit key from API key if present, otherwise use IP.
    This allows per-key rate limiting for authenticated requests.
    """
    # Try Authorization header
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return f"key:{auth[7:]}"

    # Try X-API-Key header
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"

    # Fall back to IP address
    return f"ip:{get_remote_address(request)}"


# Create limiter with custom key function
limiter = Limiter(key_func=get_api_key_or_ip)


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> JSONResponse:
    """Custom handler for rate limit exceeded errors"""
    logger.warning(f"Rate limit exceeded for {get_api_key_or_ip(request)}")
    return JSONResponse(
        status_code=429,
        content={
            "error": "RateLimitExceeded",
            "message": f"Rate limit exceeded: {exc.detail}",
            "detail": "Please wait before making more requests",
        },
    )


# Rate limit decorators for different endpoint tiers
def rembg_rate_limit():
    """Rate limit decorator for RemBG endpoints"""
    return limiter.limit(settings.rate_limit_rembg)


def trellis_rate_limit():
    """Rate limit decorator for TRELLIS endpoints"""
    return limiter.limit(settings.rate_limit_trellis)


def default_rate_limit():
    """Rate limit decorator for general endpoints"""
    return limiter.limit(settings.rate_limit_default)
