"""
API Key Authentication Middleware
"""

import hashlib
import logging
from typing import Optional, Dict, Any
from fastapi import Security, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.config import settings

logger = logging.getLogger(__name__)

# HTTP Bearer security scheme
security = HTTPBearer(auto_error=False)


class APIKeyValidator:
    """Validates API keys from environment or Redis"""

    def __init__(self):
        self._env_keys = set(settings.api_keys_list)
        logger.info(f"Loaded {len(self._env_keys)} API keys from environment")

    @staticmethod
    def _hash_key(key: str) -> str:
        """Hash API key for secure storage/comparison"""
        return hashlib.sha256(key.encode()).hexdigest()

    def validate(self, token: str) -> Dict[str, Any]:
        """
        Validate API key and return key metadata.

        Returns:
            Dict with key metadata (key_id, tier, etc.)

        Raises:
            HTTPException if key is invalid
        """
        # Check environment keys
        if token in self._env_keys:
            return {
                "key_id": "env",
                "tier": "standard",
                "rate_limit": settings.rate_limit_default,
            }

        # If no valid key found, raise error
        logger.warning(f"Invalid API key attempt: {token[:8]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Global validator instance
_validator: Optional[APIKeyValidator] = None


def get_validator() -> APIKeyValidator:
    """Get or create validator instance"""
    global _validator
    if _validator is None:
        _validator = APIKeyValidator()
    return _validator


async def get_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Dict[str, Any]:
    """
    FastAPI dependency for API key validation.

    Usage:
        @router.get("/endpoint")
        async def endpoint(api_key: dict = Depends(get_api_key)):
            ...
    """
    # Check for Authorization header
    if credentials is None:
        # Also check for X-API-Key header as alternative
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header:
            token = api_key_header
        else:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing API key. Provide via Authorization: Bearer <key> or X-API-Key header",
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        token = credentials.credentials

    # Validate the token
    validator = get_validator()
    return validator.validate(token)


async def get_optional_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security),
) -> Optional[Dict[str, Any]]:
    """
    Optional API key dependency - returns None if no key provided.
    Useful for endpoints that work differently for authenticated vs anonymous users.
    """
    try:
        return await get_api_key(request, credentials)
    except HTTPException:
        return None
