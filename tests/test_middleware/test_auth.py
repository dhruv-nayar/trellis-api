"""
Tests for API Key Authentication Middleware
"""

import pytest
from unittest.mock import Mock, AsyncMock
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials


class TestAPIKeyValidator:
    """Tests for APIKeyValidator class"""

    def test_validate_valid_env_key(self, test_api_key, monkeypatch):
        """Test validation of valid environment key"""
        # Set environment before importing
        monkeypatch.setenv("API_KEYS", test_api_key)

        # Reload config to pick up the env change
        import importlib
        import api.config
        importlib.reload(api.config)

        from api.middleware.auth import APIKeyValidator
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        validator = APIKeyValidator()
        result = validator.validate(test_api_key)

        assert result["key_id"] == "env"
        assert result["tier"] == "standard"
        assert "rate_limit" in result

        # Reset
        auth_module._validator = None

    def test_validate_invalid_key(self, monkeypatch):
        """Test validation of invalid key raises HTTPException"""
        from api.middleware.auth import APIKeyValidator
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        validator = APIKeyValidator()

        with pytest.raises(HTTPException) as exc_info:
            validator.validate("invalid-key-12345")

        assert exc_info.value.status_code == 401
        assert "Invalid API key" in str(exc_info.value.detail)

        # Reset
        auth_module._validator = None

    def test_validate_empty_key(self, monkeypatch):
        """Test validation of empty key raises HTTPException"""
        from api.middleware.auth import APIKeyValidator
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        validator = APIKeyValidator()

        with pytest.raises(HTTPException) as exc_info:
            validator.validate("")

        assert exc_info.value.status_code == 401

        # Reset
        auth_module._validator = None


class TestGetApiKey:
    """Tests for get_api_key dependency"""

    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = Mock()
        request.headers = {}
        return request

    @pytest.mark.asyncio
    async def test_valid_bearer_token(self, test_api_key, mock_request, monkeypatch):
        """Test validation with valid Bearer token"""
        # Set environment before importing
        monkeypatch.setenv("API_KEYS", test_api_key)

        import importlib
        import api.config
        importlib.reload(api.config)

        from api.middleware.auth import get_api_key
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=test_api_key
        )

        result = await get_api_key(mock_request, credentials)

        assert result["key_id"] == "env"

        # Reset
        auth_module._validator = None

    @pytest.mark.asyncio
    async def test_valid_x_api_key_header(self, test_api_key, mock_request, monkeypatch):
        """Test validation with X-API-Key header"""
        # Set environment before importing
        monkeypatch.setenv("API_KEYS", test_api_key)

        import importlib
        import api.config
        importlib.reload(api.config)

        from api.middleware.auth import get_api_key
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        mock_request.headers = {"X-API-Key": test_api_key}

        result = await get_api_key(mock_request, credentials=None)

        assert result["key_id"] == "env"

        # Reset
        auth_module._validator = None

    @pytest.mark.asyncio
    async def test_invalid_token(self, mock_request, monkeypatch):
        """Test validation with invalid token"""
        from api.middleware.auth import get_api_key
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="wrong-key"
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_api_key(mock_request, credentials)

        assert exc_info.value.status_code == 401

        # Reset
        auth_module._validator = None

    @pytest.mark.asyncio
    async def test_missing_auth(self, mock_request, monkeypatch):
        """Test missing authentication raises HTTPException"""
        from api.middleware.auth import get_api_key
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        with pytest.raises(HTTPException) as exc_info:
            await get_api_key(mock_request, credentials=None)

        assert exc_info.value.status_code == 401
        assert "Missing API key" in str(exc_info.value.detail)

        # Reset
        auth_module._validator = None


class TestGetOptionalApiKey:
    """Tests for get_optional_api_key dependency"""

    @pytest.fixture
    def mock_request(self):
        """Create mock request"""
        request = Mock()
        request.headers = {}
        return request

    @pytest.mark.asyncio
    async def test_returns_key_data_when_valid(self, test_api_key, mock_request, monkeypatch):
        """Test returns key data when valid key provided"""
        # Set environment before importing
        monkeypatch.setenv("API_KEYS", test_api_key)

        import importlib
        import api.config
        importlib.reload(api.config)

        from api.middleware.auth import get_optional_api_key
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials=test_api_key
        )

        result = await get_optional_api_key(mock_request, credentials)

        assert result is not None
        assert result["key_id"] == "env"

        # Reset
        auth_module._validator = None

    @pytest.mark.asyncio
    async def test_returns_none_when_no_key(self, mock_request, monkeypatch):
        """Test returns None when no key provided"""
        from api.middleware.auth import get_optional_api_key
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        result = await get_optional_api_key(mock_request, credentials=None)

        assert result is None

        # Reset
        auth_module._validator = None

    @pytest.mark.asyncio
    async def test_returns_none_when_invalid_key(self, mock_request, monkeypatch):
        """Test returns None when invalid key provided"""
        from api.middleware.auth import get_optional_api_key
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer",
            credentials="invalid-key"
        )

        result = await get_optional_api_key(mock_request, credentials)

        assert result is None

        # Reset
        auth_module._validator = None


class TestGetValidator:
    """Tests for get_validator singleton"""

    def test_returns_same_instance(self, monkeypatch):
        """Test get_validator returns same instance"""
        from api.middleware.auth import get_validator
        import api.middleware.auth as auth_module

        # Reset global validator
        auth_module._validator = None

        validator1 = get_validator()
        validator2 = get_validator()

        assert validator1 is validator2

        # Reset
        auth_module._validator = None
