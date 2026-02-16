"""
Pytest configuration and shared fixtures for API tests
"""

import sys
from unittest.mock import Mock, MagicMock, AsyncMock, patch

# =============================================================================
# Early Module Mocking (MUST be before any api imports)
# =============================================================================

# Mock rembg module to avoid onnxruntime dependency
mock_rembg = MagicMock()
mock_rembg.remove = MagicMock()
mock_rembg.new_session = MagicMock()
sys.modules['rembg'] = mock_rembg
sys.modules['rembg.bg'] = MagicMock()
sys.modules['rembg.session_factory'] = MagicMock()
sys.modules['rembg.sessions'] = MagicMock()

# Mock gradio_client module
mock_gradio = MagicMock()
mock_gradio.Client = MagicMock()
mock_gradio.handle_file = MagicMock(side_effect=lambda x: x)
sys.modules['gradio_client'] = mock_gradio

import pytest
import asyncio
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime, timedelta
from io import BytesIO

import fakeredis
from fastapi.testclient import TestClient
from fastapi import UploadFile
from PIL import Image


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def test_api_key():
    """Test API key for authentication"""
    return "test-key-12345"


@pytest.fixture
def auth_headers(test_api_key):
    """Headers with Bearer authorization"""
    return {"Authorization": f"Bearer {test_api_key}"}


@pytest.fixture
def x_api_key_headers(test_api_key):
    """Headers with X-API-Key authorization"""
    return {"X-API-Key": test_api_key}


# =============================================================================
# Redis Fixtures
# =============================================================================

@pytest.fixture
def fake_redis():
    """In-memory Redis for testing using fakeredis"""
    return fakeredis.FakeRedis(decode_responses=True)


@pytest.fixture
def mock_redis_connection(fake_redis, monkeypatch):
    """Patch redis.from_url to return fake_redis"""
    def mock_from_url(*args, **kwargs):
        return fake_redis

    monkeypatch.setattr("redis.from_url", mock_from_url)
    return fake_redis


# =============================================================================
# JobStore Fixtures
# =============================================================================

@pytest.fixture
def mock_job_store(mock_redis_connection):
    """JobStore instance with mocked Redis"""
    from api.services.job_store import JobStore
    store = JobStore()
    store._redis = mock_redis_connection
    return store


@pytest.fixture
def sample_job_data():
    """Sample job data for testing"""
    return {
        "job_id": "test-job-123",
        "job_type": "rembg",
        "status": "pending",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "completed_at": None,
        "progress": 0,
        "message": "Queued for processing",
        "error": None,
        "input_count": 2,
        "output_count": 0,
        "filenames": ["image1.jpg", "image2.png"],
        "download_urls": [],
        "celery_task_id": None,
    }


# =============================================================================
# Storage Fixtures
# =============================================================================

@pytest.fixture
def temp_storage_dirs(tmp_path):
    """Create temporary upload and output directories"""
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    upload_dir.mkdir()
    output_dir.mkdir()
    return {"upload": upload_dir, "output": output_dir}


@pytest.fixture
def mock_storage_service(temp_storage_dirs, monkeypatch):
    """StorageService with temporary directories"""
    from api.services.storage import StorageService

    service = StorageService(
        upload_dir=temp_storage_dirs["upload"],
        output_dir=temp_storage_dirs["output"],
        max_file_size=10 * 1024 * 1024,  # 10MB
        cleanup_after_hours=24,
    )
    return service


# =============================================================================
# Image Fixtures
# =============================================================================

@pytest.fixture
def sample_image_bytes():
    """Create sample image bytes for testing"""
    img = Image.new("RGB", (100, 100), color="red")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_images(tmp_path):
    """Create test images in various formats"""
    images = {}

    # RGB JPEG
    img_jpg = Image.new("RGB", (100, 100), color="red")
    path_jpg = tmp_path / "test.jpg"
    img_jpg.save(path_jpg, "JPEG")
    images[".jpg"] = path_jpg

    # RGBA PNG
    img_png = Image.new("RGBA", (100, 100), color=(255, 0, 0, 255))
    path_png = tmp_path / "test.png"
    img_png.save(path_png, "PNG")
    images[".png"] = path_png

    # RGB WebP
    img_webp = Image.new("RGB", (100, 100), color="blue")
    path_webp = tmp_path / "test.webp"
    img_webp.save(path_webp, "WEBP")
    images[".webp"] = path_webp

    return images


@pytest.fixture
def mock_upload_file(sample_image_bytes):
    """Create a mock UploadFile for testing"""
    def _create_upload(filename="test.jpg", content_type="image/jpeg", content=None):
        file_content = content or sample_image_bytes
        file = Mock(spec=UploadFile)
        file.filename = filename
        file.content_type = content_type
        file.read = AsyncMock(return_value=file_content)
        file.seek = AsyncMock()
        file.file = BytesIO(file_content)
        return file

    return _create_upload


# =============================================================================
# RemBG Service Fixtures
# =============================================================================

@pytest.fixture
def mock_rembg(monkeypatch):
    """Mock rembg library"""
    mock_session = Mock()
    mock_remove = Mock(return_value=Image.new("RGBA", (100, 100), (0, 0, 0, 0)))
    mock_new_session = Mock(return_value=mock_session)

    # Patch at both levels - the library and where it's imported
    monkeypatch.setattr("rembg.new_session", mock_new_session)
    monkeypatch.setattr("rembg.remove", mock_remove)
    monkeypatch.setattr("api.services.rembg_service.new_session", mock_new_session)
    monkeypatch.setattr("api.services.rembg_service.remove", mock_remove)

    return {
        "session": mock_session,
        "remove": mock_remove,
        "new_session": mock_new_session,
    }


@pytest.fixture
def mock_rembg_service(mock_rembg, monkeypatch):
    """RemBGService with mocked rembg"""
    from api.services.rembg_service import RemBGService
    service = RemBGService(model_name="u2net")
    return service


# =============================================================================
# TRELLIS Fixtures
# =============================================================================

@pytest.fixture
def mock_gradio_client(monkeypatch):
    """Mock Gradio client for TrellisV1"""
    mock_client = Mock()
    mock_client.predict = Mock(return_value={"glb": "/tmp/output.glb"})

    mock_client_class = Mock(return_value=mock_client)
    mock_handle_file = Mock(side_effect=lambda x: x)

    # Patch at both levels
    monkeypatch.setattr("gradio_client.Client", mock_client_class)
    monkeypatch.setattr("gradio_client.handle_file", mock_handle_file)
    monkeypatch.setattr("api.services.trellis_v1.Client", mock_client_class)
    monkeypatch.setattr("api.services.trellis_v1.handle_file", mock_handle_file)

    return {
        "client": mock_client,
        "Client": mock_client_class,
        "handle_file": mock_handle_file,
    }


@pytest.fixture
def mock_trellis_v1_client(mock_gradio_client, tmp_path, monkeypatch):
    """TrellisV1Client with mocked Gradio"""
    # Create a fake GLB file for the mock to return
    fake_glb = tmp_path / "output.glb"
    fake_glb.write_bytes(b"fake glb content")
    mock_gradio_client["client"].predict.return_value = {"glb": str(fake_glb)}

    from api.services.trellis_v1 import TrellisV1Client
    client = TrellisV1Client()
    return client


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for TrellisV2"""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"id": "job-123", "status": "COMPLETED", "output": {"glb": "base64data"}}
    mock_response.raise_for_status = Mock()

    mock_client = Mock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.get = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock()

    return mock_client


# =============================================================================
# Celery Task Fixtures
# =============================================================================

@pytest.fixture
def mock_celery_task():
    """Mock Celery task context"""
    task = Mock()
    task.request = Mock()
    task.request.id = "celery-task-123"
    task.request.retries = 0
    task.max_retries = 3
    task.update_state = Mock()
    task.retry = Mock(side_effect=Exception("Retry called"))

    return task


@pytest.fixture
def mock_celery_app(monkeypatch):
    """Mock Celery app"""
    mock_app = Mock()
    mock_app.control = Mock()
    mock_app.control.revoke = Mock()
    mock_app.control.inspect = Mock()
    mock_app.control.inspect.return_value.active.return_value = {"worker1": []}

    return mock_app


# =============================================================================
# FastAPI Test Client Fixtures
# =============================================================================

@pytest.fixture
def app_settings(test_api_key, monkeypatch):
    """Configure app settings for testing"""
    monkeypatch.setenv("API_KEYS", test_api_key)
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/1")
    monkeypatch.setenv("DEBUG", "true")


@pytest.fixture
def client(app_settings, mock_redis_connection, tmp_path, monkeypatch):
    """Test client with mocked dependencies"""
    # Patch settings before importing app
    upload_dir = tmp_path / "uploads"
    output_dir = tmp_path / "outputs"
    upload_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)

    monkeypatch.setenv("UPLOAD_DIR", str(upload_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))

    # Need to reload settings to pick up env changes
    import importlib
    import api.config
    importlib.reload(api.config)

    # Reset auth validator to pick up new API keys
    import api.middleware.auth as auth_module
    auth_module._validator = None

    from api.main import app
    from api.services.job_store import JobStore, get_job_store
    from api.services.storage import StorageService, get_storage_service

    # Create mock services that will be used by the app
    mock_job_store = JobStore()
    mock_job_store._redis = mock_redis_connection

    mock_storage = StorageService(
        upload_dir=upload_dir,
        output_dir=output_dir,
        max_file_size=10 * 1024 * 1024,
        cleanup_after_hours=24,
    )

    # Override dependencies
    app.dependency_overrides[get_job_store] = lambda: mock_job_store
    app.dependency_overrides[get_storage_service] = lambda: mock_storage

    test_client = TestClient(app)

    # Store mocks on the client for test access
    test_client.mock_job_store = mock_job_store
    test_client.mock_storage_service = mock_storage

    yield test_client

    # Cleanup
    app.dependency_overrides.clear()
    auth_module._validator = None


@pytest.fixture
def authenticated_client(client, auth_headers):
    """Helper for making authenticated requests"""
    class AuthenticatedClient:
        def __init__(self, test_client, headers):
            self._client = test_client
            self._headers = headers
            # Expose mock services for test access
            self.mock_job_store = test_client.mock_job_store
            self.mock_storage_service = test_client.mock_storage_service

        def get(self, url, **kwargs):
            headers = {**self._headers, **kwargs.pop("headers", {})}
            return self._client.get(url, headers=headers, **kwargs)

        def post(self, url, **kwargs):
            headers = {**self._headers, **kwargs.pop("headers", {})}
            return self._client.post(url, headers=headers, **kwargs)

        def delete(self, url, **kwargs):
            headers = {**self._headers, **kwargs.pop("headers", {})}
            return self._client.delete(url, headers=headers, **kwargs)

    return AuthenticatedClient(client, auth_headers)


# =============================================================================
# Async Fixtures
# =============================================================================

@pytest.fixture
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# =============================================================================
# Test Data Fixtures
# =============================================================================

@pytest.fixture
def job_id():
    """Standard test job ID"""
    return "550e8400-e29b-41d4-a716-446655440000"


@pytest.fixture
def completed_job_data(job_id):
    """Sample completed job data"""
    return {
        "job_id": job_id,
        "job_type": "rembg",
        "status": "completed",
        "created_at": datetime.utcnow().isoformat(),
        "updated_at": datetime.utcnow().isoformat(),
        "completed_at": datetime.utcnow().isoformat(),
        "progress": 100,
        "message": "Successfully processed 2 images",
        "error": None,
        "input_count": 2,
        "output_count": 2,
        "filenames": ["image1.jpg", "image2.png"],
        "download_urls": [
            f"/api/v1/jobs/{job_id}/download/image1_nobg.png",
            f"/api/v1/jobs/{job_id}/download/image2_nobg.png",
        ],
        "celery_task_id": "celery-task-456",
    }


# =============================================================================
# Utility Functions
# =============================================================================

def create_test_image(path: Path, format: str = "PNG", size: tuple = (100, 100), mode: str = "RGB"):
    """Helper to create test images"""
    img = Image.new(mode, size, color="red")
    img.save(path, format)
    return path
