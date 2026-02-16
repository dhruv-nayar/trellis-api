"""
Tests for RemBG router endpoints
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from io import BytesIO


class TestRemBGEndpoint:
    """Tests for POST /api/v1/rembg"""

    @pytest.fixture
    def mock_task(self, monkeypatch):
        """Mock Celery task"""
        mock_delay = Mock()
        mock_delay.id = "celery-task-id-123"

        mock_process_rembg = Mock()
        mock_process_rembg.delay = Mock(return_value=mock_delay)

        # Patch at the tasks module level since it's imported inside the router function
        monkeypatch.setattr(
            "api.tasks.rembg_tasks.process_rembg",
            mock_process_rembg
        )

        return mock_process_rembg

    def test_rembg_success(self, authenticated_client, sample_image_bytes, mock_task, monkeypatch):
        """Test successful RemBG request"""
        response = authenticated_client.post(
            "/api/v1/rembg",
            files=[("files", ("test.jpg", BytesIO(sample_image_bytes), "image/jpeg"))],
        )

        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert data["status"] == "pending"
        assert data["job_type"] == "rembg"

    def test_rembg_multiple_files(self, authenticated_client, sample_image_bytes, mock_task, monkeypatch):
        """Test RemBG with multiple files"""
        response = authenticated_client.post(
            "/api/v1/rembg",
            files=[
                ("files", ("test1.jpg", BytesIO(sample_image_bytes), "image/jpeg")),
                ("files", ("test2.png", BytesIO(sample_image_bytes), "image/png")),
            ],
        )

        assert response.status_code == 200
        data = response.json()
        assert "Processing 2 image(s)" in data["message"]

    def test_rembg_no_files(self, authenticated_client):
        """Test RemBG with no files returns 400"""
        response = authenticated_client.post(
            "/api/v1/rembg",
            files=[],
        )

        # FastAPI returns 422 for missing required field
        assert response.status_code == 422

    def test_rembg_too_many_files(self, authenticated_client, sample_image_bytes, monkeypatch):
        """Test RemBG with too many files returns 400"""
        # Create 11 files (assuming max is 10)
        files = [
            ("files", (f"test{i}.jpg", BytesIO(sample_image_bytes), "image/jpeg"))
            for i in range(11)
        ]

        response = authenticated_client.post(
            "/api/v1/rembg",
            files=files,
        )

        assert response.status_code == 400
        assert "Maximum" in response.json()["detail"]

    def test_rembg_invalid_format(self, authenticated_client):
        """Test RemBG with invalid file format returns 400"""
        response = authenticated_client.post(
            "/api/v1/rembg",
            files=[("files", ("test.txt", BytesIO(b"not an image"), "text/plain"))],
        )

        assert response.status_code == 400
        assert "Unsupported file format" in response.json()["detail"]

    def test_rembg_no_auth(self, client):
        """Test RemBG without authentication returns 401"""
        response = client.post(
            "/api/v1/rembg",
            files=[("files", ("test.jpg", BytesIO(b"fake"), "image/jpeg"))],
        )

        assert response.status_code == 401

    def test_rembg_with_model_param(self, authenticated_client, sample_image_bytes, mock_task, monkeypatch):
        """Test RemBG with custom model parameter"""
        response = authenticated_client.post(
            "/api/v1/rembg",
            files=[("files", ("test.jpg", BytesIO(sample_image_bytes), "image/jpeg"))],
            data={"model": "isnet-general-use"},
        )

        assert response.status_code == 200

    def test_rembg_with_alpha_matting(self, authenticated_client, sample_image_bytes, mock_task, monkeypatch):
        """Test RemBG with alpha matting enabled"""
        response = authenticated_client.post(
            "/api/v1/rembg",
            files=[("files", ("test.jpg", BytesIO(sample_image_bytes), "image/jpeg"))],
            data={
                "alpha_matting": "true",
                "alpha_matting_foreground_threshold": 230,
                "alpha_matting_background_threshold": 20,
            },
        )

        assert response.status_code == 200
