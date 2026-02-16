"""
Tests for Jobs router endpoints
"""

import pytest
import json
from unittest.mock import Mock, patch
from datetime import datetime


class TestGetJobStatus:
    """Tests for GET /api/v1/jobs/{job_id}"""

    def test_get_job_status_exists(self, authenticated_client, sample_job_data, monkeypatch):
        """Test getting status of existing job"""
        # Use the mock_job_store from authenticated_client
        authenticated_client.mock_job_store._redis.set(
            f"job:{sample_job_data['job_id']}",
            json.dumps(sample_job_data)
        )

        response = authenticated_client.get(f"/api/v1/jobs/{sample_job_data['job_id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == sample_job_data["job_id"]
        assert data["status"] == "pending"

    def test_get_job_status_not_found(self, authenticated_client):
        """Test getting status of non-existent job"""
        response = authenticated_client.get("/api/v1/jobs/non-existent-job-id")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_job_status_completed(self, authenticated_client, completed_job_data, monkeypatch):
        """Test getting status of completed job includes download URLs"""
        authenticated_client.mock_job_store._redis.set(
            f"job:{completed_job_data['job_id']}",
            json.dumps(completed_job_data)
        )

        response = authenticated_client.get(f"/api/v1/jobs/{completed_job_data['job_id']}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["progress"] == 100
        assert len(data["download_urls"]) > 0

    def test_get_job_status_no_auth(self, client, job_id):
        """Test getting job status without auth"""
        response = client.get(f"/api/v1/jobs/{job_id}")

        assert response.status_code == 401


class TestDownloadFile:
    """Tests for GET /api/v1/jobs/{job_id}/download/{filename}"""

    def test_download_completed_job(self, authenticated_client, completed_job_data, monkeypatch):
        """Test downloading file from completed job"""
        job_id = completed_job_data["job_id"]

        # Set up job in store (use mock from authenticated_client)
        authenticated_client.mock_job_store._redis.set(f"job:{job_id}", json.dumps(completed_job_data))

        # Create actual output file
        output_dir = authenticated_client.mock_storage_service.get_job_output_dir(job_id)
        output_file = output_dir / "output.png"
        output_file.write_bytes(b"fake png content")

        response = authenticated_client.get(f"/api/v1/jobs/{job_id}/download/output.png")

        assert response.status_code == 200
        assert response.headers["content-type"] == "image/png"

    def test_download_pending_job(self, authenticated_client, sample_job_data, monkeypatch):
        """Test downloading from pending job returns 400"""
        job_id = sample_job_data["job_id"]
        authenticated_client.mock_job_store._redis.set(f"job:{job_id}", json.dumps(sample_job_data))

        response = authenticated_client.get(f"/api/v1/jobs/{job_id}/download/output.png")

        assert response.status_code == 400
        assert "not completed" in response.json()["detail"]

    def test_download_nonexistent_job(self, authenticated_client):
        """Test downloading from non-existent job"""
        response = authenticated_client.get("/api/v1/jobs/fake-job/download/output.png")

        assert response.status_code == 404

    def test_download_nonexistent_file(self, authenticated_client, completed_job_data, monkeypatch):
        """Test downloading non-existent file"""
        job_id = completed_job_data["job_id"]
        authenticated_client.mock_job_store._redis.set(f"job:{job_id}", json.dumps(completed_job_data))

        response = authenticated_client.get(f"/api/v1/jobs/{job_id}/download/does_not_exist.png")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_download_glb_file(self, authenticated_client, monkeypatch):
        """Test downloading GLB file has correct content type"""
        from datetime import datetime, UTC
        job_id = "glb-test-job"
        job_data = {
            "job_id": job_id,
            "job_type": "trellis",
            "status": "completed",
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
            "completed_at": datetime.now(UTC).isoformat(),
            "progress": 100,
            "message": "Done",
            "error": None,
            "input_count": 1,
            "output_count": 1,
            "filenames": ["input.jpg"],
            "download_urls": [f"/api/v1/jobs/{job_id}/download/model.glb"],
            "celery_task_id": "task-123",
        }
        authenticated_client.mock_job_store._redis.set(f"job:{job_id}", json.dumps(job_data))

        # Create GLB file
        output_dir = authenticated_client.mock_storage_service.get_job_output_dir(job_id)
        (output_dir / "model.glb").write_bytes(b"glTF binary")

        response = authenticated_client.get(f"/api/v1/jobs/{job_id}/download/model.glb")

        assert response.status_code == 200
        assert response.headers["content-type"] == "model/gltf-binary"


class TestDeleteJob:
    """Tests for DELETE /api/v1/jobs/{job_id}"""

    def test_delete_job_success(self, authenticated_client, sample_job_data, monkeypatch):
        """Test successful job deletion"""
        job_id = sample_job_data["job_id"]
        # sample_job_data has celery_task_id=None, so no celery revoke will happen
        authenticated_client.mock_job_store._redis.set(f"job:{job_id}", json.dumps(sample_job_data))

        # Create some files
        authenticated_client.mock_storage_service.get_job_upload_dir(job_id)
        authenticated_client.mock_storage_service.get_job_output_dir(job_id)

        response = authenticated_client.delete(f"/api/v1/jobs/{job_id}")

        assert response.status_code == 200
        assert "deleted" in response.json()["message"]

        # Job should be removed from store
        assert authenticated_client.mock_job_store._redis.get(f"job:{job_id}") is None

    def test_delete_nonexistent_job(self, authenticated_client):
        """Test deleting non-existent job"""
        response = authenticated_client.delete("/api/v1/jobs/does-not-exist")

        assert response.status_code == 404

    def test_delete_job_no_auth(self, client, job_id):
        """Test deleting job without auth"""
        response = client.delete(f"/api/v1/jobs/{job_id}")

        assert response.status_code == 401

    def test_delete_cancels_celery_task(self, authenticated_client, sample_job_data, monkeypatch):
        """Test deletion revokes Celery task"""
        import sys
        from types import ModuleType

        job_id = sample_job_data["job_id"]
        sample_job_data["celery_task_id"] = "celery-task-to-cancel"
        authenticated_client.mock_job_store._redis.set(f"job:{job_id}", json.dumps(sample_job_data))

        # Create a mock module with mock celery_app
        mock_celery = Mock()
        mock_celery.control.revoke = Mock()
        mock_module = ModuleType("api.tasks.celery_app")
        mock_module.celery_app = mock_celery

        # Save original and replace in sys.modules
        original_module = sys.modules.get("api.tasks.celery_app")
        sys.modules["api.tasks.celery_app"] = mock_module

        try:
            response = authenticated_client.delete(f"/api/v1/jobs/{job_id}")

            assert response.status_code == 200
            mock_celery.control.revoke.assert_called_once_with(
                "celery-task-to-cancel",
                terminate=True
            )
        finally:
            # Restore original
            if original_module is not None:
                sys.modules["api.tasks.celery_app"] = original_module
            else:
                sys.modules.pop("api.tasks.celery_app", None)
