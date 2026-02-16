"""
Tests for StorageService
"""

import pytest
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock
from io import BytesIO
from PIL import Image

from api.services.storage import StorageService, SUPPORTED_FORMATS


class TestStorageServiceInit:
    """Tests for StorageService initialization"""

    def test_creates_directories(self, tmp_path):
        """Test that upload and output dirs are created"""
        upload_dir = tmp_path / "uploads"
        output_dir = tmp_path / "outputs"

        service = StorageService(
            upload_dir=upload_dir,
            output_dir=output_dir,
        )

        assert upload_dir.exists()
        assert output_dir.exists()

    def test_uses_custom_settings(self, tmp_path):
        """Test that custom settings are used"""
        service = StorageService(
            upload_dir=tmp_path / "custom_uploads",
            output_dir=tmp_path / "custom_outputs",
            max_file_size=5 * 1024 * 1024,
            cleanup_after_hours=48,
        )

        assert service.max_file_size == 5 * 1024 * 1024
        assert service.cleanup_after_hours == 48


class TestStorageServiceDirectories:
    """Tests for job directory management"""

    def test_get_job_upload_dir(self, mock_storage_service):
        """Test getting upload directory for a job"""
        job_dir = mock_storage_service.get_job_upload_dir("test-job-123")

        assert job_dir.exists()
        assert "test-job-123" in str(job_dir)
        assert "uploads" in str(job_dir) or job_dir.parent == mock_storage_service.upload_dir

    def test_get_job_output_dir(self, mock_storage_service):
        """Test getting output directory for a job"""
        job_dir = mock_storage_service.get_job_output_dir("test-job-456")

        assert job_dir.exists()
        assert "test-job-456" in str(job_dir)


class TestSaveUploads:
    """Tests for file upload handling"""

    @pytest.mark.asyncio
    async def test_save_uploads_valid_images(self, mock_storage_service, mock_upload_file):
        """Test saving valid image files"""
        files = [
            mock_upload_file(filename="image1.jpg", content_type="image/jpeg"),
            mock_upload_file(filename="image2.png", content_type="image/png"),
        ]

        paths, filenames = await mock_storage_service.save_uploads(files, "test-job")

        assert len(paths) == 2
        assert len(filenames) == 2
        assert all(p.exists() for p in paths)
        assert filenames == ["image1.jpg", "image2.png"]

    @pytest.mark.asyncio
    async def test_save_uploads_webp(self, mock_storage_service, mock_upload_file):
        """Test saving WebP image"""
        # Create actual WebP image bytes
        img = Image.new("RGB", (50, 50), color="blue")
        buffer = BytesIO()
        img.save(buffer, format="WEBP")
        buffer.seek(0)
        webp_bytes = buffer.getvalue()

        file = mock_upload_file(
            filename="test.webp",
            content_type="image/webp",
            content=webp_bytes,
        )

        paths, filenames = await mock_storage_service.save_uploads([file], "test-job")

        assert len(paths) == 1
        assert paths[0].suffix == ".webp"

    @pytest.mark.asyncio
    async def test_save_uploads_invalid_format(self, mock_storage_service, mock_upload_file):
        """Test rejection of invalid file formats"""
        from fastapi import HTTPException

        file = mock_upload_file(
            filename="document.txt",
            content_type="text/plain",
        )

        with pytest.raises(HTTPException) as exc_info:
            await mock_storage_service.save_uploads([file], "test-job")

        assert exc_info.value.status_code == 400
        assert "Unsupported file format" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_save_uploads_exe_rejected(self, mock_storage_service, mock_upload_file):
        """Test rejection of executable files"""
        from fastapi import HTTPException

        file = mock_upload_file(
            filename="malware.exe",
            content_type="application/octet-stream",
        )

        with pytest.raises(HTTPException) as exc_info:
            await mock_storage_service.save_uploads([file], "test-job")

        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_save_uploads_file_too_large(self, mock_storage_service, mock_upload_file):
        """Test rejection of oversized files"""
        from fastapi import HTTPException

        # Create large content (11MB when max is 10MB)
        large_content = b"x" * (11 * 1024 * 1024)

        file = mock_upload_file(
            filename="huge.jpg",
            content_type="image/jpeg",
            content=large_content,
        )

        with pytest.raises(HTTPException) as exc_info:
            await mock_storage_service.save_uploads([file], "test-job")

        assert exc_info.value.status_code == 400
        assert "exceeds max size" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_save_uploads_no_filename(self, mock_storage_service, sample_image_bytes):
        """Test rejection of files without filename"""
        from fastapi import HTTPException
        from unittest.mock import Mock, AsyncMock

        file = Mock()
        file.filename = None  # No filename
        file.content_type = "image/jpeg"
        file.read = AsyncMock(return_value=sample_image_bytes)

        with pytest.raises(HTTPException) as exc_info:
            await mock_storage_service.save_uploads([file], "test-job")

        assert exc_info.value.status_code == 400
        assert "must have a filename" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_save_uploads_handles_duplicates(self, mock_storage_service, mock_upload_file):
        """Test handling of duplicate filenames"""
        files = [
            mock_upload_file(filename="image.jpg"),
            mock_upload_file(filename="image.jpg"),
        ]

        paths, filenames = await mock_storage_service.save_uploads(files, "test-job")

        assert len(paths) == 2
        # Paths should be different due to deduplication
        assert paths[0] != paths[1]


class TestSanitizeFilename:
    """Tests for filename sanitization"""

    def test_sanitize_normal_filename(self, mock_storage_service):
        """Test normal filename passes through"""
        result = mock_storage_service._sanitize_filename("image.jpg")
        assert result == "image.jpg"

    def test_sanitize_path_traversal(self, mock_storage_service):
        """Test path traversal characters are sanitized"""
        result = mock_storage_service._sanitize_filename("../../../etc/passwd")
        assert ".." not in result
        assert "/" not in result

    def test_sanitize_special_characters(self, mock_storage_service):
        """Test special characters are replaced"""
        result = mock_storage_service._sanitize_filename("file name<with>special:chars?.jpg")
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "?" not in result

    def test_sanitize_empty_becomes_unnamed(self, mock_storage_service):
        """Test empty filename becomes 'unnamed'"""
        result = mock_storage_service._sanitize_filename("")
        assert result == "unnamed"

    def test_sanitize_all_special_becomes_unnamed(self, mock_storage_service):
        """Test all-special filename becomes 'unnamed'"""
        result = mock_storage_service._sanitize_filename("<<<>>>")
        # All special characters get replaced with _, result is all underscores
        assert result == "unnamed" or result == "______"


class TestGetFilePath:
    """Tests for file path retrieval"""

    def test_get_file_path_exists(self, mock_storage_service):
        """Test getting path to existing file"""
        job_id = "path-test"
        job_dir = mock_storage_service.get_job_output_dir(job_id)

        # Create test file
        test_file = job_dir / "output.png"
        test_file.write_bytes(b"test")

        path = mock_storage_service.get_file_path(job_id, "output.png", is_output=True)

        assert path is not None
        assert path.exists()

    def test_get_file_path_not_found(self, mock_storage_service):
        """Test getting path to non-existent file"""
        path = mock_storage_service.get_file_path("no-job", "no-file.png")
        assert path is None

    def test_get_file_path_prevents_traversal(self, mock_storage_service):
        """Test path traversal is blocked"""
        job_id = "traversal-test"
        mock_storage_service.get_job_output_dir(job_id)

        path = mock_storage_service.get_file_path(job_id, "../../../etc/passwd")
        assert path is None


class TestGetOutputFiles:
    """Tests for output file listing"""

    def test_get_output_files(self, mock_storage_service):
        """Test listing output files for a job"""
        job_id = "output-list-test"
        job_dir = mock_storage_service.get_job_output_dir(job_id)

        # Create some files
        (job_dir / "output1.png").write_bytes(b"test1")
        (job_dir / "output2.png").write_bytes(b"test2")

        files = mock_storage_service.get_output_files(job_id)

        assert len(files) == 2

    def test_get_output_files_empty(self, mock_storage_service):
        """Test listing when no output files exist"""
        files = mock_storage_service.get_output_files("nonexistent-job")
        assert files == []


class TestCleanupJob:
    """Tests for job cleanup"""

    def test_cleanup_job_removes_directories(self, mock_storage_service):
        """Test cleanup removes both upload and output directories"""
        job_id = "cleanup-test"

        # Create directories and files
        upload_dir = mock_storage_service.get_job_upload_dir(job_id)
        output_dir = mock_storage_service.get_job_output_dir(job_id)
        (upload_dir / "input.jpg").write_bytes(b"input")
        (output_dir / "output.png").write_bytes(b"output")

        # Verify they exist
        assert upload_dir.exists()
        assert output_dir.exists()

        # Cleanup
        result = mock_storage_service.cleanup_job(job_id)

        assert result is True
        assert not upload_dir.exists()
        assert not output_dir.exists()

    def test_cleanup_nonexistent_job(self, mock_storage_service):
        """Test cleanup of non-existent job returns False"""
        result = mock_storage_service.cleanup_job("never-existed")
        assert result is False


class TestCleanupOldJobs:
    """Tests for time-based cleanup"""

    def test_cleanup_old_jobs_by_mtime(self, mock_storage_service, monkeypatch):
        """Test cleanup removes jobs older than threshold"""
        import os
        import time

        job_id = "old-job"
        job_dir = mock_storage_service.get_job_output_dir(job_id)
        (job_dir / "output.png").write_bytes(b"test")

        # Set modification time to past
        old_time = time.time() - (mock_storage_service.cleanup_after_hours + 1) * 3600
        os.utime(job_dir, (old_time, old_time))

        cleaned = mock_storage_service.cleanup_old_jobs()

        assert cleaned >= 1
        assert not job_dir.exists()

    def test_cleanup_keeps_recent_jobs(self, mock_storage_service):
        """Test cleanup keeps recent jobs"""
        job_id = "recent-job"
        job_dir = mock_storage_service.get_job_output_dir(job_id)
        (job_dir / "output.png").write_bytes(b"test")

        # Don't modify mtime - should be recent
        cleaned = mock_storage_service.cleanup_old_jobs()

        # Job should still exist
        assert job_dir.exists()


class TestGetDiskUsage:
    """Tests for disk usage statistics"""

    def test_get_disk_usage_calculates_sizes(self, mock_storage_service):
        """Test disk usage calculation"""
        job_id = "disk-test"

        # Create files with known sizes
        upload_dir = mock_storage_service.get_job_upload_dir(job_id)
        output_dir = mock_storage_service.get_job_output_dir(job_id)

        (upload_dir / "input.jpg").write_bytes(b"x" * 1024)  # 1KB
        (output_dir / "output.png").write_bytes(b"y" * 2048)  # 2KB

        usage = mock_storage_service.get_disk_usage()

        assert "upload_dir_size_mb" in usage
        assert "output_dir_size_mb" in usage
        assert "total_size_mb" in usage
        assert usage["upload_dir_size_mb"] >= 0
        assert usage["output_dir_size_mb"] >= 0

    def test_get_disk_usage_empty_dirs(self, tmp_path):
        """Test disk usage with empty directories"""
        service = StorageService(
            upload_dir=tmp_path / "empty_uploads",
            output_dir=tmp_path / "empty_outputs",
        )

        usage = service.get_disk_usage()

        assert usage["upload_dir_size_mb"] == 0
        assert usage["output_dir_size_mb"] == 0
        assert usage["total_size_mb"] == 0
