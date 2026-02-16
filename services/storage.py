"""
File Storage Service
Handles file uploads, downloads, and cleanup
"""

import shutil
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Optional, Tuple
from fastapi import UploadFile, HTTPException
from api.config import settings

logger = logging.getLogger(__name__)

# Supported image formats
SUPPORTED_FORMATS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff", ".tif"}


class StorageService:
    """Manages file storage for jobs"""

    def __init__(
        self,
        upload_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        max_file_size: Optional[int] = None,
        cleanup_after_hours: Optional[int] = None,
    ):
        self.upload_dir = upload_dir or settings.upload_dir
        self.output_dir = output_dir or settings.output_dir
        self.max_file_size = max_file_size or settings.max_file_size_bytes
        self.cleanup_after_hours = cleanup_after_hours or settings.cleanup_after_hours

        # Ensure directories exist
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_job_upload_dir(self, job_id: str) -> Path:
        """Get upload directory for a specific job"""
        job_dir = self.upload_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    def get_job_output_dir(self, job_id: str) -> Path:
        """Get output directory for a specific job"""
        job_dir = self.output_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        return job_dir

    async def save_uploads(
        self,
        files: List[UploadFile],
        job_id: str,
    ) -> Tuple[List[Path], List[str]]:
        """
        Save uploaded files to job-specific directory.

        Returns:
            Tuple of (saved_paths, original_filenames)

        Raises:
            HTTPException if validation fails
        """
        job_upload_dir = self.get_job_upload_dir(job_id)
        saved_paths = []
        filenames = []

        for file in files:
            # Validate file type
            if not file.filename:
                raise HTTPException(
                    status_code=400,
                    detail="File must have a filename"
                )

            suffix = Path(file.filename).suffix.lower()
            if suffix not in SUPPORTED_FORMATS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file format: {suffix}. Supported: {', '.join(SUPPORTED_FORMATS)}"
                )

            # Validate content type
            if file.content_type and not file.content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid content type: {file.content_type}. Expected image/*"
                )

            # Read and validate file size
            content = await file.read()
            if len(content) > self.max_file_size:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} exceeds max size of {self.max_file_size // (1024*1024)}MB"
                )

            # Generate safe filename
            safe_filename = self._sanitize_filename(file.filename)
            file_path = job_upload_dir / safe_filename

            # Handle duplicates
            counter = 1
            original_stem = file_path.stem
            while file_path.exists():
                file_path = job_upload_dir / f"{original_stem}_{counter}{file_path.suffix}"
                counter += 1

            # Save file
            with open(file_path, "wb") as f:
                f.write(content)

            saved_paths.append(file_path)
            filenames.append(file.filename)
            logger.debug(f"Saved upload: {file_path}")

        logger.info(f"Saved {len(saved_paths)} files for job {job_id}")
        return saved_paths, filenames

    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename to prevent path traversal"""
        # Get just the filename without any path components
        name = Path(filename).name
        # Replace any potentially dangerous characters
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
        sanitized = "".join(c if c in safe_chars else "_" for c in name)
        return sanitized or "unnamed"

    def get_output_files(self, job_id: str) -> List[Path]:
        """Get all output files for a job"""
        job_output_dir = self.get_job_output_dir(job_id)
        if not job_output_dir.exists():
            return []
        return list(job_output_dir.iterdir())

    def get_file_path(self, job_id: str, filename: str, is_output: bool = True) -> Optional[Path]:
        """Get path to a specific file"""
        base_dir = self.output_dir if is_output else self.upload_dir
        file_path = base_dir / job_id / filename

        # Security check - ensure path is within expected directory
        try:
            file_path.resolve().relative_to(base_dir.resolve())
        except ValueError:
            logger.warning(f"Path traversal attempt: {file_path}")
            return None

        if file_path.exists():
            return file_path
        return None

    def cleanup_job(self, job_id: str) -> bool:
        """Remove all files for a job"""
        cleaned = False
        for base_dir in [self.upload_dir, self.output_dir]:
            job_dir = base_dir / job_id
            if job_dir.exists():
                shutil.rmtree(job_dir)
                cleaned = True
                logger.info(f"Cleaned up {job_dir}")
        return cleaned

    def cleanup_old_jobs(self) -> int:
        """
        Remove job directories older than cleanup_after_hours.
        Returns the number of jobs cleaned up.
        """
        cutoff = datetime.now() - timedelta(hours=self.cleanup_after_hours)
        cleaned = 0

        for base_dir in [self.upload_dir, self.output_dir]:
            if not base_dir.exists():
                continue

            for job_dir in base_dir.iterdir():
                if not job_dir.is_dir():
                    continue

                try:
                    mtime = datetime.fromtimestamp(job_dir.stat().st_mtime)
                    if mtime < cutoff:
                        shutil.rmtree(job_dir)
                        cleaned += 1
                        logger.info(f"Cleaned up expired job directory: {job_dir}")
                except Exception as e:
                    logger.error(f"Error cleaning up {job_dir}: {e}")

        return cleaned

    def get_disk_usage(self) -> dict:
        """Get disk usage statistics"""
        def dir_size(path: Path) -> int:
            if not path.exists():
                return 0
            return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())

        upload_size = dir_size(self.upload_dir)
        output_size = dir_size(self.output_dir)

        return {
            "upload_dir_size_mb": round(upload_size / (1024 * 1024), 2),
            "output_dir_size_mb": round(output_size / (1024 * 1024), 2),
            "total_size_mb": round((upload_size + output_size) / (1024 * 1024), 2),
        }


# Global instance for dependency injection
_storage_service: Optional[StorageService] = None


def get_storage_service() -> StorageService:
    """Get or create StorageService instance"""
    global _storage_service
    if _storage_service is None:
        _storage_service = StorageService()
    return _storage_service
