"""
Tests for TRELLIS Celery tasks
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image


class TestProcessTrellisTask:
    """Tests for process_trellis Celery task"""

    @pytest.fixture(autouse=True)
    def setup_celery_eager(self):
        """Configure Celery for eager execution (synchronous without broker)"""
        from api.tasks.celery_app import celery_app

        # Save original settings
        original_always_eager = celery_app.conf.task_always_eager
        original_eager_propagates = celery_app.conf.task_eager_propagates
        original_result_backend = celery_app.conf.result_backend

        # Configure for testing
        celery_app.conf.task_always_eager = True
        celery_app.conf.task_eager_propagates = False  # Don't propagate exceptions
        celery_app.conf.result_backend = 'cache+memory://'  # In-memory results

        yield

        # Restore original settings
        celery_app.conf.task_always_eager = original_always_eager
        celery_app.conf.task_eager_propagates = original_eager_propagates
        celery_app.conf.result_backend = original_result_backend

    @pytest.fixture
    def mock_services(self, monkeypatch, tmp_path):
        """Set up mock services for task testing"""
        # Create test input files
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        input_file = input_dir / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(input_file, "JPEG")

        # Create output path
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()
        output_path = output_dir / "model.glb"

        # Mock job store
        mock_job_store = Mock()
        mock_job_store.set_processing = Mock()
        mock_job_store.update_job = Mock()
        mock_job_store.set_completed = Mock()
        mock_job_store.set_failed = Mock()

        # Mock trellis service
        mock_trellis = Mock()

        def mock_process(image_paths, output_path, **kwargs):
            # Create fake output file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(b"fake glb content")
            return output_path

        mock_trellis.process = Mock(side_effect=mock_process)

        monkeypatch.setattr("api.tasks.trellis_tasks.get_job_store", lambda: mock_job_store)
        monkeypatch.setattr("api.tasks.trellis_tasks.get_trellis_service", lambda: mock_trellis)

        return {
            "job_store": mock_job_store,
            "trellis": mock_trellis,
            "input_file": input_file,
            "output_path": output_path,
        }

    @pytest.fixture
    def mock_current_task(self, monkeypatch):
        """Mock current_task used for getting task ID"""
        mock_task = Mock()
        mock_task.request = Mock()
        mock_task.request.id = "celery-trellis-123"
        mock_task.request.retries = 0
        monkeypatch.setattr("api.tasks.trellis_tasks.current_task", mock_task)
        return mock_task

    def test_process_trellis_success(self, mock_services, mock_current_task, monkeypatch):
        """Test successful TRELLIS task execution"""
        from api.tasks.trellis_tasks import process_trellis

        job_id = "test-trellis-123"
        input_paths = [str(mock_services["input_file"])]
        output_path = str(mock_services["output_path"])

        # Call the task directly (eager mode executes synchronously)
        process_trellis.delay(job_id, input_paths, output_path)

        # Verify job was marked as processing
        mock_services["job_store"].set_processing.assert_called_once_with(
            job_id, "celery-trellis-123"
        )

        # Verify job was marked as completed
        mock_services["job_store"].set_completed.assert_called_once()
        call_kwargs = mock_services["job_store"].set_completed.call_args[1]
        assert call_kwargs["job_id"] == job_id
        assert call_kwargs["output_count"] == 1

    def test_process_trellis_backend_selection(self, mock_services, mock_current_task, monkeypatch):
        """Test correct backend is passed to service"""
        from api.tasks.trellis_tasks import process_trellis
        from api.models.enums import TrellisBackend

        process_trellis.delay(
            "test-job",
            [str(mock_services["input_file"])],
            str(mock_services["output_path"]),
            "runpod",  # backend
        )

        # Verify backend was passed
        call_kwargs = mock_services["trellis"].process.call_args[1]
        assert call_kwargs["backend"] == TrellisBackend.RUNPOD

    def test_process_trellis_single_output(self, mock_services, mock_current_task, monkeypatch):
        """Test task produces single GLB output"""
        from api.tasks.trellis_tasks import process_trellis

        process_trellis.delay(
            "test-job",
            [str(mock_services["input_file"])],
            str(mock_services["output_path"]),
        )

        # Check completed was called with exactly 1 output
        call_kwargs = mock_services["job_store"].set_completed.call_args[1]
        assert call_kwargs["output_count"] == 1
        assert len(call_kwargs["download_urls"]) == 1

    def test_process_trellis_fails_after_max_retries(self, mock_services, mock_current_task, monkeypatch):
        """Test task fails after max retries"""
        from api.tasks.trellis_tasks import process_trellis

        # Make service fail
        mock_services["trellis"].process.side_effect = Exception("Persistent GPU error")

        # Set retries to max via the current_task mock
        mock_current_task.request.retries = 3

        # In eager mode with max retries reached, task should mark job as failed
        with patch.object(process_trellis, 'max_retries', 3):
            process_trellis.delay(
                "test-job",
                [str(mock_services["input_file"])],
                str(mock_services["output_path"]),
            )

        # Job should be marked as failed
        mock_services["job_store"].set_failed.assert_called_once()

    def test_process_trellis_with_custom_params(self, mock_services, mock_current_task, monkeypatch):
        """Test task passes custom parameters"""
        from api.tasks.trellis_tasks import process_trellis

        process_trellis.delay(
            "test-job",
            [str(mock_services["input_file"])],
            str(mock_services["output_path"]),
            "huggingface",  # backend
            42,  # seed
            1024,  # texture_size
            False,  # optimize
        )

        # Verify params were passed to service
        call_kwargs = mock_services["trellis"].process.call_args[1]
        assert call_kwargs["seed"] == 42
        assert call_kwargs["texture_size"] == 1024
        assert call_kwargs["optimize"] is False

    def test_process_trellis_multiple_images(self, mock_services, mock_current_task, monkeypatch, tmp_path):
        """Test task with multiple input images"""
        from api.tasks.trellis_tasks import process_trellis

        # Create multiple input files
        input_dir = tmp_path / "multi_inputs"
        input_dir.mkdir()
        input_files = []
        for i in range(3):
            f = input_dir / f"view{i}.jpg"
            img = Image.new("RGB", (100, 100), color="blue")
            img.save(f, "JPEG")
            input_files.append(str(f))

        process_trellis.delay(
            "multi-view-job",
            input_files,
            str(mock_services["output_path"]),
        )

        # Verify multiple images were passed
        call_args = mock_services["trellis"].process.call_args[1]
        assert len(call_args["image_paths"]) == 3

    def test_process_trellis_download_url_format(self, mock_services, mock_current_task, monkeypatch):
        """Test download URL is correctly formatted"""
        from api.tasks.trellis_tasks import process_trellis

        job_id = "url-test-job"

        process_trellis.delay(
            job_id,
            [str(mock_services["input_file"])],
            str(mock_services["output_path"]),
        )

        # Check download URL format in set_completed call
        call_kwargs = mock_services["job_store"].set_completed.call_args[1]
        download_urls = call_kwargs["download_urls"]
        assert len(download_urls) == 1
        assert f"/api/v1/jobs/{job_id}/download/" in download_urls[0]
        assert download_urls[0].endswith(".glb")

    def test_process_trellis_progress_updates(self, mock_services, mock_current_task, monkeypatch):
        """Test progress is updated during processing"""
        from api.tasks.trellis_tasks import process_trellis

        process_trellis.delay(
            "progress-job",
            [str(mock_services["input_file"])],
            str(mock_services["output_path"]),
        )

        # Should update progress at least twice (connecting, processing)
        assert mock_services["job_store"].update_job.call_count >= 2
