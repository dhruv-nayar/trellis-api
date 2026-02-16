"""
Tests for RemBG Celery tasks
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, PropertyMock
from PIL import Image


class TestProcessRembgTask:
    """Tests for process_rembg Celery task"""

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
    def mock_services(self, monkeypatch, tmp_path, sample_image_bytes):
        """Set up mock services for task testing"""
        # Create test input files
        input_dir = tmp_path / "inputs"
        input_dir.mkdir()
        input_file = input_dir / "test.jpg"
        img = Image.new("RGB", (100, 100), color="red")
        img.save(input_file, "JPEG")

        # Create output directory
        output_dir = tmp_path / "outputs"
        output_dir.mkdir()

        # Mock job store
        mock_job_store = Mock()
        mock_job_store.set_processing = Mock()
        mock_job_store.update_job = Mock()
        mock_job_store.set_completed = Mock()
        mock_job_store.set_failed = Mock()

        # Mock rembg service
        mock_rembg = Mock()

        def mock_process_batch(input_paths, output_dir, **kwargs):
            # Create fake output files
            output_paths = []
            for p in input_paths:
                out_path = output_dir / f"{p.stem}_nobg.png"
                out_img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
                out_img.save(out_path, "PNG")
                output_paths.append(out_path)

            # Call progress callback if provided
            if kwargs.get("progress_callback"):
                for i, _ in enumerate(input_paths):
                    kwargs["progress_callback"](i + 1, len(input_paths))

            return output_paths

        mock_rembg.process_batch = Mock(side_effect=mock_process_batch)

        monkeypatch.setattr("api.tasks.rembg_tasks.get_job_store", lambda: mock_job_store)
        monkeypatch.setattr("api.tasks.rembg_tasks.get_rembg_service", lambda **kwargs: mock_rembg)

        return {
            "job_store": mock_job_store,
            "rembg": mock_rembg,
            "input_file": input_file,
            "output_dir": output_dir,
        }

    @pytest.fixture
    def mock_current_task(self, monkeypatch):
        """Mock current_task used for getting task ID"""
        mock_task = Mock()
        mock_task.request = Mock()
        mock_task.request.id = "celery-task-123"
        mock_task.request.retries = 0
        monkeypatch.setattr("api.tasks.rembg_tasks.current_task", mock_task)
        return mock_task

    def test_process_rembg_success(self, mock_services, mock_current_task, monkeypatch):
        """Test successful RemBG task execution"""
        from api.tasks.rembg_tasks import process_rembg

        job_id = "test-job-123"
        input_paths = [str(mock_services["input_file"])]
        output_dir = str(mock_services["output_dir"])

        # Call the task directly (eager mode executes synchronously)
        async_result = process_rembg.delay(job_id, input_paths, output_dir)

        # Verify job was marked as processing
        mock_services["job_store"].set_processing.assert_called_once_with(
            job_id, "celery-task-123"
        )

        # Verify job was marked as completed
        mock_services["job_store"].set_completed.assert_called_once()
        call_kwargs = mock_services["job_store"].set_completed.call_args[1]
        assert call_kwargs["job_id"] == job_id
        assert call_kwargs["output_count"] == 1

    def test_process_rembg_updates_progress(self, mock_services, mock_current_task, monkeypatch, tmp_path):
        """Test progress updates during RemBG task"""
        from api.tasks.rembg_tasks import process_rembg

        # Create multiple input files
        input_dir = tmp_path / "multi_inputs"
        input_dir.mkdir(exist_ok=True)
        input_files = []
        for i in range(3):
            f = input_dir / f"test{i}.jpg"
            img = Image.new("RGB", (100, 100), color="red")
            img.save(f, "JPEG")
            input_files.append(str(f))

        process_rembg.delay("test-job", input_files, str(mock_services["output_dir"]))

        # Verify progress was updated
        assert mock_services["job_store"].update_job.call_count >= 3

    def test_process_rembg_download_urls_format(self, mock_services, mock_current_task, monkeypatch):
        """Test download URLs are formatted correctly"""
        from api.tasks.rembg_tasks import process_rembg

        job_id = "test-job-456"

        process_rembg.delay(job_id, [str(mock_services["input_file"])], str(mock_services["output_dir"]))

        # Check download URLs
        call_kwargs = mock_services["job_store"].set_completed.call_args[1]
        download_urls = call_kwargs["download_urls"]

        assert len(download_urls) == 1
        assert f"/api/v1/jobs/{job_id}/download/" in download_urls[0]
        assert download_urls[0].endswith(".png")

    def test_process_rembg_fails_after_max_retries(self, mock_services, mock_current_task, monkeypatch):
        """Test task fails after max retries"""
        from api.tasks.rembg_tasks import process_rembg

        # Make service fail
        mock_services["rembg"].process_batch.side_effect = Exception("Persistent error")

        # Set retries to max via the current_task mock
        mock_current_task.request.retries = 3

        # In eager mode with max retries reached, task should mark job as failed
        with patch.object(process_rembg, 'max_retries', 3):
            process_rembg.delay(
                "test-job",
                [str(mock_services["input_file"])],
                str(mock_services["output_dir"])
            )

        # Job should be marked as failed
        mock_services["job_store"].set_failed.assert_called_once()

    def test_process_rembg_with_alpha_matting(self, mock_services, mock_current_task, monkeypatch):
        """Test task passes alpha matting params"""
        from api.tasks.rembg_tasks import process_rembg

        process_rembg.delay(
            "test-job",
            [str(mock_services["input_file"])],
            str(mock_services["output_dir"]),
            "u2net",  # model
            True,  # alpha_matting
            230,  # alpha_matting_foreground_threshold
            20,  # alpha_matting_background_threshold
        )

        # Verify params were passed to service
        call_kwargs = mock_services["rembg"].process_batch.call_args[1]
        assert call_kwargs["alpha_matting"] is True
        assert call_kwargs["alpha_matting_foreground_threshold"] == 230
        assert call_kwargs["alpha_matting_background_threshold"] == 20
