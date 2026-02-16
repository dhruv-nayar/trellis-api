"""
Tests for RemBGService
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from PIL import Image


class TestRemBGServiceInit:
    """Tests for RemBG service initialization"""

    def test_init_default_model(self, mock_rembg, monkeypatch):
        """Test initialization with default model"""
        from api.services.rembg_service import RemBGService

        service = RemBGService()

        assert service.model_name == "u2net"
        assert service._session is None  # Lazy loaded

    def test_init_custom_model(self, mock_rembg, monkeypatch):
        """Test initialization with custom model"""
        from api.services.rembg_service import RemBGService

        service = RemBGService(model_name="isnet-general-use")

        assert service.model_name == "isnet-general-use"


class TestLazySessionLoading:
    """Tests for lazy session loading"""

    def test_session_not_loaded_on_init(self, mock_rembg, monkeypatch):
        """Test session is not loaded on initialization"""
        from api.services.rembg_service import RemBGService

        service = RemBGService()

        # new_session should not be called yet
        mock_rembg["new_session"].assert_not_called()
        assert service._session is None

    def test_session_loaded_on_first_access(self, mock_rembg, monkeypatch):
        """Test session is loaded on first access"""
        from api.services.rembg_service import RemBGService

        service = RemBGService()
        session = service.session

        mock_rembg["new_session"].assert_called_once_with("u2net")
        assert session is not None

    def test_session_reused_on_subsequent_access(self, mock_rembg, monkeypatch):
        """Test session is reused on subsequent access"""
        from api.services.rembg_service import RemBGService

        service = RemBGService()
        session1 = service.session
        session2 = service.session

        # Should only be called once
        mock_rembg["new_session"].assert_called_once()
        assert session1 is session2


class TestProcessSingle:
    """Tests for single image processing"""

    def test_process_single_creates_png(self, mock_rembg_service, sample_images, tmp_path):
        """Test output is PNG with transparency"""
        input_path = sample_images[".jpg"]
        output_path = tmp_path / "output.png"

        result = mock_rembg_service.process_single(input_path, output_path)

        assert result == output_path
        assert output_path.exists()
        assert output_path.suffix == ".png"

    def test_process_single_converts_mode(self, mock_rembg, tmp_path, monkeypatch):
        """Test images are converted to RGB if needed"""
        from api.services.rembg_service import RemBGService

        # Create L mode (grayscale) image
        gray_img = Image.new("L", (100, 100), color=128)
        gray_path = tmp_path / "gray.png"
        gray_img.save(gray_path, "PNG")

        output_path = tmp_path / "output.png"

        service = RemBGService()
        result = service.process_single(gray_path, output_path)

        # rembg.remove should have been called
        mock_rembg["remove"].assert_called()

    def test_process_single_creates_output_directory(self, mock_rembg_service, sample_images, tmp_path):
        """Test output directory is created if it doesn't exist"""
        input_path = sample_images[".png"]
        output_path = tmp_path / "nested" / "deep" / "output.png"

        result = mock_rembg_service.process_single(input_path, output_path)

        assert output_path.parent.exists()
        assert output_path.exists()

    def test_process_single_with_alpha_matting(self, mock_rembg, sample_images, tmp_path, monkeypatch):
        """Test processing with alpha matting enabled"""
        from api.services.rembg_service import RemBGService

        input_path = sample_images[".jpg"]
        output_path = tmp_path / "output.png"

        service = RemBGService()
        service.process_single(
            input_path,
            output_path,
            alpha_matting=True,
            alpha_matting_foreground_threshold=230,
            alpha_matting_background_threshold=20,
        )

        # Verify remove was called with correct params
        call_kwargs = mock_rembg["remove"].call_args[1]
        assert call_kwargs["alpha_matting"] is True
        assert call_kwargs["alpha_matting_foreground_threshold"] == 230
        assert call_kwargs["alpha_matting_background_threshold"] == 20


class TestProcessBatch:
    """Tests for batch processing"""

    def test_process_batch_all_success(self, mock_rembg_service, sample_images, tmp_path):
        """Test all images processed successfully"""
        input_paths = [sample_images[".jpg"], sample_images[".png"]]
        output_dir = tmp_path / "batch_output"

        results = mock_rembg_service.process_batch(input_paths, output_dir)

        assert len(results) == 2
        assert all(p.exists() for p in results)
        # All outputs should be PNG
        assert all(p.suffix == ".png" for p in results)

    def test_process_batch_output_naming(self, mock_rembg_service, sample_images, tmp_path):
        """Test output files are named correctly"""
        input_paths = [sample_images[".jpg"]]
        output_dir = tmp_path / "batch_output"

        results = mock_rembg_service.process_batch(input_paths, output_dir)

        assert len(results) == 1
        assert results[0].name == "test_nobg.png"

    def test_process_batch_partial_failure(self, mock_rembg, sample_images, tmp_path, monkeypatch):
        """Test batch continues on individual failures"""
        from api.services.rembg_service import RemBGService

        # Set up mock to fail on second call
        call_count = [0]

        def mock_remove(img, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise Exception("Processing failed")
            return Image.new("RGBA", (100, 100), (0, 0, 0, 0))

        mock_rembg["remove"].side_effect = mock_remove

        input_paths = [sample_images[".jpg"], sample_images[".png"], sample_images[".webp"]]
        output_dir = tmp_path / "batch_output"

        service = RemBGService()
        results = service.process_batch(input_paths, output_dir)

        # Should have processed 2 out of 3
        assert len(results) == 2

    def test_process_batch_progress_callback(self, mock_rembg_service, sample_images, tmp_path):
        """Test progress callback is invoked"""
        callback = Mock()

        input_paths = [sample_images[".jpg"], sample_images[".png"]]
        output_dir = tmp_path / "batch_output"

        mock_rembg_service.process_batch(
            input_paths,
            output_dir,
            progress_callback=callback,
        )

        # Should be called twice (once per image)
        assert callback.call_count == 2
        # Check call arguments
        callback.assert_any_call(1, 2)  # First image
        callback.assert_any_call(2, 2)  # Second image

    def test_process_batch_creates_output_dir(self, mock_rembg_service, sample_images, tmp_path):
        """Test output directory is created"""
        output_dir = tmp_path / "new" / "output" / "dir"
        assert not output_dir.exists()

        mock_rembg_service.process_batch([sample_images[".jpg"]], output_dir)

        assert output_dir.exists()

    def test_process_batch_empty_input(self, mock_rembg_service, tmp_path):
        """Test handling of empty input list"""
        output_dir = tmp_path / "output"

        results = mock_rembg_service.process_batch([], output_dir)

        assert results == []


class TestCleanup:
    """Tests for service cleanup"""

    def test_cleanup_releases_session(self, mock_rembg, monkeypatch):
        """Test cleanup releases session"""
        from api.services.rembg_service import RemBGService

        service = RemBGService()

        # Access session to load it
        _ = service.session
        assert service._session is not None

        # Cleanup
        service.cleanup()

        assert service._session is None

    def test_cleanup_safe_when_no_session(self, mock_rembg, monkeypatch):
        """Test cleanup is safe when no session loaded"""
        from api.services.rembg_service import RemBGService

        service = RemBGService()
        # Don't access session

        # Should not raise
        service.cleanup()

        assert service._session is None


class TestGetRemBGService:
    """Tests for get_rembg_service dependency"""

    def test_get_rembg_service_returns_instance(self, mock_rembg, monkeypatch):
        """Test get_rembg_service returns instance"""
        from api.services.rembg_service import get_rembg_service
        import api.services.rembg_service as rembg_module

        # Reset global
        rembg_module._rembg_service = None

        service = get_rembg_service()
        assert service is not None
        assert service.model_name == "u2net"

        # Reset
        rembg_module._rembg_service = None

    def test_get_rembg_service_same_instance(self, mock_rembg, monkeypatch):
        """Test get_rembg_service returns same instance"""
        from api.services.rembg_service import get_rembg_service
        import api.services.rembg_service as rembg_module

        # Reset global
        rembg_module._rembg_service = None

        service1 = get_rembg_service()
        service2 = get_rembg_service()

        assert service1 is service2

        # Reset
        rembg_module._rembg_service = None

    def test_get_rembg_service_new_model(self, mock_rembg, monkeypatch):
        """Test get_rembg_service creates new instance for different model"""
        from api.services.rembg_service import get_rembg_service
        import api.services.rembg_service as rembg_module

        # Reset global
        rembg_module._rembg_service = None

        service1 = get_rembg_service("u2net")
        service2 = get_rembg_service("isnet-general-use")

        # Should be different instances
        assert service1 is not service2
        assert service2.model_name == "isnet-general-use"

        # Reset
        rembg_module._rembg_service = None
