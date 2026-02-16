"""
Tests for TrellisV1Client (HuggingFace-hosted TRELLIS)
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock


class TestTrellisV1Init:
    """Tests for TrellisV1Client initialization"""

    def test_init_default_space(self, mock_gradio_client, monkeypatch):
        """Test initialization with default HuggingFace space"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        assert client._client is None  # Lazy loaded

    def test_init_custom_space(self, mock_gradio_client, monkeypatch):
        """Test initialization with custom HuggingFace space"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client(hf_space="custom/space")

        assert client.hf_space == "custom/space"


class TestLazyClientLoading:
    """Tests for lazy Gradio client loading"""

    def test_client_not_loaded_on_init(self, mock_gradio_client, monkeypatch):
        """Test client is not loaded on initialization"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        mock_gradio_client["Client"].assert_not_called()

    def test_client_loaded_on_first_access(self, mock_gradio_client, monkeypatch):
        """Test client is loaded on first access"""
        from api.services.trellis_v1 import TrellisV1Client

        trellis = TrellisV1Client()
        _ = trellis.client

        mock_gradio_client["Client"].assert_called_once()

    def test_client_reused_on_subsequent_access(self, mock_gradio_client, monkeypatch):
        """Test client is reused on subsequent access"""
        from api.services.trellis_v1 import TrellisV1Client

        trellis = TrellisV1Client()
        client1 = trellis.client
        client2 = trellis.client

        mock_gradio_client["Client"].assert_called_once()


class TestProcessSingle:
    """Tests for single image processing"""

    def test_process_single_success(self, mock_gradio_client, sample_images, tmp_path, monkeypatch):
        """Test successful single image processing"""
        # Create a mock GLB file in a different location
        mock_glb = tmp_path / "mock_output" / "result.glb"
        mock_glb.parent.mkdir(parents=True, exist_ok=True)
        mock_glb.write_bytes(b"fake glb content")

        mock_gradio_client["client"].predict.return_value = {"glb": str(mock_glb)}

        from api.services.trellis_v1 import TrellisV1Client
        client = TrellisV1Client()

        input_path = sample_images[".jpg"]
        output_path = tmp_path / "output.glb"

        result = client.process_single(input_path, output_path)

        assert result == output_path
        assert output_path.exists()

    def test_process_single_creates_output_dir(self, mock_gradio_client, sample_images, tmp_path, monkeypatch):
        """Test output directory is created"""
        # Create a mock GLB file
        mock_glb = tmp_path / "mock_output" / "result.glb"
        mock_glb.parent.mkdir(parents=True, exist_ok=True)
        mock_glb.write_bytes(b"fake glb content")

        mock_gradio_client["client"].predict.return_value = {"glb": str(mock_glb)}

        from api.services.trellis_v1 import TrellisV1Client
        client = TrellisV1Client()

        input_path = sample_images[".jpg"]
        output_path = tmp_path / "nested" / "dir" / "output.glb"

        client.process_single(input_path, output_path)

        assert output_path.parent.exists()

    def test_process_single_with_seed(self, mock_gradio_client, sample_images, tmp_path, monkeypatch):
        """Test processing with custom seed"""
        from api.services.trellis_v1 import TrellisV1Client

        # Create fake GLB for return
        fake_glb = tmp_path / "fake.glb"
        fake_glb.write_bytes(b"fake glb")
        mock_gradio_client["client"].predict.return_value = {"glb": str(fake_glb)}

        client = TrellisV1Client()
        input_path = sample_images[".jpg"]
        output_path = tmp_path / "output.glb"

        client.process_single(input_path, output_path, seed=42)

        # Verify predict was called with correct seed
        call_kwargs = mock_gradio_client["client"].predict.call_args[1]
        assert call_kwargs["seed"] == 42


class TestProcessMulti:
    """Tests for multi-image processing"""

    def test_process_multi_success(self, mock_gradio_client, sample_images, tmp_path, monkeypatch):
        """Test successful multi-image processing"""
        from api.services.trellis_v1 import TrellisV1Client

        # Create fake GLB for return
        fake_glb = tmp_path / "fake.glb"
        fake_glb.write_bytes(b"fake glb")
        mock_gradio_client["client"].predict.return_value = {"glb": str(fake_glb)}

        client = TrellisV1Client()
        input_paths = [sample_images[".jpg"], sample_images[".png"]]
        output_path = tmp_path / "output.glb"

        result = client.process_multi(input_paths, output_path)

        assert result == output_path

    def test_process_multi_fallback_to_single(self, mock_gradio_client, sample_images, tmp_path, monkeypatch):
        """Test fallback to single image when multi-image API fails"""
        from api.services.trellis_v1 import TrellisV1Client

        # Create fake GLB for return
        fake_glb = tmp_path / "fake.glb"
        fake_glb.write_bytes(b"fake glb")

        # First call (multi-image) fails, second call (single) succeeds
        call_count = [0]

        def mock_predict(**kwargs):
            call_count[0] += 1
            if kwargs.get("api_name") == "/multiimage_to_3d":
                raise Exception("Multi-image API not available")
            return {"glb": str(fake_glb)}

        mock_gradio_client["client"].predict.side_effect = mock_predict

        client = TrellisV1Client()
        input_paths = [sample_images[".jpg"], sample_images[".png"]]
        output_path = tmp_path / "output.glb"

        result = client.process_multi(input_paths, output_path)

        # Should still succeed via fallback
        assert result == output_path
        # Should have tried both endpoints
        assert call_count[0] == 2


class TestProcess:
    """Tests for the main process method"""

    def test_process_single_image(self, mock_gradio_client, sample_images, tmp_path, monkeypatch):
        """Test process with single image uses process_single"""
        # Create a mock GLB file
        mock_glb = tmp_path / "mock_output" / "result.glb"
        mock_glb.parent.mkdir(parents=True, exist_ok=True)
        mock_glb.write_bytes(b"fake glb content")

        mock_gradio_client["client"].predict.return_value = {"glb": str(mock_glb)}

        from api.services.trellis_v1 import TrellisV1Client
        client = TrellisV1Client()

        input_paths = [sample_images[".jpg"]]
        output_path = tmp_path / "output.glb"

        # Spy on process_single
        original_process_single = client.process_single
        client.process_single = Mock(side_effect=original_process_single)

        client.process(input_paths, output_path)

        client.process_single.assert_called_once()

    def test_process_multiple_images(self, mock_gradio_client, sample_images, tmp_path, monkeypatch):
        """Test process with multiple images uses process_multi"""
        from api.services.trellis_v1 import TrellisV1Client

        # Create fake GLB for return
        fake_glb = tmp_path / "fake.glb"
        fake_glb.write_bytes(b"fake glb")
        mock_gradio_client["client"].predict.return_value = {"glb": str(fake_glb)}

        client = TrellisV1Client()

        # Spy on process_multi
        original_process_multi = client.process_multi
        client.process_multi = Mock(side_effect=original_process_multi)

        input_paths = [sample_images[".jpg"], sample_images[".png"]]
        output_path = tmp_path / "output.glb"

        client.process(input_paths, output_path)

        client.process_multi.assert_called_once()


class TestExtractGlbPath:
    """Tests for GLB path extraction from various result formats"""

    def test_extract_dict_with_glb_key(self, mock_gradio_client, monkeypatch):
        """Test extracting from dict with 'glb' key"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        result = {"glb": "/path/to/output.glb", "other": "data"}
        path = client._extract_glb_path(result)

        assert path == "/path/to/output.glb"

    def test_extract_dict_with_output_key(self, mock_gradio_client, monkeypatch):
        """Test extracting from dict with 'output' key"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        result = {"output": "/path/to/model.glb"}
        path = client._extract_glb_path(result)

        assert path == "/path/to/model.glb"

    def test_extract_string_path(self, mock_gradio_client, monkeypatch):
        """Test extracting from direct string path"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        result = "/path/to/output.glb"
        path = client._extract_glb_path(result)

        assert path == "/path/to/output.glb"

    def test_extract_tuple_first_element(self, mock_gradio_client, monkeypatch):
        """Test extracting from tuple (first element)"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        result = ("/path/to/output.glb", "other_data", 123)
        path = client._extract_glb_path(result)

        assert path == "/path/to/output.glb"

    def test_extract_list_first_element(self, mock_gradio_client, monkeypatch):
        """Test extracting from list (first element)"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        result = ["/path/to/output.glb", "extra"]
        path = client._extract_glb_path(result)

        assert path == "/path/to/output.glb"

    def test_extract_dict_no_valid_key_raises(self, mock_gradio_client, monkeypatch):
        """Test extracting from dict without valid key raises ValueError"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        result = {"unknown_key": "value"}

        with pytest.raises(ValueError) as exc_info:
            client._extract_glb_path(result)

        assert "No GLB path found" in str(exc_info.value)

    def test_extract_unexpected_type_raises(self, mock_gradio_client, monkeypatch):
        """Test extracting from unexpected type raises ValueError"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        result = 12345  # Integer - unexpected type

        with pytest.raises(ValueError) as exc_info:
            client._extract_glb_path(result)

        assert "Unexpected result format" in str(exc_info.value)


class TestHealthCheck:
    """Tests for health check"""

    def test_health_check_space_accessible(self, mock_gradio_client, monkeypatch):
        """Test health check returns True when space is accessible"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        result = client.health_check()

        assert result is True

    def test_health_check_space_down(self, mock_gradio_client, monkeypatch):
        """Test health check returns False when space is down"""
        from api.services.trellis_v1 import TrellisV1Client

        # Make client creation fail
        mock_gradio_client["Client"].side_effect = Exception("Space unavailable")

        client = TrellisV1Client()

        result = client.health_check()

        assert result is False


class TestCleanup:
    """Tests for client cleanup"""

    def test_cleanup_releases_client(self, mock_gradio_client, monkeypatch):
        """Test cleanup releases client"""
        from api.services.trellis_v1 import TrellisV1Client

        client = TrellisV1Client()

        # Access client to load it
        _ = client.client
        assert client._client is not None

        # Cleanup
        client.cleanup()

        assert client._client is None


class TestGetTrellisV1Client:
    """Tests for get_trellis_v1_client dependency"""

    def test_get_trellis_v1_client_returns_instance(self, mock_gradio_client, monkeypatch):
        """Test get_trellis_v1_client returns instance"""
        from api.services.trellis_v1 import get_trellis_v1_client
        import api.services.trellis_v1 as trellis_module

        # Reset global
        trellis_module._trellis_v1_client = None

        client = get_trellis_v1_client()
        assert client is not None

        # Reset
        trellis_module._trellis_v1_client = None

    def test_get_trellis_v1_client_same_instance(self, mock_gradio_client, monkeypatch):
        """Test get_trellis_v1_client returns same instance"""
        from api.services.trellis_v1 import get_trellis_v1_client
        import api.services.trellis_v1 as trellis_module

        # Reset global
        trellis_module._trellis_v1_client = None

        client1 = get_trellis_v1_client()
        client2 = get_trellis_v1_client()

        assert client1 is client2

        # Reset
        trellis_module._trellis_v1_client = None
