"""
TRELLIS Service
Unified interface for TRELLIS processing (V1 HuggingFace, V2 RunPod)
"""

import logging
from pathlib import Path
from typing import List, Optional, Protocol

from api.models.enums import TrellisBackend
from api.config import settings

logger = logging.getLogger(__name__)


class TrellisClient(Protocol):
    """Protocol for TRELLIS clients"""

    def process(
        self,
        image_paths: List[Path],
        output_path: Path,
        seed: int = 1,
    ) -> Path:
        """Process images to 3D GLB"""
        ...

    def health_check(self) -> bool:
        """Check client health"""
        ...


class TrellisService:
    """
    Unified interface for TRELLIS processing.

    Supports multiple backends:
    - V1: HuggingFace Gradio client (default)
    - V2: RunPod/Modal self-hosted
    """

    def __init__(self):
        self._v1_client: Optional[TrellisClient] = None
        self._v2_client: Optional[TrellisClient] = None

    def get_client(self, backend: TrellisBackend) -> TrellisClient:
        """Get appropriate client for the specified backend"""
        if backend == TrellisBackend.HUGGINGFACE:
            if self._v1_client is None:
                from api.services.trellis_v1 import TrellisV1Client
                self._v1_client = TrellisV1Client()
            return self._v1_client

        elif backend in (TrellisBackend.RUNPOD, TrellisBackend.MODAL):
            if self._v2_client is None:
                from api.services.trellis_v2 import TrellisV2Client
                self._v2_client = TrellisV2Client()
            return self._v2_client

        else:
            raise ValueError(f"Unknown backend: {backend}")

    def process(
        self,
        image_paths: List[Path],
        output_path: Path,
        backend: Optional[TrellisBackend] = None,
        seed: int = 1,
        texture_size: int = 2048,
        optimize: bool = True,
    ) -> Path:
        """
        Process images to 3D GLB using the specified backend.

        Args:
            image_paths: List of input image paths
            output_path: Path for output GLB file
            backend: Backend to use (default: from settings)
            seed: Random seed for reproducibility
            texture_size: Texture resolution (V2 only)
            optimize: Whether to optimize mesh (V2 only)

        Returns:
            Path to output GLB file
        """
        if backend is None:
            backend = TrellisBackend(settings.trellis_backend)

        logger.info(f"Processing {len(image_paths)} image(s) with backend: {backend.value}")

        client = self.get_client(backend)
        return client.process(image_paths, output_path, seed)

    def health_check(self, backend: Optional[TrellisBackend] = None) -> bool:
        """Check health of the specified backend"""
        if backend is None:
            backend = TrellisBackend(settings.trellis_backend)

        try:
            client = self.get_client(backend)
            return client.health_check()
        except Exception as e:
            logger.error(f"TRELLIS health check failed: {e}")
            return False


# Global instance
_trellis_service: Optional[TrellisService] = None


def get_trellis_service() -> TrellisService:
    """Get or create TrellisService instance"""
    global _trellis_service
    if _trellis_service is None:
        _trellis_service = TrellisService()
    return _trellis_service
