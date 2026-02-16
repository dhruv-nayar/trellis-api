"""
TRELLIS V1 Service
HuggingFace-hosted TRELLIS via Gradio client
"""

import shutil
import logging
from pathlib import Path
from typing import List, Optional, Union
from gradio_client import Client, handle_file

from api.config import settings

logger = logging.getLogger(__name__)


class TrellisV1Client:
    """
    HuggingFace-hosted TRELLIS via Gradio client.

    This client connects to the JeffreyXiang/TRELLIS HuggingFace Space
    and uses its API for image-to-3D conversion. No local GPU required.
    """

    def __init__(self, hf_space: Optional[str] = None):
        """
        Initialize TRELLIS V1 client.

        Args:
            hf_space: HuggingFace Space ID (default: JeffreyXiang/TRELLIS)
        """
        self.hf_space = hf_space or settings.huggingface_space
        self._client: Optional[Client] = None
        logger.info(f"TrellisV1Client initialized for space: {self.hf_space}")

    @property
    def client(self) -> Client:
        """Lazy-load the Gradio client"""
        if self._client is None:
            logger.info(f"Connecting to HuggingFace Space: {self.hf_space}")
            self._client = Client(self.hf_space)
        return self._client

    def process_single(
        self,
        image_path: Path,
        output_path: Path,
        seed: int = 1,
    ) -> Path:
        """
        Process a single image to 3D GLB.

        Args:
            image_path: Path to input image
            output_path: Path for output GLB file
            seed: Random seed for reproducibility

        Returns:
            Path to output GLB file
        """
        logger.info(f"Processing single image: {image_path}")

        # Call HuggingFace Space
        result = self.client.predict(
            image=handle_file(str(image_path)),
            seed=seed,
            api_name="/image_to_3d"
        )

        # Extract GLB path from result
        glb_path = self._extract_glb_path(result)

        # Copy result to output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(glb_path, output_path)

        logger.info(f"Saved GLB to: {output_path}")
        return output_path

    def process_multi(
        self,
        image_paths: List[Path],
        output_path: Path,
        seed: int = 1,
    ) -> Path:
        """
        Process multiple images for multi-view reconstruction.

        Note: Multi-image support in TRELLIS is experimental.
        It uses conditioning embedding averaging and works best when
        images are similar compositions (same subject, different angles).

        Args:
            image_paths: List of input image paths
            output_path: Path for output GLB file
            seed: Random seed for reproducibility

        Returns:
            Path to output GLB file
        """
        logger.info(f"Processing {len(image_paths)} images for multi-view reconstruction")

        # Try multi-image API first
        try:
            # Build multi-image input
            multi_files = [handle_file(str(p)) for p in image_paths]

            # Attempt multi-image endpoint
            # Note: The actual endpoint name may vary depending on the HF Space
            result = self.client.predict(
                multiimages=multi_files,
                seed=seed,
                api_name="/multiimage_to_3d"
            )

            glb_path = self._extract_glb_path(result)

        except Exception as e:
            logger.warning(f"Multi-image API failed ({e}), falling back to single image")
            # Fall back to processing just the first image
            return self.process_single(image_paths[0], output_path, seed)

        # Copy result to output path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(glb_path, output_path)

        logger.info(f"Saved multi-view GLB to: {output_path}")
        return output_path

    def process(
        self,
        image_paths: List[Path],
        output_path: Path,
        seed: int = 1,
    ) -> Path:
        """
        Process image(s) to 3D GLB.
        Automatically chooses single or multi-image mode.

        Args:
            image_paths: List of input image paths (1 or more)
            output_path: Path for output GLB file
            seed: Random seed for reproducibility

        Returns:
            Path to output GLB file
        """
        if len(image_paths) == 1:
            return self.process_single(image_paths[0], output_path, seed)
        else:
            return self.process_multi(image_paths, output_path, seed)

    def _extract_glb_path(self, result: Union[dict, str, tuple]) -> str:
        """
        Extract GLB path from Gradio API result.

        The result format can vary:
        - dict with 'glb' key
        - string path directly
        - tuple with path as first element
        """
        if isinstance(result, dict):
            if 'glb' in result:
                return result['glb']
            # Try other common keys
            for key in ['output', 'file', 'path', 'model']:
                if key in result:
                    return result[key]
            raise ValueError(f"No GLB path found in result dict: {result.keys()}")

        elif isinstance(result, str):
            return result

        elif isinstance(result, (tuple, list)):
            # Usually the GLB is the first element
            return result[0]

        else:
            raise ValueError(f"Unexpected result format: {type(result)}")

    def health_check(self) -> bool:
        """Check if the HuggingFace Space is accessible"""
        try:
            # Just try to create the client connection
            _ = self.client
            return True
        except Exception as e:
            logger.error(f"TRELLIS V1 health check failed: {e}")
            return False

    def cleanup(self):
        """Release client resources"""
        self._client = None
        logger.info("TrellisV1Client resources released")


# Global instance (lazy-loaded)
_trellis_v1_client: Optional[TrellisV1Client] = None


def get_trellis_v1_client() -> TrellisV1Client:
    """Get or create TrellisV1Client instance"""
    global _trellis_v1_client
    if _trellis_v1_client is None:
        _trellis_v1_client = TrellisV1Client()
    return _trellis_v1_client
