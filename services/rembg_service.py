"""
RemBG Service
Background removal using rembg package
"""

import logging
from pathlib import Path
from typing import List, Optional
from PIL import Image

# Lazy imports for rembg to avoid slow startup
_rembg_remove = None
_rembg_new_session = None

logger = logging.getLogger(__name__)


def _get_rembg():
    """Lazy load rembg to avoid slow import at startup"""
    global _rembg_remove, _rembg_new_session
    if _rembg_remove is None:
        logger.info("Loading rembg module...")
        from rembg import remove, new_session
        _rembg_remove = remove
        _rembg_new_session = new_session
    return _rembg_remove, _rembg_new_session


class RemBGService:
    """Service for removing backgrounds from images using rembg"""

    def __init__(self, model_name: str = "u2net"):
        """
        Initialize RemBG service.

        Args:
            model_name: rembg model to use. Options:
                - u2net (default, general purpose)
                - u2netp (lighter version)
                - u2net_human_seg (optimized for humans)
                - u2net_cloth_seg (for clothing)
                - silueta (fast, lower quality)
                - isnet-general-use
                - isnet-anime
        """
        self.model_name = model_name
        self._session = None
        logger.info(f"RemBG service initialized with model: {model_name}")

    @property
    def session(self):
        """Lazy-load the rembg session"""
        if self._session is None:
            logger.info(f"Loading rembg model: {self.model_name}")
            _, new_session = _get_rembg()
            self._session = new_session(self.model_name)
        return self._session

    def process_single(
        self,
        input_path: Path,
        output_path: Path,
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
    ) -> Path:
        """
        Remove background from a single image.

        Args:
            input_path: Path to input image
            output_path: Path for output image (will be PNG)
            alpha_matting: Enable alpha matting for better edge quality
            alpha_matting_foreground_threshold: Foreground threshold (0-255)
            alpha_matting_background_threshold: Background threshold (0-255)

        Returns:
            Path to output image
        """
        logger.debug(f"Processing image: {input_path}")

        # Open and process image
        with Image.open(input_path) as img:
            # Convert to RGB if necessary (rembg expects RGB)
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

            # Remove background
            remove, _ = _get_rembg()
            output = remove(
                img,
                session=self.session,
                alpha_matting=alpha_matting,
                alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                alpha_matting_background_threshold=alpha_matting_background_threshold,
            )

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Save as PNG (to preserve transparency)
            output.save(output_path, "PNG")

        logger.debug(f"Saved output: {output_path}")
        return output_path

    def process_batch(
        self,
        input_paths: List[Path],
        output_dir: Path,
        alpha_matting: bool = False,
        alpha_matting_foreground_threshold: int = 240,
        alpha_matting_background_threshold: int = 10,
        progress_callback: Optional[callable] = None,
    ) -> List[Path]:
        """
        Process multiple images for background removal.

        Args:
            input_paths: List of input image paths
            output_dir: Directory for output images
            alpha_matting: Enable alpha matting
            alpha_matting_foreground_threshold: Foreground threshold
            alpha_matting_background_threshold: Background threshold
            progress_callback: Optional callback(current, total) for progress updates

        Returns:
            List of output paths
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        total = len(input_paths)

        for i, input_path in enumerate(input_paths):
            # Generate output filename (always PNG for transparency)
            output_name = f"{input_path.stem}_nobg.png"
            output_path = output_dir / output_name

            try:
                self.process_single(
                    input_path=input_path,
                    output_path=output_path,
                    alpha_matting=alpha_matting,
                    alpha_matting_foreground_threshold=alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold=alpha_matting_background_threshold,
                )
                output_paths.append(output_path)

                if progress_callback:
                    progress_callback(i + 1, total)

            except Exception as e:
                logger.error(f"Failed to process {input_path}: {e}")
                # Continue processing other images
                continue

        logger.info(f"Processed {len(output_paths)}/{total} images successfully")
        return output_paths

    def cleanup(self):
        """Release model resources"""
        if self._session is not None:
            self._session = None
            logger.info("RemBG session released")


# Global instance (lazy-loaded)
_rembg_service: Optional[RemBGService] = None


def get_rembg_service(model_name: str = "u2net") -> RemBGService:
    """Get or create RemBG service instance"""
    global _rembg_service
    if _rembg_service is None or _rembg_service.model_name != model_name:
        _rembg_service = RemBGService(model_name=model_name)
    return _rembg_service
