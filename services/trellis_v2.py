"""
TRELLIS V2 Service
Self-hosted TRELLIS on RunPod/Modal
"""

import asyncio
import base64
import logging
from pathlib import Path
from typing import List, Optional
import httpx

from api.config import settings

logger = logging.getLogger(__name__)


class TrellisV2Client:
    """
    Self-hosted TRELLIS client for RunPod/Modal deployment.

    This client connects to a self-hosted TRELLIS endpoint for
    image-to-3D conversion with full control over the model.
    """

    def __init__(
        self,
        endpoint_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 600.0,  # 10 minutes
    ):
        """
        Initialize TRELLIS V2 client.

        Args:
            endpoint_url: RunPod/Modal endpoint URL
            api_key: API key for the endpoint
            timeout: Request timeout in seconds
        """
        self.endpoint_url = endpoint_url or settings.runpod_endpoint
        self.api_key = api_key or settings.runpod_api_key
        self.timeout = httpx.Timeout(timeout)

        if not self.endpoint_url:
            raise ValueError("RunPod endpoint URL is required")
        if not self.api_key:
            raise ValueError("RunPod API key is required")

        logger.info(f"TrellisV2Client initialized for endpoint: {self.endpoint_url}")

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def _decode_glb(self, glb_data: str, output_path: Path) -> Path:
        """Decode base64 GLB data and save to file"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(base64.b64decode(glb_data))
        return output_path

    def process(
        self,
        image_paths: List[Path],
        output_path: Path,
        seed: int = 1,
        texture_size: int = 2048,
        optimize: bool = True,
    ) -> Path:
        """
        Process images via self-hosted TRELLIS endpoint.

        This is a synchronous wrapper around the async implementation
        for compatibility with Celery tasks.

        Args:
            image_paths: List of input image paths
            output_path: Path for output GLB file
            seed: Random seed for reproducibility
            texture_size: Texture resolution
            optimize: Whether to optimize mesh

        Returns:
            Path to output GLB file
        """
        # Run async method in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self._process_async(
                    image_paths=image_paths,
                    output_path=output_path,
                    seed=seed,
                    texture_size=texture_size,
                    optimize=optimize,
                )
            )
        finally:
            loop.close()

    async def _process_async(
        self,
        image_paths: List[Path],
        output_path: Path,
        seed: int = 1,
        texture_size: int = 2048,
        optimize: bool = True,
    ) -> Path:
        """
        Async implementation of image processing.

        Args:
            image_paths: List of input image paths
            output_path: Path for output GLB file
            seed: Random seed for reproducibility
            texture_size: Texture resolution
            optimize: Whether to optimize mesh

        Returns:
            Path to output GLB file
        """
        logger.info(f"Processing {len(image_paths)} images via RunPod")

        # Encode images to base64
        images_b64 = [self._encode_image(p) for p in image_paths]

        # Build request payload
        payload = {
            "input": {
                "images": images_b64,
                "seed": seed,
                "texture_size": texture_size,
                "optimize": optimize,
            }
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            # Submit job to RunPod
            logger.debug(f"Submitting job to {self.endpoint_url}/run")
            response = await client.post(
                f"{self.endpoint_url}/run",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            result = response.json()

            # Get job ID
            job_id = result.get("id")
            if not job_id:
                raise ValueError(f"No job ID in response: {result}")

            logger.info(f"RunPod job submitted: {job_id}")

            # Poll for completion
            glb_data = await self._poll_result(client, headers, job_id)

        # Save GLB file
        self._decode_glb(glb_data, output_path)

        logger.info(f"Saved GLB to: {output_path}")
        return output_path

    async def _poll_result(
        self,
        client: httpx.AsyncClient,
        headers: dict,
        job_id: str,
        poll_interval: float = 5.0,
    ) -> str:
        """
        Poll RunPod for job completion.

        Args:
            client: HTTP client
            headers: Request headers
            job_id: RunPod job ID
            poll_interval: Seconds between polls

        Returns:
            Base64-encoded GLB data
        """
        status_url = f"{self.endpoint_url}/status/{job_id}"

        while True:
            response = await client.get(status_url, headers=headers)
            response.raise_for_status()
            result = response.json()

            status = result.get("status")
            logger.debug(f"Job {job_id} status: {status}")

            if status == "COMPLETED":
                output = result.get("output", {})
                glb_data = output.get("glb")
                if not glb_data:
                    raise ValueError(f"No GLB data in completed job: {output.keys()}")
                return glb_data

            elif status == "FAILED":
                error = result.get("error", "Unknown error")
                raise Exception(f"RunPod job failed: {error}")

            elif status == "CANCELLED":
                raise Exception("RunPod job was cancelled")

            elif status in ("IN_QUEUE", "IN_PROGRESS"):
                await asyncio.sleep(poll_interval)

            else:
                logger.warning(f"Unknown status: {status}")
                await asyncio.sleep(poll_interval)

    def health_check(self) -> bool:
        """Check if the RunPod endpoint is accessible"""
        try:
            with httpx.Client(timeout=httpx.Timeout(10.0)) as client:
                response = client.get(
                    f"{self.endpoint_url}/health",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                )
                return response.status_code == 200
        except Exception as e:
            logger.error(f"TRELLIS V2 health check failed: {e}")
            return False

    def cleanup(self):
        """Release client resources (no-op for HTTP client)"""
        logger.info("TrellisV2Client resources released")


# Global instance (lazy-loaded)
_trellis_v2_client: Optional[TrellisV2Client] = None


def get_trellis_v2_client() -> TrellisV2Client:
    """Get or create TrellisV2Client instance"""
    global _trellis_v2_client
    if _trellis_v2_client is None:
        _trellis_v2_client = TrellisV2Client()
    return _trellis_v2_client
