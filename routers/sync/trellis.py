"""
Synchronous TRELLIS Router
Returns GLB file directly without job queuing
"""

import logging
import tempfile
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import Response

from api.services.trellis_service import get_trellis_service
from api.models.enums import TrellisBackend

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trellis", tags=["TRELLIS (Sync)"])


@router.post(
    "/",
    summary="Convert image to 3D model",
    description="Synchronously convert image(s) to a 3D GLB model. This may take 1-5 minutes.",
    responses={
        200: {
            "description": "3D model in GLB format",
            "content": {
                "model/gltf-binary": {"schema": {"type": "string", "format": "binary"}},
            },
        },
        400: {"description": "Invalid request"},
        500: {"description": "Processing error"},
    },
)
async def image_to_3d(
    files: List[UploadFile] = File(..., description="Image file(s) to convert"),
    seed: int = Query(0, ge=0, description="Random seed for reproducibility"),
    backend: str = Query(
        "huggingface",
        description="Backend to use: huggingface (default), runpod, modal",
    ),
) -> Response:
    """
    Convert image(s) to 3D model synchronously.

    **Note:** This operation takes 1-5 minutes depending on the backend.

    - Single image: Standard image-to-3D conversion
    - Multiple images (2-4): Multi-view reconstruction for better quality

    Returns a GLB (GL Transmission Format Binary) file.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 images for multi-view")

    # Validate backend
    try:
        backend_enum = TrellisBackend(backend)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid backend: {backend}. Options: huggingface, runpod, modal",
        )

    # Validate file types
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    for f in files:
        if f.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {f.content_type}. Allowed: {allowed_types}",
            )

    service = get_trellis_service()

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_dir = temp_path / "input"
            output_dir = temp_path / "output"
            input_dir.mkdir()
            output_dir.mkdir()

            # Save uploaded files
            input_paths = []
            for i, file in enumerate(files):
                ext = Path(file.filename).suffix or ".png"
                input_file = input_dir / f"input_{i}{ext}"
                content = await file.read()
                input_file.write_bytes(content)
                input_paths.append(input_file)

            output_path = output_dir / "model.glb"

            logger.info(
                f"Processing {len(input_paths)} image(s) with backend {backend}, seed={seed}"
            )

            # Process (this is the slow part - 1-5 minutes)
            result_path = service.process(
                image_paths=input_paths,
                output_path=output_path,
                backend=backend_enum,
                seed=seed,
            )

            if not result_path.exists():
                raise HTTPException(status_code=500, detail="Failed to generate 3D model")

            # Return GLB file
            glb_bytes = result_path.read_bytes()
            original_name = Path(files[0].filename).stem

            return Response(
                content=glb_bytes,
                media_type="model/gltf-binary",
                headers={
                    "Content-Disposition": f'attachment; filename="{original_name}.glb"'
                },
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing TRELLIS: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")


@router.get("/health")
def trellis_health():
    """Check TRELLIS backend health"""
    service = get_trellis_service()
    healthy = service.health_check()
    return {
        "status": "healthy" if healthy else "unhealthy",
        "backend": "huggingface",
    }
