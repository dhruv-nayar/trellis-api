"""
Synchronous RemBG Router
Returns processed images directly without job queuing
"""

import logging
import tempfile
import zipfile
import io
from pathlib import Path
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import Response

from api.services.rembg_service import get_rembg_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/rembg", tags=["RemBG (Sync)"])


@router.post(
    "/",
    summary="Remove background from images",
    description="Synchronously remove backgrounds from images. Returns PNG directly for single image, or ZIP for multiple images.",
    responses={
        200: {
            "description": "Processed image(s)",
            "content": {
                "image/png": {"schema": {"type": "string", "format": "binary"}},
                "application/zip": {"schema": {"type": "string", "format": "binary"}},
            },
        },
        400: {"description": "Invalid request"},
        500: {"description": "Processing error"},
    },
)
async def remove_background(
    files: List[UploadFile] = File(..., description="Image files to process"),
    model: str = Query(
        "u2net",
        description="Model to use: u2net, u2netp, u2net_human_seg, isnet-general-use, isnet-anime",
    ),
    alpha_matting: bool = Query(False, description="Enable alpha matting for better edges"),
) -> Response:
    """
    Remove background from uploaded images synchronously.

    - Single image: Returns PNG directly
    - Multiple images: Returns ZIP archive containing all processed images
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 files per request")

    # Validate file types
    allowed_types = {"image/jpeg", "image/png", "image/webp", "image/jpg"}
    for f in files:
        if f.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {f.content_type}. Allowed: {allowed_types}",
            )

    service = get_rembg_service(model_name=model)

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

            logger.info(f"Processing {len(input_paths)} image(s) with model {model}")

            # Process images
            output_paths = service.process_batch(
                input_paths=input_paths,
                output_dir=output_dir,
                alpha_matting=alpha_matting,
            )

            if not output_paths:
                raise HTTPException(status_code=500, detail="Failed to process images")

            # Return response
            if len(output_paths) == 1:
                # Single file: return PNG directly
                png_bytes = output_paths[0].read_bytes()
                original_name = Path(files[0].filename).stem
                return Response(
                    content=png_bytes,
                    media_type="image/png",
                    headers={
                        "Content-Disposition": f'attachment; filename="{original_name}_nobg.png"'
                    },
                )
            else:
                # Multiple files: return ZIP
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
                    for output_path in output_paths:
                        zf.write(output_path, output_path.name)

                zip_buffer.seek(0)
                return Response(
                    content=zip_buffer.getvalue(),
                    media_type="application/zip",
                    headers={
                        "Content-Disposition": 'attachment; filename="results.zip"'
                    },
                )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error processing images: {e}")
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
