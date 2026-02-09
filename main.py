"""
TRELLIS API Server
FastAPI server for image to 3D conversion
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
import uuid
import shutil
from typing import Optional
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="TRELLIS API",
    description="Convert images to 3D GLB files using TRELLIS",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration
UPLOAD_DIR = Path("./uploads")
OUTPUT_DIR = Path("./outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# In-memory job storage (use Redis/DB for production)
jobs = {}


class ConversionRequest(BaseModel):
    seed: Optional[int] = 1
    texture_size: Optional[int] = 2048
    optimize: Optional[bool] = True


class JobStatus(BaseModel):
    job_id: str
    status: str  # 'pending', 'processing', 'completed', 'failed'
    message: Optional[str] = None
    download_url: Optional[str] = None
    error: Optional[str] = None


def cleanup_file(path: Path):
    """Delete file after some time"""
    try:
        if path.exists():
            path.unlink()
            logger.info(f"Cleaned up: {path}")
    except Exception as e:
        logger.error(f"Cleanup failed for {path}: {e}")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "service": "TRELLIS API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "convert": "/api/convert",
            "status": "/api/status/{job_id}",
            "download": "/api/download/{job_id}"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {"status": "healthy"}


@app.post("/api/convert", response_model=JobStatus)
async def convert_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    seed: int = 1
):
    """
    Convert an image to 3D GLB format

    - **file**: Image file (JPG, PNG, WebP)
    - **seed**: Random seed for reproducibility (default: 1)

    Returns job ID for tracking progress
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    input_path = UPLOAD_DIR / f"{job_id}{Path(file.filename).suffix}"
    output_path = OUTPUT_DIR / f"{job_id}.glb"

    try:
        with input_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        logger.info(f"Job {job_id}: Image uploaded - {file.filename}")

        # Initialize job status
        jobs[job_id] = {
            "status": "pending",
            "input_path": str(input_path),
            "output_path": str(output_path),
            "seed": seed
        }

        # Start processing in background
        background_tasks.add_task(process_image, job_id, input_path, output_path, seed)

        # Schedule cleanup (delete files after 1 hour)
        background_tasks.add_task(cleanup_file, input_path)

        return JobStatus(
            job_id=job_id,
            status="pending",
            message="Image uploaded. Processing will begin shortly."
        )

    except Exception as e:
        logger.error(f"Job {job_id}: Upload failed - {e}")
        if input_path.exists():
            input_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/status/{job_id}", response_model=JobStatus)
async def get_status(job_id: str):
    """
    Check the status of a conversion job

    - **job_id**: Job ID from /api/convert
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    response = JobStatus(
        job_id=job_id,
        status=job["status"],
        message=job.get("message")
    )

    if job["status"] == "completed":
        response.download_url = f"/api/download/{job_id}"
    elif job["status"] == "failed":
        response.error = job.get("error")

    return response


@app.get("/api/download/{job_id}")
async def download_result(job_id: str, background_tasks: BackgroundTasks):
    """
    Download the generated GLB file

    - **job_id**: Job ID from /api/convert
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is {job['status']}, not ready for download"
        )

    output_path = Path(job["output_path"])

    if not output_path.exists():
        raise HTTPException(status_code=404, detail="Output file not found")

    # Schedule cleanup after download
    background_tasks.add_task(cleanup_file, output_path)

    return FileResponse(
        path=output_path,
        media_type="model/gltf-binary",
        filename=f"model_{job_id}.glb"
    )


@app.delete("/api/jobs/{job_id}")
async def cancel_job(job_id: str):
    """
    Cancel a job and clean up files

    - **job_id**: Job ID to cancel
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    # Clean up files
    for path_key in ["input_path", "output_path"]:
        if path_key in job:
            path = Path(job[path_key])
            if path.exists():
                path.unlink()

    # Remove from jobs
    del jobs[job_id]

    return {"message": f"Job {job_id} cancelled and cleaned up"}


# ============================================================================
# PROCESSING FUNCTION - Choose implementation below
# ============================================================================

def process_image(job_id: str, input_path: Path, output_path: Path, seed: int):
    """
    Process image to 3D. Choose implementation:
    - Option 1: Use Gradio client (proxy to HuggingFace)
    - Option 2: Use local TRELLIS installation (requires GPU)
    """
    jobs[job_id]["status"] = "processing"
    jobs[job_id]["message"] = "Converting image to 3D..."

    try:
        # OPTION 1: Proxy to HuggingFace (No GPU needed)
        from gradio_client import Client, handle_file

        logger.info(f"Job {job_id}: Connecting to HuggingFace Space...")
        client = Client("JeffreyXiang/TRELLIS")

        logger.info(f"Job {job_id}: Processing image...")
        result = client.predict(
            image=handle_file(str(input_path)),
            seed=seed,
            api_name="/image_to_3d"
        )

        # Get GLB path from result
        if isinstance(result, dict) and 'glb' in result:
            glb_path = result['glb']
        elif isinstance(result, str):
            glb_path = result
        elif isinstance(result, tuple):
            glb_path = result[0]
        else:
            raise ValueError(f"Unexpected result format: {type(result)}")

        # Copy result to output path
        shutil.copy(glb_path, output_path)

        # OPTION 2: Local TRELLIS (Requires GPU and TRELLIS installed)
        # Uncomment this block to use local TRELLIS:
        """
        import sys
        sys.path.insert(0, str(Path.home() / ".cache/trellis/repo"))

        from trellis.pipelines import TrellisImageTo3DPipeline
        from PIL import Image
        import torch

        logger.info(f"Job {job_id}: Loading TRELLIS model...")
        pipeline = TrellisImageTo3DPipeline.from_pretrained("JeffreyXiang/TRELLIS-image-large")
        pipeline = pipeline.to("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Job {job_id}: Processing image...")
        image = Image.open(input_path).convert("RGB")
        outputs = pipeline.run(image, seed=seed)

        logger.info(f"Job {job_id}: Exporting GLB...")
        # Export based on TRELLIS output structure
        if hasattr(outputs, 'mesh'):
            outputs.mesh.export(str(output_path))
        else:
            raise ValueError("Could not extract mesh from outputs")
        """

        # Mark as completed
        jobs[job_id]["status"] = "completed"
        jobs[job_id]["message"] = "Conversion completed successfully"
        logger.info(f"Job {job_id}: Completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id}: Failed - {e}")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
        jobs[job_id]["message"] = "Conversion failed"


if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
