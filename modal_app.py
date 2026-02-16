"""
TRELLIS API - Modal Deployment

Deploy with:
    modal deploy modal_app.py

Test locally:
    modal serve modal_app.py
"""

import modal

# Define the Modal app
app = modal.App("trellis-api")

# Create image with dependencies and copy local code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
    )
    .pip_install(
        "fastapi>=0.109.0",
        "uvicorn[standard]>=0.27.0",
        "python-multipart>=0.0.6",
        "pydantic>=2.0.0",
        "pydantic-settings>=2.0.0",
        "pillow>=9.0.0",
        "rembg[cpu]>=2.0.50",
        "onnxruntime>=1.16.0",
        "gradio_client>=0.10.0",
        "httpx>=0.25.0",
        "python-dotenv>=1.0.0",
    )
    .add_local_dir(".", "/root/api", ignore=[".venv", "__pycache__", ".git", ".env", ".env.example", ".env.tmp"])
)


@app.function(
    image=image,
    timeout=600,  # 10 minutes for long TRELLIS operations
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI application."""
    # Create a minimal FastAPI app directly here to avoid import delays
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import Response
    import tempfile
    from pathlib import Path

    fastapi = FastAPI(
        title="TRELLIS API (Modal)",
        version="2.0.0-modal",
    )

    fastapi.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @fastapi.get("/")
    def root():
        return {
            "name": "TRELLIS API (Modal)",
            "version": "2.0.0-modal",
            "endpoints": {
                "health": "/health",
                "rembg": "/api/v1/rembg/",
            }
        }

    @fastapi.get("/health")
    def health():
        return {"status": "healthy", "version": "2.0.0-modal"}

    @fastapi.post("/api/v1/rembg/")
    async def remove_background(files: list[UploadFile] = File(...)):
        """Remove background from images"""
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")

        # Lazy import rembg only when needed
        from rembg import remove, new_session
        from PIL import Image
        import io

        # Process first file
        file = files[0]
        content = await file.read()

        # Process image
        img = Image.open(io.BytesIO(content))
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        session = new_session("u2net")
        output = remove(img, session=session)

        # Convert to bytes
        output_buffer = io.BytesIO()
        output.save(output_buffer, format="PNG")
        output_buffer.seek(0)

        return Response(
            content=output_buffer.getvalue(),
            media_type="image/png",
            headers={"Content-Disposition": f'attachment; filename="{Path(file.filename).stem}_nobg.png"'}
        )

    return fastapi


# For local testing with `modal serve`
if __name__ == "__main__":
    print("Run with: modal serve modal_app.py")
    print("Or deploy with: modal deploy modal_app.py")
