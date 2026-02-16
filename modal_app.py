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
    allow_concurrent_inputs=10,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the FastAPI application."""
    import sys
    sys.path.insert(0, "/root")

    from api.main_sync import app
    return app


# For local testing with `modal serve`
if __name__ == "__main__":
    print("Run with: modal serve modal_app.py")
    print("Or deploy with: modal deploy modal_app.py")
