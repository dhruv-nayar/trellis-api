"""
TRELLIS API - Synchronous Version
Single-service deployment without Redis/Celery

This version processes requests synchronously and returns results directly.
Ideal for simple deployments on Render, Fly.io, or other single-container platforms.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler"""
    logger.info("Starting TRELLIS API v2.0.0-sync")
    yield
    logger.info("Shutting down TRELLIS API")


app = FastAPI(
    title="TRELLIS API",
    description="""
## Synchronous Image Processing API

This API provides synchronous endpoints for:
- **Background Removal** (RemBG): Remove backgrounds from images
- **Image to 3D** (TRELLIS): Convert images to 3D GLB models

### Key Differences from Async Version
- No job polling required - results returned directly
- Simpler architecture - single service, no Redis/Celery
- Longer request timeouts (5-10 minutes for TRELLIS)

### Usage
1. POST your image(s) to the endpoint
2. Wait for processing (may take 1-5 minutes for TRELLIS)
3. Receive the processed file directly in the response
    """,
    version="2.0.0-sync",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Import and include routers
from api.routers.sync import rembg_router, trellis_router

app.include_router(rembg_router, prefix="/api/v1")
app.include_router(trellis_router, prefix="/api/v1")


@app.get("/", tags=["Root"])
def root():
    """API root - returns available endpoints"""
    return {
        "name": "TRELLIS API (Sync)",
        "version": "2.0.0-sync",
        "mode": "synchronous",
        "description": "Single-service deployment without Redis/Celery",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
            "rembg": "/api/v1/rembg",
            "trellis": "/api/v1/trellis",
        },
    }


@app.get("/health", tags=["Health"])
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0-sync",
        "mode": "synchronous",
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.exception(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": str(exc),
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main_sync:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        timeout_keep_alive=600,  # 10 minutes for long TRELLIS requests
    )
