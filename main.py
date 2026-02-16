"""
TRELLIS API Server v2
FastAPI server for image processing and 3D generation with Celery queue
"""

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from api.config import settings
from api.routers import rembg_router, trellis_router, jobs_router, health_router
from api.middleware.rate_limit import limiter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info(f"Starting {settings.app_name} v{settings.app_version}")

    # Ensure directories exist
    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    yield

    # Shutdown
    logger.info("Shutting down...")


# Create FastAPI app
app = FastAPI(
    title=settings.app_name,
    description="""
## TRELLIS API

Image processing and 3D generation API with background removal and image-to-3D conversion.

### Features

- **RemBG**: Remove backgrounds from images using AI
- **TRELLIS**: Convert images to 3D GLB models
- **Multi-view**: Support for multi-image 3D reconstruction
- **Queue-based**: Async processing with job tracking

### Authentication

All endpoints require an API key via:
- `Authorization: Bearer <your-api-key>` header
- `X-API-Key: <your-api-key>` header

### Workflow

1. POST to `/api/v1/rembg` or `/api/v1/trellis` with images
2. Receive a `job_id`
3. Poll `GET /api/v1/jobs/{job_id}` for status
4. Download results from `download_urls` when complete
    """,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle uncaught exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred",
            "detail": str(exc) if settings.debug else None,
        },
    )


# Include routers
app.include_router(health_router)
app.include_router(rembg_router, prefix="/api/v1")
app.include_router(trellis_router, prefix="/api/v1")
app.include_router(jobs_router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=settings.debug,
    )
