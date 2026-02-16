"""API Routers package"""

from api.routers.rembg import router as rembg_router
from api.routers.trellis import router as trellis_router
from api.routers.jobs import router as jobs_router
from api.routers.health import router as health_router

__all__ = ["rembg_router", "trellis_router", "jobs_router", "health_router"]
