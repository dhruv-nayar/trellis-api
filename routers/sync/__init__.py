"""
Sync routers for single-service deployment
"""

from api.routers.sync.rembg import router as rembg_router
from api.routers.sync.trellis import router as trellis_router

__all__ = ["rembg_router", "trellis_router"]
