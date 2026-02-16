"""
Request models for API endpoints
"""

from pydantic import BaseModel, Field
from typing import Optional
from api.models.enums import TrellisBackend


class TrellisRequest(BaseModel):
    """Request parameters for TRELLIS endpoint (passed as form fields)"""
    seed: int = Field(default=1, ge=0, description="Random seed for reproducibility")
    texture_size: int = Field(
        default=2048,
        ge=512,
        le=4096,
        description="Texture resolution (512, 1024, 2048, or 4096)"
    )
    optimize: bool = Field(
        default=True,
        description="Whether to optimize/simplify the mesh"
    )
    backend: TrellisBackend = Field(
        default=TrellisBackend.HUGGINGFACE,
        description="Backend to use for processing"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "seed": 42,
                "texture_size": 2048,
                "optimize": True,
                "backend": "huggingface"
            }
        }


class RemBGRequest(BaseModel):
    """Request parameters for RemBG endpoint (passed as form fields)"""
    model: str = Field(
        default="u2net",
        description="RemBG model to use (u2net, u2netp, u2net_human_seg, etc.)"
    )
    alpha_matting: bool = Field(
        default=False,
        description="Enable alpha matting for better edges"
    )
    alpha_matting_foreground_threshold: int = Field(
        default=240,
        ge=0,
        le=255,
        description="Foreground threshold for alpha matting"
    )
    alpha_matting_background_threshold: int = Field(
        default=10,
        ge=0,
        le=255,
        description="Background threshold for alpha matting"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "model": "u2net",
                "alpha_matting": False
            }
        }
