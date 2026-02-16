"""
API Configuration using Pydantic Settings
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
from functools import lru_cache
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""

    # API Settings
    app_name: str = "TRELLIS API"
    app_version: str = "2.0.0"
    debug: bool = False

    # Authentication
    api_keys: str = Field(default="", description="Comma-separated list of valid API keys")

    # Redis Configuration
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # TRELLIS Configuration
    trellis_backend: str = Field(
        default="huggingface",
        description="Backend to use: 'huggingface' (V1) or 'runpod' (V2)"
    )
    huggingface_space: str = "JeffreyXiang/TRELLIS"

    # RunPod Configuration (V2)
    runpod_endpoint: Optional[str] = None
    runpod_api_key: Optional[str] = None

    # Storage Configuration
    upload_dir: Path = Path("./uploads")
    output_dir: Path = Path("./outputs")
    max_file_size_mb: int = 10
    max_files_per_request: int = 10
    cleanup_after_hours: int = 24

    # Rate Limiting
    rate_limit_default: str = "60/minute"
    rate_limit_rembg: str = "30/minute"
    rate_limit_trellis: str = "10/minute"

    # Celery Task Settings
    rembg_task_timeout: int = 300  # 5 minutes
    trellis_task_timeout: int = 600  # 10 minutes
    rembg_max_retries: int = 3
    trellis_max_retries: int = 2

    # CORS Settings
    cors_origins: str = "*"
    cors_allow_credentials: bool = True

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False

    @property
    def api_keys_list(self) -> List[str]:
        """Parse comma-separated API keys into a list"""
        if not self.api_keys:
            return []
        return [k.strip() for k in self.api_keys.split(",") if k.strip()]

    @property
    def max_file_size_bytes(self) -> int:
        """Convert MB to bytes"""
        return self.max_file_size_mb * 1024 * 1024

    @property
    def cors_origins_list(self) -> List[str]:
        """Parse CORS origins"""
        if self.cors_origins == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Convenience alias
settings = get_settings()
