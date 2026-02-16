"""
Enum definitions for the API
"""

from enum import Enum


class JobStatus(str, Enum):
    """Status of a processing job"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobType(str, Enum):
    """Type of processing job"""
    REMBG = "rembg"
    TRELLIS = "trellis"


class TrellisBackend(str, Enum):
    """Backend for TRELLIS processing"""
    HUGGINGFACE = "huggingface"  # V1: HuggingFace Gradio client
    RUNPOD = "runpod"            # V2: Self-hosted on RunPod
    MODAL = "modal"              # V2 alternative: Modal.com
