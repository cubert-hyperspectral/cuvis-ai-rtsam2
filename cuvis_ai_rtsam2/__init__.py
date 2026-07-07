"""cuvis_ai_rtsam2: realtime SAM2 wrapper and cuvis.ai plugin package."""

from efficient_track_anything.build_efficienttam import (  # noqa: F401
    build_efficienttam,
    build_efficienttam_camera_predictor,
)
from sam2.build_sam import build_sam2, build_sam2_camera_predictor  # noqa: F401


def register_all_nodes() -> int:
    """Register all cuvis_ai_rtsam2 nodes in the cuvis.ai NodeRegistry."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    registry = NodeRegistry()
    return int(registry.auto_register_package("cuvis_ai_rtsam2.node"))


__all__ = [
    "build_sam2",
    "build_sam2_camera_predictor",
    "build_efficienttam",
    "build_efficienttam_camera_predictor",
    "register_all_nodes",
]
