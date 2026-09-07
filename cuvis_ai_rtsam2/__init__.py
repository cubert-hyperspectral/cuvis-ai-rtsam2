"""cuvis_ai_rtsam2: realtime SAM2 wrapper and cuvis.ai plugin package.

Importing the package registers the plugin's weight declarations
(:mod:`cuvis_ai_rtsam2.weights`) with cuvis-ai-core's model-weight registry, so the
tracker nodes' cache lookup and ``download-model`` in the same environment share the
mirror pins without a plugin manifest on disk.
"""

from cuvis_ai_core.data.model_weights import ModelWeights

from cuvis_ai_rtsam2.weights import PLUGIN_NAME, WEIGHTS
from efficient_track_anything.build_efficienttam import (  # noqa: F401
    build_efficienttam,
    build_efficienttam_camera_predictor,
)
from sam2.build_sam import build_sam2, build_sam2_camera_predictor  # noqa: F401

ModelWeights.register(PLUGIN_NAME, WEIGHTS)


def register_all_nodes() -> int:
    """Register all cuvis_ai_rtsam2 nodes in the cuvis.ai NodeRegistry."""
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    registry = NodeRegistry()
    return int(registry.auto_register_package("cuvis_ai_rtsam2.node"))


__all__ = [
    "PLUGIN_NAME",
    "WEIGHTS",
    "build_sam2",
    "build_sam2_camera_predictor",
    "build_efficienttam",
    "build_efficienttam_camera_predictor",
    "register_all_nodes",
]
