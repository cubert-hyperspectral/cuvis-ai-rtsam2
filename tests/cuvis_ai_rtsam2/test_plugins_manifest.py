"""Manifest guard: the in-repo plugins.yaml must resolve exactly the concrete nodes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from cuvis_ai_core.utils.node_registry import NodeRegistry

from cuvis_ai_rtsam2.node import (
    RTSAM2BboxPropagation,
    RTSAM2MaskPropagation,
    RTSAM2PointExpansion,
    rtsam2_point_expansion,
    rtsam2_streaming_propagation,
)

pytestmark = pytest.mark.unit

_MANIFEST = Path(__file__).resolve().parents[2] / "cuvis_ai_rtsam2" / "plugins.yaml"


def test_manifest_registers_and_resolves_the_concrete_nodes() -> None:
    registry = NodeRegistry()
    registry.register_plugin(_MANIFEST)

    assert registry.list_plugins() == ["rtsam2"]
    assert registry.get("RTSAM2BboxPropagation") is RTSAM2BboxPropagation
    assert registry.get("RTSAM2MaskPropagation") is RTSAM2MaskPropagation
    assert registry.get("RTSAM2PointExpansion") is RTSAM2PointExpansion


def test_manifest_capabilities_match_package_exports() -> None:
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    declared = {entry["class_name"].rsplit(".", 1)[-1] for entry in manifest["capabilities"]}

    # Filter each module's __all__ by defining module: the abstract base is
    # re-exported in the streaming module's __all__ but must not be demanded
    # from the manifest.
    concrete = {
        name
        for module in (rtsam2_streaming_propagation, rtsam2_point_expansion)
        for name in module.__all__
        if getattr(module, name).__module__ == module.__name__
    }

    assert declared == concrete
