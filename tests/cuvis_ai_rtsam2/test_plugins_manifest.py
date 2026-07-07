"""Manifest guard: the in-repo plugins.yaml must resolve exactly the concrete nodes."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from cuvis_ai_core.utils.node_registry import NodeRegistry

from cuvis_ai_rtsam2.node import (
    RTSAM2BboxPropagation,
    RTSAM2MaskPropagation,
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


def test_manifest_capabilities_match_package_exports() -> None:
    manifest = yaml.safe_load(_MANIFEST.read_text(encoding="utf-8"))
    declared = {entry["class_name"].rsplit(".", 1)[-1] for entry in manifest["capabilities"]}

    concrete = {
        name
        for name in rtsam2_streaming_propagation.__all__
        if getattr(rtsam2_streaming_propagation, name).__module__
        == rtsam2_streaming_propagation.__name__
    }

    assert declared == concrete
