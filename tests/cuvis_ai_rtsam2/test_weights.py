"""The plugin's weight declarations, their registration, and the variant-table drift guards.

No network and no checkpoint: the declarations are data and the registry is in-process.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

from cuvis_ai_core.data.model_weights import ModelWeights
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.plugin import PluginWeightEntry

import cuvis_ai_rtsam2
import cuvis_ai_rtsam2.node as node_pkg
import cuvis_ai_rtsam2.weights as weights_mod
from cuvis_ai_rtsam2.node._rtsam2_tracker_base import _MODEL_ALIASES, _MODEL_VARIANT_REGISTRY
from cuvis_ai_rtsam2.weights import PLUGIN_NAME, WEIGHTS

EFFICIENTTAM_VARIANTS = [
    "efficienttam_s",
    "efficienttam_ti",
    "efficienttam_s_512x512",
    "efficienttam_ti_512x512",
]


def _node_classes() -> list[type[Node]]:
    return [
        cls
        for cls in vars(node_pkg).values()
        if isinstance(cls, type) and issubclass(cls, Node) and cls is not Node
    ]


def _constructor_params(cls: type) -> set[str]:
    return set(inspect.signature(cls.__init__).parameters) - {"self", "args", "kwargs"}


def test_declares_the_four_efficienttam_variants_with_full_pins() -> None:
    assert [entry.name for entry in WEIGHTS] == EFFICIENTTAM_VARIANTS
    for entry in WEIGHTS:
        assert isinstance(entry, PluginWeightEntry)
        assert entry.repo_id == "cubert-gmbh/efficient-track-anything"
        assert entry.filename == f"{entry.name}.pt"
        assert len(entry.revision) == 40 and len(entry.sha256) == 64
        assert entry.size_bytes > 50_000_000
        assert entry.selected_by == "model_type"
        assert entry.explicit_path_hparams == ["model_dir"]
        assert entry.license == "Apache-2.0" and entry.license_file == "LICENSE"
        assert entry.used_for and entry.summary and entry.description
    assert len({entry.revision for entry in WEIGHTS}) == 1, "one mirror commit for all four"
    assert len({entry.sha256 for entry in WEIGHTS}) == 4


def test_register_called_at_import() -> None:
    assert PLUGIN_NAME == "rtsam2"
    for entry in WEIGHTS:
        row = ModelWeights.get(entry.name)
        assert row.plugin == PLUGIN_NAME
        assert row.source == "plugin"
        assert row.entry == entry
    assert cuvis_ai_rtsam2.WEIGHTS is WEIGHTS


def test_every_cache_backed_variant_is_declared() -> None:
    """A variant that resolves from the shared cache must have a declaration, and vice versa."""
    declared = {entry.name for entry in WEIGHTS}
    cache_backed = {
        key: spec.weights_name
        for key, spec in _MODEL_VARIANT_REGISTRY.items()
        if spec.weights_name is not None
    }
    assert cache_backed == {name: name for name in declared}


def test_sam21_variants_have_no_registry_row() -> None:
    """The SAM 2.1 variants are explicit-path only (``model_dir`` or the vendored checkpoints)."""
    sam21 = [key for key in _MODEL_VARIANT_REGISTRY if key.startswith("sam2.1_")]
    assert len(sam21) == 4
    known = {entry.name for entry in WEIGHTS} | {a for entry in WEIGHTS for a in entry.aliases}
    for key in sam21:
        assert _MODEL_VARIANT_REGISTRY[key].weights_name is None
        assert key not in known
    assert "sam2" not in known, "the sam2 alias stays plugin-internal"


def test_alias_efficienttam_is_the_default_row() -> None:
    (default,) = [entry for entry in WEIGHTS if entry.default]
    assert default.name == _MODEL_ALIASES["efficienttam"] == "efficienttam_s"
    assert default.aliases == ["efficienttam"]
    assert ModelWeights.get("efficienttam").entry == default
    assert all(not entry.aliases for entry in WEIGHTS if entry is not default)


def test_selector_and_explicit_path_are_node_constructor_params() -> None:
    """What emit_metadata validates: every declared hparam exists on at least one node."""
    classes = _node_classes()
    assert classes
    for entry in WEIGHTS:
        for hparam in [*entry.explicit_path_hparams, entry.selected_by]:
            assert any(hparam in _constructor_params(cls) for cls in classes), hparam


def test_weights_module_is_side_effect_free() -> None:
    """The module declares only: loading it alone must not import torch, core or the plugin."""
    path = Path(weights_mod.__file__)
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('weights_probe', {str(path)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert len(mod.WEIGHTS) == 4\n"
        "for heavy in ('torch', 'cuvis_ai_core', 'sam2', 'efficient_track_anything',"
        " 'cuvis_ai_rtsam2'):\n"
        "    assert heavy not in sys.modules, heavy\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stderr
