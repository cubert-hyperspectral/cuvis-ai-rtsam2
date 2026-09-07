"""Weight declarations of the rtsam2 plugin.

Side-effect free on purpose: this module only declares. ``cuvis_ai_rtsam2/__init__``
registers the tuple with cuvis-ai-core's ``ModelWeights`` at import, and cuvis-ai's
``emit_metadata`` projects it into the plugin manifest's ``weights:`` block, so
CuvisNEXT and the installer know what to provision without importing the plugin.

The four EfficientTAM variants are the rows a pipeline's ``model_type`` picks
(``efficienttam`` is the alias of the default, ``efficienttam_s``). The SAM 2.1
variants have no mirror and no row: they resolve from an explicit ``model_dir`` or
the vendored ``checkpoints/`` folder only. The pins come from
``tools/mirror_weights.py`` in cuvis-ai-core (the mirror is
``cubert-gmbh/efficient-track-anything``, a byte-identical copy of
``yunyangx/efficient-track-anything`` at 9bdd8ab5).
"""

from __future__ import annotations

from cuvis_ai_schemas.plugin import PluginWeightEntry

PLUGIN_NAME = "rtsam2"
"""The manifest name of this plugin (what pipelines list under ``plugins:``)."""

_REPO = "cubert-gmbh/efficient-track-anything"
_REVISION = "3dfd0228d7774b94c24116cf729e03c209ff448a"


def _variant(
    name: str,
    display_name: str,
    summary: str,
    *,
    sha256: str,
    size_bytes: int,
    default: bool = False,
    aliases: tuple[str, ...] = (),
    description: str,
) -> PluginWeightEntry:
    return PluginWeightEntry(
        name=name,
        display_name=display_name,
        summary=summary,
        used_for=["Point expansion", "Propagation"],
        repo_id=_REPO,
        filename=f"{name}.pt",
        revision=_REVISION,
        sha256=sha256,
        size_bytes=size_bytes,
        license="Apache-2.0",
        license_file="LICENSE",
        aliases=list(aliases),
        selected_by="model_type",
        default=default,
        explicit_path_hparams=["model_dir"],
        description=description,
    )


WEIGHTS: tuple[PluginWeightEntry, ...] = (
    _variant(
        "efficienttam_s",
        "RTSAM (EfficientTAM small)",
        "Point expansion, propagation; the default RTSAM variant",
        sha256="2b572be30d9e96ee29c8d785fe157c6b079ede7d56fbc8a3671d4120e63c89cd",
        size_bytes=136_375_868,
        default=True,
        aliases=("efficienttam",),
        description=(
            "EfficientTAM small checkpoint, the variant an rtsam2 pipeline gets when it "
            "sets no model_type (or model_type: efficienttam)."
        ),
    ),
    _variant(
        "efficienttam_ti",
        "RTSAM (EfficientTAM tiny)",
        "Point expansion, propagation; smaller and faster",
        sha256="acbb17b28cca1f860acee09c9ecb6efdb732080dc7a85a07292c31813175fa7d",
        size_bytes=71_616_316,
        description="EfficientTAM tiny checkpoint (model_type: efficienttam_ti).",
    ),
    _variant(
        "efficienttam_s_512x512",
        "RTSAM (EfficientTAM small, 512 px)",
        "Point expansion, propagation at 512 x 512 input",
        sha256="67b5840012737ed2c94a4cb8787c5c1b27b3a946045d2e602e65ef77230b6085",
        size_bytes=136_379_145,
        description=(
            "EfficientTAM small checkpoint for 512 x 512 input (model_type: "
            "efficienttam_s_512x512)."
        ),
    ),
    _variant(
        "efficienttam_ti_512x512",
        "RTSAM (EfficientTAM tiny, 512 px)",
        "Point expansion, propagation at 512 x 512 input, smallest",
        sha256="7d4d652a465f0081391050932f45d8a66768ccc99c8ea393ce8a5927e83f3b9b",
        size_bytes=71_620_052,
        description=(
            "EfficientTAM tiny checkpoint for 512 x 512 input (model_type: "
            "efficienttam_ti_512x512)."
        ),
    ),
)
"""Every weight the rtsam2 nodes load from the shared cache, keyed by ``model_type``."""
