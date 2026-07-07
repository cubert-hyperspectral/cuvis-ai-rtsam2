![image](https://raw.githubusercontent.com/cubert-hyperspectral/cuvis.sdk/main/branding/logo/banner.png)

# CUVIS.AI RTSAM2

This repository provides real-time SAM2 / EfficientTAM streaming tracking as a cuvis.ai plugin,
enabling prompt-seeded object tracking pipelines that process one frame per step. It is a fork of
[robrosinc/REALTIME_SAM2](https://github.com/robrosinc/REALTIME_SAM2) (which vendors Meta's SAM2
and EfficientTAM) and is maintained by Cubert GmbH as part of the cuvis.ai ecosystem.

## Platform

cuvis.ai is split across multiple repositories:

| Repository | Role |
|---|---|
| [cuvis-ai-core](https://github.com/cubert-hyperspectral/cuvis-ai-core) | Framework — base `Node` class, pipeline orchestration, services, and plugin system |
| [cuvis-ai-schemas](https://github.com/cubert-hyperspectral/cuvis-ai-schemas) | Shared schema definitions and generated types |
| [cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai) | Node catalog and end-user pipeline examples |
| **cuvis-ai-rtsam2** (this repo) | RTSAM2 plugin — cuvis.ai nodes for real-time prompt-seeded tracking |

## Nodes

| Node | Description |
|---|---|
| `RTSAM2BboxPropagation` | Bounding-box prompt seeding, then per-frame streaming tracking |
| `RTSAM2MaskPropagation` | Label-map (mask) prompt seeding, then per-frame streaming tracking |

Both nodes seed once per stream: prompts are applied on the first prompt frame, subsequent
frames are tracked. `reset()` (driven automatically by `Predictor` between runs) starts a
fresh stream; it also releases the loaded predictor, which is rebuilt on the next prompt
frame. Mid-stream re-prompting raises.

Supported `model_type` values: `efficienttam` (alias for `efficienttam_s`),
`efficienttam_s`, `efficienttam_s_512x512`, `efficienttam_ti`, `efficienttam_ti_512x512`,
`sam2` (alias for `sam2.1_hiera_t`), `sam2.1_hiera_t`, `sam2.1_hiera_s`, `sam2.1_hiera_b+`,
`sam2.1_hiera_l`.

## Checkpoints

Model weights are not downloaded at runtime. Place them under `checkpoints/` (or pass an
absolute `model_dir` hparam — recommended for installed, non-editable deployments):

- EfficientTAM: `checkpoints/download_checkpoints.sh` (Hugging Face)
- SAM2.1: <https://github.com/facebookresearch/sam2#download-checkpoints>

## Quick Start

For local development in this repository:

```bash
git clone https://github.com/cubert-hyperspectral/cuvis-ai-rtsam2.git
cd cuvis-ai-rtsam2
uv sync --extra dev
```

For cuvis.ai usage, see the RTSAM2 mask-propagation pipelines in
[cuvis-ai](https://github.com/cubert-hyperspectral/cuvis-ai) under
`cuvis_ai/configs/pipeline/rtsam2/`.

## Plugin manifest

One yaml file is one plugin. Local-path manifest (development; this repo ships one at
[cuvis_ai_rtsam2/plugins.yaml](cuvis_ai_rtsam2/plugins.yaml)):

```yaml
name: rtsam2
path: "../cuvis-ai-rtsam2"
capabilities:
  - class_name: cuvis_ai_rtsam2.node.rtsam2_streaming_propagation.RTSAM2BboxPropagation
  - class_name: cuvis_ai_rtsam2.node.rtsam2_streaming_propagation.RTSAM2MaskPropagation
```

Git-tag manifest (frozen, reproducible installs — available once the first release tag exists):

```yaml
name: rtsam2
repo: "https://github.com/cubert-hyperspectral/cuvis-ai-rtsam2.git"
tag: "v0.1.0"
package_name: "cuvis-ai-rtsam2"
capabilities:
  - class_name: cuvis_ai_rtsam2.node.rtsam2_streaming_propagation.RTSAM2BboxPropagation
  - class_name: cuvis_ai_rtsam2.node.rtsam2_streaming_propagation.RTSAM2MaskPropagation
```

Verify a manifest resolves:

```bash
uv run python -c "from cuvis_ai_core.utils.node_registry import NodeRegistry; r = NodeRegistry(); r.register_plugin('cuvis_ai_rtsam2/plugins.yaml'); print(r.list_plugins())"
```

## Upstream demos

The upstream gradio demo apps (`notebooks/tam_app.py`, `notebooks/sam_app.py`) are not
installable alongside this plugin: gradio 4.44 pins `pillow<11` while cuvis-ai-core requires
`pillow>=12.2`. Run them in a separate environment built from upstream's own requirements (see
[README_original.md](README_original.md)). The `notebooks` extra ships headless OpenCV; demos
expecting GUI OpenCV features are best-effort.

## Links

- **Documentation:** https://docs.cuvis.ai/latest/
- **Website:** https://www.cubert-hyperspectral.com/
- **Support:** http://support.cubert-hyperspectral.com/
- **Issues:** https://github.com/cubert-hyperspectral/cuvis-ai-rtsam2/issues
- **Changelog:** [CHANGELOG.md](CHANGELOG.md)
- **Original REALTIME_SAM2 README:** [README_original.md](README_original.md)

---

See [LICENSE](LICENSE) for repository licensing details.
