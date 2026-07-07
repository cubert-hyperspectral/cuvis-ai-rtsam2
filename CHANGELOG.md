# Changelog

All notable changes to this project will be documented in this file.

## 0.1.0 - 2026-07-07

- Added the `cuvis_ai_rtsam2` plugin package with the `RTSAM2BboxPropagation` and `RTSAM2MaskPropagation` streaming tracker nodes (shared `RTSAM2TrackerInference` base) wrapping the vendored SAM2.1 / EfficientTAM camera predictors: prompt once on the first frame (bbox list or int32 label-map mask), then track frame by frame; pure-tensor-mocked tests alongside.
- Added the `efficienttam_s` / `efficienttam_s_512x512` model configs missing from the vendored upstream tree.
- Added `reset()` and `cleanup()` to the tracker base (driven by `Predictor` between runs); both drop the loaded predictor, since the upstream camera predictors keep model state outside `condition_state` that `load_first_frame` does not rebuild (verified on real weights).
- Added palette metadata (`_category`, `_tags`) to the concrete nodes.
- Added the in-repo local-path manifest `cuvis_ai_rtsam2/plugins.yaml` plus a drift test asserting it resolves exactly the concrete nodes.
- Added CI (test, lint, security, typecheck, build, uv.lock cu128 torch guard), a tag-driven release workflow, and the cuvis-ai-core dependency-compatibility workflow.
- Added contributor docs (`CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`), a codecov config, a detect-secrets baseline, and `.upstream-sync.yml` recording the upstream fork state.
- Renamed the RGB input port `rgb_frame` to `rgb_image` on all nodes, matching the sam3 plugin.
- Converted packaging to the `cuvis-ai-rtsam2` distribution with setuptools-scm versioning; require `cuvis-ai-core>=0.10.0` and `cuvis-ai-schemas>=0.7.0`.
- Pinned torch/torchvision to the explicit pytorch-cu128 uv index; declared triton (`triton-windows` on Windows) for the predictors' `torch.compile` paths; swapped `opencv-python` for `opencv-python-headless`.
- Moved the abstract `RTSAM2TrackerInference` into an underscore module so package auto-registration exposes only the two concrete nodes; import paths are unchanged.
- Replaced the upstream README with a cuvis.ai landing page; the original is preserved as `README_original.md`.
- Fixed frame scaling: the upstream predictors divide numpy frames by 255, so the node now converts the port's float [0,1] frames to uint8 before handing them over (feeding floats froze the mask at the seed location with decaying scores).
- Removed the manual shape re-asserts that duplicated the framework's port validation.
- Removed the upstream gradio demo stack from the installable dependencies (gradio 4.44 pins `pillow<11`, which cannot co-install with cuvis-ai-core); the demos need a separate env.
