# Changelog

## [Unreleased]

### Added

- Added `reset()` and `cleanup()` to the tracker base (driven by `Predictor` between runs).
  Both release the loaded predictor: the upstream camera predictors keep model state outside
  `condition_state` that `load_first_frame` does not rebuild, so a fresh stream needs a fresh
  predictor build (verified on real weights).
- Added palette metadata (`_category`, `_tags`) to the concrete nodes.
- Added the in-repo local-path manifest `cuvis_ai_rtsam2/plugins.yaml` plus a drift test that
  asserts it resolves exactly the concrete nodes.
- Added CI (test, lint, security, typecheck, build, uv.lock cu128 torch guard), a tag-driven
  release workflow, and the plugin-vs-core dependency compatibility workflow.
- Added `.upstream-sync.yml` recording the upstream fork state.

### Changed

- Renamed the RGB input port `rgb_frame` to `rgb_image` on all nodes, matching the sam3 plugin.
- Converted packaging to the `cuvis-ai-rtsam2` distribution with setuptools-scm versioning;
  raised floors to `cuvis-ai-core>=0.10.0` and declared `cuvis-ai-schemas>=0.7.0` directly.
- Pinned torch/torchvision to the explicit pytorch-cu128 uv index; declared triton
  (`triton-windows` on Windows) for the predictors' `torch.compile` paths.
- Swapped `opencv-python` for `opencv-python-headless`.
- Moved the abstract `RTSAM2TrackerInference` into an underscore module so package
  auto-registration exposes only the two concrete nodes; import paths are unchanged.
- Replaced the upstream README with a cuvis.ai landing page; the original is preserved as
  `README_original.md`.

### Fixed

- Fixed frame scaling: the upstream camera predictors divide numpy frames by 255, so the node
  now converts the port's float [0,1] frames to uint8 0-255 before handing them over. Feeding
  floats directly left the model tracking on a nearly black image, freezing the mask at the
  seed location with decaying scores.

### Removed

- Removed the manual shape re-asserts that duplicated the framework's port validation.
- Removed the upstream gradio demo stack from the installable dependencies (gradio 4.44 pins
  `pillow<11`, which cannot co-install with cuvis-ai-core); the demos need a separate env.
