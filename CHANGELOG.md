# Changelog

All notable changes to this project will be documented in this file.

## 0.3.0 - 2026-08-21

- Added the `RTSAM2PointExpansion` node: interactive positive/negative click points expanded into one object mask on a single frame, port-compatible with the sam3 plugin's `SAM3PointExpansion` (`points` dicts with `element_id`/`x`/`y`/`type`; `frame_id` accepted for contract parity, unused). It reuses the vendored camera predictors' point API in a per-click lifecycle (`load_first_frame` + decoder-only prompt, never `finalize_new_input`/`track`), so clicks are deterministic and `reset()` keeps the loaded predictor. Out-of-bounds coordinates clamp into the frame with one warning per node instance; malformed or non-finite points raise.
- Patched the vendored camera predictors to derive the inference-state device from the model weights instead of probing CUDA/MPS and raising: the point-expansion node now runs on CPU-only machines (force with `CUDA_VISIBLE_DEVICES=-1`; an empty string is not a valid override). The three `.cuda(non_blocking=True)` calls on `prev_sam_mask_logits` were retargeted to that device, and EfficientTAM's stray `print("frame 0")` in `load_first_frame` was removed. Side effect: the EfficientTAM streaming tracker nodes on a CPU-only machine now run slowly instead of raising at state init (sam2.1-family trackers still raise on CPU, now at the first tracked frame via an unpatched `pin_memory()` call); tracker CPU support remains out of scope. The full local-patch list lives in `.upstream-sync.yml` and must be re-applied after every upstream sync.
- Extracted the shared camera-predictor fake into `tests/cuvis_ai_rtsam2/_mock_predictors.py`, with point-prompt support and EfficientTAM's dummy-object fidelity (`load_first_frame` registers object 0; its channel returns at `NO_OBJ_SCORE`).
- Security: refreshed the `aiohttp` (3.14.3), `cryptography` (50.0.0), and `gitpython` (3.1.59) locks, clearing the freshly published advisories that failed the CI security scan on main.
## 0.2.1 - 2026-08-20

- Documented the torch cu128 index tables as local-development-only: installs of this package as a git or registry dependency never read them, and composed child environments mirror the host's torch build (cuvis-ai-core >= 0.12.1).

## 0.2.0 - 2026-07-17

- The EfficientTAM checkpoint now resolves from the shared HuggingFace cache when it is absent from the checkout, so the sandboxed runtime loads a weight provisioned out of band (`download-model efficienttam_s`) offline instead of failing. It is a pure cache lookup (never a download); an explicit `model_dir` still takes precedence as a deterministic override, and families with no single canonical HF repo (SAM2.1) are unaffected. The missing-asset guidance now points at the shared-cache provisioning path too.
- Require `cuvis-ai-schemas>=0.8.0` and `cuvis-ai-core>=0.11.2`, adopting the released framework versions. Core 0.11.2's floors transitively pull the security-fixed `click` 8.4.2 (PYSEC-2026-2132) and `pillow` 12.3.0 (PYSEC-2026-2253/2254/2255/2256/2257/3451/3452/3453) into the lock.
- Ignored PYSEC-2026-3447 (setuptools, fixed only in 83.0.0) in the pip-audit step: torch 2.11 (cu128) caps setuptools below the fix, so it needs a torch upgrade.

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
