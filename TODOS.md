# TODOS

## Mid-stream mask re-prompting (SAM3 parity)

- **What:** Accept a new or edited label map on any frame while tracking (add/edit objects
  mid-stream) instead of the current prompt-once-then-raise behavior.
- **Why:** Interactive host sessions (CuvisNEXT view pipelines) want to add or correct objects
  without restarting the stream; `SAM3MaskPropagation` already supports this.
- **Where to start:** rtsam2 injection lives in
  `cuvis_ai_rtsam2/node/_rtsam2_tracker_base.py::_add_mask_prompt` (writes
  `condition_state["mask_inputs_per_obj"]`, calls `_run_single_frame_inference(new_input=True)`);
  the mid-stream guard is the `has_prompt` raise in `_forward_stream`. SAM3's reference flow is
  `_apply_runtime_mask` in `cuvis_ai_sam3/node/sam3_streaming_propagation.py` (per-component
  interior points, action-history reset).
- **Risks:** new engineering on the private predictor-state path; every upstream sync of the
  camera predictors can break it; needs real-weights validation.
- **When:** after v0.1.0, as a minor bump.

## Bbox-propagation pipelines in cuvis-ai

- **What:** `rtsam2_bbox_propagation{,_video,_view}.yaml` in cuvis-ai's
  `cuvis_ai/configs/pipeline/rtsam2/`, mirroring the sam3 bbox set and wired to
  `RTSAM2BboxPropagation`.
- **Why:** the node is exposed in the manifests but ships without a reference pipeline.
- **Where to start:** adapt the mask yamls; the builtin `BBoxPrompt` node
  (`cuvis_ai/node/prompts.py`) is the drop-in prompt producer, same pattern as
  `MaskPrompt` -> `mask`.
- **Risks:** none structural; unvalidated against real weights until first run.
- **Blocked by:** this standards pass landing (rgb_image ports, manifests).
