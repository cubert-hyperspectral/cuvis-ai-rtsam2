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
## Same-frame embedding cache for RTSAM2PointExpansion

- **What:** Cache the frame embedding so repeat clicks on the same frame run decoder-only
  instead of re-seeding (`load_first_frame`) on every click.
- **Why:** Measured on EfficientTAM-S: a full click costs ~64 ms on GPU but 480-580 ms on CPU;
  a cached same-frame click would be ~55 ms. On the GPU production path the win is marginal,
  on CPU it is ~10x.
- **Where to start:** the upstream predictors already keep `cached_features`
  (`efficient_track_anything/efficienttam_camera_predictor.py`, `_get_image_feature`); a cache
  needs a frame-content key (see SAM3PointExpansion's `_content_signature`) plus popping the
  per-object temp outputs between clicks so repeat clicks stay deterministic (no
  `prev_sam_mask_logits` feedback).
- **Risks:** determinism regressions if the temp-output pop misses a state slot; more coupling
  to the predictors' private state.
- **When:** only if user feedback says CPU clicking feels sluggish; the re-seed-per-click
  design is deliberately the simple, deterministic default.

## Point-seeded propagation (click, then track)

- **What:** Let the streaming trackers accept point prompts at seed time (positive/negative
  clicks on the first prompt frame), alongside the existing bbox and mask seeding.
- **Why:** upstream's `add_new_points_or_box` / `add_new_prompts` already take points at seed
  time; hosts could then start a tracked stream from the same click UX the point-expansion
  node uses.
- **Where to start:** mirror `_add_box_prompt` in
  `cuvis_ai_rtsam2/node/_rtsam2_tracker_base.py`; reuse `RTSAM2PointExpansion._parse_points`
  for validation/clamping. Seed-time only; mid-stream re-prompting stays the separate TODO
  above.
- **Risks:** none structural; needs real-weights validation of seeded-then-tracked quality.
- **When:** separate minor bump, after point expansion ships.
