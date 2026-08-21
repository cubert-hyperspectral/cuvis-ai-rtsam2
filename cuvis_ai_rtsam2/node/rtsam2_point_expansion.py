"""Interactive single-frame RTSAM2 point-expansion node.

Unlike the streaming propagation nodes (which advance an internal frame index on
every ``forward`` call), this node segments ONE object on a SINGLE frame from
positive/negative click points. It reuses the vendored camera predictors' point
API in a per-click lifecycle: seed the frame, decode the click, and never call
``finalize_new_input`` or ``track``. Re-sending an updated point set re-seeds the
same predictor, so the mask is a deterministic function of the current point set
(no ``prev_sam_mask_logits`` feedback across clicks).

Per-click data flow::

     host click set ─┐
                     │  points: [{element_id,x,y,type}]     (element_id ignored: single object)
     rgb_image ──────┼──> _frame_from_tensor ──> uint8 HWC
     frame_id  ──────┘        (float[0,1] guard)
                     │
                     v
            _parse_points ──> coords (N,2) clamped to frame, labels (N,) 1=pos 0=neg
                     │
                     v
            any positive point? ──no──> _empty_output(H,W)   [model never built]
                     │yes
                     v
            _build_object_id_maps([prompt_obj_id]) ──> ext→int, internal ids start at 1
                     │
                     v
            load_first_frame(uint8, num_classes=1)   << EVERY click: fresh condition_state
                     │                                  (EfficientTAM also seeds dummy obj 0)
                     v
            sam2: add_new_points_or_box(...)  |  efficienttam: add_new_prompts(...)
                     │                            (decoder-only pass)
                     v
            (frame_idx=0, obj_ids=[0?, int_id], video_res_masks[N,1,H,W])
                     │
                     v
            _pack_output ──> mask int32[1,H,W] (dummy dropped) + object_ids + detection_scores

The re-seed costs one encoder pass per click (EfficientTAM-S: ~64 ms on GPU,
~0.5 s on CPU); both families' model configs already select the best multimask
candidate for a single positive click, so no ``multimask_output`` hparam exists.
Because this node never calls ``track()``/``finalize_new_input()``, the upstream
frame-index leak that forces the streaming trackers to drop their predictor on
``reset()`` does not apply, so ``reset()`` here keeps the loaded model.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from loguru import logger

from cuvis_ai_rtsam2.node._rtsam2_tracker_base import RTSAM2TrackerInference

_POSITIVE = "positive"
_NEGATIVE = "negative"
_NEUTRAL = "neutral"


class RTSAM2PointExpansion(RTSAM2TrackerInference):
    """Expand positive/negative click points into one object mask on a single frame.

    The optional ``points`` input is a per-frame list of prompt dicts with keys
    ``element_id``, ``x``, ``y`` (pixel coordinates), and ``type`` (one of
    ``positive``, ``negative``, ``neutral``). Positive points mark the object,
    negative points mark background, neutral points are ignored. All points
    address a single object whose id is the ``prompt_obj_id`` hparam
    (``element_id`` is ignored).

    Frames with no positive point emit an empty mask without building the model.
    Every clicked frame re-seeds the predictor (one encoder pass per click), so
    the same points always yield the same mask and ``frame_id`` is accepted for
    contract parity with ``SAM3PointExpansion`` but unused. Out-of-bounds
    coordinates are clamped into the frame with one warning per node instance.
    """

    _tags = frozenset(
        {
            NodeTag.RGB,
            NodeTag.IMAGE,
            NodeTag.MASK,
            NodeTag.KEYPOINTS,
            NodeTag.SEGMENTATION,
            NodeTag.INFERENCE,
            NodeTag.LEARNABLE,
            NodeTag.BATCHED,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, 3),
            description="RGB frame [1,H,W,3] in float32 with values in [0, 1].",
        ),
        "points": PortSpec(
            dtype=list,
            shape=(),
            description=(
                "Optional per-frame list of point prompt dicts with keys element_id, x, y, type "
                "(type in {positive, negative, neutral}). Positive=object, negative=background."
            ),
            optional=True,
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Optional source frame index [1]; accepted for contract parity, unused.",
            optional=True,
        ),
    }

    def __init__(
        self,
        model_type: str = "efficienttam",
        model_dir: str | None = None,
        prompt_obj_id: int = 1,
        **kwargs: Any,
    ) -> None:
        """Configure the interactive point-expansion node.

        Args:
            model_type: Model variant or family alias (``efficienttam`` resolves
                to ``efficienttam_s``, ``sam2`` to ``sam2.1_hiera_t``).
            model_dir: Optional directory holding the checkpoint file.
            prompt_obj_id: Object id written into the output label map (integer > 0).
        """
        if isinstance(prompt_obj_id, bool) or not isinstance(prompt_obj_id, int):
            raise ValueError(
                f"prompt_obj_id must be an integer, got {type(prompt_obj_id).__name__} "
                f"{prompt_obj_id!r}."
            )
        if prompt_obj_id <= 0:
            raise ValueError(f"prompt_obj_id must be > 0, got {prompt_obj_id}.")

        self._prompt_obj_id = int(prompt_obj_id)
        self._warned_out_of_bounds = False

        super().__init__(
            model_type=model_type,
            model_dir=model_dir,
            prompt_obj_id=prompt_obj_id,
            **kwargs,
        )

    def reset(self) -> None:
        """Clear the per-click object-id maps but keep the loaded predictor.

        This node never calls ``track()``/``finalize_new_input()``, so the
        upstream frame-index leak that forces the streaming trackers to rebuild
        their predictor per stream does not apply; ``load_first_frame`` on the
        next click rebuilds ``condition_state`` on the retained model. The
        out-of-bounds warning stays once-per-instance across resets.
        """
        self._stream_frame_idx = 0
        self._tracking_started = False
        self._ext_to_int = {}
        self._int_to_ext = {}
        self._maybe_clear_cuda_cache()

    def cleanup(self) -> None:
        """Release the loaded predictor and all per-click state on teardown."""
        self._predictor = None
        self.reset()

    def _parse_points(
        self,
        points: list[dict[str, Any]] | None,
        frame_shape: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert the runtime point list into pixel coords [M,2] and labels [M] (1=pos, 0=neg).

        Neutral points are dropped. Malformed entries (non-dict, missing x/y,
        unknown type, non-finite coordinates) raise even when no positive point
        is present. Out-of-bounds coordinates are clamped into
        [0, W-1] x [0, H-1] with one warning per node instance.
        """
        if points is None:
            return np.zeros((0, 2), dtype=np.float32), np.zeros((0,), dtype=np.int32)
        if not isinstance(points, list):
            raise ValueError(f"Expected points to be a list of dicts, got {type(points).__name__}.")

        height, width = int(frame_shape[0]), int(frame_shape[1])
        coords: list[list[float]] = []
        labels: list[int] = []
        for idx, raw in enumerate(points):
            if not isinstance(raw, dict):
                raise ValueError(
                    f"Expected point prompt at index {idx} to be a dict, got {type(raw).__name__}."
                )
            point_type = str(raw.get("type", _POSITIVE)).lower()
            if point_type == _NEUTRAL:
                continue
            if point_type not in (_POSITIVE, _NEGATIVE):
                raise ValueError(
                    f"Point prompt at index {idx} has unknown type {point_type!r}; "
                    "expected positive, negative, or neutral."
                )
            if "x" not in raw or "y" not in raw:
                raise ValueError(f"Point prompt at index {idx} is missing 'x' or 'y'.")
            x = float(raw["x"])
            y = float(raw["y"])
            if not (math.isfinite(x) and math.isfinite(y)):
                raise ValueError(
                    f"Point prompt at index {idx} has non-finite coordinates (x={x}, y={y})."
                )
            clamped_x = min(max(x, 0.0), float(width - 1))
            clamped_y = min(max(y, 0.0), float(height - 1))
            if (clamped_x != x or clamped_y != y) and not self._warned_out_of_bounds:
                self._warned_out_of_bounds = True
                logger.warning(
                    "RTSAM2PointExpansion clamped an out-of-bounds point ({x}, {y}) to "
                    "({cx}, {cy}) for a {width}x{height} frame; further clamps on this "
                    "node instance are silent. A persistent mismatch usually means the "
                    "host sends display-resolution coordinates for a different-resolution "
                    "frame.",
                    x=x,
                    y=y,
                    cx=clamped_x,
                    cy=clamped_y,
                    width=width,
                    height=height,
                )
            coords.append([clamped_x, clamped_y])
            labels.append(1 if point_type == _POSITIVE else 0)

        return (
            np.asarray(coords, dtype=np.float32).reshape(-1, 2),
            np.asarray(labels, dtype=np.int32),
        )

    def forward(
        self,
        rgb_image: torch.Tensor,
        points: list[dict[str, Any]] | None = None,
        frame_id: torch.Tensor | None = None,  # noqa: ARG002 -- contract parity, unused
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Expand the current point set into one object mask on this frame."""
        frame_np, frame_shape = self._frame_from_tensor(rgb_image)
        height, width = frame_shape

        # Parse before the empty-guard so malformed entries always raise, even
        # when no positive point is present.
        point_coords, point_labels = self._parse_points(points, frame_shape)
        if not bool((point_labels == 1).any()):
            return self._empty_output(height, width)

        self._ensure_model()
        self._build_object_id_maps([self._prompt_obj_id])
        internal_id = int(self._ext_to_int[self._prompt_obj_id])

        with torch.inference_mode():
            # Re-seed every click: rebuilds condition_state and embeds the frame,
            # keeping the click deterministic (no prev-mask feedback possible).
            self._predictor.load_first_frame(frame_np, num_classes=1)
            if self._model_type == "sam2":
                _, obj_ids, video_res_masks = self._predictor.add_new_points_or_box(
                    frame_idx=0,
                    obj_id=internal_id,
                    points=point_coords,
                    labels=point_labels,
                    box=None,
                    normalize_coords=True,
                )
            else:
                _, obj_ids, video_res_masks = self._predictor.add_new_prompts(
                    frame_idx=0,
                    obj_id=internal_id,
                    points=point_coords,
                    labels=point_labels,
                    normalize_coords=True,
                )

        return self._pack_output(obj_ids, video_res_masks, frame_shape)


__all__ = [
    "RTSAM2PointExpansion",
]
