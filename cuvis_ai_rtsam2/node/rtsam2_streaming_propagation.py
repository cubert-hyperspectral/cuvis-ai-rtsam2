"""Streaming RTSAM2 propagation nodes for cuvis.ai pipelines."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from cuvis_ai_schemas.enums import NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec

from cuvis_ai_rtsam2.node._rtsam2_tracker_base import RTSAM2TrackerInference


class RTSAM2BboxPropagation(RTSAM2TrackerInference):
    """RTSAM2 propagation with runtime bbox prompts."""

    _tags = RTSAM2TrackerInference._tags | {NodeTag.BBOX}

    INPUT_SPECS = {
        **RTSAM2TrackerInference.INPUT_SPECS,
        "bboxes": PortSpec(
            dtype=list,
            shape=(),
            description=(
                "Optional per-frame list of bbox prompt dicts with keys "
                "element_id, object_id, x_min, y_min, x_max, y_max."
            ),
            optional=True,
        ),
    }

    @staticmethod
    def _normalize_runtime_bboxes(
        bboxes: list[dict[str, Any]] | None,
        expected_hw: tuple[int, int],
    ) -> list[dict[str, float | int]]:
        if bboxes is None:
            return []
        if not isinstance(bboxes, list):
            raise ValueError(
                f"Expected runtime bboxes to be a list of dicts, got {type(bboxes).__name__}."
            )

        height, width = int(expected_hw[0]), int(expected_hw[1])
        deduped_by_object_id: dict[int, dict[str, float | int]] = {}
        for idx, raw_box in enumerate(bboxes):
            if not isinstance(raw_box, dict):
                raise ValueError(
                    f"Expected bbox prompt at index {idx} to be a dict, got {type(raw_box).__name__}."
                )
            element_id = int(raw_box.get("element_id", 0))
            if element_id != 0:
                raise ValueError(
                    f"Runtime bbox prompt at index {idx} has element_id={element_id}; "
                    "RTSAM2 expects element_id=0."
                )
            if "object_id" not in raw_box:
                raise ValueError(f"Runtime bbox prompt at index {idx} is missing 'object_id'.")
            object_id = int(raw_box["object_id"])
            if object_id <= 0:
                raise ValueError(
                    f"Runtime bbox prompt at index {idx} has invalid object_id={object_id}; "
                    "object_id must be > 0."
                )

            missing = [key for key in ("x_min", "y_min", "x_max", "y_max") if key not in raw_box]
            if missing:
                raise ValueError(
                    f"Runtime bbox prompt at index {idx} is missing required keys: {missing}."
                )

            x_min = float(raw_box["x_min"])
            y_min = float(raw_box["y_min"])
            x_max = float(raw_box["x_max"])
            y_max = float(raw_box["y_max"])
            if not (0.0 <= x_min < x_max <= float(width)):
                raise ValueError(
                    "Runtime bbox x-range is invalid for the current RGB frame: "
                    f"(x_min={x_min}, x_max={x_max}, width={width})."
                )
            if not (0.0 <= y_min < y_max <= float(height)):
                raise ValueError(
                    "Runtime bbox y-range is invalid for the current RGB frame: "
                    f"(y_min={y_min}, y_max={y_max}, height={height})."
                )

            deduped_by_object_id[object_id] = {
                "element_id": element_id,
                "object_id": object_id,
                "x_min": x_min,
                "y_min": y_min,
                "x_max": x_max,
                "y_max": y_max,
            }

        return list(deduped_by_object_id.values())

    def forward(
        self,
        rgb_image: torch.Tensor,
        bboxes: list[dict[str, Any]] | None = None,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Seed tracking from bbox prompts on the first prompt frame, then track."""
        _, frame_shape = self._frame_from_tensor(rgb_image)
        prompt_bboxes = self._normalize_runtime_bboxes(bboxes, frame_shape)
        prompt_object_ids = [int(prompt_bbox["object_id"]) for prompt_bbox in prompt_bboxes]

        def _apply_prompt(frame_idx: int) -> None:
            for prompt_bbox in prompt_bboxes:
                external_id = int(prompt_bbox["object_id"])
                self._add_box_prompt(
                    frame_idx=int(frame_idx),
                    obj_id=int(self._ext_to_int[external_id]),
                    box_coords=[
                        float(prompt_bbox["x_min"]),
                        float(prompt_bbox["y_min"]),
                        float(prompt_bbox["x_max"]),
                        float(prompt_bbox["y_max"]),
                    ],
                )

        return self._forward_stream(
            rgb_image,
            frame_id,
            has_prompt=len(prompt_bboxes) > 0,
            external_object_ids=prompt_object_ids,
            prompt_callback=_apply_prompt,
        )


class RTSAM2MaskPropagation(RTSAM2TrackerInference):
    """RTSAM2 propagation with runtime label-map prompts."""

    INPUT_SPECS = {
        **RTSAM2TrackerInference.INPUT_SPECS,
        "mask": PortSpec(
            dtype=torch.int32,
            shape=(1, -1, -1),
            description=(
                "Optional int32 label map [1,H,W]. 0=background, each positive "
                "label is treated as an object ID prompt on that frame."
            ),
            optional=True,
        ),
    }

    @staticmethod
    def _normalize_runtime_mask(
        mask: torch.Tensor | None,
        expected_hw: tuple[int, int],
    ) -> np.ndarray | None:
        if mask is None:
            return None
        if mask.ndim != 3 or int(mask.shape[0]) != 1:
            raise ValueError(
                f"Expected runtime mask shape [1,H,W], got {tuple(int(v) for v in mask.shape)}."
            )

        expected_h, expected_w = int(expected_hw[0]), int(expected_hw[1])
        actual_h, actual_w = int(mask.shape[1]), int(mask.shape[2])
        if (actual_h, actual_w) != (expected_h, expected_w):
            raise ValueError(
                "Runtime mask shape does not match current RGB frame: "
                f"mask={(actual_h, actual_w)}, rgb={(expected_h, expected_w)}."
            )

        label_map = np.asarray(mask[0].detach().cpu().numpy(), dtype=np.int32)
        if np.any(label_map < 0):
            raise ValueError("RTSAM2 mask prompts must not contain negative object IDs.")
        return label_map

    @staticmethod
    def _prompt_object_ids(label_map: np.ndarray | None) -> list[int]:
        if label_map is None:
            return []
        return sorted(int(obj_id) for obj_id in np.unique(label_map) if int(obj_id) > 0)

    def forward(
        self,
        rgb_image: torch.Tensor,
        mask: torch.Tensor | None = None,
        frame_id: torch.Tensor | None = None,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Seed tracking from a label-map prompt on the first prompt frame, then track."""
        _, frame_shape = self._frame_from_tensor(rgb_image)
        label_map = self._normalize_runtime_mask(mask, frame_shape)
        prompt_object_ids = self._prompt_object_ids(label_map)

        def _apply_prompt(frame_idx: int) -> None:
            if label_map is None:
                return
            for external_id in prompt_object_ids:
                self._add_mask_prompt(
                    frame_idx=int(frame_idx),
                    obj_id=int(self._ext_to_int[external_id]),
                    mask_object_id=int(external_id),
                    label_map=label_map,
                )

        return self._forward_stream(
            rgb_image,
            frame_id,
            has_prompt=len(prompt_object_ids) > 0,
            external_object_ids=prompt_object_ids,
            prompt_callback=_apply_prompt,
        )


__all__ = [
    "RTSAM2TrackerInference",
    "RTSAM2BboxPropagation",
    "RTSAM2MaskPropagation",
]
