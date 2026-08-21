"""Shared camera-predictor fakes for the RTSAM2 node tests.

Lives in an underscore-prefixed plain module (not conftest.py, whose globals are
not importable) so the streaming-propagation and point-expansion test files can
use the same fake; pytest puts this directory on ``sys.path``, so it imports as
``from _mock_predictors import ...``.

EfficientTAM fidelity note: upstream ``load_first_frame`` registers one dummy
object per class (id 0 for ``num_classes=1``) with a neutral (0, 0) point, so
prompt calls return an extra leading mask channel pinned at ``NO_OBJ_SCORE``.
The fake reproduces the dummy in state and in the returned logits, but registers
it directly rather than through the ``add_new_prompts`` mock, so call-count
assertions in tests only ever see real prompts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from cuvis_ai_rtsam2.node import RTSAM2TrackerInference
from cuvis_ai_rtsam2.node import _rtsam2_tracker_base as rtsam2_base

# Upstream's fill value for absent-object mask channels; EfficientTAM's dummy
# object comes back at this level instead of as a plausible mask.
NO_OBJ_SCORE = -1024.0


def _make_logits_for_ids(
    obj_ids: list[int],
    *,
    frame_idx: int,
    height: int = 10,
    width: int = 12,
) -> torch.Tensor:
    logits = torch.full((len(obj_ids), 1, height, width), -8.0, dtype=torch.float32)
    for idx, _obj_id in enumerate(obj_ids):
        y0 = min(height - 2, 1 + idx * 2)
        x0 = min(width - 3, 1 + frame_idx + idx)
        y1 = min(height, y0 + 2)
        x1 = min(width, x0 + 3)
        logits[idx, 0, y0:y1, x0:x1] = 4.0 + float(idx)
    return logits


class _MockCameraPredictor:
    def __init__(self, family: str) -> None:
        self.family = family
        self.image_size = 8
        self.object_of_interest = -1
        self.condition_state = {}
        self._frame_idx = 0
        self._dummy_obj_ids: set[int] = set()
        self.load_first_frame = MagicMock(side_effect=self._load_first_frame_impl)
        self.finalize_new_input = MagicMock(side_effect=self._finalize_new_input_impl)
        self.track = MagicMock(side_effect=self._track_impl)
        self.add_new_points_or_box = MagicMock(side_effect=self._add_sam2_prompt_impl)
        self.add_new_prompts = MagicMock(side_effect=self._add_efficienttam_prompt_impl)
        self._obj_id_to_idx = MagicMock(side_effect=self._obj_id_to_idx_impl)
        self._run_single_frame_inference = MagicMock(
            side_effect=self._run_single_frame_inference_impl
        )

    def to(self, device: str | torch.device) -> _MockCameraPredictor:
        del device
        return self

    def _load_first_frame_impl(self, _img, num_classes: int = 1) -> torch.Tensor | None:
        self._frame_idx = 0
        self._dummy_obj_ids = set()
        self.condition_state = {
            "obj_ids": [],
            "obj_id_to_idx": {},
            "obj_idx_to_id": {},
            "prompt_boxes": [],
            "prompt_points": [],
            "mask_prompt_calls": [],
            "num_classes": int(num_classes),
            "device": torch.device("cpu"),
            "storage_device": torch.device("cpu"),
            "point_inputs_per_obj": {},
            "mask_inputs_per_obj": {},
            "output_dict_per_obj": {},
            "temp_output_dict_per_obj": {},
            "frames_already_tracked": {},
            "tracking_has_started": False,
        }
        if self.family == "efficienttam":
            # Upstream registers a neutral-point dummy object per class here.
            for cls in range(int(num_classes)):
                self._dummy_obj_ids.add(cls)
                if cls not in self.condition_state["obj_ids"]:
                    self.condition_state["obj_ids"].append(cls)
            return self._logits_for(self.condition_state["obj_ids"], frame_idx=0)
        return None

    def _obj_id_to_idx_impl(self, obj_id: int) -> int:
        obj_id = int(obj_id)
        obj_idx = self.condition_state["obj_id_to_idx"].get(obj_id)
        if obj_idx is not None:
            return obj_idx

        obj_idx = len(self.condition_state["obj_id_to_idx"])
        self.condition_state["obj_id_to_idx"][obj_id] = obj_idx
        self.condition_state["obj_idx_to_id"][obj_idx] = obj_id
        self.condition_state["obj_ids"] = list(self.condition_state["obj_id_to_idx"].keys())
        self.condition_state["point_inputs_per_obj"][obj_idx] = {}
        self.condition_state["mask_inputs_per_obj"][obj_idx] = {}
        self.condition_state["output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        self.condition_state["temp_output_dict_per_obj"][obj_idx] = {
            "cond_frame_outputs": {},
            "non_cond_frame_outputs": {},
        }
        return obj_idx

    def _logits_for(self, obj_ids: list[int], *, frame_idx: int) -> torch.Tensor:
        # Real objects keep the exact geometry _make_logits_for_ids gives a
        # dummy-free id list; dummy channels sit at NO_OBJ_SCORE.
        real_ids = [obj_id for obj_id in obj_ids if obj_id not in self._dummy_obj_ids]
        real_logits = _make_logits_for_ids(real_ids, frame_idx=frame_idx)
        logits = torch.full((len(obj_ids), 1, 10, 12), NO_OBJ_SCORE, dtype=torch.float32)
        real_idx = 0
        for idx, obj_id in enumerate(obj_ids):
            if obj_id not in self._dummy_obj_ids:
                logits[idx] = real_logits[real_idx]
                real_idx += 1
        return logits

    def _prompt_return(self, frame_idx: int) -> tuple[int, list[int], torch.Tensor]:
        obj_ids = list(self.condition_state["obj_ids"])
        # Accumulated point prompts shift the returned geometry (load_first_frame
        # resets the record), so a caller that skips the per-click re-seed gets
        # different logits for the same click and equality tests catch it.
        shift = max(0, len(self.condition_state.get("prompt_points", ())) - 1)
        return (
            int(frame_idx),
            obj_ids,
            self._logits_for(obj_ids, frame_idx=int(frame_idx) + shift),
        )

    def _record_prompt(
        self,
        frame_idx: int,
        obj_id: int,
        *,
        box=None,
        points=None,
        labels=None,
        normalize_coords: bool = True,
    ) -> tuple[int, list[int], torch.Tensor]:
        if int(obj_id) not in self.condition_state["obj_ids"]:
            self.condition_state["obj_ids"].append(int(obj_id))
        if box is not None:
            self.condition_state["prompt_boxes"].append(
                {
                    "frame_idx": int(frame_idx),
                    "obj_id": int(obj_id),
                    "box": [float(value) for value in box],
                }
            )
        if points is not None:
            self.condition_state["prompt_points"].append(
                {
                    "frame_idx": int(frame_idx),
                    "obj_id": int(obj_id),
                    "points": np.asarray(points, dtype=np.float32),
                    "labels": np.asarray(labels, dtype=np.int32),
                    "normalize_coords": bool(normalize_coords),
                }
            )
        return self._prompt_return(frame_idx)

    def _add_sam2_prompt_impl(
        self,
        *,
        frame_idx: int,
        obj_id: int,
        points=None,
        labels=None,
        box=None,
        normalize_coords: bool = True,
    ) -> tuple[int, list[int], torch.Tensor]:
        return self._record_prompt(
            frame_idx,
            obj_id,
            box=box,
            points=points,
            labels=labels,
            normalize_coords=normalize_coords,
        )

    def _add_efficienttam_prompt_impl(
        self,
        *,
        frame_idx: int,
        obj_id: int,
        points=None,
        labels=None,
        boxes=None,
        normalize_coords: bool = True,
    ) -> tuple[int, list[int], torch.Tensor]:
        return self._record_prompt(
            frame_idx,
            obj_id,
            box=boxes,
            points=points,
            labels=labels,
            normalize_coords=normalize_coords,
        )

    def _run_single_frame_inference_impl(
        self,
        *,
        output_dict,
        frame_idx: int,
        batch_size: int,
        is_init_cond_frame: bool,
        point_inputs,
        mask_inputs,
        reverse: bool,
        run_mem_encoder: bool,
        prev_sam_mask_logits,
        new_input: bool,
    ) -> tuple[dict[str, torch.Tensor | None], torch.Tensor]:
        del (
            batch_size,
            is_init_cond_frame,
            point_inputs,
            reverse,
            run_mem_encoder,
            prev_sam_mask_logits,
            new_input,
        )
        obj_idx = next(
            idx
            for idx, obj_output_dict in self.condition_state["output_dict_per_obj"].items()
            if obj_output_dict is output_dict
        )
        obj_id = int(self.condition_state["obj_idx_to_id"][obj_idx])
        if mask_inputs is not None:
            self.condition_state["mask_prompt_calls"].append(
                {
                    "frame_idx": int(frame_idx),
                    "obj_id": obj_id,
                    "mask_shape": tuple(int(v) for v in mask_inputs.shape),
                    "positive_pixels": int((mask_inputs > 0).sum().item()),
                }
            )
        pred_masks = _make_logits_for_ids([obj_id], frame_idx=int(frame_idx))
        return (
            {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": pred_masks,
                "obj_ptr": torch.zeros((1, 1), dtype=torch.float32),
                "object_score_logits": torch.full((1, 1), 10.0, dtype=torch.float32),
            },
            pred_masks,
        )

    def _finalize_new_input_impl(self) -> tuple[int, list[int], torch.Tensor]:
        obj_ids = list(self.condition_state["obj_ids"])
        self.condition_state["tracking_has_started"] = True
        return (
            self._frame_idx,
            obj_ids,
            self._logits_for(obj_ids, frame_idx=self._frame_idx),
        )

    def _track_impl(self, _img) -> tuple[list[int], torch.Tensor]:
        self._frame_idx += 1
        obj_ids = list(self.condition_state["obj_ids"])
        return obj_ids, self._logits_for(obj_ids, frame_idx=self._frame_idx)


def _attach_mock_predictor(
    node: RTSAM2TrackerInference,
    *,
    family: str,
) -> _MockCameraPredictor:
    predictor = _MockCameraPredictor(family)
    node._predictor = predictor
    node._ensure_model = MagicMock()
    return predictor


def _random_rgb(height: int = 10, width: int = 12) -> torch.Tensor:
    return torch.rand(1, height, width, 3, dtype=torch.float32)


def _patch_model_package_root(
    monkeypatch: pytest.MonkeyPatch,
    repo_root: Path,
) -> None:
    def _package_root(cls, model_type: str) -> Path:
        package_name = "sam2" if model_type == "sam2" else "efficient_track_anything"
        return repo_root / package_name

    monkeypatch.setattr(
        RTSAM2TrackerInference,
        "_model_package_root",
        classmethod(_package_root),
    )


def _materialize_variant_layout(
    tmp_path: Path,
    *,
    variant: str,
    include_config: bool = True,
    checkpoint_dir: Path | None = None,
    include_checkpoint: bool = True,
) -> tuple[Path, Path, Path]:
    repo_root = tmp_path / "cuvis-ai-rtsam2"
    spec = rtsam2_base._MODEL_VARIANT_REGISTRY[variant]
    package_root = repo_root / ("sam2" if spec.family == "sam2" else "efficient_track_anything")
    config_path = package_root / spec.config_name
    if include_config:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text("# test\n", encoding="utf-8")

    model_dir = checkpoint_dir or (repo_root / "checkpoints")
    checkpoint_path = model_dir / spec.checkpoint_name
    if include_checkpoint:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text("weights\n", encoding="utf-8")

    return repo_root, config_path.resolve(strict=False), checkpoint_path.resolve(strict=False)
