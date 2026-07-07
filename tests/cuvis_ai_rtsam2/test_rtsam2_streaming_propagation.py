"""Tests for RTSAM2 streaming propagation nodes with mocked camera predictors."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from cuvis_ai_core.utils.node_registry import NodeRegistry
from cuvis_ai_schemas.enums import NodeCategory, NodeTag

from cuvis_ai_rtsam2 import register_all_nodes
from cuvis_ai_rtsam2.node import (
    RTSAM2BboxPropagation,
    RTSAM2MaskPropagation,
    RTSAM2TrackerInference,
)
from cuvis_ai_rtsam2.node import _rtsam2_tracker_base as rtsam2_base

pytestmark = pytest.mark.unit


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

    def _load_first_frame_impl(self, _img, num_classes: int = 1) -> None:
        self._frame_idx = 0
        self.condition_state = {
            "obj_ids": [],
            "obj_id_to_idx": {},
            "obj_idx_to_id": {},
            "prompt_boxes": [],
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

    def _record_prompt(self, frame_idx: int, obj_id: int, box: list[float]) -> None:
        if int(obj_id) not in self.condition_state["obj_ids"]:
            self.condition_state["obj_ids"].append(int(obj_id))
        self.condition_state["prompt_boxes"].append(
            {
                "frame_idx": int(frame_idx),
                "obj_id": int(obj_id),
                "box": [float(value) for value in box],
            }
        )

    def _add_sam2_prompt_impl(
        self,
        *,
        frame_idx: int,
        obj_id: int,
        points,
        labels,
        box,
        normalize_coords: bool,
    ) -> None:
        del points, labels, normalize_coords
        self._record_prompt(frame_idx, obj_id, list(box))

    def _add_efficienttam_prompt_impl(
        self,
        *,
        frame_idx: int,
        obj_id: int,
        points,
        labels,
        boxes,
        normalize_coords: bool,
    ) -> None:
        del points, labels, normalize_coords
        self._record_prompt(frame_idx, obj_id, list(boxes))

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
            _make_logits_for_ids(obj_ids, frame_idx=self._frame_idx),
        )

    def _track_impl(self, _img) -> tuple[list[int], torch.Tensor]:
        self._frame_idx += 1
        obj_ids = list(self.condition_state["obj_ids"])
        return obj_ids, _make_logits_for_ids(obj_ids, frame_idx=self._frame_idx)


def _attach_mock_predictor(
    node: RTSAM2BboxPropagation | RTSAM2MaskPropagation,
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


def test_registration_round_trip() -> None:
    count = register_all_nodes()
    registry = NodeRegistry()

    assert count == 2
    assert registry.get("RTSAM2BboxPropagation") is RTSAM2BboxPropagation
    assert registry.get("RTSAM2MaskPropagation") is RTSAM2MaskPropagation
    with pytest.raises(KeyError):
        registry.get("RTSAM2TrackerInference")


def test_frame_from_tensor_converts_to_uint8_for_upstream() -> None:
    # Upstream prepare_data divides numpy frames by 255, i.e. it expects uint8
    # 0-255 input; handing it the port's float [0,1] frames directly leaves the
    # model tracking on a nearly black image (mask freezes at the seed location).
    frame = torch.full((1, 4, 6, 3), 0.5, dtype=torch.float32)
    frame_np, (height, width) = RTSAM2TrackerInference._frame_from_tensor(frame)

    assert frame_np.dtype == np.uint8
    assert (height, width) == (4, 6)
    assert int(frame_np[0, 0, 0]) == 128


def test_input_specs_declare_rgb_image_port() -> None:
    for node_cls in (RTSAM2TrackerInference, RTSAM2BboxPropagation, RTSAM2MaskPropagation):
        assert "rgb_image" in node_cls.INPUT_SPECS
        assert "rgb_frame" not in node_cls.INPUT_SPECS


def test_palette_metadata_uses_schema_enums() -> None:
    assert RTSAM2BboxPropagation._category is NodeCategory.MODEL
    assert RTSAM2MaskPropagation._category is NodeCategory.MODEL
    assert NodeTag.MASK in RTSAM2MaskPropagation._tags
    assert NodeTag.BBOX in RTSAM2BboxPropagation._tags
    assert NodeTag.BBOX not in RTSAM2MaskPropagation._tags


def test_forward_binds_inputs_by_port_name() -> None:
    bbox_node = RTSAM2BboxPropagation(model_type="efficienttam", name="test_kwarg_bbox")
    bbox_node._ensure_model = MagicMock()
    result = bbox_node.forward(
        rgb_image=_random_rgb(),
        bboxes=None,
        frame_id=torch.tensor([0], dtype=torch.int64),
    )
    assert result["mask"].shape == (1, 10, 12)

    mask_node = RTSAM2MaskPropagation(model_type="sam2", name="test_kwarg_mask")
    mask_node._ensure_model = MagicMock()
    result = mask_node.forward(
        rgb_image=_random_rgb(),
        mask=None,
        frame_id=torch.tensor([0], dtype=torch.int64),
    )
    assert result["mask"].shape == (1, 10, 12)


def test_reset_drops_predictor_and_allows_a_fresh_stream() -> None:
    node = RTSAM2MaskPropagation(model_type="sam2", name="test_reset_fresh_stream")
    first_predictor = _attach_mock_predictor(node, family="sam2")

    first_mask = torch.zeros((1, 10, 12), dtype=torch.int32)
    first_mask[:, 2:5, 4:8] = 7
    node.forward(_random_rgb(), mask=first_mask)
    node.forward(_random_rgb())

    node.reset()

    assert node._tracking_started is False
    assert node._stream_frame_idx == 0
    assert node._ext_to_int == {}
    assert node._int_to_ext == {}
    # The upstream predictors keep model state outside condition_state that
    # load_first_frame does not rebuild, so reset() must drop the predictor.
    assert node._predictor is None

    second_predictor = _attach_mock_predictor(node, family="sam2")
    second_mask = torch.zeros((1, 10, 12), dtype=torch.int32)
    second_mask[:, 6:8, 2:5] = 9
    result = node.forward(_random_rgb(), mask=second_mask)

    assert result["object_ids"].tolist() == [[9]]
    assert second_predictor is not first_predictor
    assert second_predictor.load_first_frame.call_count == 1


def test_cleanup_releases_predictor_and_is_idempotent() -> None:
    node = RTSAM2MaskPropagation(model_type="sam2", name="test_cleanup_releases")
    _attach_mock_predictor(node, family="sam2")

    prompt_mask = torch.zeros((1, 10, 12), dtype=torch.int32)
    prompt_mask[:, 2:5, 4:8] = 5
    node.forward(_random_rgb(), mask=prompt_mask)
    ensure_model_calls = node._ensure_model.call_count

    node.cleanup()
    node.cleanup()

    assert node._predictor is None
    assert node._tracking_started is False

    result = node.forward(_random_rgb())
    assert torch.count_nonzero(result["mask"]).item() == 0
    assert result["object_ids"].shape == (1, 0)
    assert node._ensure_model.call_count == ensure_model_calls


def test_midstream_mask_prompt_raises() -> None:
    node = RTSAM2MaskPropagation(model_type="sam2", name="test_midstream_prompt")
    _attach_mock_predictor(node, family="sam2")

    prompt_mask = torch.zeros((1, 10, 12), dtype=torch.int32)
    prompt_mask[:, 2:5, 4:8] = 3
    node.forward(_random_rgb(), mask=prompt_mask)

    with pytest.raises(ValueError, match="Mid-stream re-prompting"):
        node.forward(_random_rgb(), mask=prompt_mask)


def test_default_model_type_resolves_to_efficienttam_s() -> None:
    node = RTSAM2BboxPropagation(name="test_default_model_type")

    assert node._requested_model_type == "efficienttam"
    assert node._model_variant == "efficienttam_s"
    assert node._model_type == "efficienttam"


@pytest.mark.parametrize(
    ("requested_model_type", "expected_variant", "expected_family"),
    [
        ("efficienttam", "efficienttam_s", "efficienttam"),
        ("sam2", "sam2.1_hiera_t", "sam2"),
        ("efficienttam_s", "efficienttam_s", "efficienttam"),
        ("efficienttam_s_512x512", "efficienttam_s_512x512", "efficienttam"),
        ("efficienttam_ti", "efficienttam_ti", "efficienttam"),
        ("efficienttam_ti_512x512", "efficienttam_ti_512x512", "efficienttam"),
        ("sam2.1_hiera_t", "sam2.1_hiera_t", "sam2"),
        ("sam2.1_hiera_s", "sam2.1_hiera_s", "sam2"),
        ("sam2.1_hiera_b+", "sam2.1_hiera_b+", "sam2"),
        ("sam2.1_hiera_l", "sam2.1_hiera_l", "sam2"),
    ],
)
def test_model_type_resolution(
    requested_model_type: str,
    expected_variant: str,
    expected_family: str,
) -> None:
    node = RTSAM2BboxPropagation(
        model_type=requested_model_type,
        name=f"test_{expected_variant.replace('.', '_').replace('+', 'plus')}",
    )

    assert node._model_variant == expected_variant
    assert node._model_type == expected_family


def test_empty_before_prompt_returns_zero_outputs_without_model_init() -> None:
    node = RTSAM2BboxPropagation(model_type="efficienttam", name="test_empty_before_prompt")
    node._ensure_model = MagicMock()

    for _ in range(3):
        result = node.forward(_random_rgb(), bboxes=None)
        assert result["mask"].shape == (1, 10, 12)
        assert torch.count_nonzero(result["mask"]).item() == 0
        assert result["object_ids"].shape == (1, 0)
        assert result["detection_scores"].shape == (1, 0)

    node._ensure_model.assert_not_called()


def test_prompt_initializes_state_and_returns_non_zero_mask() -> None:
    node = RTSAM2BboxPropagation(model_type="efficienttam", name="test_prompt_initializes")
    predictor = _attach_mock_predictor(node, family="efficienttam")

    result = node.forward(
        _random_rgb(),
        bboxes=[
            {
                "element_id": 0,
                "object_id": 7,
                "x_min": 2,
                "y_min": 1,
                "x_max": 6,
                "y_max": 4,
            }
        ],
    )

    assert node._tracking_started is True
    assert torch.count_nonzero(result["mask"]).item() > 0
    assert int(result["object_ids"][0, 0].item()) == 7
    assert predictor.load_first_frame.call_count == 1
    assert predictor.load_first_frame.call_args.kwargs["num_classes"] == 1
    predictor.add_new_prompts.assert_called_once()


def test_score_semantics_match_positive_sigmoid_mean() -> None:
    node = RTSAM2BboxPropagation(model_type="efficienttam", name="test_score_semantics")
    node._int_to_ext = {0: 4, 1: 9}
    logits = torch.tensor(
        [
            [[[2.0, -1.0], [0.5, 3.0]]],
            [[[0.2, -2.0], [4.0, -3.0]]],
        ],
        dtype=torch.float32,
    )

    result = node._pack_output([0, 1], logits, (2, 2))
    expected_0 = float(torch.sigmoid(torch.tensor([2.0, 0.5, 3.0])).mean().item())
    expected_1 = float(torch.sigmoid(torch.tensor([0.2, 4.0])).mean().item())

    assert torch.allclose(
        result["detection_scores"],
        torch.tensor([[expected_0, expected_1]], dtype=torch.float32),
    )


def test_empty_object_filtering_removes_all_negative_logits() -> None:
    node = RTSAM2BboxPropagation(
        model_type="efficienttam",
        name="test_empty_object_filtering",
    )
    node._int_to_ext = {0: 1, 1: 2}
    logits = torch.tensor(
        [
            [[[-1.0, -2.0], [-3.0, -4.0]]],
            [[[3.0, -2.0], [-3.0, 2.0]]],
        ],
        dtype=torch.float32,
    )

    result = node._pack_output([0, 1], logits, (2, 2))

    assert result["object_ids"].tolist() == [[2]]
    assert result["detection_scores"].shape == (1, 1)


def test_label_map_overlap_uses_highest_logit_wins() -> None:
    node = RTSAM2BboxPropagation(model_type="efficienttam", name="test_overlap_resolution")
    node._int_to_ext = {0: 11, 1: 22}
    logits = torch.tensor(
        [
            [[[1.0, -1.0], [-1.0, -1.0]]],
            [[[2.0, -1.0], [-1.0, -1.0]]],
        ],
        dtype=torch.float32,
    )

    result = node._pack_output([0, 1], logits, (2, 2))

    assert int(result["mask"][0, 0, 0].item()) == 22


def test_mask_prompt_uses_direct_mask_conditioning() -> None:
    node = RTSAM2MaskPropagation(
        model_type="sam2",
        name="test_mask_prompt_uses_direct_mask_conditioning",
    )
    predictor = _attach_mock_predictor(node, family="sam2")

    prompt_mask = torch.zeros((1, 10, 12), dtype=torch.int32)
    prompt_mask[:, 2:5, 4:8] = 42

    result = node.forward(_random_rgb(), mask=prompt_mask)

    assert result["object_ids"].tolist() == [[42]]
    predictor.add_new_points_or_box.assert_not_called()
    predictor.add_new_prompts.assert_not_called()
    predictor._run_single_frame_inference.assert_called_once()
    assert predictor.condition_state["mask_prompt_calls"][0]["obj_id"] == 1
    assert predictor.condition_state["mask_prompt_calls"][0]["mask_shape"] == (1, 1, 10, 12)
    assert predictor.condition_state["mask_prompt_calls"][0]["positive_pixels"] == 12


def test_object_id_preservation_uses_dense_internal_mapping() -> None:
    node = RTSAM2BboxPropagation(
        model_type="efficienttam",
        name="test_bbox_object_id_preservation",
    )
    predictor = _attach_mock_predictor(node, family="efficienttam")

    result = node.forward(
        _random_rgb(),
        bboxes=[
            {
                "element_id": 0,
                "object_id": 42,
                "x_min": 2,
                "y_min": 1,
                "x_max": 6,
                "y_max": 4,
            }
        ],
    )

    assert result["object_ids"].tolist() == [[42]]
    assert predictor.condition_state["prompt_boxes"][0]["obj_id"] == 1


def test_mask_multi_label_seeding_preserves_external_object_ids() -> None:
    node = RTSAM2MaskPropagation(model_type="sam2", name="test_mask_multi_label")
    _attach_mock_predictor(node, family="sam2")

    prompt_mask = torch.zeros((1, 10, 12), dtype=torch.int32)
    prompt_mask[:, 1:3, 1:3] = 3
    prompt_mask[:, 4:6, 4:6] = 7
    prompt_mask[:, 7:9, 8:10] = 11

    result = node.forward(_random_rgb(), mask=prompt_mask)

    assert result["object_ids"].tolist() == [[3, 7, 11]]


def test_repo_relative_model_dir_resolves_from_repo_root(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root, config_path, checkpoint_path = _materialize_variant_layout(
        tmp_path,
        variant="efficienttam_s",
        checkpoint_dir=tmp_path / "cuvis-ai-rtsam2" / "relative_models",
    )
    _patch_model_package_root(monkeypatch, repo_root)

    node = RTSAM2BboxPropagation(
        model_type="efficienttam",
        model_dir="relative_models",
        name="test_repo_relative_model_dir",
    )

    assert node._resolve_config_path() == config_path
    assert node._resolve_checkpoint_path() == checkpoint_path
    assert node._resolve_model_assets() == (
        "configs/efficienttam/efficienttam_s.yaml",
        str(checkpoint_path),
    )


def test_absolute_model_dir_is_preserved(
    monkeypatch,
    tmp_path: Path,
) -> None:
    absolute_model_dir = tmp_path / "absolute_models"
    repo_root, _, checkpoint_path = _materialize_variant_layout(
        tmp_path,
        variant="sam2.1_hiera_t",
        checkpoint_dir=absolute_model_dir,
    )
    _patch_model_package_root(monkeypatch, repo_root)

    node = RTSAM2MaskPropagation(
        model_type="sam2",
        model_dir=str(absolute_model_dir),
        name="test_absolute_model_dir",
    )

    assert node._resolve_checkpoint_path() == checkpoint_path


def test_missing_config_error_includes_resolved_paths(
    monkeypatch,
    tmp_path: Path,
) -> None:
    custom_model_dir = tmp_path / "custom_models"
    repo_root, config_path, checkpoint_path = _materialize_variant_layout(
        tmp_path,
        variant="efficienttam_s",
        include_config=False,
        checkpoint_dir=custom_model_dir,
    )
    _patch_model_package_root(monkeypatch, repo_root)

    node = RTSAM2BboxPropagation(
        model_type="efficienttam_s",
        model_dir=str(custom_model_dir),
        name="test_missing_config_error",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        node._ensure_model()

    message = str(exc_info.value)
    assert str(config_path) in message
    assert str(checkpoint_path) in message
    assert "RTSAM2 checkout" in message


def test_missing_efficienttam_checkpoint_error_includes_download_guidance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root, _, checkpoint_path = _materialize_variant_layout(
        tmp_path,
        variant="efficienttam_s_512x512",
        include_checkpoint=False,
    )
    _patch_model_package_root(monkeypatch, repo_root)

    node = RTSAM2BboxPropagation(
        model_type="efficienttam_s_512x512",
        name="test_missing_efficienttam_checkpoint_error",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        node._ensure_model()

    message = str(exc_info.value)
    assert str(checkpoint_path) in message
    assert "download_checkpoints.sh" in message
    assert "efficienttam_s_512x512" in message


def test_missing_sam2_checkpoint_error_includes_download_guidance(
    monkeypatch,
    tmp_path: Path,
) -> None:
    repo_root, _, checkpoint_path = _materialize_variant_layout(
        tmp_path,
        variant="sam2.1_hiera_l",
        include_checkpoint=False,
    )
    _patch_model_package_root(monkeypatch, repo_root)

    node = RTSAM2MaskPropagation(
        model_type="sam2.1_hiera_l",
        name="test_missing_sam2_checkpoint_error",
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        node._ensure_model()

    message = str(exc_info.value)
    assert str(checkpoint_path) in message
    assert "facebookresearch/sam2#download-checkpoints" in message
    assert "sam2.1_hiera_large.pt" in message
