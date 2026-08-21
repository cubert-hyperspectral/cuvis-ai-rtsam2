"""Tests for RTSAM2PointExpansion with mocked camera predictors plus real-weights smokes."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from _mock_predictors import (
    NO_OBJ_SCORE,
    _attach_mock_predictor,
    _make_logits_for_ids,
    _materialize_variant_layout,
    _patch_model_package_root,
    _random_rgb,
)
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from loguru import logger

from cuvis_ai_rtsam2.node import RTSAM2PointExpansion, RTSAM2TrackerInference

pytestmark = pytest.mark.unit


def _positive_point(x: float = 2.0, y: float = 2.0) -> dict[str, float | int | str]:
    return {"element_id": 0, "x": x, "y": y, "type": "positive"}


def _negative_point(x: float = 8.0, y: float = 8.0) -> dict[str, float | int | str]:
    return {"element_id": 0, "x": x, "y": y, "type": "negative"}


@pytest.fixture
def warning_messages():
    messages: list[str] = []
    sink_id = logger.add(lambda message: messages.append(str(message)), level="WARNING")
    yield messages
    logger.remove(sink_id)


# --- port contract -------------------------------------------------------


def test_port_contract_matches_sam3_point_expansion() -> None:
    assert set(RTSAM2PointExpansion.INPUT_SPECS) == {"rgb_image", "points", "frame_id"}
    assert RTSAM2PointExpansion.INPUT_SPECS["points"].optional
    assert RTSAM2PointExpansion.INPUT_SPECS["frame_id"].optional
    assert not RTSAM2PointExpansion.INPUT_SPECS["rgb_image"].optional
    # Outputs are inherited from the tracker base unchanged.
    assert RTSAM2PointExpansion.OUTPUT_SPECS is RTSAM2TrackerInference.OUTPUT_SPECS
    assert set(RTSAM2PointExpansion.OUTPUT_SPECS) == {"mask", "object_ids", "detection_scores"}


def test_palette_metadata_is_single_frame_not_streaming() -> None:
    assert RTSAM2PointExpansion._category is NodeCategory.MODEL
    assert NodeTag.KEYPOINTS in RTSAM2PointExpansion._tags
    assert NodeTag.IMAGE in RTSAM2PointExpansion._tags
    for streaming_tag in (NodeTag.VIDEO, NodeTag.STATEFUL, NodeTag.TRACKING):
        assert streaming_tag not in RTSAM2PointExpansion._tags


# --- constructor validation ----------------------------------------------


@pytest.mark.parametrize("bad_id", [0, -1, -7])
def test_ctor_rejects_non_positive_prompt_obj_id(bad_id: int) -> None:
    with pytest.raises(ValueError, match="prompt_obj_id must be > 0"):
        RTSAM2PointExpansion(prompt_obj_id=bad_id, name="test_bad_id")


@pytest.mark.parametrize("bad_id", [1.5, "2", True, None])
def test_ctor_rejects_non_integer_prompt_obj_id(bad_id: object) -> None:
    with pytest.raises(ValueError, match="prompt_obj_id must be an integer"):
        RTSAM2PointExpansion(prompt_obj_id=bad_id, name="test_nonint_id")


def test_ctor_rejects_unknown_model_type() -> None:
    with pytest.raises(ValueError, match="Unsupported model_type"):
        RTSAM2PointExpansion(model_type="sam9000", name="test_bad_model")


@pytest.mark.parametrize(
    ("alias", "variant"),
    [("efficienttam", "efficienttam_s"), ("sam2", "sam2.1_hiera_t")],
)
def test_model_type_alias_resolution(alias: str, variant: str) -> None:
    node = RTSAM2PointExpansion(model_type=alias, name=f"test_alias_{variant}")
    assert node._model_variant == variant


# --- empty paths: model never built --------------------------------------


@pytest.mark.parametrize(
    "points",
    [
        None,
        [],
        [{"element_id": 0, "x": 2, "y": 2, "type": "neutral"}],
        [_negative_point()],
    ],
    ids=["none", "empty-list", "neutral-only", "negative-only"],
)
def test_no_positive_point_returns_empty_without_model(points: list | None) -> None:
    node = RTSAM2PointExpansion(name="test_empty_paths")
    predictor = _attach_mock_predictor(node, family="efficienttam")

    result = node.forward(_random_rgb(), points=points)

    node._ensure_model.assert_not_called()
    predictor.load_first_frame.assert_not_called()
    assert result["mask"].shape == (1, 10, 12)
    assert int(result["mask"].sum()) == 0
    assert result["object_ids"].shape == (1, 0)
    assert result["detection_scores"].shape == (1, 0)


def test_empty_forward_needs_no_checkpoint_on_disk() -> None:
    # A bare node (no mock, no weights anywhere) must serve the empty path.
    node = RTSAM2PointExpansion(name="test_empty_bare")
    result = node.forward(_random_rgb(), points=[])
    assert int(result["mask"].sum()) == 0
    assert node._predictor is None


# --- point parsing -------------------------------------------------------


def test_malformed_points_raise_even_without_positive_point() -> None:
    node = RTSAM2PointExpansion(name="test_parse_before_guard")
    predictor = _attach_mock_predictor(node, family="efficienttam")

    with pytest.raises(ValueError, match="missing 'x' or 'y'"):
        node.forward(_random_rgb(), points=[{"element_id": 0, "type": "negative", "x": 1}])
    with pytest.raises(ValueError, match="to be a dict"):
        node.forward(_random_rgb(), points=["not-a-dict"])
    with pytest.raises(ValueError, match="unknown type"):
        node.forward(_random_rgb(), points=[{"element_id": 0, "x": 1, "y": 1, "type": "maybe"}])
    with pytest.raises(ValueError, match="to be a list"):
        node.forward(_random_rgb(), points={"x": 1, "y": 1})
    node._ensure_model.assert_not_called()
    predictor.load_first_frame.assert_not_called()


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_coordinates_raise(bad_value: float) -> None:
    node = RTSAM2PointExpansion(name="test_nonfinite")
    _attach_mock_predictor(node, family="efficienttam")

    with pytest.raises(ValueError, match="non-finite coordinates"):
        node.forward(_random_rgb(), points=[_positive_point(x=bad_value)])


def test_out_of_bounds_points_clamp_to_pixel_index_range(warning_messages) -> None:
    node = RTSAM2PointExpansion(name="test_clamp")
    predictor = _attach_mock_predictor(node, family="efficienttam")

    # Frame is 10x12: valid pixel indices are x in [0, 11], y in [0, 9].
    node.forward(
        _random_rgb(),
        points=[_positive_point(x=12.0, y=-3.0)],
    )

    recorded = predictor.condition_state["prompt_points"][-1]
    assert recorded["points"].tolist() == [[11.0, 0.0]]
    assert len(warning_messages) == 1
    assert "clamped" in warning_messages[0]
    assert "12.0" in warning_messages[0]
    assert "12x10 frame" in warning_messages[0]


def test_clamp_warns_once_per_instance_and_survives_reset(warning_messages) -> None:
    node = RTSAM2PointExpansion(name="test_warn_once")
    _attach_mock_predictor(node, family="efficienttam")

    node.forward(_random_rgb(), points=[_positive_point(x=999.0)])
    node.forward(_random_rgb(), points=[_positive_point(y=999.0)])
    assert len(warning_messages) == 1

    node.reset()
    node.forward(_random_rgb(), points=[_positive_point(x=-1.0)])
    assert len(warning_messages) == 1

    fresh = RTSAM2PointExpansion(name="test_warn_once_fresh")
    _attach_mock_predictor(fresh, family="efficienttam")
    fresh.forward(_random_rgb(), points=[_positive_point(x=-1.0)])
    assert len(warning_messages) == 2


def test_in_bounds_points_do_not_warn(warning_messages) -> None:
    node = RTSAM2PointExpansion(name="test_no_warn")
    _attach_mock_predictor(node, family="efficienttam")

    node.forward(_random_rgb(), points=[_positive_point(x=11.0, y=9.0)])
    assert warning_messages == []


# --- clicked path through the predictor ----------------------------------


@pytest.mark.parametrize("family", ["sam2", "efficienttam"])
def test_click_calls_family_prompt_api_and_never_tracks(family: str) -> None:
    node = RTSAM2PointExpansion(model_type=family, name=f"test_click_{family}")
    predictor = _attach_mock_predictor(node, family=family)

    points = [_positive_point(x=3.0, y=2.0), _negative_point(x=9.0, y=7.0)]
    result = node.forward(_random_rgb(), points=points)

    assert predictor.load_first_frame.call_count == 1
    assert predictor.load_first_frame.call_args.kwargs["num_classes"] == 1
    predictor.finalize_new_input.assert_not_called()
    predictor.track.assert_not_called()

    if family == "sam2":
        predictor.add_new_points_or_box.assert_called_once()
        kwargs = predictor.add_new_points_or_box.call_args.kwargs
        assert kwargs["box"] is None
        predictor.add_new_prompts.assert_not_called()
    else:
        predictor.add_new_prompts.assert_called_once()
        kwargs = predictor.add_new_prompts.call_args.kwargs
        assert "boxes" not in kwargs
        predictor.add_new_points_or_box.assert_not_called()

    assert kwargs["frame_idx"] == 0
    assert kwargs["obj_id"] == 1  # internal id for the single external object
    assert kwargs["normalize_coords"] is True
    assert kwargs["points"].tolist() == [[3.0, 2.0], [9.0, 7.0]]
    assert kwargs["labels"].tolist() == [1, 0]  # positive=1, negative=0, input order

    assert int(result["mask"].max()) == 1
    assert result["object_ids"].tolist() == [[1]]


def test_output_label_is_prompt_obj_id() -> None:
    node = RTSAM2PointExpansion(prompt_obj_id=7, name="test_label_seven")
    _attach_mock_predictor(node, family="efficienttam")

    result = node.forward(_random_rgb(), points=[_positive_point()])

    labels = set(np.unique(result["mask"].numpy()).tolist())
    assert labels == {0, 7}
    assert result["object_ids"].tolist() == [[7]]
    assert result["detection_scores"].shape == (1, 1)
    score = float(result["detection_scores"][0, 0])
    # Mock real channel emits constant positive logits of 4.0.
    assert score == pytest.approx(float(torch.sigmoid(torch.tensor(4.0))))


def test_each_click_reseeds_the_frame() -> None:
    node = RTSAM2PointExpansion(name="test_reseed")
    predictor = _attach_mock_predictor(node, family="efficienttam")

    node.forward(_random_rgb(), points=[_positive_point()])
    node.forward(_random_rgb(), points=[_positive_point(), _negative_point()])

    assert predictor.load_first_frame.call_count == 2
    assert predictor.add_new_prompts.call_count == 2


def test_same_points_yield_identical_mask() -> None:
    node = RTSAM2PointExpansion(name="test_deterministic")
    _attach_mock_predictor(node, family="efficienttam")

    frame = _random_rgb()
    points = [_positive_point(), _negative_point()]
    first = node.forward(frame, points=points)
    second = node.forward(frame, points=points)

    assert torch.equal(first["mask"], second["mask"])
    assert torch.equal(first["detection_scores"], second["detection_scores"])


def test_efficienttam_dummy_channel_is_filtered_and_inert() -> None:
    node = RTSAM2PointExpansion(name="test_dummy_filtered")
    predictor = _attach_mock_predictor(node, family="efficienttam")

    result = node.forward(_random_rgb(), points=[_positive_point()])

    # The fake registers upstream's dummy object 0 during load_first_frame ...
    assert 0 in predictor.condition_state["obj_ids"]
    assert 0 in predictor._dummy_obj_ids
    # ... and its NO_OBJ_SCORE channel neither appears in the outputs nor
    # perturbs the real mask: the label map equals the single-object geometry.
    expected = (_make_logits_for_ids([1], frame_idx=0)[0, 0] > 0).numpy()
    assert NO_OBJ_SCORE < 0
    np.testing.assert_array_equal(result["mask"][0].numpy() == 1, expected)
    assert result["object_ids"].tolist() == [[1]]


def test_frame_id_is_accepted_and_unused() -> None:
    node = RTSAM2PointExpansion(name="test_frame_id")
    predictor = _attach_mock_predictor(node, family="efficienttam")

    first = node.forward(
        _random_rgb(), points=[_positive_point()], frame_id=torch.tensor([3], dtype=torch.int64)
    )
    second = node.forward(
        _random_rgb(), points=[_positive_point()], frame_id=torch.tensor([9], dtype=torch.int64)
    )

    # Different frame ids change nothing: every click re-seeds regardless.
    assert predictor.load_first_frame.call_count == 2
    assert torch.equal(first["mask"], second["mask"])


# --- lifecycle -----------------------------------------------------------


def test_reset_keeps_the_predictor() -> None:
    node = RTSAM2PointExpansion(name="test_reset_keeps")
    predictor = _attach_mock_predictor(node, family="efficienttam")

    node.forward(_random_rgb(), points=[_positive_point()])
    node.reset()

    assert node._predictor is predictor
    assert node._ext_to_int == {}
    assert node._int_to_ext == {}

    result = node.forward(_random_rgb(), points=[_positive_point()])
    assert node._predictor is predictor
    assert result["object_ids"].tolist() == [[1]]


def test_cleanup_drops_the_predictor() -> None:
    node = RTSAM2PointExpansion(name="test_cleanup_drops")
    _attach_mock_predictor(node, family="efficienttam")

    node.forward(_random_rgb(), points=[_positive_point()])
    node.cleanup()

    assert node._predictor is None


def test_to_moves_node_and_keeps_predictor() -> None:
    node = RTSAM2PointExpansion(name="test_to_moves")
    predictor = _attach_mock_predictor(node, family="efficienttam")

    node.to("cpu")

    assert node._device == torch.device("cpu")
    assert node._predictor is predictor  # mock .to returns self


# --- asset resolution ----------------------------------------------------


def test_missing_checkpoint_error_includes_download_guidance(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root, _, checkpoint_path = _materialize_variant_layout(
        tmp_path,
        variant="efficienttam_s",
        include_checkpoint=False,
    )
    _patch_model_package_root(monkeypatch, repo_root)
    # No weight provisioned in the shared cache either -> the guidance must fire.
    monkeypatch.setattr("huggingface_hub.try_to_load_from_cache", lambda *a, **k: None)

    node = RTSAM2PointExpansion(name="test_missing_ckpt")
    with pytest.raises(FileNotFoundError) as exc_info:
        node.forward(_random_rgb(), points=[_positive_point()])

    message = str(exc_info.value)
    assert "download-model efficienttam_s" in message
    assert str(checkpoint_path) in message


# --- real-weights smokes (T6: required release gate on this box) ----------


def _efficienttam_s_resolvable() -> bool:
    probe = RTSAM2PointExpansion(name="_smoke_probe")
    try:
        probe._resolve_model_assets()
    except FileNotFoundError:
        return False
    return True


def _textured_disc_frame(
    height: int = 240,
    width: int = 320,
    center: tuple[int, int] = (120, 160),
    radius: int = 70,
    seed: int = 0,
) -> tuple[torch.Tensor, np.ndarray]:
    """Dark noisy background with a bright, textured disc.

    A flat bright disc fails point expansion (the multimask head picks a
    fragment); texture is what makes the disc segment as one object.
    """
    rng = np.random.default_rng(seed)
    frame = rng.uniform(0.02, 0.18, size=(height, width, 3)).astype(np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    disc = (yy - center[0]) ** 2 + (xx - center[1]) ** 2 <= radius**2
    texture = rng.uniform(0.55, 0.95, size=(height, width, 3)).astype(np.float32)
    texture[..., 0] *= (0.6 + 0.4 * np.sin(xx / 7.0)).astype(np.float32)
    texture[..., 1] *= (0.6 + 0.4 * np.cos(yy / 9.0)).astype(np.float32)
    frame[disc] = texture[disc]
    return torch.from_numpy(frame).unsqueeze(0), disc


def _mask_iou(mask: np.ndarray, reference: np.ndarray) -> float:
    intersection = float(np.logical_and(mask, reference).sum())
    union = float(np.logical_or(mask, reference).sum())
    return intersection / union if union else 0.0


@pytest.mark.slow
@pytest.mark.skipif(
    not _efficienttam_s_resolvable(),
    reason="efficienttam_s checkpoint not resolvable (run 'download-model efficienttam_s')",
)
class TestRealWeightsSmoke:
    """End-to-end clicks on real EfficientTAM-S weights.

    Runs on whatever device is visible: execute once on CUDA and once with
    CUDA_VISIBLE_DEVICES=-1 (the CPU run is the end-to-end proof of the
    vendored derived-device patch).
    """

    @pytest.fixture(scope="class")
    def node(self):
        node = RTSAM2PointExpansion(name="smoke_point_expansion")
        yield node
        node.cleanup()

    def test_positive_click_covers_the_disc(self, node) -> None:
        frame, disc = _textured_disc_frame()
        result = node.forward(frame, points=[_positive_point(x=160.0, y=120.0)])
        mask = result["mask"][0].numpy() == 1
        assert _mask_iou(mask, disc) >= 0.8
        assert result["object_ids"].tolist() == [[1]]

    def test_negative_click_shrinks_the_mask(self, node) -> None:
        frame, disc = _textured_disc_frame()
        positive = _positive_point(x=130.0, y=120.0)
        pos_only = node.forward(frame, points=[positive])["mask"][0].numpy() == 1
        result = node.forward(frame, points=[positive, _negative_point(x=225.0, y=120.0)])
        mask = result["mask"][0].numpy() == 1
        # Clicks are soft guidance to the decoder: a negative click inside a
        # coherent object pulls the boundary away from it rather than carving
        # out the exact pixel, so assert measurable shrinkage on that side.
        assert mask[120, 130]
        assert float(mask.sum()) <= 0.98 * float(pos_only.sum())
        assert _mask_iou(mask, disc) >= 0.5

    def test_identical_point_set_is_deterministic(self, node) -> None:
        frame, _ = _textured_disc_frame()
        points = [_positive_point(x=160.0, y=120.0), _negative_point(x=185.0, y=120.0)]
        first = node.forward(frame, points=points)
        second = node.forward(frame, points=points)
        assert torch.equal(first["mask"], second["mask"])

    def test_reset_then_reclick_is_identical(self, node) -> None:
        frame, _ = _textured_disc_frame()
        points = [_positive_point(x=160.0, y=120.0)]
        before = node.forward(frame, points=points)
        predictor = node._predictor
        node.reset()
        assert node._predictor is predictor
        after = node.forward(frame, points=points)
        assert torch.equal(before["mask"], after["mask"])

    def test_frame_switch_and_return_stays_correct(self, node) -> None:
        frame_a, disc_a = _textured_disc_frame(center=(120, 160), seed=0)
        frame_b, disc_b = _textured_disc_frame(center=(80, 100), seed=1)

        first_a = node.forward(frame_a, points=[_positive_point(x=160.0, y=120.0)])
        result_b = node.forward(frame_b, points=[_positive_point(x=100.0, y=80.0)])
        second_a = node.forward(frame_a, points=[_positive_point(x=160.0, y=120.0)])

        assert _mask_iou(result_b["mask"][0].numpy() == 1, disc_b) >= 0.8
        assert _mask_iou(second_a["mask"][0].numpy() == 1, disc_a) >= 0.8
        assert torch.equal(first_a["mask"], second_a["mask"])
