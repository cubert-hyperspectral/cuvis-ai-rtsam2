"""Shared streaming base for the RTSAM2 propagation nodes.

Underscore-prefixed module so ``NodeRegistry.auto_register_package`` skips it;
only the concrete nodes in ``rtsam2_streaming_propagation`` are registerable.

Stream lifecycle (one ``forward`` call per frame)::

    forward(rgb_image, prompt)          forward(rgb_image)
            |                                   |
            v                                   v
    load_first_frame ─> prompt(s) ─> finalize_new_input ─> track() ─> track() ─> ...
            ^                                                            |
            |               (mid-stream re-prompt raises) ───────────────┘
            |
    reset()  clears per-stream state, KEEPS the loaded predictor
    cleanup() = reset() + drop the predictor (session teardown)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec


@dataclass(frozen=True)
class _ModelSpec:
    family: str
    config_name: str
    checkpoint_name: str


_MODEL_VARIANT_REGISTRY: dict[str, _ModelSpec] = {
    "efficienttam_s": _ModelSpec(
        family="efficienttam",
        config_name="configs/efficienttam/efficienttam_s.yaml",
        checkpoint_name="efficienttam_s.pt",
    ),
    "efficienttam_s_512x512": _ModelSpec(
        family="efficienttam",
        config_name="configs/efficienttam/efficienttam_s_512x512.yaml",
        checkpoint_name="efficienttam_s_512x512.pt",
    ),
    "efficienttam_ti": _ModelSpec(
        family="efficienttam",
        config_name="configs/efficienttam/efficienttam_ti.yaml",
        checkpoint_name="efficienttam_ti.pt",
    ),
    "efficienttam_ti_512x512": _ModelSpec(
        family="efficienttam",
        config_name="configs/efficienttam/efficienttam_ti_512x512.yaml",
        checkpoint_name="efficienttam_ti_512x512.pt",
    ),
    "sam2.1_hiera_t": _ModelSpec(
        family="sam2",
        config_name="configs/sam2.1/sam2.1_hiera_t.yaml",
        checkpoint_name="sam2.1_hiera_tiny.pt",
    ),
    "sam2.1_hiera_s": _ModelSpec(
        family="sam2",
        config_name="configs/sam2.1/sam2.1_hiera_s.yaml",
        checkpoint_name="sam2.1_hiera_small.pt",
    ),
    "sam2.1_hiera_b+": _ModelSpec(
        family="sam2",
        config_name="configs/sam2.1/sam2.1_hiera_b+.yaml",
        checkpoint_name="sam2.1_hiera_base_plus.pt",
    ),
    "sam2.1_hiera_l": _ModelSpec(
        family="sam2",
        config_name="configs/sam2.1/sam2.1_hiera_l.yaml",
        checkpoint_name="sam2.1_hiera_large.pt",
    ),
}

_MODEL_ALIASES = {
    "efficienttam": "efficienttam_s",
    "sam2": "sam2.1_hiera_t",
}

_SUPPORTED_MODEL_TYPES = tuple(sorted({*_MODEL_ALIASES.keys(), *_MODEL_VARIANT_REGISTRY.keys()}))


class RTSAM2TrackerInference(Node):
    """Base class for realtime SAM2 bbox and mask propagation."""

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.VIDEO,
            NodeTag.RGB,
            NodeTag.MASK,
            NodeTag.TRACKING,
            NodeTag.SEGMENTATION,
            NodeTag.INFERENCE,
            NodeTag.LEARNABLE,
            NodeTag.BATCHED,
            NodeTag.STATEFUL,
            NodeTag.TORCH,
        }
    )

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(1, -1, -1, 3),
            description="RGB frame [1,H,W,3] in float32 with values in [0, 1].",
        ),
        "frame_id": PortSpec(
            dtype=torch.int64,
            shape=(1,),
            description="Source frame index [1]. Preserved by upstream sinks.",
            optional=True,
        ),
    }
    OUTPUT_SPECS = {
        "mask": PortSpec(dtype=torch.int32, shape=(1, -1, -1)),
        "object_ids": PortSpec(dtype=torch.int64, shape=(1, -1)),
        "detection_scores": PortSpec(dtype=torch.float32, shape=(1, -1)),
    }

    def __init__(
        self,
        model_type: str = "efficienttam",
        model_dir: str | None = None,
        **kwargs: Any,
    ) -> None:
        normalized_model_type = str(model_type).strip().lower()
        resolved_variant = _MODEL_ALIASES.get(normalized_model_type, normalized_model_type)
        model_spec = _MODEL_VARIANT_REGISTRY.get(resolved_variant)
        if model_spec is None:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Expected one of {sorted(_SUPPORTED_MODEL_TYPES)}."
            )

        self._requested_model_type = normalized_model_type
        self._model_variant = resolved_variant
        self._model_spec = model_spec
        self._model_type = model_spec.family
        self._model_dir = model_dir
        self._predictor: Any = None
        self._stream_frame_idx = 0
        self._tracking_started = False
        self._ext_to_int: dict[int, int] = {}
        self._int_to_ext: dict[int, int] = {}
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        super().__init__(
            model_type=model_type,
            model_dir=model_dir,
            **kwargs,
        )

    def to(self, *args: Any, **kwargs: Any) -> RTSAM2TrackerInference:
        """Move the node and any loaded predictor to the requested device."""
        moved = super().to(*args, **kwargs)
        device_arg = args[0] if args else kwargs.get("device")
        if device_arg is not None:
            self._device = torch.device(device_arg)
            if self._predictor is not None:
                self._predictor = self._predictor.to(self._device)
        return moved

    def reset(self) -> None:
        """Reset per-stream state so the node can process a fresh stream.

        Called automatically by ``Predictor._reset_nodes()`` at the start of
        each predict run. Keeps the loaded predictor; the next prompt frame
        goes through ``load_first_frame``, which rebuilds the camera
        predictor's tracking state from scratch.
        """
        self._stream_frame_idx = 0
        self._tracking_started = False
        self._ext_to_int = {}
        self._int_to_ext = {}

    def cleanup(self) -> None:
        """Release the loaded predictor on pipeline/session teardown."""
        self.reset()
        self._predictor = None
        self._maybe_clear_cuda_cache()

    @staticmethod
    def _frame_from_tensor(
        rgb_image: torch.Tensor,
    ) -> tuple[np.ndarray, tuple[int, int]]:
        frame = rgb_image[0].detach().cpu().to(dtype=torch.float32)
        height, width = int(frame.shape[0]), int(frame.shape[1])
        return np.asarray(frame.numpy(), dtype=np.float32), (height, width)

    @staticmethod
    def _empty_output(height: int, width: int) -> dict[str, torch.Tensor]:
        return {
            "mask": torch.zeros((1, int(height), int(width)), dtype=torch.int32),
            "object_ids": torch.zeros((1, 0), dtype=torch.int64),
            "detection_scores": torch.zeros((1, 0), dtype=torch.float32),
        }

    @staticmethod
    def _maybe_clear_cuda_cache() -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _obj_ids_to_list(obj_ids: Sequence[Any] | torch.Tensor) -> list[int]:
        if isinstance(obj_ids, torch.Tensor):
            return [int(value) for value in obj_ids.detach().cpu().reshape(-1).tolist()]
        return [int(value) for value in obj_ids]

    @classmethod
    def _model_package_root(cls, model_type: str) -> Path:
        if model_type == "sam2":
            import sam2

            return Path(sam2.__file__).resolve().parent

        import efficient_track_anything

        return Path(efficient_track_anything.__file__).resolve().parent

    @classmethod
    def _repo_root_for_model_type(cls, model_type: str) -> Path:
        return cls._model_package_root(model_type).parent

    @staticmethod
    def _config_module_for_model_type(model_type: str) -> str:
        if model_type == "sam2":
            return "sam2"
        return "efficient_track_anything"

    @classmethod
    def _activate_hydra_config_module(cls, model_type: str) -> None:
        from hydra import initialize_config_module
        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()
        initialize_config_module(
            cls._config_module_for_model_type(model_type),
            version_base="1.2",
        )

    def _resolve_config_path(self) -> Path:
        return (self._model_package_root(self._model_type) / self._model_spec.config_name).resolve(
            strict=False
        )

    def _resolve_checkpoint_path(self) -> Path:
        repo_root = self._repo_root_for_model_type(self._model_type)
        if self._model_dir is None:
            model_dir = repo_root / "checkpoints"
        else:
            model_dir = Path(self._model_dir)
            if not model_dir.is_absolute():
                model_dir = repo_root / model_dir
        return (model_dir / self._model_spec.checkpoint_name).resolve(strict=False)

    def _missing_asset_guidance(
        self,
        *,
        config_path: Path,
        checkpoint_path: Path,
    ) -> str:
        repo_root = self._repo_root_for_model_type(self._model_type)
        guidance = [
            "RTSAM2 model assets are missing.",
            (
                "Requested model_type "
                f"'{self._requested_model_type}' resolved to variant '{self._model_variant}'."
            ),
            f"Expected config file: {config_path}",
            f"Expected checkpoint file: {checkpoint_path}",
            "The config file must be present inside the RTSAM2 checkout. "
            "If it is missing, restore or reinstall the RTSAM2 repository contents.",
        ]
        if self._model_type == "efficienttam":
            guidance.append(
                "Download the EfficientTAM checkpoints into "
                f"'{(repo_root / 'checkpoints').resolve(strict=False)}', for example via "
                f"'{(repo_root / 'checkpoints' / 'download_checkpoints.sh').resolve(strict=False)}'."
            )
        else:
            guidance.append(
                "Download the SAM2.1 checkpoint from "
                "https://github.com/facebookresearch/sam2#download-checkpoints "
                f"and place it in '{checkpoint_path.parent}'."
            )
        return " ".join(guidance)

    def _resolve_model_assets(self) -> tuple[str, str]:
        config_path = self._resolve_config_path()
        checkpoint_path = self._resolve_checkpoint_path()
        if not config_path.is_file() or not checkpoint_path.is_file():
            raise FileNotFoundError(
                self._missing_asset_guidance(
                    config_path=config_path,
                    checkpoint_path=checkpoint_path,
                )
            )
        return self._model_spec.config_name, str(checkpoint_path)

    def _ensure_model(self) -> None:
        if self._predictor is not None:
            return

        config_name, resolved_checkpoint = self._resolve_model_assets()
        self._activate_hydra_config_module(self._model_type)
        device = str(self._device)

        if self._model_type == "sam2":
            from sam2.build_sam import build_sam2_camera_predictor

            self._predictor = build_sam2_camera_predictor(
                config_name,
                ckpt_path=resolved_checkpoint,
                device=device,
            )
        else:
            from efficient_track_anything.build_efficienttam import (
                build_efficienttam_camera_predictor,
            )

            self._predictor = build_efficienttam_camera_predictor(
                config_name,
                ckpt_path=resolved_checkpoint,
                device=device,
            )

    def _build_object_id_maps(self, external_object_ids: Sequence[int]) -> int:
        unique_external_ids = sorted({int(obj_id) for obj_id in external_object_ids})
        if not unique_external_ids:
            raise ValueError("RTSAM2 prompt frames must include at least one object ID.")
        if any(obj_id <= 0 for obj_id in unique_external_ids):
            raise ValueError("RTSAM2 object IDs must be positive integers.")

        self._ext_to_int = {}
        self._int_to_ext = {}
        internal_id_start = max(1, len(unique_external_ids))
        for dense_idx, external_id in enumerate(unique_external_ids):
            internal_id = internal_id_start + dense_idx
            self._ext_to_int[external_id] = internal_id
            self._int_to_ext[internal_id] = external_id
        return len(unique_external_ids)

    def _init_first_frame(self, frame_np: np.ndarray, num_classes: int) -> None:
        self._ensure_model()
        if self._predictor is None:
            raise RuntimeError("RTSAM2 predictor failed to initialize.")

        self._predictor.load_first_frame(frame_np, num_classes=int(num_classes))
        self._tracking_started = True

    def _apply_initial_prompts(
        self,
        frame_np: np.ndarray,
        num_classes: int,
        prompt_callback: Callable[[int], None],
    ) -> tuple[int, Sequence[Any] | torch.Tensor, torch.Tensor]:
        self._init_first_frame(frame_np, num_classes)
        prompt_callback(0)
        return self._predictor.finalize_new_input()

    def _track_frame(
        self,
        frame_np: np.ndarray,
    ) -> tuple[Sequence[Any] | torch.Tensor, torch.Tensor]:
        if self._predictor is None:
            raise RuntimeError("RTSAM2 predictor must be initialized before tracking.")
        return self._predictor.track(frame_np)

    def _add_box_prompt(
        self,
        *,
        frame_idx: int,
        obj_id: int,
        box_coords: Sequence[float],
    ) -> Any:
        if self._predictor is None:
            raise RuntimeError("RTSAM2 predictor must exist before adding prompts.")

        box = [float(coord) for coord in box_coords]
        if self._model_type == "sam2":
            return self._predictor.add_new_points_or_box(
                frame_idx=int(frame_idx),
                obj_id=int(obj_id),
                points=None,
                labels=None,
                box=box,
                normalize_coords=True,
            )
        return self._predictor.add_new_prompts(
            frame_idx=int(frame_idx),
            obj_id=int(obj_id),
            points=None,
            labels=None,
            boxes=box,
            normalize_coords=True,
        )

    @staticmethod
    def _mask_prompt_tensor(label_map: np.ndarray, obj_id: int) -> torch.Tensor:
        binary_mask = torch.from_numpy(np.asarray(label_map == int(obj_id), dtype=np.float32))
        return binary_mask.unsqueeze(0).unsqueeze(0)

    def _add_mask_prompt(
        self,
        *,
        frame_idx: int,
        obj_id: int,
        mask_object_id: int,
        label_map: np.ndarray,
    ) -> None:
        if self._predictor is None:
            raise RuntimeError("RTSAM2 predictor must exist before adding prompts.")

        mask_inputs = self._mask_prompt_tensor(label_map, mask_object_id)
        if not bool(mask_inputs.any().item()):
            raise ValueError(
                f"Mask prompt for object_id={mask_object_id} is empty on frame {frame_idx}."
            )

        if hasattr(self._predictor, "object_of_interest"):
            self._predictor.object_of_interest = int(obj_id)

        condition_state = self._predictor.condition_state
        obj_idx = self._predictor._obj_id_to_idx(int(obj_id))
        point_inputs_per_frame = condition_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = condition_state["mask_inputs_per_obj"][obj_idx]

        point_inputs_per_frame.pop(int(frame_idx), None)
        mask_inputs = mask_inputs.to(
            device=condition_state["device"],
            dtype=torch.float32,
        )
        mask_inputs_per_frame[int(frame_idx)] = mask_inputs

        is_init_cond_frame = int(frame_idx) not in condition_state["frames_already_tracked"]
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = bool(condition_state["frames_already_tracked"][int(frame_idx)]["reverse"])

        obj_output_dict = condition_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = condition_state["temp_output_dict_per_obj"][obj_idx]
        is_cond = is_init_cond_frame
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        current_out, _ = self._predictor._run_single_frame_inference(
            output_dict=obj_output_dict,
            frame_idx=int(frame_idx),
            batch_size=1,
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=None,
            mask_inputs=mask_inputs,
            reverse=reverse,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
            new_input=True,
        )
        obj_temp_output_dict[storage_key][int(frame_idx)] = current_out

    def _pack_output(
        self,
        obj_ids: Sequence[Any] | torch.Tensor,
        mask_logits: torch.Tensor | np.ndarray,
        frame_shape: tuple[int, int],
    ) -> dict[str, torch.Tensor]:
        internal_obj_ids = self._obj_ids_to_list(obj_ids)
        selected_indices = [
            idx for idx, obj_id in enumerate(internal_obj_ids) if int(obj_id) in self._int_to_ext
        ]
        if len(selected_indices) == 0:
            return self._empty_output(*frame_shape)
        external_obj_ids = [
            int(self._int_to_ext[int(internal_obj_ids[idx])]) for idx in selected_indices
        ]

        if not isinstance(mask_logits, torch.Tensor):
            mask_logits = torch.as_tensor(mask_logits, dtype=torch.float32)
        if mask_logits.ndim == 3:
            mask_logits = mask_logits.unsqueeze(1)
        if mask_logits.ndim != 4 or int(mask_logits.shape[1]) != 1:
            raise ValueError(
                "mask_logits must have shape [N, 1, H, W], "
                f"got {tuple(int(v) for v in mask_logits.shape)}."
            )

        height, width = frame_shape
        all_logits = mask_logits[selected_indices, 0]
        binary_masks = all_logits > 0.0

        label_map = np.zeros((height, width), dtype=np.int32)
        if bool(binary_masks.any().item()):
            masked_logits = all_logits.clone()
            masked_logits[~binary_masks] = float("-inf")
            winner_idx = masked_logits.argmax(dim=0).detach().cpu().numpy()
            any_positive = binary_masks.any(dim=0).detach().cpu().numpy()
            for idx, obj_id in enumerate(external_obj_ids):
                label_map[(winner_idx == idx) & any_positive] = int(obj_id)

        kept_ids: list[int] = []
        scores: list[float] = []
        for idx, obj_id in enumerate(external_obj_ids):
            positive_logits = all_logits[idx][binary_masks[idx]]
            if positive_logits.numel() == 0:
                continue
            kept_ids.append(int(obj_id))
            scores.append(float(torch.sigmoid(positive_logits).mean().item()))

        return {
            "mask": torch.from_numpy(label_map).unsqueeze(0),
            "object_ids": torch.tensor([kept_ids], dtype=torch.int64),
            "detection_scores": torch.tensor([scores], dtype=torch.float32),
        }

    def _forward_stream(
        self,
        rgb_image: torch.Tensor,
        frame_id: torch.Tensor | None,  # noqa: ARG002 -- carried by sinks, unused here
        *,
        has_prompt: bool,
        external_object_ids: Sequence[int],
        prompt_callback: Callable[[int], None],
    ) -> dict[str, torch.Tensor]:
        frame_float_hwc, frame_shape = self._frame_from_tensor(rgb_image)

        if not self._tracking_started:
            if not has_prompt:
                self._stream_frame_idx += 1
                return self._empty_output(*frame_shape)

            num_classes = self._build_object_id_maps(external_object_ids)
            _, obj_ids, video_res_masks = self._apply_initial_prompts(
                frame_float_hwc,
                num_classes,
                prompt_callback,
            )
            result = self._pack_output(obj_ids, video_res_masks, frame_shape)
            self._stream_frame_idx += 1
            self._maybe_clear_cuda_cache()
            return result

        if has_prompt:
            raise ValueError(
                "RTSAM2 only accepts prompts on the first prompt frame. "
                "Mid-stream re-prompting is not supported."
            )

        obj_ids, video_res_masks = self._track_frame(frame_float_hwc)
        result = self._pack_output(obj_ids, video_res_masks, frame_shape)
        self._stream_frame_idx += 1
        self._maybe_clear_cuda_cache()
        return result
