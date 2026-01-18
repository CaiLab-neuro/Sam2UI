"""
Video Segmentation Module
=========================

Shared video segmentation functionality for both UI and CLI usage.
Provides a unified VideoSegmenter class that wraps SAM2/SAM3 models.

This module extracts common segmentation logic from sam2_ui.py and
process_annotations.py to reduce code duplication and provide a
consistent interface for video segmentation.
"""

import os
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, Union
import cv2
import numpy as np

try:
    import torch
except ImportError:
    torch = None

# Import quality metrics calculator from utils
from utils import IncrementalQualityMetricsCalculator, save_quality_metrics


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class PointAnnotation:
    """
    Unified annotation format for point-based segmentation prompts.

    Supports conversion from both UI tuple format and JSON dict format.
    """
    x: float
    y: float
    is_positive: bool
    object_id: int
    frame_idx: int
    object_name: Optional[str] = None

    @classmethod
    def from_ui_tuple(cls, data: Tuple[float, float, bool, int, int],
                      object_name: Optional[str] = None) -> 'PointAnnotation':
        """
        Convert from UI format: (x, y, is_positive, obj_id, frame_idx)

        Args:
            data: Tuple of (x, y, is_positive, obj_id, frame_idx)
            object_name: Optional name for the object

        Returns:
            PointAnnotation instance
        """
        x, y, is_pos, obj_id, frame_idx = data
        return cls(
            x=float(x),
            y=float(y),
            is_positive=bool(is_pos),
            object_id=int(obj_id),
            frame_idx=int(frame_idx),
            object_name=object_name
        )

    @classmethod
    def from_json_dict(cls, data: Dict[str, Any]) -> 'PointAnnotation':
        """
        Convert from JSON format: {"x", "y", "is_positive", "object_id", "frame_index"}

        Args:
            data: Dictionary with annotation data

        Returns:
            PointAnnotation instance
        """
        return cls(
            x=float(data["x"]),
            y=float(data["y"]),
            is_positive=bool(data["is_positive"]),
            object_id=int(data["object_id"]),
            frame_idx=int(data["frame_index"]),
            object_name=data.get("object_name")
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for JSON serialization."""
        return {
            "x": self.x,
            "y": self.y,
            "is_positive": self.is_positive,
            "object_id": self.object_id,
            "frame_index": self.frame_idx,
            "object_name": self.object_name
        }


@dataclass
class SegmentationConfig:
    """
    Configuration for video segmentation.

    Attributes:
        output_dir: Directory to save segmentation results
        masks_subdir: Subdirectory name for mask files (default: "masks")
        frame_dir: Persistent frame directory (None = use temp directory)
        frame_range: Optional (start, end) for partial video processing
        frame_offset: Offset for global frame numbering in output filenames
        enable_backward_propagation: Whether to propagate masks backward
        frames_to_keep: Number of frames to keep in memory during propagation
        offload_video_to_cpu: Offload video frames to CPU to save GPU memory
        offload_state_to_cpu: Offload model state to CPU when not in use
        calculate_quality_metrics: Whether to calculate quality metrics
        cleanup_temp_frames: Whether to cleanup temporary frame directory
        frame_format: Format for extracted frames ("jpg" for lossy/smaller, "png" for lossless)
    """
    output_dir: str
    masks_subdir: str = "masks"
    frame_dir: Optional[str] = None
    frame_range: Optional[Tuple[int, int]] = None
    frame_offset: int = 0
    enable_backward_propagation: bool = True
    frames_to_keep: int = 20
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True
    calculate_quality_metrics: bool = True
    cleanup_temp_frames: bool = True
    frame_format: str = "jpg"  # "jpg" (default, smaller) or "png" (lossless)

    @property
    def masks_output_dir(self) -> str:
        """Full path to masks output directory."""
        return os.path.join(self.output_dir, self.masks_subdir)


@dataclass
class SegmentationResult:
    """
    Results from video segmentation.

    Attributes:
        masks_metadata: Dict mapping frame_idx -> {obj_id -> mask_metadata}
        object_names: Dict mapping object_id -> name string
        object_colors: Dict mapping object_id -> (R, G, B) tuple
        num_frames_processed: Total number of frames processed
        frame_dimensions: (height, width) of processed frames
        quality_metrics: Optional tuple of (inter_frame_changes, background_ratios)
        inference_state: Optional reference to SAM inference state (for UI reuse)
    """
    masks_metadata: Dict[int, Dict[int, Any]]
    object_names: Dict[int, str]
    object_colors: Dict[int, Tuple[int, int, int]]
    num_frames_processed: int
    frame_dimensions: Tuple[int, int]
    quality_metrics: Optional[Tuple[List[float], List[float]]] = None
    inference_state: Optional[Any] = None


# =============================================================================
# Progress Callback Protocol
# =============================================================================

class ProgressCallback(Protocol):
    """
    Protocol for progress reporting during segmentation.

    Implementations can update UI progress bars, print to console, etc.
    """

    def on_progress(self, phase: str, current: int, total: int, message: str) -> None:
        """
        Report progress within a phase.

        Args:
            phase: Current phase name (e.g., "extracting", "forward", "backward")
            current: Current step number
            total: Total steps in this phase
            message: Human-readable progress message
        """
        ...

    def on_phase_start(self, phase: str, total_steps: int) -> None:
        """
        Called when a new phase begins.

        Args:
            phase: Phase name starting
            total_steps: Expected total steps in this phase
        """
        ...

    def on_phase_complete(self, phase: str) -> None:
        """
        Called when a phase completes.

        Args:
            phase: Phase name that completed
        """
        ...


class NullProgressCallback:
    """No-op progress callback for silent operation."""

    def on_progress(self, phase: str, current: int, total: int, message: str) -> None:
        pass

    def on_phase_start(self, phase: str, total_steps: int) -> None:
        pass

    def on_phase_complete(self, phase: str) -> None:
        pass


# =============================================================================
# VideoSegmenter Class
# =============================================================================

class VideoSegmenter:
    """
    Video segmentation engine using SAM2/SAM3 models.

    Provides a unified interface for both UI and CLI segmentation workflows.
    Handles frame extraction, annotation processing, mask propagation, and
    streaming mask export.

    Usage:
        # Load predictor externally (allows caller to control model loading)
        predictor = build_sam2_video_predictor(config, checkpoint, device="cuda")

        # Create segmenter
        segmenter = VideoSegmenter(predictor, device="cuda", use_bfloat16=True)

        # Configure progress reporting (optional)
        segmenter.set_progress_callback(my_callback)

        # Run segmentation
        result = segmenter.segment(
            video_path="video.mp4",
            annotations=[...],
            object_names={1: "Person"},
            object_colors={1: (255, 0, 0)},
            config=SegmentationConfig(output_dir="./output")
        )
    """

    def __init__(
        self,
        predictor: Any,
        device: str = "cuda",
        use_bfloat16: bool = False
    ):
        """
        Initialize the segmenter with a pre-loaded predictor.

        Args:
            predictor: SAM2/SAM3 video predictor (already loaded)
            device: Device string (e.g., "cuda", "cuda:0", "cpu")
            use_bfloat16: Whether to use bfloat16 precision
        """
        self.predictor = predictor
        self.device = device
        self.use_bfloat16 = use_bfloat16
        self._progress_callback: ProgressCallback = NullProgressCallback()
        self._inference_state: Optional[Any] = None

        # Detect if this is a SAM3 model
        self._is_sam3 = self._detect_sam3()

    def _detect_sam3(self) -> bool:
        """Detect if the predictor is a SAM3 model."""
        if hasattr(self.predictor, '__class__'):
            class_name = self.predictor.__class__.__name__
            return 'Sam3' in class_name or 'SAM3' in class_name
        return False

    @property
    def is_sam3(self) -> bool:
        """Whether the predictor is a SAM3 model."""
        return self._is_sam3

    @property
    def inference_state(self) -> Optional[Any]:
        """Current inference state (available after segmentation)."""
        return self._inference_state

    def set_progress_callback(self, callback: ProgressCallback) -> None:
        """Set the progress callback for reporting."""
        self._progress_callback = callback

    def reset(self) -> None:
        """Reset the segmenter state."""
        if self._inference_state is not None:
            try:
                self.predictor.reset_state(self._inference_state)
            except Exception:
                pass
            self._inference_state = None

    # -------------------------------------------------------------------------
    # Frame Extraction
    # -------------------------------------------------------------------------

    def _extract_frames(
        self,
        video_path: str,
        output_dir: str,
        frame_range: Optional[Tuple[int, int]] = None,
        frame_format: str = "jpg"
    ) -> int:
        """
        Extract frames from video to image files.

        Args:
            video_path: Path to input video
            output_dir: Directory to save extracted frames
            frame_range: Optional (start, end) for partial extraction
            frame_format: Format for extracted frames ("jpg" or "png")

        Returns:
            Number of frames extracted
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video file: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # Determine frame range
        start_frame = 0
        end_frame = total_frames
        if frame_range:
            start_frame, end_frame = frame_range
            start_frame = max(0, start_frame)
            end_frame = min(total_frames, end_frame)

        num_frames = end_frame - start_frame
        self._progress_callback.on_phase_start("extracting", num_frames)

        os.makedirs(output_dir, exist_ok=True)

        # Seek to start frame if needed
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Validate frame format
        if frame_format not in ("jpg", "png"):
            print(f"WARNING: Unknown frame format '{frame_format}', defaulting to 'jpg'")
            frame_format = "jpg"

        extracted = 0
        for frame_num in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame with sequential naming (SAM expects 00000.jpg, 00001.jpg, ...)
            frame_path = os.path.join(output_dir, f"{extracted:05d}.{frame_format}")
            if frame_format == "png":
                # Use fast PNG compression (level 1 is fast, level 9 is slow but smaller)
                cv2.imwrite(frame_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
            else:
                # Default JPEG quality
                cv2.imwrite(frame_path, frame)
            extracted += 1

            if extracted % 100 == 0:
                self._progress_callback.on_progress(
                    "extracting", extracted, num_frames,
                    f"Extracted {extracted}/{num_frames} frames"
                )

        cap.release()
        self._progress_callback.on_phase_complete("extracting")

        return extracted

    # -------------------------------------------------------------------------
    # Inference State Management
    # -------------------------------------------------------------------------

    def _init_inference_state(
        self,
        frame_dir: str,
        offload_video_to_cpu: bool = True,
        offload_state_to_cpu: bool = True
    ) -> Any:
        """
        Initialize SAM inference state with the frame directory.

        Args:
            frame_dir: Directory containing extracted JPEG frames
            offload_video_to_cpu: Offload video to CPU for memory savings
            offload_state_to_cpu: Offload model state to CPU when not in use

        Returns:
            SAM inference state object
        """
        init_params = {'video_path': frame_dir}

        if offload_video_to_cpu:
            init_params['offload_video_to_cpu'] = True
        if offload_state_to_cpu:
            init_params['offload_state_to_cpu'] = True

        # Set CUDA device if using specific GPU
        original_device = None
        if self.device.startswith("cuda:") and torch.cuda.is_available():
            gpu_id = int(self.device.split(":")[1])
            original_device = torch.cuda.current_device()
            torch.cuda.set_device(gpu_id)

        try:
            self._inference_state = self.predictor.init_state(**init_params)
        finally:
            # Restore original CUDA device
            if original_device is not None:
                try:
                    torch.cuda.set_device(original_device)
                except Exception:
                    pass

        return self._inference_state

    def _cleanup_inference_state(
        self,
        inference_state: Any,
        current_frame_idx: int,
        frames_to_keep: int = 20,
        reverse: bool = False
    ) -> None:
        """
        Clean up old frames from inference state to prevent memory growth.

        Handles both SAM2 and SAM3 inference state structures.

        Args:
            inference_state: SAM inference state object
            current_frame_idx: Current frame being processed
            frames_to_keep: Number of recent frames to keep
            reverse: If True, propagating backward (delete frames ahead)
        """
        # Direction-aware cleanup logic
        if reverse:
            should_delete = lambda f: f > current_frame_idx + frames_to_keep
        else:
            should_delete = lambda f: f < current_frame_idx - frames_to_keep

        # Try SAM3 structure first (direct attribute access)
        if hasattr(inference_state, 'non_cond_frame_outputs'):
            non_cond = inference_state.non_cond_frame_outputs
            old_frames = [f for f in non_cond.keys() if should_delete(f)]
            for old_frame in old_frames:
                del non_cond[old_frame]

            if hasattr(inference_state, 'cond_frame_outputs'):
                cond = inference_state.cond_frame_outputs
                old_frames = [f for f in cond.keys() if should_delete(f)]
                for old_frame in old_frames:
                    del cond[old_frame]

            # Clean cached_frame_outputs
            if hasattr(inference_state, '__dict__') and 'cached_frame_outputs' in inference_state.__dict__:
                cached = inference_state.__dict__['cached_frame_outputs']
                if isinstance(cached, dict):
                    old_frames = [f for f in cached.keys() if isinstance(f, int) and should_delete(f)]
                    for old_frame in old_frames:
                        del cached[old_frame]

            # Clean tracker inference states
            if hasattr(inference_state, '__dict__') and 'tracker_inference_states' in inference_state.__dict__:
                tracker_states = inference_state.__dict__['tracker_inference_states']
                if isinstance(tracker_states, list):
                    for tracker_state in tracker_states:
                        if hasattr(tracker_state, 'output_dict'):
                            output_dict = tracker_state.output_dict
                            for cache_name in ['non_cond_frame_outputs', 'cond_frame_outputs']:
                                if cache_name in output_dict:
                                    cache = output_dict[cache_name]
                                    old_frames = [f for f in cache.keys() if should_delete(f)]
                                    for old_frame in old_frames:
                                        del cache[old_frame]

        # Fall back to SAM2 dict structure
        elif isinstance(inference_state, dict) and "output_dict_per_obj" in inference_state:
            # Clean main output_dict (only non_cond_frame_outputs, NOT cond_frame_outputs)
            if "output_dict" in inference_state:
                output_dict = inference_state["output_dict"]
                if "non_cond_frame_outputs" in output_dict:
                    cache = output_dict["non_cond_frame_outputs"]
                    old_frames = [f for f in cache.keys() if should_delete(f)]
                    for old_frame in old_frames:
                        del cache[old_frame]

            # Clean per-object dicts
            for obj_idx in range(len(inference_state.get("obj_ids", []))):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                non_cond = obj_output_dict.get("non_cond_frame_outputs", {})
                old_frames = [f for f in non_cond.keys() if should_delete(f)]
                for old_frame in old_frames:
                    del non_cond[old_frame]

        # Periodic GPU memory cleanup
        if current_frame_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # -------------------------------------------------------------------------
    # Annotation Processing
    # -------------------------------------------------------------------------

    def _group_annotations(
        self,
        annotations: List[PointAnnotation]
    ) -> Dict[int, Dict[int, Dict[str, List]]]:
        """
        Group annotations by frame index and object ID.

        Args:
            annotations: List of PointAnnotation objects

        Returns:
            Nested dict: frame_idx -> obj_id -> {'points': [...], 'labels': [...]}
        """
        grouped: Dict[int, Dict[int, Dict[str, List]]] = {}

        for ann in annotations:
            if ann.frame_idx not in grouped:
                grouped[ann.frame_idx] = {}
            if ann.object_id not in grouped[ann.frame_idx]:
                grouped[ann.frame_idx][ann.object_id] = {'points': [], 'labels': []}

            grouped[ann.frame_idx][ann.object_id]['points'].append([ann.x, ann.y])
            grouped[ann.frame_idx][ann.object_id]['labels'].append(1 if ann.is_positive else 0)

        return grouped

    def _add_annotation_points(
        self,
        inference_state: Any,
        frame_idx: int,
        obj_id: int,
        points: np.ndarray,
        labels: np.ndarray,
        frame_width: int,
        frame_height: int
    ) -> None:
        """
        Add annotation points to the SAM model.

        Handles API differences between SAM2 and SAM3:
        - SAM3: Requires relative [0-1] coordinates, uses add_new_points()
        - SAM2: Uses pixel coordinates, uses add_new_points_or_box()

        Args:
            inference_state: SAM inference state
            frame_idx: Frame index to add points to
            obj_id: Object ID
            points: Array of (x, y) coordinates (pixel coords)
            labels: Array of labels (1=positive, 0=negative)
            frame_width: Width of frames for coordinate conversion
            frame_height: Height of frames for coordinate conversion
        """
        if self.is_sam3:
            # SAM3 requires relative [0-1] coordinates
            rel_points = [[x / frame_width, y / frame_height] for x, y in points]
            points_to_use = np.array(rel_points, dtype=np.float32)
        else:
            # SAM2 uses pixel coordinates
            points_to_use = points.astype(np.float32)

        # Convert to tensors
        points_tensor = torch.from_numpy(points_to_use).to(dtype=torch.float32, device=self.device)
        labels_tensor = torch.from_numpy(labels).to(dtype=torch.int64, device=self.device)

        # Ensure tensors are contiguous
        points_tensor = points_tensor.contiguous()
        labels_tensor = labels_tensor.contiguous()

        if self.is_sam3:
            # SAM3 uses add_new_points
            _, out_obj_ids, low_res_masks, video_res_masks = self.predictor.add_new_points(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points_tensor,
                labels=labels_tensor,
                clear_old_points=False,
            )
        else:
            # SAM2 uses add_new_points_or_box
            _ = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=obj_id,
                points=points_tensor,
                labels=labels_tensor,
                clear_old_points=False,
            )

    # -------------------------------------------------------------------------
    # Mask Propagation
    # -------------------------------------------------------------------------

    def _propagate_masks(
        self,
        inference_state: Any,
        num_frames: int,
        masks_dir: str,
        object_names: Dict[int, str],
        object_colors: Dict[int, Tuple[int, int, int]],
        annotated_objects: set,
        quality_calculator: Optional[IncrementalQualityMetricsCalculator],
        frames_to_keep: int,
        enable_backward: bool,
        frame_offset: int = 0
    ) -> Dict[int, Dict[int, Any]]:
        """
        Propagate masks through the video in forward and backward directions.

        Args:
            inference_state: SAM inference state
            num_frames: Total number of frames
            masks_dir: Directory to save mask files
            object_names: Dict mapping obj_id -> name
            object_colors: Dict mapping obj_id -> (R, G, B)
            annotated_objects: Set of object IDs that have annotations
            quality_calculator: Optional calculator for quality metrics
            frames_to_keep: Number of frames to keep in memory
            enable_backward: Whether to run backward propagation
            frame_offset: Offset for frame numbering in output filenames

        Returns:
            Dict mapping frame_idx -> {obj_id -> mask_metadata}
        """
        masks_metadata: Dict[int, Dict[int, Any]] = {}
        os.makedirs(masks_dir, exist_ok=True)

        # Forward propagation
        masks_metadata = self._propagate_direction(
            inference_state=inference_state,
            num_frames=num_frames,
            masks_dir=masks_dir,
            object_names=object_names,
            object_colors=object_colors,
            annotated_objects=annotated_objects,
            quality_calculator=quality_calculator,
            frames_to_keep=frames_to_keep,
            reverse=False,
            masks_metadata=masks_metadata,
            frame_offset=frame_offset
        )

        # Backward propagation (optional)
        if enable_backward:
            masks_metadata = self._propagate_direction(
                inference_state=inference_state,
                num_frames=num_frames,
                masks_dir=masks_dir,
                object_names=object_names,
                object_colors=object_colors,
                annotated_objects=annotated_objects,
                quality_calculator=quality_calculator,
                frames_to_keep=frames_to_keep,
                reverse=True,
                masks_metadata=masks_metadata,
                frame_offset=frame_offset
            )

        return masks_metadata

    def _propagate_direction(
        self,
        inference_state: Any,
        num_frames: int,
        masks_dir: str,
        object_names: Dict[int, str],
        object_colors: Dict[int, Tuple[int, int, int]],
        annotated_objects: set,
        quality_calculator: Optional[IncrementalQualityMetricsCalculator],
        frames_to_keep: int,
        reverse: bool,
        masks_metadata: Dict[int, Dict[int, Any]],
        frame_offset: int = 0
    ) -> Dict[int, Dict[int, Any]]:
        """
        Propagate masks in a single direction (forward or backward).

        Args:
            inference_state: SAM inference state
            num_frames: Total number of frames
            masks_dir: Directory to save mask files
            object_names: Dict mapping obj_id -> name
            object_colors: Dict mapping obj_id -> (R, G, B)
            annotated_objects: Set of object IDs that have annotations
            quality_calculator: Optional quality metrics calculator
            frames_to_keep: Number of frames to keep in memory
            reverse: True for backward propagation
            masks_metadata: Existing metadata dict (updated in place)
            frame_offset: Offset for frame numbering in output filenames

        Returns:
            Updated masks_metadata dict
        """
        direction = "backward" if reverse else "forward"
        self._progress_callback.on_phase_start(direction, num_frames)

        # Use autocast for bfloat16 if enabled
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.use_bfloat16 and self.device.startswith("cuda")
            else nullcontext()
        )

        with autocast_context:
            # Construct propagation call based on model type
            if self.is_sam3:
                if reverse:
                    propagate_iterator = self.predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=num_frames - 1,
                        max_frame_num_to_track=num_frames,
                        reverse=True,
                        propagate_preflight=True
                    )
                else:
                    propagate_iterator = self.predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=0,
                        max_frame_num_to_track=num_frames,
                        reverse=False,
                        propagate_preflight=True
                    )
            else:
                # SAM2 propagation
                propagate_iterator = self.predictor.propagate_in_video(
                    inference_state, reverse=reverse
                )

            processed = 0
            for result in propagate_iterator:
                # Unpack based on model type
                if self.is_sam3:
                    out_frame_idx, out_obj_ids, out_low_res_masks, out_mask_logits, out_obj_scores = result
                else:
                    out_frame_idx, out_obj_ids, out_mask_logits = result

                frame_masks: Dict[int, Any] = {}
                frame_masks_for_quality: Dict[int, np.ndarray] = {}

                for i, obj_id in enumerate(out_obj_ids):
                    # Only process annotated objects
                    if obj_id not in annotated_objects:
                        continue

                    # Convert mask to numpy
                    mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

                    # Get object info
                    obj_name = object_names.get(obj_id, object_names.get(str(obj_id), f"Object_{obj_id}"))
                    obj_color = object_colors.get(obj_id, object_colors.get(str(obj_id), (255, 0, 0)))

                    # Apply frame offset for output filename
                    output_frame_idx = out_frame_idx + frame_offset

                    # Export mask to disk
                    mask_filename = f"mask_f{output_frame_idx:06d}_{obj_name}_id{obj_id}.png"
                    mask_path = os.path.join(masks_dir, mask_filename)
                    mask_uint8 = (mask * 255).astype(np.uint8)
                    cv2.imwrite(mask_path, mask_uint8)

                    # Store metadata
                    score = float(out_obj_scores[i]) if self.is_sam3 else 1.0
                    frame_masks[obj_id] = {
                        'filename': mask_filename,
                        'score': score,
                        'name': obj_name,
                        'color': obj_color
                    }

                    # Store for quality metrics
                    frame_masks_for_quality[obj_id] = mask_uint8

                    del mask

                masks_metadata[out_frame_idx] = frame_masks

                # Update quality metrics
                if quality_calculator is not None:
                    if reverse:
                        quality_calculator.update_backward(out_frame_idx, frame_masks_for_quality)
                    else:
                        quality_calculator.update_forward(out_frame_idx, frame_masks_for_quality)

                del frame_masks_for_quality

                # Delete tensors to free memory
                del out_mask_logits
                if self.is_sam3:
                    del out_low_res_masks
                    del out_obj_scores
                del result

                # Clean up old frames from inference state
                self._cleanup_inference_state(
                    inference_state, out_frame_idx,
                    frames_to_keep=frames_to_keep, reverse=reverse
                )

                processed += 1
                if processed % 50 == 0:
                    self._progress_callback.on_progress(
                        direction, processed, num_frames,
                        f"{direction.capitalize()}: Frame {processed}/{num_frames}"
                    )

        self._progress_callback.on_phase_complete(direction)
        return masks_metadata

    # -------------------------------------------------------------------------
    # Main Segmentation Method
    # -------------------------------------------------------------------------

    def segment(
        self,
        video_path: str,
        annotations: List[PointAnnotation],
        object_names: Dict[int, str],
        object_colors: Dict[int, Tuple[int, int, int]],
        config: SegmentationConfig
    ) -> SegmentationResult:
        """
        Run full video segmentation pipeline.

        Args:
            video_path: Path to input video file
            annotations: List of point annotations
            object_names: Dict mapping object_id -> name string
            object_colors: Dict mapping object_id -> (R, G, B) color tuple
            config: Segmentation configuration

        Returns:
            SegmentationResult with masks metadata and quality metrics
        """
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        masks_dir = config.masks_output_dir
        os.makedirs(masks_dir, exist_ok=True)

        # Determine frame directory
        temp_dir_obj = None
        if config.frame_dir:
            frame_dir = config.frame_dir
            os.makedirs(frame_dir, exist_ok=True)
            is_persistent = True
        else:
            temp_dir_obj = tempfile.mkdtemp(prefix='sam2_frames_')
            frame_dir = temp_dir_obj
            is_persistent = False

        try:
            # Check if frames already exist (support both jpg and png formats)
            existing_frames = sorted(Path(frame_dir).glob("*.jpg"))
            if not existing_frames:
                existing_frames = sorted(Path(frame_dir).glob("*.png"))

            if existing_frames:
                print(f"Reusing {len(existing_frames)} existing frames from: {frame_dir}")
                num_frames = len(existing_frames)
            else:
                # Extract frames from video using configured format
                print(f"Extracting frames to: {frame_dir} (format: {config.frame_format})")
                num_frames = self._extract_frames(
                    video_path, frame_dir, config.frame_range, config.frame_format
                )
                print(f"Extracted {num_frames} frames")

            # Get frame dimensions (find first frame file)
            frame_files = sorted(Path(frame_dir).glob("*.jpg"))
            if not frame_files:
                frame_files = sorted(Path(frame_dir).glob("*.png"))
            if not frame_files:
                raise ValueError(f"No frame files found in: {frame_dir}")

            first_frame_path = frame_files[0]
            first_frame = cv2.imread(str(first_frame_path))
            if first_frame is None:
                raise ValueError(f"Cannot read first frame: {first_frame_path}")
            frame_height, frame_width = first_frame.shape[:2]
            del first_frame

            print(f"Frame dimensions: {frame_width}x{frame_height}")

            # Initialize inference state
            self._progress_callback.on_phase_start("initializing", 1)
            inference_state = self._init_inference_state(
                frame_dir,
                offload_video_to_cpu=config.offload_video_to_cpu,
                offload_state_to_cpu=config.offload_state_to_cpu
            )

            # Get actual number of frames from inference state
            if isinstance(inference_state, dict) and "num_frames" in inference_state:
                num_frames = inference_state["num_frames"]

            self._progress_callback.on_phase_complete("initializing")
            print(f"Initialized inference state for {num_frames} frames")

            # Initialize quality metrics calculator
            quality_calculator = None
            if config.calculate_quality_metrics:
                quality_calculator = IncrementalQualityMetricsCalculator(
                    frame_dimensions=(frame_height, frame_width),
                    num_frames=num_frames
                )

            # Group annotations by frame and object
            grouped_annotations = self._group_annotations(annotations)

            # Collect all annotated object IDs
            annotated_objects = set()
            for frame_data in grouped_annotations.values():
                annotated_objects.update(frame_data.keys())

            print(f"Processing {len(annotated_objects)} objects across {len(grouped_annotations)} annotated frames")

            # Add annotation points
            self._progress_callback.on_phase_start("adding_points", len(grouped_annotations))
            for frame_idx, obj_dict in sorted(grouped_annotations.items()):
                if frame_idx >= num_frames:
                    print(f"WARNING: Skipping frame {frame_idx} (beyond video length)")
                    continue

                for obj_id, point_data in obj_dict.items():
                    points = np.array(point_data['points'], dtype=np.float32)
                    labels = np.array(point_data['labels'], dtype=np.int32)

                    self._add_annotation_points(
                        inference_state, frame_idx, obj_id,
                        points, labels, frame_width, frame_height
                    )

                    obj_name = object_names.get(obj_id, object_names.get(str(obj_id), f"Object_{obj_id}"))
                    print(f"  Added {len(points)} points for {obj_name} on frame {frame_idx}")

            self._progress_callback.on_phase_complete("adding_points")

            # Propagate masks
            masks_metadata = self._propagate_masks(
                inference_state=inference_state,
                num_frames=num_frames,
                masks_dir=masks_dir,
                object_names=object_names,
                object_colors=object_colors,
                annotated_objects=annotated_objects,
                quality_calculator=quality_calculator,
                frames_to_keep=config.frames_to_keep,
                enable_backward=config.enable_backward_propagation,
                frame_offset=config.frame_offset
            )

            # Get quality metrics
            quality_metrics = None
            if quality_calculator is not None:
                inter_frame_changes, background_ratios = quality_calculator.get_results()
                quality_metrics = (inter_frame_changes, background_ratios)
                quality_calculator.print_summary()

                # Save quality metrics to disk
                save_quality_metrics(config.output_dir, inter_frame_changes, background_ratios)

            # Clean up inference state memory
            if isinstance(inference_state, dict) and "output_dict_per_obj" in inference_state:
                for obj_idx in range(len(annotated_objects)):
                    if obj_idx < len(inference_state.get("output_dict_per_obj", [])):
                        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                        non_cond = obj_output_dict.get("non_cond_frame_outputs", {})
                        non_cond.clear()
            elif hasattr(inference_state, 'non_cond_frame_outputs'):
                inference_state.non_cond_frame_outputs.clear()

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            print(f"Generated masks for {len(masks_metadata)} frames")

            return SegmentationResult(
                masks_metadata=masks_metadata,
                object_names=object_names,
                object_colors=object_colors,
                num_frames_processed=num_frames,
                frame_dimensions=(frame_height, frame_width),
                quality_metrics=quality_metrics,
                inference_state=inference_state
            )

        finally:
            # Clean up temporary frame directory
            if not is_persistent and temp_dir_obj and config.cleanup_temp_frames:
                try:
                    import shutil
                    shutil.rmtree(temp_dir_obj)
                    print(f"Cleaned up temporary frames: {temp_dir_obj}")
                except Exception as e:
                    print(f"WARNING: Could not clean up temp directory: {e}")


# =============================================================================
# Utility: Null Context Manager
# =============================================================================

class nullcontext:
    """Minimal null context manager for Python < 3.7 compatibility."""
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False
