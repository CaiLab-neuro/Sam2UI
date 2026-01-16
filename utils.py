"""
Utility functions for SAM2 Video UI.

This module contains reusable functions for mask I/O operations,
quality metrics calculation, and other common utilities.
"""

import os
import re
import numpy as np
import cv2
from typing import Dict, List, Tuple, Callable, Optional, Any


# =============================================================================
# Mask Filename Utilities
# =============================================================================

def generate_mask_filename(frame_idx: int, obj_id: int, obj_name: str) -> str:
    """
    Generate standardized mask filename.

    Args:
        frame_idx: Frame index
        obj_id: Object ID
        obj_name: Object name (will be sanitized)

    Returns:
        Filename string in format: mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png
    """
    return f"mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png"


def generate_legacy_mask_filename(frame_idx: int, obj_id: int) -> str:
    """
    Generate legacy mask filename for backward compatibility.

    Args:
        frame_idx: Frame index
        obj_id: Object ID

    Returns:
        Filename string in legacy format: frame_{frame_idx:05d}_obj_{obj_id}.png
    """
    return f"frame_{frame_idx:05d}_obj_{obj_id}.png"


def parse_mask_filename(filename: str) -> Optional[Tuple[int, str, int]]:
    """
    Parse mask filename to extract frame index, object name, and object ID.

    Args:
        filename: Mask filename to parse

    Returns:
        Tuple of (frame_idx, obj_name, obj_id) or None if parsing fails
    """
    # Try new format: mask_f{frame_idx:06d}_{obj_name}_id{obj_id}.png
    match = re.match(r'mask_f(\d{6})_(.+)_id(\d+)\.png', filename)
    if match:
        return int(match.group(1)), match.group(2), int(match.group(3))

    # Try legacy format: frame_{frame_idx:05d}_obj_{obj_id}.png
    match = re.match(r'frame_(\d{5})_obj_(\d+)\.png', filename)
    if match:
        return int(match.group(1)), f"Object_{match.group(2)}", int(match.group(2))

    return None


# =============================================================================
# Mask I/O Functions
# =============================================================================

def save_mask(
    mask: np.ndarray,
    output_dir: str,
    frame_idx: int,
    obj_id: int,
    obj_name: str
) -> str:
    """
    Save mask to disk as PNG file.

    Args:
        mask: Binary mask (H, W) numpy array
        output_dir: Directory to save mask
        frame_idx: Frame index
        obj_id: Object ID
        obj_name: Object name

    Returns:
        Path to saved mask file
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_filename = generate_mask_filename(frame_idx, obj_id, obj_name)
    mask_path = os.path.join(output_dir, mask_filename)

    cv2.imwrite(mask_path, mask)
    return mask_path


def load_mask(
    mask_dir: str,
    frame_idx: int,
    obj_id: int,
    obj_name: Optional[str] = None
) -> Optional[np.ndarray]:
    """
    Load mask from disk.

    Tries multiple filename patterns for compatibility:
    1. New pattern with exact name
    2. Legacy pattern
    3. Pattern matching by frame and ID (name mismatch fallback)

    Args:
        mask_dir: Directory containing mask files
        frame_idx: Frame index
        obj_id: Object ID
        obj_name: Object name (optional, defaults to "Object_{obj_id}")

    Returns:
        Mask as numpy array (H, W) or None if not found
    """
    if obj_name is None:
        obj_name = f"Object_{obj_id}"

    # Try 1: Exact name match with new pattern
    mask_filename = generate_mask_filename(frame_idx, obj_id, obj_name)
    mask_path = os.path.join(mask_dir, mask_filename)

    if os.path.exists(mask_path):
        return cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Try 2: Legacy pattern (for backward compatibility)
    mask_filename_legacy = generate_legacy_mask_filename(frame_idx, obj_id)
    mask_path_legacy = os.path.join(mask_dir, mask_filename_legacy)

    if os.path.exists(mask_path_legacy):
        print(f"WARNING: Using legacy mask pattern for frame {frame_idx}, obj {obj_id}")
        return cv2.imread(mask_path_legacy, cv2.IMREAD_GRAYSCALE)

    # Try 3: Pattern matching by frame and ID (name mismatch fallback)
    if os.path.isdir(mask_dir):
        for filename in os.listdir(mask_dir):
            match = re.match(rf'mask_f{frame_idx:06d}_(.+)_id{obj_id}\.png', filename)
            if match:
                found_name = match.group(1)
                print(f"WARNING: Mask name mismatch for obj {obj_id}. "
                      f"Expected '{obj_name}', found '{found_name}'. Using found mask.")
                mask_path_fallback = os.path.join(mask_dir, filename)
                return cv2.imread(mask_path_fallback, cv2.IMREAD_GRAYSCALE)

    print(f"WARNING: Mask file not found for frame {frame_idx}, obj {obj_id}")
    return None


def export_mask_to_disk(
    mask: np.ndarray,
    output_dir: str,
    frame_idx: int,
    obj_id: int,
    obj_name: str,
    obj_color: Tuple[int, int, int] = (255, 0, 0)
) -> Dict[str, Any]:
    """
    Export mask to disk and return metadata.

    Args:
        mask: Binary mask (H, W) numpy array
        output_dir: Directory to save mask
        frame_idx: Frame index
        obj_id: Object ID
        obj_name: Object name
        obj_color: Object color as (R, G, B) tuple

    Returns:
        Metadata dictionary containing mask information
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_filename = generate_mask_filename(frame_idx, obj_id, obj_name)
    mask_path = os.path.join(output_dir, mask_filename)

    cv2.imwrite(mask_path, mask)

    metadata = {
        'frame_idx': frame_idx,
        'obj_id': obj_id,
        'mask_file': mask_filename,
        'mask_path': mask_path,
        'object_name': obj_name,
        'color': obj_color,
        'mask_shape': mask.shape
    }

    return metadata


# =============================================================================
# Quality Metrics Functions
# =============================================================================

class IncrementalQualityMetricsCalculator:
    """
    Calculate quality metrics incrementally during video propagation.

    This class enables calculating inter-frame changes and background ratios
    during forward/backward propagation passes, avoiding the need to load
    all masks from disk after processing.

    Usage:
        calculator = IncrementalQualityMetricsCalculator((height, width), num_frames)

        # During forward propagation (frames 0 to N-1):
        for frame_idx, masks in forward_propagation():
            calculator.update_forward(frame_idx, masks)

        # During backward propagation (frames N-1 to 0):
        for frame_idx, masks in backward_propagation():
            calculator.update_backward(frame_idx, masks)

        # Get final results
        inter_frame_changes, background_ratios = calculator.get_results()
    """

    def __init__(self, frame_dimensions: Tuple[int, int], num_frames: int):
        """
        Initialize the incremental calculator.

        Args:
            frame_dimensions: (height, width) of frames
            num_frames: Total number of frames in the video
        """
        self.height, self.width = frame_dimensions
        self.total_pixels = self.height * self.width
        self.num_frames = num_frames

        # Initialize arrays with zeros (will be filled during propagation)
        self.inter_frame_changes: List[float] = [0.0] * num_frames
        self.background_ratios: List[float] = [0.0] * num_frames

        # State for forward pass
        self._forward_prev_mask: Optional[np.ndarray] = None

        # State for backward pass
        self._backward_next_mask: Optional[np.ndarray] = None

        # Track which pass has been run
        self._forward_completed = False
        self._backward_completed = False

    def _create_combined_mask(self, object_masks: Dict[int, np.ndarray]) -> np.ndarray:
        """
        Create a combined mask from individual object masks.

        Args:
            object_masks: Dictionary mapping obj_id -> mask array (H, W)

        Returns:
            Combined mask where each pixel contains the obj_id (0 for background)
        """
        combined = np.zeros((self.height, self.width), dtype=np.uint8)

        for obj_id, mask in object_masks.items():
            if mask is None:
                continue

            # Ensure mask matches frame dimensions
            if mask.shape != (self.height, self.width):
                from PIL import Image as PILImage
                mask_pil = PILImage.fromarray(mask.astype(np.uint8))
                mask_pil = mask_pil.resize((self.width, self.height), PILImage.NEAREST)
                mask = np.array(mask_pil)

            # Mark object pixels
            combined[mask > 0] = obj_id

        return combined

    def _calculate_background_ratio(self, combined_mask: np.ndarray) -> float:
        """Calculate the ratio of background pixels."""
        object_pixels = np.count_nonzero(combined_mask)
        return 1.0 - (object_pixels / self.total_pixels)

    def _calculate_change_ratio(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Calculate the ratio of pixels that changed between two masks."""
        changed_pixels = np.count_nonzero(mask1 != mask2)
        return changed_pixels / self.total_pixels

    def update_forward(self, frame_idx: int, object_masks: Dict[int, np.ndarray]) -> None:
        """
        Update metrics during forward propagation (frames 0 to N-1).

        Call this for each frame in forward order as masks are generated.

        Args:
            frame_idx: Current frame index
            object_masks: Dictionary mapping obj_id -> mask array (H, W)
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            return

        combined_mask = self._create_combined_mask(object_masks)

        # Calculate background ratio
        self.background_ratios[frame_idx] = self._calculate_background_ratio(combined_mask)

        # Calculate inter-frame change
        if self._forward_prev_mask is None:
            # First frame: no previous to compare
            self.inter_frame_changes[frame_idx] = 0.0
        else:
            self.inter_frame_changes[frame_idx] = self._calculate_change_ratio(
                self._forward_prev_mask, combined_mask
            )

        # Store current mask for next iteration
        self._forward_prev_mask = combined_mask.copy()

    def update_backward(self, frame_idx: int, object_masks: Dict[int, np.ndarray]) -> None:
        """
        Update metrics during backward propagation (frames N-1 to 0).

        Call this for each frame in backward order. This updates the metrics
        calculated during forward pass with the final mask values.

        Args:
            frame_idx: Current frame index
            object_masks: Dictionary mapping obj_id -> mask array (H, W)
        """
        if frame_idx < 0 or frame_idx >= self.num_frames:
            return

        combined_mask = self._create_combined_mask(object_masks)

        # Update background ratio with final mask
        self.background_ratios[frame_idx] = self._calculate_background_ratio(combined_mask)

        # Update inter-frame change for the NEXT frame (in forward order)
        # When processing frame i in backward order, we have:
        # - combined_mask: mask for frame i
        # - _backward_next_mask: mask for frame i+1 (from previous backward iteration)
        # inter_frame_changes[i+1] = change from frame i to frame i+1
        if self._backward_next_mask is not None and frame_idx + 1 < self.num_frames:
            self.inter_frame_changes[frame_idx + 1] = self._calculate_change_ratio(
                combined_mask, self._backward_next_mask
            )

        # Store current mask for next backward iteration
        self._backward_next_mask = combined_mask.copy()

        # Mark backward pass for frame 0 as completed
        if frame_idx == 0:
            self._backward_completed = True

    def update_single_pass(self, frame_idx: int, object_masks: Dict[int, np.ndarray],
                           reverse: bool = False) -> None:
        """
        Update metrics for a single propagation pass (forward or backward only).

        This is useful when only one direction propagation is performed.

        Args:
            frame_idx: Current frame index
            object_masks: Dictionary mapping obj_id -> mask array (H, W)
            reverse: True if processing in backward direction
        """
        if reverse:
            self.update_backward(frame_idx, object_masks)
        else:
            self.update_forward(frame_idx, object_masks)

    def get_results(self) -> Tuple[List[float], List[float]]:
        """
        Get the calculated quality metrics.

        Returns:
            Tuple of (inter_frame_changes, background_ratios)
        """
        return self.inter_frame_changes, self.background_ratios

    def print_summary(self) -> None:
        """Print a summary of the calculated metrics."""
        print(f"Quality metrics calculated for {self.num_frames} frames")
        if len(self.inter_frame_changes) > 1:
            # Skip frame 0 for mean inter-frame change calculation
            mean_change = np.mean(self.inter_frame_changes[1:])
            print(f"  Mean inter-frame change: {mean_change:.3f}")
        print(f"  Mean background ratio: {np.mean(self.background_ratios):.3f}")


class QualityMetricsCalculator:
    """
    Calculate segmentation quality metrics: inter-frame changes and background ratios.

    This class is designed to be reusable across different contexts (UI, batch processing).
    """

    def __init__(self, frame_dimensions: Tuple[int, int], num_frames: int):
        """
        Initialize the calculator.

        Args:
            frame_dimensions: (height, width) of frames
            num_frames: Total number of frames in the video
        """
        self.height, self.width = frame_dimensions
        self.total_pixels = self.height * self.width
        self.num_frames = num_frames

        self.inter_frame_changes: List[float] = []
        self.background_ratios: List[float] = []

    def calculate(
        self,
        masks: Dict[int, Dict[int, Any]],
        load_mask_func: Callable[[int, int], Optional[np.ndarray]]
    ) -> Tuple[List[float], List[float]]:
        """
        Calculate quality metrics for all frames.

        Args:
            masks: Dictionary mapping frame_idx -> {obj_id -> mask_data}
            load_mask_func: Function(frame_idx, obj_id) -> numpy array mask or None

        Returns:
            Tuple of (inter_frame_changes, background_ratios)
        """
        print("Calculating segmentation quality metrics...")

        self.inter_frame_changes = []
        self.background_ratios = []

        prev_combined_mask = None

        for frame_idx in range(self.num_frames):
            # Create combined mask for this frame (all objects)
            combined_mask = np.zeros((self.height, self.width), dtype=np.uint8)

            if frame_idx in masks:
                for obj_id in masks[frame_idx]:
                    mask = load_mask_func(frame_idx, obj_id)
                    if mask is not None:
                        # Resize if needed
                        if mask.shape != (self.height, self.width):
                            from PIL import Image as PILImage
                            mask_pil = PILImage.fromarray(mask)
                            mask_pil = mask_pil.resize((self.width, self.height), PILImage.NEAREST)
                            mask = np.array(mask_pil)

                        # Mark object pixels (any non-zero value)
                        combined_mask[mask > 0] = obj_id

            # Calculate background ratio
            object_pixels = np.count_nonzero(combined_mask)
            bg_ratio = 1.0 - (object_pixels / self.total_pixels)
            self.background_ratios.append(bg_ratio)

            # Calculate inter-frame change
            if prev_combined_mask is None:
                # First frame: no previous frame to compare
                self.inter_frame_changes.append(0.0)
            else:
                # Count pixels that changed category
                changed_pixels = np.count_nonzero(combined_mask != prev_combined_mask)
                change_ratio = changed_pixels / self.total_pixels
                self.inter_frame_changes.append(change_ratio)

            prev_combined_mask = combined_mask.copy()

        print(f"Calculated metrics for {self.num_frames} frames")
        if len(self.inter_frame_changes) > 1:
            print(f"  Mean inter-frame change: {np.mean(self.inter_frame_changes[1:]):.3f}")
        print(f"  Mean background ratio: {np.mean(self.background_ratios):.3f}")

        return self.inter_frame_changes, self.background_ratios


def calculate_quality_metrics(
    masks: Dict[int, Dict[int, Any]],
    load_mask_func: Callable[[int, int], Optional[np.ndarray]],
    frame_dimensions: Tuple[int, int],
    num_frames: int
) -> Tuple[List[float], List[float]]:
    """
    Convenience function to calculate quality metrics without instantiating a class.

    Args:
        masks: Dictionary mapping frame_idx -> {obj_id -> mask_data}
        load_mask_func: Function(frame_idx, obj_id) -> numpy array mask or None
        frame_dimensions: (height, width) of frames
        num_frames: Total number of frames

    Returns:
        Tuple of (inter_frame_changes, background_ratios)
    """
    calculator = QualityMetricsCalculator(frame_dimensions, num_frames)
    return calculator.calculate(masks, load_mask_func)


def save_quality_metrics(
    output_dir: str,
    inter_frame_changes: List[float],
    background_ratios: List[float]
) -> bool:
    """
    Save quality metrics to quality_metrics.npz in output directory.

    Args:
        output_dir: Directory to save the metrics file
        inter_frame_changes: List of inter-frame change ratios
        background_ratios: List of background ratios

    Returns:
        True if saved successfully, False otherwise
    """
    if not inter_frame_changes or not background_ratios:
        print("No quality metrics to save")
        return False

    try:
        metrics_path = os.path.join(output_dir, "quality_metrics.npz")

        np.savez_compressed(
            metrics_path,
            inter_frame_changes=np.array(inter_frame_changes),
            background_ratios=np.array(background_ratios),
            frame_count=len(inter_frame_changes)
        )

        print(f"Saved quality metrics to {metrics_path}")
        return True
    except Exception as e:
        print(f"WARNING: Failed to save quality metrics: {e}")
        return False


def load_quality_metrics(output_dir: str) -> Tuple[Optional[List[float]], Optional[List[float]]]:
    """
    Load quality metrics from quality_metrics.npz if available.

    Args:
        output_dir: Directory containing the metrics file

    Returns:
        Tuple of (inter_frame_changes, background_ratios), or (None, None) if not found/failed
    """
    metrics_path = os.path.join(output_dir, "quality_metrics.npz")

    if not os.path.exists(metrics_path):
        print("No quality metrics file found (OK for older results)")
        return None, None

    try:
        data = np.load(metrics_path)

        inter_frame_changes = data['inter_frame_changes'].tolist()
        background_ratios = data['background_ratios'].tolist()

        print(f"Loaded quality metrics: {len(inter_frame_changes)} values")
        return inter_frame_changes, background_ratios
    except Exception as e:
        print(f"WARNING: Failed to load quality metrics: {e}")
        return None, None
