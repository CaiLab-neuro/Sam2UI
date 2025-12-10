#!/usr/bin/env python3
"""
SAM2 Lazy Frame Loader
======================

Implements lazy loading for SAM2 video frames to reduce memory usage.
Instead of loading all frames into RAM at once (~178GB for 36K frames),
loads frames on-demand with LRU caching (~2-5GB for 20-frame cache).

Usage:
    from sam2_lazy_loader import enable_lazy_loading

    # Before creating SAM2VideoPredictor
    enable_lazy_loading(cache_size=20)

    # Then use SAM2 normally
    predictor = build_sam2_video_predictor(...)
"""

import os
import torch
from collections import OrderedDict
from pathlib import Path


class LazyVideoFrameLoader:
    """
    Lazy video frame loader with LRU cache.
    Drop-in replacement for SAM2's eager frame loading.

    This class mimics a tensor but loads frames on-demand via __getitem__.
    SAM2 accesses frames sequentially, so we can cache recent frames and
    evict old ones, keeping memory bounded.
    """

    def __init__(self, img_paths, image_size, offload_video_to_cpu,
                 img_mean, img_std, compute_device, cache_size=20):
        """
        Args:
            img_paths: List of paths to JPEG frames
            image_size: Target size for frame resizing
            offload_video_to_cpu: If True, keep frames on CPU
            img_mean: Normalization mean (tensor)
            img_std: Normalization std (tensor)
            compute_device: Device for computation (cuda/cpu)
            cache_size: Number of frames to keep in memory (default 20)
                       SAM2 looks back at most 6 frames, so 20 is safe
        """
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.img_mean = img_mean
        self.img_std = img_std
        self.compute_device = compute_device
        self.cache_size = cache_size

        # LRU cache: OrderedDict maintains insertion order
        # Most recently accessed items are moved to the end
        self.cache = OrderedDict()

        # Get video dimensions from first frame
        from sam2.utils.misc import _load_img_as_tensor
        self._load_img_as_tensor = _load_img_as_tensor

        img, self.video_height, self.video_width = self._load_img_as_tensor(
            img_paths[0], image_size
        )

        # Cache first frame (likely where user will click)
        self._add_to_cache(0, self._normalize_frame(img))

        print(f"LazyVideoFrameLoader initialized:")
        print(f"  Total frames: {len(img_paths)}")
        print(f"  Cache size: {cache_size} frames")
        print(f"  Video dimensions: {self.video_width}x{self.video_height}")
        print(f"  Target size: {image_size}x{image_size}")

    def __len__(self):
        """Return total number of frames"""
        return len(self.img_paths)

    def __getitem__(self, index):
        """
        Load frame on-demand with LRU caching.
        This is the only method SAM2 calls to access frames.
        """
        # Check cache first
        if index in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(index)
            return self.cache[index]

        # Load frame from disk
        img, _, _ = self._load_img_as_tensor(self.img_paths[index], self.image_size)

        # Normalize and move to device
        img = self._normalize_frame(img)

        # Add to cache with LRU eviction
        self._add_to_cache(index, img)

        return img

    def _normalize_frame(self, img):
        """Normalize frame and move to appropriate device"""
        # Normalize by mean and std
        img = img - self.img_mean
        img = img / self.img_std

        # Move to device if not offloading to CPU
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device)

        return img

    def _add_to_cache(self, index, img):
        """Add frame to cache with LRU eviction"""
        # Evict oldest frame if cache is full
        if len(self.cache) >= self.cache_size:
            oldest_idx = next(iter(self.cache))
            del self.cache[oldest_idx]

        # Add new frame
        self.cache[index] = img

    # Implement minimal tensor-like interface for compatibility
    @property
    def device(self):
        """Return device of cached frames (for compatibility)"""
        if self.cache:
            return next(iter(self.cache.values())).device
        return self.compute_device if not self.offload_video_to_cpu else torch.device('cpu')


def enable_lazy_loading(cache_size=20):
    """
    Monkey-patch SAM2 to use lazy loading.
    Call this BEFORE creating SAM2VideoPredictor.

    Args:
        cache_size: Number of frames to cache (default 20)
                   - Minimum: 10 (6 for lookback + buffer)
                   - Recommended: 20 (~2GB)
                   - Maximum: 50 (~5GB)

    Example:
        enable_lazy_loading(cache_size=20)
        predictor = build_sam2_video_predictor(...)
    """
    import sam2.utils.misc as sam2_misc

    # Save original function for potential restoration
    if not hasattr(sam2_misc, '_original_load_video_frames_from_jpg_images'):
        sam2_misc._original_load_video_frames_from_jpg_images = \
            sam2_misc.load_video_frames_from_jpg_images

    def lazy_load_video_frames_from_jpg_images(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,  # Ignored - we always lazy load
        compute_device=torch.device("cuda"),
    ):
        """
        Lazy loading replacement for SAM2's eager loading.
        Returns a LazyVideoFrameLoader instead of a pre-loaded tensor.
        """
        if isinstance(video_path, str) and os.path.isdir(video_path):
            jpg_folder = video_path
        else:
            raise NotImplementedError(
                "Only JPEG folder supported. "
                "Use --frame-dir to extract frames first."
            )

        # Get frame paths
        frame_names = [
            p for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        if len(frame_names) == 0:
            raise RuntimeError(f"No JPEG images found in {jpg_folder}")

        img_paths = [os.path.join(jpg_folder, fn) for fn in frame_names]

        # Convert mean/std to tensors
        img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        # Move mean/std to device if not offloading
        if not offload_video_to_cpu:
            img_mean = img_mean.to(compute_device)
            img_std = img_std.to(compute_device)

        # Create lazy loader
        lazy_images = LazyVideoFrameLoader(
            img_paths, image_size, offload_video_to_cpu,
            img_mean, img_std, compute_device, cache_size
        )

        return lazy_images, lazy_images.video_height, lazy_images.video_width

    # Monkey-patch SAM2
    sam2_misc.load_video_frames_from_jpg_images = lazy_load_video_frames_from_jpg_images

    print("=" * 60)
    print("SAM2 LAZY LOADING ENABLED")
    print("=" * 60)
    print(f"Frame cache size: {cache_size}")
    print(f"Expected memory usage: ~{cache_size * 0.1:.1f}GB (vs ~178GB eager)")
    print()


def disable_lazy_loading():
    """
    Restore SAM2's original frame loading (for debugging).
    """
    import sam2.utils.misc as sam2_misc

    if hasattr(sam2_misc, '_original_load_video_frames_from_jpg_images'):
        sam2_misc.load_video_frames_from_jpg_images = \
            sam2_misc._original_load_video_frames_from_jpg_images
        print("SAM2 lazy loading disabled - using eager loading")
    else:
        print("WARNING: Original function not found, lazy loading may not be disabled")


if __name__ == "__main__":
    print(__doc__)
