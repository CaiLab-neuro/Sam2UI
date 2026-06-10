#!/usr/bin/env python3
"""
Lazy Frame Loader
=================

Implements lazy loading for SAM2/SAM3 video frames to reduce memory usage.
Instead of loading all frames into RAM at once (~178GB for 36K frames),
loads frames on-demand with LRU caching (~2-5GB for 20-frame cache).

Usage:
    from sam_lazy_loader import enable_lazy_loading

    # Before creating SAM2VideoPredictor or SAM3 session
    enable_lazy_loading(cache_size=20)

    # Then use SAM2/SAM3 normally
    predictor = build_sam2_video_predictor(...)
    sam3_model.start_session(resource_path=frames_dir, ...)
"""

import os
import torch
from collections import OrderedDict
from pathlib import Path


class LazyVideoFrameLoader:
    """
    Lazy video frame loader with LRU cache.
    Drop-in replacement for SAM2/SAM3 eager frame loading.

    Mimics a list/tensor of frames but loads from disk on demand.
    SAM2 and SAM3 access frames sequentially (one at a time), so a small
    LRU cache keeps memory bounded regardless of video length.

    Works for both SAM2 (float32, mean≈0.48) and SAM3 (float16, mean=0.5).
    """

    def __init__(self, img_paths, image_size, offload_video_to_cpu,
                 img_mean, img_std, compute_device, cache_size=20,
                 frame_loader_fn=None, frame_dtype=None):
        """
        Args:
            img_paths: List of paths to image frames
            image_size: Target square size for frame resizing
            offload_video_to_cpu: If True, keep cached frames on CPU
            img_mean: Normalization mean tensor (shape [3,1,1])
            img_std: Normalization std tensor (shape [3,1,1])
            compute_device: GPU device for non-offload mode
            cache_size: Number of frames to keep in LRU cache (default 20)
            frame_loader_fn: Optional custom image loader fn(path, size) ->
                             (tensor, height, width). Defaults to SAM2's loader.
            frame_dtype: Optional dtype to cast frames to after loading
                        (e.g. torch.float16 for SAM3). None = keep as-is.
        """
        self.img_paths = img_paths
        self.image_size = image_size
        self.offload_video_to_cpu = offload_video_to_cpu
        self.compute_device = compute_device
        self.cache_size = cache_size
        self.frame_dtype = frame_dtype

        # Ensure mean/std are on the correct device
        target_device = torch.device('cpu') if offload_video_to_cpu else compute_device
        self.img_mean = img_mean.to(target_device)
        self.img_std = img_std.to(target_device)

        # Choose frame loader function
        if frame_loader_fn is not None:
            self._load_img_as_tensor = frame_loader_fn
        else:
            from sam2.utils.misc import _load_img_as_tensor
            self._load_img_as_tensor = _load_img_as_tensor

        # LRU cache: OrderedDict (most recently used at end)
        self.cache = OrderedDict()

        # Load first frame to obtain video dimensions and seed the cache
        img, self.video_height, self.video_width = self._load_img_as_tensor(
            img_paths[0], image_size
        )
        self._add_to_cache(0, self._normalize_frame(img))

        print(f"LazyVideoFrameLoader initialized:")
        print(f"  Total frames: {len(img_paths)}")
        print(f"  Cache size: {cache_size} frames")
        print(f"  Video dimensions: {self.video_width}x{self.video_height}")
        print(f"  Target size: {image_size}x{image_size}")
        print(f"  Frame dtype: {frame_dtype or 'default (float32)'}")

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        """Load frame on demand with LRU caching."""
        if index in self.cache:
            self.cache.move_to_end(index)
            return self.cache[index]

        img, _, _ = self._load_img_as_tensor(self.img_paths[index], self.image_size)
        img = self._normalize_frame(img)
        self._add_to_cache(index, img)
        return img

    def _normalize_frame(self, img):
        """Cast dtype, move to device, normalize."""
        # Cast dtype first (e.g. float16 for SAM3)
        if self.frame_dtype is not None:
            img = img.to(dtype=self.frame_dtype)

        # Move to compute device if not offloading
        if not self.offload_video_to_cpu:
            img = img.to(self.compute_device)

        img = img - self.img_mean
        img = img / self.img_std
        return img

    def _add_to_cache(self, index, img):
        """Add to LRU cache, evicting oldest entry if full."""
        if len(self.cache) >= self.cache_size:
            self.cache.popitem(last=False)  # evict least recently used
        self.cache[index] = img

    @property
    def device(self):
        """Device of cached frames (tensor-like compatibility)."""
        if self.cache:
            return next(iter(self.cache.values())).device
        return self.compute_device if not self.offload_video_to_cpu else torch.device('cpu')


def enable_lazy_loading(cache_size=20, enable_sam3=True):
    """
    Monkey-patch SAM2 and SAM3 to use lazy frame loading.
    Call this BEFORE creating any SAM2VideoPredictor or SAM3 session.

    Patches two code paths:
      1. SAM2: sam2.utils.misc.load_video_frames_from_jpg_images
         (also patched in sam3.model.utils.sam2_utils for SAM2-compat API)
      2. SAM3 text pipeline: sam3.model.io_utils.load_video_frames_from_image_folder
         (used by start_session → init_state for text-based segmentation)

    Args:
        cache_size: Frames to keep in LRU cache (default 20, ~2GB for 1008px frames)
        enable_sam3: Also patch SAM3 paths (default True)

    Example:
        enable_lazy_loading(cache_size=20)
        predictor = build_sam2_video_predictor(...)
        sam3_model.start_session(resource_path=frames_dir, ...)
    """
    import sam2.utils.misc as sam2_misc

    # ------------------------------------------------------------------ #
    # Patch 1: SAM2's load_video_frames_from_jpg_images                   #
    # ------------------------------------------------------------------ #
    if not hasattr(sam2_misc, '_original_load_video_frames_from_jpg_images'):
        sam2_misc._original_load_video_frames_from_jpg_images = \
            sam2_misc.load_video_frames_from_jpg_images

    def lazy_load_video_frames_from_jpg_images(
        video_path,
        image_size,
        offload_video_to_cpu,
        img_mean=(0.485, 0.456, 0.406),
        img_std=(0.229, 0.224, 0.225),
        async_loading_frames=False,
        compute_device=torch.device("cuda"),
    ):
        if isinstance(video_path, str) and os.path.isdir(video_path):
            jpg_folder = video_path
        else:
            raise NotImplementedError(
                "Only JPEG folder supported. "
                "Use --frame-dir to extract frames first."
            )

        frame_names = [
            p for p in os.listdir(jpg_folder)
            if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        if len(frame_names) == 0:
            raise RuntimeError(f"No JPEG images found in {jpg_folder}")

        img_paths = [os.path.join(jpg_folder, fn) for fn in frame_names]

        img_mean_t = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
        img_std_t = torch.tensor(img_std, dtype=torch.float32)[:, None, None]

        lazy_images = LazyVideoFrameLoader(
            img_paths, image_size, offload_video_to_cpu,
            img_mean_t, img_std_t, compute_device, cache_size,
        )
        return lazy_images, lazy_images.video_height, lazy_images.video_width

    sam2_misc.load_video_frames_from_jpg_images = lazy_load_video_frames_from_jpg_images

    patched = ["SAM2 (jpg images)"]

    # ------------------------------------------------------------------ #
    # Patch 2 & 3: SAM3-specific paths                                    #
    # ------------------------------------------------------------------ #
    if enable_sam3:
        # Patch 2: SAM3's SAM2-compat shim (used by SAM2-style point API)
        try:
            import sam3.model.utils.sam2_utils as sam3_utils
            if not hasattr(sam3_utils, '_original_load_video_frames_from_jpg_images'):
                sam3_utils._original_load_video_frames_from_jpg_images = \
                    sam3_utils.load_video_frames_from_jpg_images
            sam3_utils.load_video_frames_from_jpg_images = \
                lazy_load_video_frames_from_jpg_images
            patched.append("SAM3 sam2_utils shim")
        except (ImportError, AttributeError):
            pass

        # Patch 3: SAM3's text pipeline loader
        # Used by start_session → init_state → load_resource_as_video_frames
        # → load_video_frames_from_image_folder (when resource_path is a directory)
        try:
            import sam3.model.io_utils as sam3_io_utils
            from sam3.model.io_utils import _load_img_as_tensor as sam3_load_img

            if not hasattr(sam3_io_utils, '_original_load_video_frames_from_image_folder'):
                sam3_io_utils._original_load_video_frames_from_image_folder = \
                    sam3_io_utils.load_video_frames_from_image_folder

            def lazy_load_video_frames_from_image_folder(
                image_folder,
                image_size,
                offload_video_to_cpu,
                img_mean,
                img_std,
                async_loading_frames,
            ):
                """
                LRU-cached replacement for SAM3's eager/async image folder loader.

                SAM3 uses float16 for frame storage (mean=std=0.5 typically).
                We replicate that: load as float32, cast to float16, normalize.
                The async_loading_frames flag is ignored — LRU is strictly better.
                """
                from sam3.model.io_utils import IMAGE_EXTS

                frame_names = [
                    p for p in os.listdir(image_folder)
                    if os.path.splitext(p)[-1].lower() in IMAGE_EXTS
                ]
                try:
                    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
                except ValueError:
                    frame_names.sort()

                if len(frame_names) == 0:
                    raise RuntimeError(f"No images found in {image_folder}")

                img_paths = [os.path.join(image_folder, fn) for fn in frame_names]

                # SAM3 normalizes with float16 tensors; replicate that here.
                # img_mean/img_std arrive as tuples of floats from init_state.
                img_mean_t = torch.tensor(img_mean, dtype=torch.float16)[:, None, None]
                img_std_t = torch.tensor(img_std, dtype=torch.float16)[:, None, None]

                lazy_images = LazyVideoFrameLoader(
                    img_paths=img_paths,
                    image_size=image_size,
                    offload_video_to_cpu=offload_video_to_cpu,
                    img_mean=img_mean_t,
                    img_std=img_std_t,
                    compute_device=torch.device("cuda"),
                    cache_size=cache_size,
                    frame_loader_fn=sam3_load_img,
                    frame_dtype=torch.float16,
                )
                return lazy_images, lazy_images.video_height, lazy_images.video_width

            sam3_io_utils.load_video_frames_from_image_folder = \
                lazy_load_video_frames_from_image_folder
            patched.append("SAM3 io_utils (text pipeline)")

        except (ImportError, AttributeError):
            pass

    print("=" * 60)
    print("LAZY LOADING ENABLED")
    print("=" * 60)
    print(f"Patched: {', '.join(patched)}")
    print(f"Cache size: {cache_size} frames")
    print(f"Approx memory: ~{cache_size * 6 / 1024:.1f} GB  "
          f"(vs ~{cache_size * 6 / 1024 * (cache_size * 50):.0f} GB eager for large videos)")
    print()


def disable_lazy_loading():
    """Restore original frame loading functions."""
    import sam2.utils.misc as sam2_misc
    if hasattr(sam2_misc, '_original_load_video_frames_from_jpg_images'):
        sam2_misc.load_video_frames_from_jpg_images = \
            sam2_misc._original_load_video_frames_from_jpg_images

    try:
        import sam3.model.utils.sam2_utils as sam3_utils
        if hasattr(sam3_utils, '_original_load_video_frames_from_jpg_images'):
            sam3_utils.load_video_frames_from_jpg_images = \
                sam3_utils._original_load_video_frames_from_jpg_images
    except ImportError:
        pass

    try:
        import sam3.model.io_utils as sam3_io_utils
        if hasattr(sam3_io_utils, '_original_load_video_frames_from_image_folder'):
            sam3_io_utils.load_video_frames_from_image_folder = \
                sam3_io_utils._original_load_video_frames_from_image_folder
    except ImportError:
        pass

    print("Lazy loading disabled — using original eager loaders.")


if __name__ == "__main__":
    print(__doc__)
