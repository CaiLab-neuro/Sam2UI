#!/usr/bin/env python3
"""
SAM2 Annotation Processor
========================

Takes annotation JSON from SAM2 Video UI and video file,
then exports segmented video and masks.

Usage:
    python process_annotations.py <annotation_file> <video_file> [options]

Examples:
    # Use default model (sam2 base+)
    python process_annotations.py annotations.json video.mp4
    
    # Use SAM2.1 large model
    python process_annotations.py annotations.json video.mp4 --model sam2.1-large
    
    # Use custom config and checkpoint
    python process_annotations.py annotations.json video.mp4 --config configs/sam2/sam2_hiera_l.yaml --checkpoint checkpoints/sam2_hiera_large.pt
    
    # Process with chunking for large videos (reduces memory usage)
    python process_annotations.py annotations.json video.mp4 --chunk-size 100
"""

import os
import sys
import json
import time
import argparse
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

import psutil

# Import lazy loader BEFORE importing SAM2
from sam2_lazy_loader import enable_lazy_loading

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please run setup.py first to install dependencies.")
    sys.exit(1)

# Check SAM3 availability
def _check_sam3_available():
    """Check if SAM3 is installed and usable"""
    try:
        script_dir = Path(__file__).parent
        sam3_path = script_dir / "sam_models" / "sam3"
        if not sam3_path.exists():
            return False
        from sam3.model_builder import build_sam3_video_predictor
        return True
    except ImportError:
        return False

SAM3_AVAILABLE = _check_sam3_available()

# Model configuration mappings
# NOTE: Config paths are relative to the sam2 package (Hydra search path: pkg://sam2)
# Checkpoint paths are absolute/relative to the script directory
MODEL_CONFIGS = {
    # SAM2.1 models (recommended)
    "sam2.1-tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "sam_models/sam2/checkpoints/sam2.1_hiera_tiny.pt"),
    "sam2.1-small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "sam_models/sam2/checkpoints/sam2.1_hiera_small.pt"),
    "sam2.1-base+": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "sam_models/sam2/checkpoints/sam2.1_hiera_base_plus.pt"),
    "sam2.1-large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "sam_models/sam2/checkpoints/sam2.1_hiera_large.pt"),

    # SAM2 models (legacy)
    "sam2-tiny": ("configs/sam2/sam2_hiera_t.yaml", "sam_models/sam2/checkpoints/sam2_hiera_tiny.pt"),
    "sam2-small": ("configs/sam2/sam2_hiera_s.yaml", "sam_models/sam2/checkpoints/sam2_hiera_small.pt"),
    "sam2-base+": ("configs/sam2/sam2_hiera_b+.yaml", "sam_models/sam2/checkpoints/sam2_hiera_base_plus.pt"),
    "sam2-large": ("configs/sam2/sam2_hiera_l.yaml", "sam_models/sam2/checkpoints/sam2_hiera_large.pt"),

    # SAM3 model
    "sam3": (None, "sam_models/sam3/checkpoints/sam3.pt"),  # Config loaded automatically
}

def _auto_select_default_model():
    """Auto-select best SAM2.1 model based on GPU memory (mimics UI behavior)"""
    preference_order = [
        "sam2.1-large",       # 898MB - Best quality
        "sam2.1-base+",       # 324MB - High quality
        "sam2.1-small",       # 184MB - Good balance
        "sam2.1-tiny",        # 156MB - Fastest
    ]

    try:
        import torch
        if torch.cuda.is_available():
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

            # Adjust preference based on GPU memory
            if gpu_mem_gb >= 8:
                pass  # Keep original order (prefer large)
            elif gpu_mem_gb >= 4:
                preference_order = [p for p in preference_order if 'large' not in p]
            else:
                preference_order = [p for p in preference_order if 'large' not in p and 'base' not in p]

            print(f"Auto-selecting model based on GPU memory ({gpu_mem_gb:.1f}GB)...")
        else:
            preference_order = ["sam2.1-small", "sam2.1-tiny", "sam2.1-base+"]
            print("Auto-selecting model for CPU mode...")
    except:
        preference_order = ["sam2.1-base+", "sam2.1-small"]

    # Find first available model
    for model_name in preference_order:
        config_path, checkpoint_path = MODEL_CONFIGS.get(model_name, (None, None))
        if config_path and os.path.exists(checkpoint_path):
            print(f"Model to use if sam3 is not chosen: {model_name}")
            return model_name

    # Fallback to any available SAM2.1 model
    for model_name in MODEL_CONFIGS:
        if model_name.startswith("sam2.1"):
            config_path, checkpoint_path = MODEL_CONFIGS[model_name]
            if os.path.exists(checkpoint_path):
                print(f"  Selected: {model_name} (fallback)")
                return model_name

    print("  Selected: sam2.1-base+ (default, checkpoint may be missing)")
    return "sam2.1-base+"

class SAM2Processor:
    def __init__(self, config_file=None, checkpoint_file=None, model_name="sam2.1-base+", offload_to_cpu=False, async_loading=False, smooth_masks=False, use_bfloat16=False):
        """
        Initialize SAM2 Processor

        Args:
            config_file: Path to model config YAML (overrides model_name)
            checkpoint_file: Path to checkpoint file (overrides model_name)
            model_name: Preset model name (e.g., 'sam2-base+', 'sam2.1-large', 'sam3')
            offload_to_cpu: Use SAM2's CPU offloading for memory optimization
            async_loading: Use async frame loading (experimental, may reduce memory)
            smooth_masks: Apply morphological smoothing to reduce pixelation in masks
            use_bfloat16: Use BFloat16 precision for faster inference (requires compatible GPU)
        """
        # Store model name for detection
        self.model_name = model_name

        # Check if SAM3 was requested but is not available
        if model_name == "sam3" and not SAM3_AVAILABLE:
            print("=" * 60)
            print("WARNING: SAM3 requested but not available")
            print("=" * 60)
            print("Possible causes:")
            print("  1. SAM3 not installed (run setup.py and choose SAM3)")
            print("  2. Missing dependencies (huggingface-hub, decord, einops)")
            print("  3. Python < 3.12 or PyTorch < 2.7")
            print()
            print("Falling back to SAM2.1 base+...")
            print("=" * 60)
            print()
            # Fallback to SAM2.1 base+
            self.model_name = "sam2.1-base+"

        # Determine config and checkpoint paths
        if config_file and checkpoint_file:
            # Use custom paths
            self.config_file = config_file
            self.checkpoint_file = checkpoint_file
        elif model_name in MODEL_CONFIGS:
            # Use preset model
            self.config_file, self.checkpoint_file = MODEL_CONFIGS[model_name]
        else:
            raise ValueError(f"Unknown model name: {model_name}. Available: {list(MODEL_CONFIGS.keys())}")

        # Validate paths exist (skip config validation for SAM3)
        if self.model_name != "sam3" and self.config_file and not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        # Checkpoint is optional for some use cases, but warn if missing
        if self.checkpoint_file and not os.path.exists(self.checkpoint_file):
            print(f"WARNING: Checkpoint file not found: {self.checkpoint_file}")
            print("Model will be initialized without pre-trained weights.")
            self.checkpoint_file = None

        self.sam2_model = None
        self.video_predictor = None

        # Memory optimization configuration
        self.offload_to_cpu = offload_to_cpu
        self.async_loading = async_loading
        self.smooth_masks = smooth_masks
        self.use_bfloat16 = use_bfloat16
        self.no_backward_propagation = False  # Will be set from command line args

    @property
    def use_sam3(self):
        """Returns True if using SAM3 model"""
        return self.model_name == "sam3"

    def load_model(self):
        """Load SAM2 or SAM3 model with correct API usage"""
        model_type = "SAM3" if self.use_sam3 else "SAM2"
        print(f"Loading {model_type} model...")

        if not self.use_sam3:
            print(f"  Config: {self.config_file}")
        print(f"  Checkpoint: {self.checkpoint_file or 'None (random init)'}")

        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Device: {device}")

            # Common optimizations for both SAM2 and SAM3
            if device == "cuda":
                # Enable TF32 for Ampere GPUs (RTX 30xx+, A100) for better performance
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    print("  TensorFloat32 (TF32) enabled for Ampere GPU")

                if self.use_bfloat16:
                    # CRITICAL: Enable GLOBAL autocast before any SAM2/SAM3 operations
                    # This stays active for entire program to handle bfloat16 memory features
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
                    print("  BFloat16 mode: GLOBAL autocast enabled")
                    print("  Model weights remain in float32 (checkpoint dtype)")
                else:
                    print("  Float32 mode: native precision (no autocast)")

            # Load model based on type
            if self.use_sam3:
                # SAM3 loading
                if not SAM3_AVAILABLE:
                    raise ImportError("SAM3 not available. Run setup.py to install.")

                from sam3.model_builder import build_sam3_video_model
                print("  Building SAM3 model ...")

                # Build SAM3 model and extract tracker with SAM2-compatible API
                # (Matches sam2_ui.py implementation)
                sam3_model = build_sam3_video_model(device=device)

                # Extract the tracker component (has init_state, add_new_points, etc.)
                self.video_predictor = sam3_model.tracker
                # Attach backbone for feature extraction
                self.video_predictor.backbone = sam3_model.detector.backbone

            else:
                # SAM2 loading
                self.video_predictor = build_sam2_video_predictor(
                    config_file=self.config_file,
                    ckpt_path=self.checkpoint_file,  # Optional parameter
                    device=device
                )

            print(f"OK: {model_type} model loaded successfully")
            return True

        except Exception as e:
            print(f"ERROR: Failed to load {model_type} model: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_annotations(self, annotation_file):
        """Load annotation data from JSON file"""
        print(f"Loading annotations from {annotation_file}...")
        try:
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            if "annotations" not in data:
                raise ValueError("Invalid annotation file: missing 'annotations' field")
            
            print(f"OK: Loaded {len(data['annotations'])} annotations")
            print(f"   Video: {data.get('video_path', 'Unknown')}")
            print(f"   Total frames: {data.get('total_frames', 'Unknown')}")
            print(f"   Objects: {len(data.get('object_names', {}))}")
            
            return data
        except Exception as e:
            print(f"ERROR: Failed to load annotations: {e}")
            return None

    def _compress_video_with_ffmpeg(self, input_path, output_path, crf=23, preset='medium'):
        """
        Re-encode video with FFmpeg using CRF compression.
        Handles FFmpeg builds that don't support -preset option.

        Args:
            input_path: Temp uncompressed video
            output_path: Final compressed video
            crf: Quality (0-51, lower=better, 23=medium)
            preset: Encoding speed (medium recommended, may not be supported by all FFmpeg builds)

        Returns:
            bool: True if successful
        """
        import subprocess
        import shutil

        ffmpeg_path = shutil.which('ffmpeg')
        if not ffmpeg_path:
            print("WARNING: ffmpeg not found, skipping compression")
            shutil.move(input_path, output_path)
            return False

        try:
            print(f"Compressing with FFmpeg (CRF={crf})...")

            # Build command with preset
            cmd_with_preset = [
                ffmpeg_path,
                '-i', input_path,
                '-c:v', 'libx264',
                '-crf', str(crf),
                '-preset', preset,
                '-movflags', '+faststart',
                '-y',
                output_path
            ]

            # Try with preset first
            result = subprocess.run(cmd_with_preset, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, text=True)

            if result.returncode != 0:
                # Check if error is due to preset option not being recognized
                stderr_lower = result.stderr.lower()
                if 'preset' in stderr_lower and ('unrecognized' in stderr_lower or 'option not found' in stderr_lower):
                    print(f"  FFmpeg doesn't support -preset option (conda build), retrying without it...")

                    # Retry without preset
                    cmd_no_preset = [
                        ffmpeg_path,
                        '-i', input_path,
                        '-c:v', 'libx264',
                        '-crf', str(crf),
                        '-movflags', '+faststart',
                        '-y',
                        output_path
                    ]

                    result = subprocess.run(cmd_no_preset, stdout=subprocess.PIPE,
                                          stderr=subprocess.PIPE, text=True)
                    if result.returncode != 0:
                        print(f"FFmpeg error: {result.stderr}")
                        shutil.move(input_path, output_path)
                        return False
                else:
                    # Different error
                    print(f"FFmpeg error: {result.stderr}")
                    shutil.move(input_path, output_path)
                    return False

            # Report compression stats
            original_size = os.path.getsize(input_path)
            compressed_size = os.path.getsize(output_path)
            reduction = (1 - compressed_size / original_size) * 100

            print(f"Compressed: {original_size/1e6:.1f}MB → {compressed_size/1e6:.1f}MB ({reduction:.1f}% reduction)")

            os.remove(input_path)  # Delete temp
            return True

        except Exception as e:
            print(f"WARNING: FFmpeg failed: {e}")
            if os.path.exists(input_path):
                shutil.move(input_path, output_path)
            return False

    def get_video_info(self, video_path):
        """Get video frame count and properties without loading all frames"""
        print(f"Reading video info from {video_path}...")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")
            
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            cap.release()
            
            print(f"OK: Video info - {frame_count} frames, {fps:.2f} fps, {width}x{height}")
            return frame_count, fps, width, height
        except Exception as e:
            print(f"ERROR: Failed to read video info: {e}")
            return None, None, None, None
    
    def group_annotations_by_frame(self, annotations):
        """Group annotations by frame index"""
        frame_annotations = {}
        for annotation in annotations:
            frame_idx = annotation["frame_index"]
            if frame_idx not in frame_annotations:
                frame_annotations[frame_idx] = []
            frame_annotations[frame_idx].append(annotation)
        return frame_annotations

    def _cleanup_inference_state(self, inference_state, current_frame_idx, frames_to_keep=20, reverse=False, verbose=False):
        """
        Clean up old frames from inference state to prevent memory growth.

        Handles both SAM2 and SAM3 inference state structures:
        - SAM3: Cleans multiple caches (non_cond_frame_outputs, cond_frame_outputs, cached_frame_outputs, tracker states)
        - SAM2: Cleans per-object output dicts

        Args:
            inference_state: SAM2/SAM3 inference state object
            current_frame_idx: Current frame being processed
            frames_to_keep: Number of recent frames to keep (default 20)
            reverse: If True, propagating backward (delete frames ahead), else forward (delete frames behind)
            verbose: If True, print debug info every frame
        """
        total_deleted = 0

        # Direction-aware cleanup logic:
        # - Forward (reverse=False): Delete frames BEHIND current (f < current - keep)
        # - Backward (reverse=True): Delete frames AHEAD of current (f > current + keep)
        if reverse:
            should_delete = lambda f: f > current_frame_idx + frames_to_keep
        else:
            should_delete = lambda f: f < current_frame_idx - frames_to_keep

        # Try SAM3 structure first (direct attribute access)
        if hasattr(inference_state, 'non_cond_frame_outputs'):
            # Clean non-conditioning frame outputs (tracker state)
            non_cond = inference_state.non_cond_frame_outputs
            old_frames = [f for f in non_cond.keys() if should_delete(f)]
            for old_frame in old_frames:
                del non_cond[old_frame]
            total_deleted += len(old_frames)

            # Clean conditioning frame outputs (tracker state)
            if hasattr(inference_state, 'cond_frame_outputs'):
                cond = inference_state.cond_frame_outputs
                old_frames = [f for f in cond.keys() if should_delete(f)]
                for old_frame in old_frames:
                    del cond[old_frame]
                total_deleted += len(old_frames)

            # Clean cached_frame_outputs (Sam3VideoInference level)
            # This is a dict attribute, not a method
            if hasattr(inference_state, '__dict__') and 'cached_frame_outputs' in inference_state.__dict__:
                cached = inference_state.__dict__['cached_frame_outputs']
                if isinstance(cached, dict):
                    old_frames = [f for f in cached.keys() if isinstance(f, int) and should_delete(f)]
                    for old_frame in old_frames:
                        del cached[old_frame]
                    total_deleted += len(old_frames)

            # Clean tracker inference states (per-object tracker states)
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
                                    total_deleted += len(old_frames)

            

        # Fall back to SAM2/SAM3 dict structure (SAM3 uses this for tracker inference_state)
        elif isinstance(inference_state, dict) and "output_dict_per_obj" in inference_state:
            # CRITICAL: Clean the MAIN output_dict first (SAM3 stores frame outputs here)
            # This is where sam3_tracking_predictor.py:859 stores outputs: output_dict[storage_key][frame_idx] = current_out
            # But ONLY clean non_cond_frame_outputs, NOT cond_frame_outputs!
            if "output_dict" in inference_state:
                output_dict = inference_state["output_dict"]

                # Only clean non-conditional frames (intermediate propagation frames - safe to delete)
                if "non_cond_frame_outputs" in output_dict:
                    cache = output_dict["non_cond_frame_outputs"]
                    old_frames = [f for f in cache.keys() if should_delete(f)]
                    for old_frame in old_frames:
                        del cache[old_frame]
                    total_deleted += len(old_frames)

                # DO NOT clean cond_frame_outputs - SAM3 needs these for entire propagation!
                # These are frames with user prompts that SAM3 references during tracking.
                # Deleting these causes: AssertionError at sam3_tracker_base.py:591

            # Then clean per-object dicts (these are slices/views of the main dict)
            for obj_idx in range(len(inference_state.get("obj_ids", []))):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                non_cond = obj_output_dict.get("non_cond_frame_outputs", {})
                old_frames = [f for f in non_cond.keys() if should_delete(f)]
                for old_frame in old_frames:
                    del non_cond[old_frame]
                total_deleted += len(old_frames)

            

        # Periodic GPU memory cleanup
        if current_frame_idx % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def process_segmentation(self, video_path, annotations_data, output_dir, frame_dir=None):
        """Process segmentation using SAM2"""
        return self.process_segmentation_full(video_path, annotations_data, output_dir, frame_dir=frame_dir)

    def process_segmentation_full(self, video_path, annotations_data, output_dir, frame_dir=None):
        """Process segmentation with streaming mask export to reduce memory usage"""
        print("Starting segmentation process (streaming export mode)...")

        # Group annotations by frame and object
        frame_annotations = self.group_annotations_by_frame(annotations_data["annotations"])

        # Create or reuse frame directory
        if frame_dir:
            # Use persistent frame directory
            temp_dir = Path(frame_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)
            is_persistent = True

            # Check if frames already exist
            existing_frames = sorted(temp_dir.glob("*.jpg"))
            if existing_frames:
                print(f"  Reusing existing frames from: {temp_dir}")
                print(f"  Found {len(existing_frames)} frames")
                skip_extraction = True
            else:
                print(f"  Extracting frames to persistent directory: {temp_dir}")
                skip_extraction = False
        else:
            # Use temporary directory (will be deleted after processing)
            temp_dir = Path(tempfile.mkdtemp(prefix='sam2_frames_'))
            is_persistent = False
            skip_extraction = False
            print(f"  Extracting frames to temporary directory: {temp_dir}")

        try:
            # Extract frames if needed
            if not skip_extraction:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video file: {video_path}")

                save_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frame_path = temp_dir / f"{save_idx:05d}.jpg"
                    cv2.imwrite(str(frame_path), frame)
                    save_idx += 1

                    if save_idx % 500 == 0:
                        print(f"  Extracted {save_idx} frames...")

                cap.release()
                print(f"  Extracted {save_idx} frames")
            else:
                print(f"  Skipping frame extraction (using existing frames)")

            # Get frame dimensions for coordinate conversion (needed for SAM3)
            # SAM3 requires relative [0-1] coordinates, so we need to know frame dimensions
            if self.use_sam3:
                # Get dimensions from first frame
                first_frame_path = sorted(temp_dir.glob("*.jpg"))[0]
                first_frame = cv2.imread(str(first_frame_path))
                if first_frame is None:
                    raise ValueError(f"Cannot read first frame: {first_frame_path}")
                frame_height, frame_width = first_frame.shape[:2]
                print(f"  Frame dimensions for SAM3 coordinate conversion: {frame_width}x{frame_height}")
                del first_frame  # Free memory
            else:
                frame_width = frame_height = None  # Not needed for SAM2

            # Initialize SAM2 inference state with JPEG directory
            print("Initializing SAM inference state...")

            # Build init_state parameters
            init_params = {'video_path': str(temp_dir)}

            # Add offloading if configured
            if hasattr(self, 'offload_to_cpu') and self.offload_to_cpu:
                init_params['offload_video_to_cpu'] = True
                init_params['offload_state_to_cpu'] = True
                print("  Using CPU offloading for memory optimization")

            # Add async loading if configured
            if hasattr(self, 'async_loading') and self.async_loading:
                init_params['async_loading_frames'] = True
                print("  Using async frame loading (experimental)")

            # Get device from the model
            device = str(self.video_predictor.device)

            # No local autocast needed - global autocast was enabled in load_model() if using bfloat16
            # (see benchmark.py line 20 for this pattern)
            if device == "cuda" and self.use_bfloat16:
                autocast_mode = "BFloat16 (global autocast)"
            else:
                autocast_mode = "Float32 (native)" if device == "cuda" else "CPU"

            # No 'with autocast_context' wrapper needed - global autocast already active
            inference_state = self.video_predictor.init_state(**init_params)
            num_frames = inference_state["num_frames"]

            print(f"  Initialized state for {num_frames} frames (autocast: {autocast_mode})")

            # Process each annotated frame
            object_names = annotations_data.get("object_names", {})
            object_colors = annotations_data.get("object_colors", {})

            # Group annotations by object ID
            objects_with_annotations = {}
            for frame_idx, annotations in frame_annotations.items():
                for annotation in annotations:
                    obj_id = annotation["object_id"]
                    if obj_id not in objects_with_annotations:
                        objects_with_annotations[obj_id] = {}
                    if frame_idx not in objects_with_annotations[obj_id]:
                        objects_with_annotations[obj_id][frame_idx] = []
                    objects_with_annotations[obj_id][frame_idx].append(annotation)

            print(f"Processing {len(objects_with_annotations)} objects...")

            # Add annotation points for each object
            for obj_id, obj_frames in objects_with_annotations.items():
                obj_name = object_names.get(str(obj_id), f"Object_{obj_id}")
                print(f"\nProcessing {obj_name} (ID: {obj_id})...")

                for frame_idx, annotations in sorted(obj_frames.items()):
                    if frame_idx >= num_frames:
                        print(f"  WARNING: Skipping frame {frame_idx} (beyond video length)")
                        continue

                    points = []
                    labels = []

                    for annotation in annotations:
                        x, y = annotation["x"], annotation["y"]
                        is_positive = annotation["is_positive"]
                        points.append([x, y])
                        labels.append(1 if is_positive else 0)

                    if not points:
                        continue

                    # Convert points for SAM3 if needed
                    if self.use_sam3:
                        # SAM3 requires relative [0-1] coordinates
                        rel_points = [[x / frame_width, y / frame_height] for x, y in points]
                        points_np = np.array(rel_points, dtype=np.float32)
                        print(f"    SAM3 coordinate conversion: Pixel {points[0]} → Relative [{rel_points[0][0]:.4f}, {rel_points[0][1]:.4f}]")
                    else:
                        # SAM2 uses pixel coordinates
                        points_np = np.array(points, dtype=np.float32)

                    labels_np = np.array(labels, dtype=np.int32)

                    print(f"  Frame {frame_idx}: {len(points)} points")

                    try:
                        # SAM3 returns 4 values, SAM2 returns 3
                        if self.use_sam3:
                            _, out_obj_ids, low_res_masks, video_res_masks = self.video_predictor.add_new_points(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                points=points_np,
                                labels=labels_np,
                            )
                        else:
                            _, out_obj_ids, out_mask_logits = self.video_predictor.add_new_points(
                                inference_state=inference_state,
                                frame_idx=frame_idx,
                                obj_id=obj_id,
                                points=points_np,
                                labels=labels_np,
                            )
                    except Exception as e:
                        print(f"    WARNING: Error adding points: {e}")
                        continue

            
            # Create output directories for streaming export
            masks_dir = Path(output_dir) / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

            # Propagate annotations - FORWARD direction (frame 0 → end)
            # IMPORTANT: Always propagate from frame 0 to ensure full video coverage
            print(f"\nPropagating annotations FORWARD (frames 0 to {num_frames-1})...")

            masks_metadata = {}  # Only metadata, not actual mask arrays

            # CRITICAL: Nested autocast context to handle bfloat16 tensors from CPU offloading
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Construct propagation call based on model type
                if self.use_sam3:
                    # SAM3 requires explicit frame range
                    propagate_iterator = self.video_predictor.propagate_in_video(
                        inference_state,
                        start_frame_idx=0,  # Start from beginning for full coverage
                        max_frame_num_to_track=num_frames,
                        reverse=False,
                        propagate_preflight=True  # Consolidate points before propagation
                    )
                else:
                    # SAM2: Use default propagation (starts from annotation frames and goes forward)
                    # Note: SAM2 doesn't support explicit start_frame_idx, so we rely on reverse=False
                    # covering from annotations forward to end
                    propagate_iterator = self.video_predictor.propagate_in_video(
                        inference_state, reverse=False
                    )

                for result in propagate_iterator:
                    # Unpack based on model type (SAM3 returns 5 values, SAM2 returns 3)
                    if self.use_sam3:
                        out_frame_idx, out_obj_ids, out_low_res_masks, out_mask_logits, out_obj_scores = result
                    else:
                        out_frame_idx, out_obj_ids, out_mask_logits = result

                    frame_masks = {}

                    for i, obj_id in enumerate(out_obj_ids):
                        mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

                        # Get object info
                        obj_name = object_names.get(str(obj_id), f"Object_{obj_id}")
                        obj_color = object_colors.get(str(obj_id), [255, 0, 0])

                        # Export mask to disk immediately (streaming export)
                        mask_filename = f"mask_f{out_frame_idx:06d}_{obj_name}_id{obj_id}.png"
                        mask_path = masks_dir / mask_filename
                        cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

                        # Store metadata only (no mask array!)
                        score = out_obj_scores[i] if self.use_sam3 else 1.0
                        frame_masks[obj_id] = {
                            'filename': mask_filename,
                            'score': float(score),
                            'name': obj_name,
                            'color': obj_color
                        }

                        # Explicitly delete mask array
                        del mask

                    masks_metadata[out_frame_idx] = frame_masks

                    # CRITICAL: Delete ALL tensor variables to prevent memory accumulation
                    # SAM3 returns 5 values, SAM2 returns 3 - delete all applicable tensors
                    del out_mask_logits  # High-res masks (always present)

                    if self.use_sam3:
                        # SAM3-specific tensors that must be deleted
                        del out_low_res_masks  # Low-res masks (can accumulate ~2-3MB per frame)
                        del out_obj_scores     # Confidence scores
                        del out_obj_ids        # Object ID tensor/list

                    del result  # Delete the unpacked tuple itself

                    # CRITICAL: Clean up old frames EVERY frame to prevent memory growth
                    # This matches the UI behavior (sam2_ui.py:3799)
                    # IMPORTANT: Pass reverse=False for forward propagation (delete frames behind, not ahead)
                    self._cleanup_inference_state(inference_state, out_frame_idx, frames_to_keep=20, reverse=False, verbose=True)

                    if (out_frame_idx + 1) % 50 == 0:
                        # Monitor GPU and RAM usage
                        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
                        gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)
                        process = psutil.Process()
                        ram_used = process.memory_info().rss / (1024**3)

                        print(f"  Forward: Frame {out_frame_idx + 1}/{num_frames} | "
                              f"GPU: {gpu_allocated:.2f}GB (peak: {gpu_peak:.2f}GB) | "
                              f"RAM: {ram_used:.2f}GB")

                        # Verify tensors are deleted (should show NameError if properly deleted)
                        if self.use_sam3:
                            try:
                                _ = out_low_res_masks
                                print(f"  WARNING: out_low_res_masks still in scope!")
                            except NameError:
                                pass  # Expected - variable was deleted

                        torch.cuda.reset_peak_memory_stats()

            # Propagate annotations - BACKWARD direction (last frame → 0)
            # IMPORTANT: Always propagate backward to ensure full video coverage from both directions
            # This provides better quality by having bidirectional temporal context

            if self.no_backward_propagation:
                print("\n  Skipping backward propagation (disabled by --no-backward flag)")
                print("  WARNING: This may result in lower quality segmentation as only forward propagation is used")
            else:
                print(f"\nPropagating annotations BACKWARD (frames {num_frames-1} to 0)...")

                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Construct propagation call based on model type
                    if self.use_sam3:
                        # SAM3 requires explicit frame range for backward propagation
                        propagate_iterator = self.video_predictor.propagate_in_video(
                            inference_state,
                            start_frame_idx=num_frames - 1,  # Start from end
                            max_frame_num_to_track=num_frames,
                            reverse=True,
                            propagate_preflight=True
                        )
                    else:
                        # SAM2: Use default backward propagation
                        propagate_iterator = self.video_predictor.propagate_in_video(
                            inference_state, reverse=True
                        )

                    for result in propagate_iterator:
                        # Unpack based on model type
                        # SAM3 returns 5 values for BOTH forward and backward
                        # SAM2 returns 3 values
                        if self.use_sam3:
                            out_frame_idx, out_obj_ids, out_low_res_masks, out_mask_logits, out_obj_scores = result
                        else:
                            out_frame_idx, out_obj_ids, out_mask_logits = result

                        frame_masks = {}

                        for i, obj_id in enumerate(out_obj_ids):
                            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()

                            # Get object info
                            obj_name = object_names.get(str(obj_id), f"Object_{obj_id}")
                            obj_color = object_colors.get(str(obj_id), [255, 0, 0])

                            # Export mask to disk immediately
                            mask_filename = f"mask_f{out_frame_idx:06d}_{obj_name}_id{obj_id}.png"
                            mask_path = masks_dir / mask_filename
                            cv2.imwrite(str(mask_path), (mask * 255).astype(np.uint8))

                            # Store metadata with confidence score for SAM3
                            score = out_obj_scores[i].item() if self.use_sam3 else 1.0
                            frame_masks[obj_id] = {
                                'filename': mask_filename,
                                'score': float(score),
                                'name': obj_name,
                                'color': obj_color
                            }

                            del mask

                        masks_metadata[out_frame_idx] = frame_masks

                        # CRITICAL: Delete ALL tensor variables to prevent memory accumulation
                        # SAM3 returns 5 values, SAM2 returns 3 - delete all applicable tensors
                        del out_mask_logits  # High-res masks (always present)

                        if self.use_sam3:
                            # SAM3-specific tensors that must be deleted
                            del out_low_res_masks  # Low-res masks (can accumulate ~2-3MB per frame)
                            del out_obj_scores     # Confidence scores
                            del out_obj_ids        # Object ID tensor/list

                        del result  # Delete the unpacked tuple itself

                        # CRITICAL: Clean up old frames EVERY frame to prevent memory growth
                        # This matches the UI behavior (sam2_ui.py:3799)
                        # IMPORTANT: Pass reverse=True for backward propagation (delete frames ahead, not behind)
                        self._cleanup_inference_state(inference_state, out_frame_idx, frames_to_keep=20, reverse=True, verbose=True)

                        if (out_frame_idx + 1) % 50 == 0:
                            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
                            gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)
                            process = psutil.Process()
                            ram_used = process.memory_info().rss / (1024**3)

                            print(f"  Backward: Frame {out_frame_idx + 1}/{num_frames} | "
                                  f"GPU: {gpu_allocated:.2f}GB (peak: {gpu_peak:.2f}GB) | "
                                  f"RAM: {ram_used:.2f}GB")

                            # Verify tensors are deleted (should show NameError if properly deleted)
                            if self.use_sam3:
                                try:
                                    _ = out_low_res_masks
                                    print(f"  WARNING: out_low_res_masks still in scope!")
                                except NameError:
                                    pass  # Expected - variable was deleted

                            torch.cuda.reset_peak_memory_stats()

            # Phase 2: Clear inference state frame outputs (safe after propagation)
            model_type = "SAM3" if self.use_sam3 else "SAM2"
            print(f"\nCleaning up {model_type} inference state...")

            # Handle SAM3 structure (direct attribute access)
            if hasattr(inference_state, 'non_cond_frame_outputs'):
                inference_state.non_cond_frame_outputs.clear()

            # Handle SAM2 structure (per-object dict access)
            elif isinstance(inference_state, dict) and "output_dict_per_obj" in inference_state:
                for obj_idx in range(len(objects_with_annotations)):
                    obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                    # Clear non-conditioning frames (keep conditioning for potential refinement)
                    non_cond = obj_output_dict.get("non_cond_frame_outputs", {})
                    non_cond.clear()

            torch.cuda.empty_cache()
            print(f"OK: Cleaned {model_type} memory")

            print(f"\nOK: Generated masks for {len(masks_metadata)} frames")
            return masks_metadata, object_names, object_colors, num_frames

        finally:
            # Clean up temporary directory (only if not persistent)
            if not is_persistent:
                try:
                    shutil.rmtree(temp_dir)
                    print(f"Cleaned up temporary frames: {temp_dir}")
                except Exception as e:
                    print(f"WARNING: Could not clean up temp directory: {e}")
            else:
                print(f"Keeping frames in persistent directory: {temp_dir}")
    
    def export_masks(self, masks_by_frame, video_path, object_names, output_dir):
        """Verify mask images (already exported during propagation)"""
        print("Verifying mask images...")

        masks_dir = Path(output_dir) / "masks"
        verified_count = 0
        missing_count = 0

        for frame_idx in sorted(masks_by_frame.keys()):
            frame_masks = masks_by_frame[frame_idx]
            for obj_id, mask_data in frame_masks.items():
                mask_path = masks_dir / mask_data['filename']
                if not mask_path.exists():
                    print(f"WARNING: Missing mask file: {mask_path}")
                    missing_count += 1
                else:
                    verified_count += 1

        if missing_count > 0:
            print(f"WARNING: {missing_count} mask files are missing!")
        print(f"OK: Verified {verified_count} mask files in {masks_dir}")
        return verified_count

    def _get_contrasting_text_color(self, bg_color):
        """Calculate contrasting text color (white or black) based on background luminance

        Args:
            bg_color: BGR color tuple (e.g., [255, 0, 0] for blue)

        Returns:
            (B, G, R) tuple: (255, 255, 255) for white or (0, 0, 0) for black
        """
        # Convert BGR to RGB
        if isinstance(bg_color, (list, tuple)) and len(bg_color) >= 3:
            r, g, b = bg_color[2], bg_color[1], bg_color[0]  # BGR to RGB
        else:
            r, g, b = 255, 255, 255  # Default white

        # Calculate relative luminance (ITU-R BT.709)
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

        # Return white for dark colors, black for bright colors
        return (255, 255, 255) if luminance < 128 else (0, 0, 0)

    def load_video_frames_for_export(self, video_path):
        """Load video frames for export (needed after segmentation)"""
        print(f"Loading video frames for export...")
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError(f"Cannot open video file: {video_path}")

            frames = []
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                frame_count += 1

                if frame_count % 100 == 0:
                    print(f"   Loaded {frame_count} frames...")

            cap.release()
            print(f"OK: Loaded {len(frames)} frames for export")
            return frames
        except Exception as e:
            print(f"ERROR: Failed to load video frames: {e}")
            return None

    def export_video(self, masks_by_frame, video_path, object_names, object_colors,
                    output_dir, fps=30, overlay_opacity=0.4, compress=True, crf=23):
        """Export segmented video with overlays and optional compression"""
        print("Exporting segmented video...")

        frames = self.load_video_frames_for_export(video_path)
        if not frames:
            return False

        video_output_path_final = Path(output_dir) / "segmented_video.avi"
        video_output_path_temp = Path(output_dir) / "segmented_video.temp.avi" if compress else video_output_path_final
        
        height, width = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        out = cv2.VideoWriter(str(video_output_path_temp.with_suffix(".avi")),
            fourcc, fps, (width, height))

        # Use H.264 codec for much better compression and quality
        # Try different codec identifiers based on system availability
        # for codec in ['avc1', 'H264', 'X264', 'mp4v']:
        #     fourcc = cv2.VideoWriter_fourcc(*codec)
        #     out = cv2.VideoWriter(str(video_output_path_temp), fourcc, fps, (width, height))
        #     if out.isOpened():
        #         if codec != 'mp4v':
        #             print(f"  Using {codec} codec for video encoding")
        #         break

        if not out.isOpened():
            print("  WARNING: Could not initialize video writer with any codec")
            return False
        
        processed_frames = 0
        masks_dir = Path(output_dir) / "masks"

        for frame_idx, frame in enumerate(frames):
            overlay_frame = frame.copy()

            if frame_idx in masks_by_frame:
                frame_masks = masks_by_frame[frame_idx]

                # Collect all mask data first
                mask_data_list = []
                for obj_id, mask_data in frame_masks.items():
                    # Load mask from disk
                    mask_path = masks_dir / mask_data['filename']
                    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                    if mask is None:
                        print(f"WARNING: Could not load mask: {mask_path}")
                        continue
                    mask = (mask > 0).astype(bool)  # Convert to boolean

                    color = mask_data['color']
                    name = mask_data['name']

                    # Resize mask if needed
                    if mask.shape != (height, width):
                        from PIL import Image as PILImage
                        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                        # Use BILINEAR interpolation for smoother edges (vs NEAREST which causes pixelation)
                        mask_pil = mask_pil.resize((width, height), PILImage.BILINEAR)
                        # Threshold at 127 to maintain binary mask after interpolation
                        mask = np.array(mask_pil) > 127

                    mask_data_list.append((mask, color, name, obj_id))

                if mask_data_list:
                    # Create overlay with averaged colors for overlapping regions
                    combined_overlay = np.zeros((height, width, 3), dtype=np.float32)
                    overlap_count = np.zeros((height, width), dtype=np.int32)

                    for mask_bool, color, name, obj_id in mask_data_list:
                        # Add color to overlapping regions (accumulate for averaging)
                        color_bgr = color[::-1] if isinstance(color, (list, tuple)) else [0, 0, 255]
                        combined_overlay[mask_bool] += color_bgr
                        overlap_count[mask_bool] += 1

                    # Average colors where masks overlap
                    mask_pixels = overlap_count > 0
                    combined_overlay[mask_pixels] /= overlap_count[mask_pixels, np.newaxis]
                    combined_overlay = combined_overlay.astype(np.uint8)

                    # Single blend operation - fixes cumulative darkening bug
                    overlay_frame = cv2.addWeighted(frame, 1-overlay_opacity,
                                                  combined_overlay, overlay_opacity, 0)

                    # Add text labels with smart contrast
                    for mask_bool, color, name, obj_id in mask_data_list:
                        if mask_bool.any():
                            y_coords, x_coords = np.where(mask_bool)
                            if len(y_coords) > 0:
                                center_x = int(np.mean(x_coords))
                                center_y = int(np.mean(y_coords))

                                # Calculate contrasting text color based on mask color luminance
                                text_color = self._get_contrasting_text_color(color)

                                cv2.putText(overlay_frame, name, (center_x, center_y),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            out.write(overlay_frame)
            processed_frames += 1
            
            if processed_frames % 100 == 0:
                print(f"   Processed {processed_frames}/{len(frames)} frames...")

        out.release()

        # Compress with FFmpeg if requested
        if compress:
            print("Compressing video with FFmpeg...")
            success = self._compress_video_with_ffmpeg(
                str(video_output_path_temp),
                str(video_output_path_final),
                crf=crf,
                preset='medium'
            )
            if success:
                print(f"OK: Exported and compressed segmented video to {video_output_path_final}")
            else:
                print(f"OK: Exported segmented video to {video_output_path_final} (compression skipped)")
        else:
            print(f"OK: Exported segmented video to {video_output_path_final}")

        return True
    
    def export_metadata(self, annotations_data, masks_by_frame, output_dir, num_frames=None, video_path=None):
        """Export processing metadata with file paths"""
        print("Exporting metadata...")

        metadata = {
            "processing_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config_file": self.config_file,
                "checkpoint_file": self.checkpoint_file,
                "offload_to_cpu": self.offload_to_cpu,
                "total_frames_processed": len(masks_by_frame),
                "total_masks_generated": sum(len(masks) for masks in masks_by_frame.values()),
                "objects_detected": list(set(
                    obj_id for masks in masks_by_frame.values()
                    for obj_id in masks.keys()
                ))
            },
            "file_paths": {
                "original_video_path": str(Path(video_path).resolve()) if video_path else None,
                "segmented_video_filename": "segmented_video.avi",
                "metadata_filename": "processing_metadata.json"
            },
            "original_annotations": annotations_data
        }

        # Save metadata
        metadata_path = Path(output_dir) / "processing_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"OK: Exported metadata to {metadata_path}")
        return True

def main():
    """Main processing function"""
    parser = argparse.ArgumentParser(
        description="Process SAM2 annotations and generate segmented output",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available preset models:
  {', '.join(MODEL_CONFIGS.keys())}

Examples:
  # Use default model (sam2-base+)
  python process_annotations.py annotations.json video.mp4

  # Use SAM2.1 large model
  python process_annotations.py annotations.json video.mp4 --model sam2.1-large

  # Use CPU offloading for memory optimization
  python process_annotations.py annotations.json video.mp4 --offload-to-cpu

  # Use custom config and checkpoint
  python process_annotations.py annotations.json video.mp4 \\
    --config configs/sam2/sam2_hiera_l.yaml \\
    --checkpoint checkpoints/sam2_hiera_large.pt
        """
    )

    parser.add_argument("annotation_file", help="Path to annotation JSON file from SAM2 Video UI")
    parser.add_argument("video_file", help="Path to input video file")
    parser.add_argument("--output_dir", default="sam2_output", help="Output directory (default: sam2_output)")

    # Model selection (mutually exclusive)
    model_group = parser.add_mutually_exclusive_group()
    model_group.add_argument("--model", default=None,
                           choices=list(MODEL_CONFIGS.keys()),
                           help="Preset model name (default: auto-select based on GPU memory)")
    model_group.add_argument("--config", help="Custom config YAML path (requires --checkpoint)")

    parser.add_argument("--checkpoint", help="Custom checkpoint path (requires --config)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30)")
    parser.add_argument("--opacity", type=float, default=0.4, help="Mask overlay opacity (default: 0.4)")
    parser.add_argument("--offload-to-cpu", action="store_true",
                       help="Offload video frames and model state to CPU to reduce GPU memory usage (slightly increases CPU memory usage and at a cost of a slightly slower speed)")
    parser.add_argument("--async-loading", action="store_true",
                       help="Use async frame loading (not much benefit. Just kept this feature as it is available)")
    parser.add_argument("--smooth-masks", action="store_true",
                       help="Apply morphological smoothing to reduce pixelation in exported masks (preserves binary masks)")
    parser.add_argument("--use-bfloat16", action="store_true",
                       help="Use BFloat16 mixed precision for faster inference and reduced memory usage (requires Ampere+ GPU with BFloat16 support, e.g., RTX 30xx+, A100). Uses torch.autocast following SAM2's official benchmark pattern.")
    parser.add_argument("--frame-dir", type=str, default=None,
                       help="Persistent directory for video frames (default: auto-generated in /tmp). If specified, frames will be reused from previous runs and not deleted after processing.")
    parser.add_argument("--frame-cache-size", type=int, default=20,
                       help="Number of frames to keep in memory cache (default: 20, ~2GB). Minimum: 10, Recommended: 20-50.")
    parser.add_argument("--no-backward", action="store_true", dest="no_backward",
                       help="Disable backward propagation (not recommended, may result in lower quality segmentation for frames before first annotation)")

    args = parser.parse_args()

    # Validate custom config/checkpoint usage
    if args.config and not args.checkpoint:
        parser.error("--config requires --checkpoint")
    if args.checkpoint and not args.config:
        parser.error("--checkpoint requires --config")

    # Auto-select model if not specified
    if args.model is None and not args.config:
        args.model = _auto_select_default_model()

    # Validate input files
    if not os.path.exists(args.annotation_file):
        print(f"ERROR: Annotation file not found: {args.annotation_file}")
        return 1

    if not os.path.exists(args.video_file):
        print(f"ERROR: Video file not found: {args.video_file}")
        return 1

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Check if output already exists
    masks_dir = output_dir / "masks"
    video_file = output_dir / "segmented_video.avi"
    metadata_file = output_dir / "processing_metadata.json"

    existing_items = []
    if masks_dir.exists() and list(masks_dir.glob("*.png")):
        existing_items.append(f"masks/ ({len(list(masks_dir.glob('*.png')))} files)")
    if video_file.exists():
        existing_items.append("segmented_video.avi")
    if metadata_file.exists():
        existing_items.append("processing_metadata.json")

    if existing_items:
        print("\nWARNING: Output directory contains existing files:")
        for item in existing_items:
            print(f"  - {item}")

        print("\nOptions:")
        print("  1. Delete all and proceed")
        print("  2. Proceed without deleting (may overwrite)")
        print("  3. Abort")

        while True:
            choice = input("\nEnter choice (1/2/3): ").strip()
            if choice == '1':
                # Delete existing
                if masks_dir.exists():
                    shutil.rmtree(masks_dir)
                if video_file.exists():
                    video_file.unlink()
                if metadata_file.exists():
                    metadata_file.unlink()
                print("Deleted existing files. Proceeding...")
                break
            elif choice == '2':
                print("Proceeding without deletion...")
                break
            elif choice == '3':
                print("Aborted by user")
                return 1
            else:
                print("Invalid choice. Please enter 1, 2, or 3")

    print("=" * 60)
    print("SAM2 Annotation Processor")
    print("=" * 60)
    print(f"Annotation file: {args.annotation_file}")
    print(f"Video file: {args.video_file}")
    print(f"Output directory: {output_dir}")
    if args.offload_to_cpu:
        print(f"Memory optimization: CPU offloading enabled")
    print()

    # Enable lazy loading BEFORE creating SAM2/SAM3 model
    # This prevents loading all frames into memory at once (huge memory savings for long videos)
    enable_lazy_loading(cache_size=args.frame_cache_size, enable_sam3=True)

    # Initialize processor
    try:
        if args.config:
            processor = SAM2Processor(config_file=args.config, checkpoint_file=args.checkpoint,
                                     offload_to_cpu=args.offload_to_cpu, async_loading=args.async_loading,
                                     smooth_masks=args.smooth_masks, use_bfloat16=args.use_bfloat16)
        else:
            processor = SAM2Processor(model_name=args.model, offload_to_cpu=args.offload_to_cpu,
                                     async_loading=args.async_loading, smooth_masks=args.smooth_masks,
                                     use_bfloat16=args.use_bfloat16)

        # Set no_backward flag
        processor.no_backward_propagation = args.no_backward

    except (ValueError, FileNotFoundError) as e:
        print(f"ERROR: {e}")
        return 1
    
    # Load model
    if not processor.load_model():
        return 1
    
    # Load annotations
    annotations_data = processor.load_annotations(args.annotation_file)
    if not annotations_data:
        return 1
    
    # Get video info
    frame_count, fps, width, height = processor.get_video_info(args.video_file)
    if frame_count is None:
        return 1
    
    # Use video FPS if not specified
    if args.fps == 30.0 and fps:
        args.fps = fps
    
    try:
        # Process segmentation
        masks_by_frame, object_names, object_colors, num_frames = processor.process_segmentation(
            args.video_file, annotations_data, output_dir, frame_dir=args.frame_dir
        )

        if masks_by_frame is None:
            print("ERROR: No masks generated")
            return 1

        print(f"\nOK: Generated masks for {len(masks_by_frame)} frames")

        # Export results
        processor.export_masks(masks_by_frame, args.video_file, object_names, output_dir)
        processor.export_video(masks_by_frame, args.video_file, object_names, object_colors,
                             output_dir, args.fps, args.opacity)
        processor.export_metadata(annotations_data, masks_by_frame, output_dir, num_frames,
                                video_path=args.video_file)

        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE!")
        print("=" * 60)
        print(f"Output directory: {output_dir}")
        print(f"Frames processed: {num_frames}")
        print()
        print("Generated files:")
        print(f"  - Masks: {output_dir}/masks/")
        print(f"  - Video: {output_dir}/segmented_video.avi")
        print(f"  - Metadata: {output_dir}/processing_metadata.json")
        print()

        return 0

    except Exception as e:
        print(f"\nProcessing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nProcessing failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
