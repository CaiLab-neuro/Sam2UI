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
}

class SAM2Processor:
    def __init__(self, config_file=None, checkpoint_file=None, model_name="sam2.1-base+", offload_to_cpu=False, async_loading=False, smooth_masks=False, use_bfloat16=False):
        """
        Initialize SAM2 Processor

        Args:
            config_file: Path to model config YAML (overrides model_name)
            checkpoint_file: Path to checkpoint file (overrides model_name)
            model_name: Preset model name (e.g., 'sam2-base+', 'sam2.1-large')
            offload_to_cpu: Use SAM2's CPU offloading for memory optimization
            async_loading: Use async frame loading (experimental, may reduce memory)
            smooth_masks: Apply morphological smoothing to reduce pixelation in masks
            use_bfloat16: Use BFloat16 precision for faster inference (requires compatible GPU)
        """
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

        # Validate paths exist
        if not os.path.exists(self.config_file):
            raise FileNotFoundError(f"Config file not found: {self.config_file}")

        # Checkpoint is optional for some use cases, but warn if missing
        if not os.path.exists(self.checkpoint_file):
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
        
    def load_model(self):
        """Load SAM2 model with correct API usage"""
        print(f"Loading SAM2 model...")
        print(f"  Config: {self.config_file}")
        print(f"  Checkpoint: {self.checkpoint_file or 'None (random init)'}")

        try:
            # FIXED: Pass config_file as first argument, ckpt_path as optional second
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Device: {device}")

            

            # DO NOT convert model dtype - keep native float32 from checkpoint
            # For bfloat16: Enable GLOBAL autocast (SAM2 benchmark pattern line 20)
            if device == "cuda":
                # Enable TF32 for Ampere GPUs (RTX 30xx+, A100) for better performance
                if torch.cuda.get_device_properties(0).major >= 8:
                    torch.backends.cuda.matmul.allow_tf32 = True
                    torch.backends.cudnn.allow_tf32 = True
                    print("  TensorFloat32 (TF32) enabled for Ampere GPU")

                if self.use_bfloat16:
                    # CRITICAL: Enable GLOBAL autocast before any SAM2 operations
                    # This stays active for entire program to handle bfloat16 memory features
                    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()
                    print("  BFloat16 mode: GLOBAL autocast enabled")
                    print("  Model weights remain in float32 (checkpoint dtype)")
                else:
                    print("  Float32 mode: native precision (no autocast)")

            self.video_predictor = build_sam2_video_predictor(
                config_file=self.config_file,
                ckpt_path=self.checkpoint_file,  # Optional parameter
                device=device
            )
            print("OK: Model loaded successfully")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
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

            # Initialize SAM2 inference state with JPEG directory
            print("Initializing SAM2 inference state...")

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

                    points_np = np.array(points, dtype=np.float32)
                    labels_np = np.array(labels, dtype=np.int32)

                    print(f"  Frame {frame_idx}: {len(points)} points")

                    try:
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

            # Determine earliest annotation frame for bidirectional propagation
            earliest_frame = min(frame_annotations.keys()) if frame_annotations else 0

            # Create output directories for streaming export
            masks_dir = Path(output_dir) / "masks"
            masks_dir.mkdir(parents=True, exist_ok=True)

            # Propagate annotations - FORWARD direction first
            print(f"\nPropagating annotations FORWARD from frame {earliest_frame}...")

            masks_metadata = {}  # Only metadata, not actual mask arrays

            # CRITICAL: Nested autocast context to handle bfloat16 tensors from CPU offloading
            # Matches SAM2 benchmark.py pattern (line 72) for proper dtype handling
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                    inference_state, reverse=False
                ):
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
                        frame_masks[obj_id] = {
                            'filename': mask_filename,
                            'score': 1.0,
                            'name': obj_name,
                            'color': obj_color
                        }

                        # Explicitly delete mask array
                        del mask

                    masks_metadata[out_frame_idx] = frame_masks

                    # Delete output tensors after each frame
                    del out_mask_logits

                    if (out_frame_idx + 1) % 50 == 0:
                        # Monitor GPU and RAM usage to track memory growth
                        gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
                        gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
                        gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)

                        process = psutil.Process()
                        ram_used = process.memory_info().rss / (1024**3)

                        print(f"  Forward: Frame {out_frame_idx + 1}/{num_frames} | "
                              f"GPU: {gpu_allocated:.2f}GB (peak: {gpu_peak:.2f}GB) | "
                              f"RAM: {ram_used:.2f}GB")

                        # CRITICAL: Periodic cleanup of old frames to prevent memory growth
                        # Keep only last 20 frames (SAM2 needs max 6-16 for memory attention)
                        frames_to_keep = 20
                        for obj_idx in range(len(inference_state["obj_ids"])):
                            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                            non_cond = obj_output_dict.get("non_cond_frame_outputs", {})

                            # Find frames older than the retention window
                            old_frames = [f for f in non_cond.keys() if f < out_frame_idx - frames_to_keep]

                            # Delete old frame outputs
                            for old_frame in old_frames:
                                del non_cond[old_frame]

                        # Reset peak stats for next interval
                        torch.cuda.reset_peak_memory_stats()
                        # Clear GPU memory fragmentation
                        torch.cuda.empty_cache()

            # Propagate annotations - BACKWARD direction (if needed)
            if earliest_frame > 0:
                print(f"\nPropagating annotations BACKWARD from frame {earliest_frame}...")

                # CRITICAL: Nested autocast context to handle bfloat16 tensors from CPU offloading
                # Matches SAM2 benchmark.py pattern (line 72) for proper dtype handling
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(
                        inference_state, reverse=True
                    ):
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
                            frame_masks[obj_id] = {
                                'filename': mask_filename,
                                'score': 1.0,
                                'name': obj_name,
                                'color': obj_color
                            }

                            # Explicitly delete mask array
                            del mask

                        masks_metadata[out_frame_idx] = frame_masks

                        # Delete output tensors after each frame
                        del out_mask_logits

                        if (out_frame_idx + 1) % 50 == 0:
                            # Monitor GPU and RAM usage to track memory growth
                            gpu_allocated = torch.cuda.memory_allocated() / (1024**3)
                            gpu_reserved = torch.cuda.memory_reserved() / (1024**3)
                            gpu_peak = torch.cuda.max_memory_allocated() / (1024**3)

                            process = psutil.Process()
                            ram_used = process.memory_info().rss / (1024**3)

                            print(f"  Backward: Frame {out_frame_idx + 1}/{num_frames} | "
                                  f"GPU: {gpu_allocated:.2f}GB (peak: {gpu_peak:.2f}GB) | "
                                  f"RAM: {ram_used:.2f}GB")

                            # CRITICAL: Periodic cleanup of old frames to prevent memory growth
                            # Keep only last 20 frames (SAM2 needs max 6-16 for memory attention)
                            frames_to_keep = 20
                            for obj_idx in range(len(inference_state["obj_ids"])):
                                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                                non_cond = obj_output_dict.get("non_cond_frame_outputs", {})

                                # Find frames older than the retention window
                                old_frames = [f for f in non_cond.keys() if f < out_frame_idx - frames_to_keep]

                                # Delete old frame outputs
                                for old_frame in old_frames:
                                    del non_cond[old_frame]

                            # Reset peak stats for next interval
                            torch.cuda.reset_peak_memory_stats()
                            # Clear GPU memory fragmentation
                            torch.cuda.empty_cache()
            else:
                print("  Skipping backward propagation (earliest annotation at frame 0)")

            # Phase 2: Clear SAM2's internal frame outputs (safe after propagation)
            print("\nCleaning up SAM2 inference state...")
            for obj_idx in range(len(objects_with_annotations)):
                obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                # Clear non-conditioning frames (keep conditioning for potential refinement)
                non_cond = obj_output_dict.get("non_cond_frame_outputs", {})
                non_cond.clear()

            torch.cuda.empty_cache()
            print("OK: Cleaned SAM2 memory")

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
                    output_dir, fps=30, overlay_opacity=0.4):
        """Export segmented video with overlays"""
        print("Exporting segmented video...")

        frames = self.load_video_frames_for_export(video_path)
        if not frames:
            return False
        
        video_output_path = Path(output_dir) / "segmented_video.mp4"
        
        height, width = frames[0].shape[:2]

        # Use H.264 codec for much better compression and quality
        # Try different codec identifiers based on system availability
        for codec in ['avc1', 'H264', 'X264', 'mp4v']:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))
            if out.isOpened():
                if codec != 'mp4v':
                    print(f"  Using {codec} codec for video encoding")
                break

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
        print(f"OK: Exported segmented video to {video_output_path}")
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
                "segmented_video_filename": "segmented_video.mp4",
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
    model_group.add_argument("--model", default="sam2-base+",
                           choices=list(MODEL_CONFIGS.keys()),
                           help="Preset model name (default: sam2-base+)")
    model_group.add_argument("--config", help="Custom config YAML path (requires --checkpoint)")

    parser.add_argument("--checkpoint", help="Custom checkpoint path (requires --config)")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30)")
    parser.add_argument("--opacity", type=float, default=0.4, help="Mask overlay opacity (default: 0.4)")
    parser.add_argument("--offload-to-cpu", action="store_true",
                       help="Offload video frames and model state to CPU to reduce GPU memory usage")
    parser.add_argument("--async-loading", action="store_true",
                       help="Use async frame loading (experimental, may reduce memory usage)")
    parser.add_argument("--smooth-masks", action="store_true",
                       help="Apply morphological smoothing to reduce pixelation in exported masks (preserves binary masks)")
    parser.add_argument("--use-bfloat16", action="store_true",
                       help="Use BFloat16 mixed precision for faster inference and reduced memory usage (requires Ampere+ GPU with BFloat16 support, e.g., RTX 30xx+, A100). Uses torch.autocast following SAM2's official benchmark pattern.")
    parser.add_argument("--frame-dir", type=str, default=None,
                       help="Persistent directory for video frames (default: auto-generated in /tmp). If specified, frames will be reused from previous runs and not deleted after processing.")
    parser.add_argument("--frame-cache-size", type=int, default=20,
                       help="Number of frames to keep in memory cache (default: 20, ~2GB). Minimum: 10, Recommended: 20-50.")

    args = parser.parse_args()

    # Validate custom config/checkpoint usage
    if args.config and not args.checkpoint:
        parser.error("--config requires --checkpoint")
    if args.checkpoint and not args.config:
        parser.error("--checkpoint requires --config")

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
    video_file = output_dir / "segmented_video.mp4"
    metadata_file = output_dir / "processing_metadata.json"

    existing_items = []
    if masks_dir.exists() and list(masks_dir.glob("*.png")):
        existing_items.append(f"masks/ ({len(list(masks_dir.glob('*.png')))} files)")
    if video_file.exists():
        existing_items.append("segmented_video.mp4")
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

    # Enable lazy loading BEFORE creating SAM2 model
    enable_lazy_loading(cache_size=args.frame_cache_size)

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
        print(f"  - Video: {output_dir}/segmented_video.mp4")
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
