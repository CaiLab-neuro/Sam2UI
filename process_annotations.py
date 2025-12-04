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
import pickle
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
import torch
from PIL import Image

# Add SAM2 to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from sam2.build_sam import build_sam2_video_predictor
except ImportError as e:
    print(f"Error importing SAM2: {e}")
    print("Please run setup.py first to install dependencies.")
    sys.exit(1)

# Model configuration mappings
MODEL_CONFIGS = {
    # SAM2 models
    "sam2-tiny": ("configs/sam2/sam2_hiera_t.yaml", "checkpoints/sam2_hiera_tiny.pt"),
    "sam2-small": ("configs/sam2/sam2_hiera_s.yaml", "checkpoints/sam2_hiera_small.pt"),
    "sam2-base+": ("configs/sam2/sam2_hiera_b+.yaml", "checkpoints/sam2_hiera_base_plus.pt"),
    "sam2-large": ("configs/sam2/sam2_hiera_l.yaml", "checkpoints/sam2_hiera_large.pt"),
    
    # SAM2.1 models (if available)
    "sam2.1-tiny": ("configs/sam2.1/sam2.1_hiera_t.yaml", "checkpoints/sam2.1_hiera_tiny.pt"),
    "sam2.1-small": ("configs/sam2.1/sam2.1_hiera_s.yaml", "checkpoints/sam2.1_hiera_small.pt"),
    "sam2.1-base+": ("configs/sam2.1/sam2.1_hiera_b+.yaml", "checkpoints/sam2.1_hiera_base_plus.pt"),
    "sam2.1-large": ("configs/sam2.1/sam2.1_hiera_l.yaml", "checkpoints/sam2.1_hiera_large.pt"),
}

class SAM2Processor:
    def __init__(self, config_file=None, checkpoint_file=None, model_name="sam2-base+", chunk_size=None):
        """
        Initialize SAM2 Processor
        
        Args:
            config_file: Path to model config YAML (overrides model_name)
            checkpoint_file: Path to checkpoint file (overrides model_name)
            model_name: Preset model name (e.g., 'sam2-base+', 'sam2.1-large')
            chunk_size: Number of frames to process per chunk (None = no chunking)
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
        
        # Chunking configuration
        self.chunk_size = chunk_size
        self.use_chunking = chunk_size is not None
        self.temp_dir = None
        self.chunk_files = []
        
    def load_model(self):
        """Load SAM2 model with correct API usage"""
        print(f"Loading SAM2 model...")
        print(f"  Config: {self.config_file}")
        print(f"  Checkpoint: {self.checkpoint_file or 'None (random init)'}")
        
        try:
            # FIXED: Pass config_file as first argument, ckpt_path as optional second
            device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"  Device: {device}")
            
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
    
    def process_segmentation(self, video_path, annotations_data):
        """Process segmentation using SAM2 with optional chunking"""
        if self.use_chunking:
            return self.process_segmentation_chunked(video_path, annotations_data)
        else:
            return self.process_segmentation_full(video_path, annotations_data)
    
    def process_segmentation_full(self, video_path, annotations_data):
        """Process segmentation - load all masks into memory (original method)"""
        print("Starting segmentation process (full memory mode)...")
        
        # Group annotations by frame and object
        frame_annotations = self.group_annotations_by_frame(annotations_data["annotations"])
        
        # Initialize SAM2 inference state
        print("Initializing SAM2 inference state...")
        print(f"  Loading video: {video_path}")
        
        inference_state = self.video_predictor.init_state(video_path=video_path)
        
        num_frames = inference_state["num_frames"]
        print(f"  Loaded {num_frames} frames into inference state")
        
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
        
        # Propagate annotations to all frames (stores ALL in memory)
        print("\nPropagating annotations across all frames...")
        
        masks_by_frame = {}
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            frame_masks = {}
            
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                
                frame_masks[obj_id] = {
                    'mask': mask,
                    'score': 1.0,
                    'name': object_names.get(str(obj_id), f"Object_{obj_id}"),
                    'color': object_colors.get(str(obj_id), [255, 0, 0])
                }
            
            masks_by_frame[out_frame_idx] = frame_masks
            
            if (out_frame_idx + 1) % 50 == 0:
                print(f"  Processed frame {out_frame_idx + 1}/{num_frames}")
        
        print(f"OK: Generated masks for {len(masks_by_frame)} frames")
        return masks_by_frame, object_names, object_colors, num_frames
    
    def process_segmentation_chunked(self, video_path, annotations_data):
        """Process segmentation with chunking to reduce memory usage"""
        print(f"Starting segmentation process (chunked mode, chunk_size={self.chunk_size})...")
        
        # Create temporary directory for chunks
        self.temp_dir = Path(tempfile.mkdtemp(prefix="sam2_chunks_"))
        print(f"  Temporary chunk storage: {self.temp_dir}")
        
        # Group annotations
        frame_annotations = self.group_annotations_by_frame(annotations_data["annotations"])
        
        # Initialize SAM2 inference state
        print("Initializing SAM2 inference state...")
        print(f"  Loading video: {video_path}")
        
        inference_state = self.video_predictor.init_state(video_path=video_path)
        
        num_frames = inference_state["num_frames"]
        print(f"  Loaded {num_frames} frames into inference state")
        
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
        
        # Add annotation points
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
        
        # Propagate with chunking - save to disk as we go
        print(f"\nPropagating annotations across all frames (chunked, {self.chunk_size} frames/chunk)...")
        
        chunk_buffer = {}
        chunk_id = 0
        total_masks = 0
        
        for out_frame_idx, out_obj_ids, out_mask_logits in self.video_predictor.propagate_in_video(inference_state):
            frame_masks = {}
            
            for i, obj_id in enumerate(out_obj_ids):
                mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()
                
                frame_masks[obj_id] = {
                    'mask': mask,
                    'score': 1.0,
                    'name': object_names.get(str(obj_id), f"Object_{obj_id}"),
                    'color': object_colors.get(str(obj_id), [255, 0, 0])
                }
                total_masks += 1
            
            chunk_buffer[out_frame_idx] = frame_masks
            
            # Save chunk when buffer is full
            if len(chunk_buffer) >= self.chunk_size:
                self._save_chunk(chunk_buffer, chunk_id)
                print(f"  Saved chunk {chunk_id} ({len(chunk_buffer)} frames)")
                chunk_buffer = {}  # Clear memory
                chunk_id += 1
            
            if (out_frame_idx + 1) % 50 == 0:
                print(f"  Processed frame {out_frame_idx + 1}/{num_frames}")
        
        # Save remaining frames
        if chunk_buffer:
            self._save_chunk(chunk_buffer, chunk_id)
            print(f"  Saved chunk {chunk_id} ({len(chunk_buffer)} frames)")
            chunk_id += 1
        
        print(f"OK: Generated {total_masks} masks across {chunk_id} chunks")
        
        # Return metadata (no actual masks in memory)
        return None, object_names, object_colors, num_frames
    
    def _save_chunk(self, chunk_data, chunk_id):
        """Save a chunk of masks to disk"""
        chunk_path = self.temp_dir / f"chunk_{chunk_id:04d}.pkl"
        with open(chunk_path, 'wb') as f:
            pickle.dump(chunk_data, f)
        self.chunk_files.append(chunk_path)
    
    def _load_chunk(self, chunk_id):
        """Load a specific chunk from disk"""
        chunk_path = self.temp_dir / f"chunk_{chunk_id:04d}.pkl"
        with open(chunk_path, 'rb') as f:
            return pickle.load(f)
    
    def _cleanup_chunks(self):
        """Clean up temporary chunk files"""
        if self.temp_dir and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary chunks: {self.temp_dir}")
    
    def load_single_frame(self, video_path, frame_idx):
        """Load a single frame from video"""
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    
    def export_masks_chunked(self, video_path, object_names, output_dir):
        """Export individual mask images from chunks"""
        print("Exporting mask images (from chunks)...")
        
        masks_dir = Path(output_dir) / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        exported_count = 0
        
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Process each chunk
        for chunk_id, chunk_path in enumerate(self.chunk_files):
            chunk_data = self._load_chunk(chunk_id)
            
            for frame_idx, frame_masks in chunk_data.items():
                for obj_id, mask_data in frame_masks.items():
                    mask = mask_data['mask']
                    name = mask_data['name']
                    
                    mask_filename = f"frame_{frame_idx:06d}_{name}_{obj_id}.png"
                    mask_path = masks_dir / mask_filename
                    
                    # Resize if needed
                    if mask.shape != (height, width):
                        from PIL import Image as PILImage
                        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                        mask_pil = mask_pil.resize((width, height), PILImage.NEAREST)
                        mask_pil.save(mask_path)
                    else:
                        mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                        mask_image.save(mask_path)
                    
                    exported_count += 1
            
            if (chunk_id + 1) % 10 == 0:
                print(f"  Exported {exported_count} masks from {chunk_id + 1}/{len(self.chunk_files)} chunks...")
        
        print(f"OK: Exported {exported_count} mask images to {masks_dir}")
        return exported_count
    
    def export_video_chunked(self, video_path, object_names, object_colors, 
                            output_dir, fps=30, overlay_opacity=0.4):
        """Export segmented video with overlays (streaming from chunks)"""
        print("Exporting segmented video (streaming from chunks)...")
        
        video_output_path = Path(output_dir) / "segmented_video.mp4"
        
        # Get video properties
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))
        
        # Build frame-to-chunk mapping
        frame_to_chunk = {}
        for chunk_id, chunk_path in enumerate(self.chunk_files):
            chunk_data = self._load_chunk(chunk_id)
            for frame_idx in chunk_data.keys():
                frame_to_chunk[frame_idx] = chunk_id
        
        # Process frames
        current_chunk_id = None
        current_chunk_data = None
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            overlay_frame = frame.copy()
            
            # Load chunk if needed
            if frame_idx in frame_to_chunk:
                chunk_id = frame_to_chunk[frame_idx]
                if chunk_id != current_chunk_id:
                    current_chunk_data = self._load_chunk(chunk_id)
                    current_chunk_id = chunk_id
                
                # Apply masks
                frame_masks = current_chunk_data.get(frame_idx, {})
                for obj_id, mask_data in frame_masks.items():
                    mask = mask_data['mask']
                    color = mask_data['color']
                    name = mask_data['name']
                    
                    # Resize mask if needed
                    if mask.shape != (height, width):
                        from PIL import Image as PILImage
                        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                        mask_pil = mask_pil.resize((width, height), PILImage.NEAREST)
                        mask = np.array(mask_pil) > 0
                    
                    # Create colored mask
                    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    if isinstance(color, (list, tuple)):
                        colored_mask[mask] = color[::-1]  # RGB to BGR
                    else:
                        colored_mask[mask] = [0, 0, 255]
                    
                    # Blend
                    overlay_frame = cv2.addWeighted(overlay_frame, 1-overlay_opacity, 
                                                  colored_mask, overlay_opacity, 0)
                    
                    # Add label
                    if mask.any():
                        y_coords, x_coords = np.where(mask)
                        if len(y_coords) > 0:
                            center_x = int(np.mean(x_coords))
                            center_y = int(np.mean(y_coords))
                            cv2.putText(overlay_frame, name, (center_x, center_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(overlay_frame)
            frame_idx += 1
            
            if frame_idx % 100 == 0:
                print(f"   Processed {frame_idx}/{total_frames} frames...")
        
        cap.release()
        out.release()
        print(f"OK: Exported segmented video to {video_output_path}")
        return True
    
    def export_masks(self, masks_by_frame, video_path, object_names, output_dir):
        """Export individual mask images"""
        if self.use_chunking:
            return self.export_masks_chunked(video_path, object_names, output_dir)
        
        print("Exporting mask images...")
        
        masks_dir = Path(output_dir) / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        exported_count = 0
        
        # Get video dimensions
        cap = cv2.VideoCapture(video_path)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        for frame_idx, frame_masks in masks_by_frame.items():
            for obj_id, mask_data in frame_masks.items():
                mask = mask_data['mask']
                name = mask_data['name']
                
                mask_filename = f"frame_{frame_idx:06d}_{name}_{obj_id}.png"
                mask_path = masks_dir / mask_filename
                
                if mask.shape != (height, width):
                    from PIL import Image as PILImage
                    mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                    mask_pil = mask_pil.resize((width, height), PILImage.NEAREST)
                    mask_pil.save(mask_path)
                else:
                    mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                    mask_image.save(mask_path)
                
                exported_count += 1
        
        print(f"OK: Exported {exported_count} mask images to {masks_dir}")
        return exported_count
    
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
    
    def group_annotations_by_frame(self, annotations):
        """Group annotations by frame index"""
        frame_annotations = {}
        for annotation in annotations:
            frame_idx = annotation["frame_index"]
            if frame_idx not in frame_annotations:
                frame_annotations[frame_idx] = []
            frame_annotations[frame_idx].append(annotation)
        return frame_annotations
    
    def process_segmentation(self, frames, annotations_data):
        """Process segmentation using SAM2"""
        print("Starting segmentation process...")
        
        # Group annotations by frame
        frame_annotations = self.group_annotations_by_frame(annotations_data["annotations"])
        print(frames[0])
        print(frames)
        # Initialize SAM2 state
        print("Initializing SAM2 state...")
        # self.video_predictor.init_state(frames[0])
        if not hasattr(self, 'inference_state') or self.inference_state is None:
            self.inference_state = self.video_predictor.init_state(video_path=temp_dir)
        
        # Process each annotated frame
        masks_by_frame = {}
        object_names = annotations_data.get("object_names", {})
        object_colors = annotations_data.get("object_colors", {})
        
        for frame_idx, annotations in frame_annotations.items():
            if frame_idx >= len(frames):
                print(f"WARNING: Skipping frame {frame_idx} (beyond video length)")
                continue
                
            print(f"Processing frame {frame_idx} with {len(annotations)} annotations...")
            
            # Prepare points and labels for this frame
            points = []
            labels = []
            object_ids = []
            
            for annotation in annotations:
                x, y = annotation["x"], annotation["y"]
                is_positive = annotation["is_positive"]
                obj_id = annotation["object_id"]
                
                points.append([x, y])
                labels.append(1 if is_positive else 0)
                object_ids.append(obj_id)
            
            if not points:
                continue
            
            # Convert to numpy arrays
            points = np.array(points)
            labels = np.array(labels)
            
            # Run SAM2 prediction
            try:
                masks, scores, logits = self.video_predictor.predict(
                    frame_idx, points, labels
                )
                
                # Store masks for this frame
                frame_masks = {}
                for i, (mask, score, obj_id) in enumerate(zip(masks, scores, object_ids)):
                    if score > 0.5:  # Threshold for mask quality
                        frame_masks[obj_id] = {
                            'mask': mask,
                            'score': score,
                            'name': object_names.get(str(obj_id), f"Object_{obj_id}"),
                            'color': object_colors.get(str(obj_id), [255, 0, 0])
                        }
                
                masks_by_frame[frame_idx] = frame_masks
                print(f"   Generated {len(frame_masks)} masks")
                
            except Exception as e:
                print(f"   WARNING: Error processing frame {frame_idx}: {e}")
                continue
        
        return masks_by_frame, object_names, object_colors
    
    def export_masks(self, masks_by_frame, frames, object_names, output_dir):
        """Export individual mask images"""
        print("Exporting mask images...")
        
        masks_dir = Path(output_dir) / "masks"
        masks_dir.mkdir(exist_ok=True)
        
        exported_count = 0
        
        for frame_idx, frame_masks in masks_by_frame.items():
            if frame_idx >= len(frames):
                continue
                
            for obj_id, mask_data in frame_masks.items():
                mask = mask_data['mask']
                name = mask_data['name']
                
                # Create mask filename
                mask_filename = f"frame_{frame_idx:06d}_{name}_{obj_id}.png"
                mask_path = masks_dir / mask_filename
                
                # Convert mask to PIL Image and save
                mask_image = Image.fromarray((mask * 255).astype(np.uint8))
                mask_image.save(mask_path)
                exported_count += 1
        
        print(f"OK: Exported {exported_count} mask images to {masks_dir}")
        return exported_count
    
    def export_video(self, masks_by_frame, frames, object_names, object_colors, 
                    output_dir, fps=30, overlay_opacity=0.4):
        """Export segmented video with overlays"""
        if self.use_chunking:
            return self.export_video_chunked(video_path, object_names, object_colors,
                                           output_dir, fps, overlay_opacity)
        
        print("Exporting segmented video...")
        
        frames = self.load_video_frames_for_export(video_path)
        if not frames:
            return False
        
        video_output_path = Path(output_dir) / "segmented_video.mp4"
        
        height, width = frames[0].shape[:2]
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_output_path), fourcc, fps, (width, height))
        
        processed_frames = 0
        
        for frame_idx, frame in enumerate(frames):
            overlay_frame = frame.copy()
            
            if frame_idx in masks_by_frame:
                frame_masks = masks_by_frame[frame_idx]
                
                for obj_id, mask_data in frame_masks.items():
                    mask = mask_data['mask']
                    color = mask_data['color']
                    name = mask_data['name']
                    
                    if mask.shape != (height, width):
                        from PIL import Image as PILImage
                        mask_pil = PILImage.fromarray((mask * 255).astype(np.uint8))
                        mask_pil = mask_pil.resize((width, height), PILImage.NEAREST)
                        mask = np.array(mask_pil) > 0
                    
                    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    if isinstance(color, (list, tuple)):
                        colored_mask[mask] = color[::-1]
                    else:
                        colored_mask[mask] = [0, 0, 255]
                    
                    overlay_frame = cv2.addWeighted(overlay_frame, 1-overlay_opacity, 
                                                  colored_mask, overlay_opacity, 0)
                    
                    if mask.any():
                        y_coords, x_coords = np.where(mask)
                        if len(y_coords) > 0:
                            center_x = int(np.mean(x_coords))
                            center_y = int(np.mean(y_coords))
                            cv2.putText(overlay_frame, name, (center_x, center_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            out.write(overlay_frame)
            processed_frames += 1
            
            if processed_frames % 100 == 0:
                print(f"   Processed {processed_frames}/{len(frames)} frames...")
        
        out.release()
        print(f"OK: Exported segmented video to {video_output_path}")
        return True
    
    def export_metadata(self, annotations_data, masks_by_frame, output_dir, num_frames=None):
        """Export processing metadata"""
        print("Exporting metadata...")
        
        metadata = {
            "processing_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config_file": self.config_file,
                "checkpoint_file": self.checkpoint_file,
                "chunked_processing": self.use_chunking,
                "chunk_size": self.chunk_size if self.use_chunking else None,
            },
            "original_annotations": annotations_data
        }
        
        if masks_by_frame is not None:
            # Full memory mode
            metadata["processing_info"]["total_frames_processed"] = len(masks_by_frame)
            metadata["processing_info"]["total_masks_generated"] = sum(len(masks) for masks in masks_by_frame.values())
            metadata["processing_info"]["objects_detected"] = list(set(
                obj_id for masks in masks_by_frame.values() 
                for obj_id in masks.keys()
            ))
        else:
            # Chunked mode - calculate from chunks
            total_frames_processed = 0
            total_masks_generated = 0
            objects_detected = set()
            
            for chunk_id in range(len(self.chunk_files)):
                chunk_data = self._load_chunk(chunk_id)
                total_frames_processed += len(chunk_data)
                for frame_masks in chunk_data.values():
                    total_masks_generated += len(frame_masks)
                    objects_detected.update(frame_masks.keys())
            
            metadata["processing_info"]["total_frames_processed"] = total_frames_processed
            metadata["processing_info"]["total_masks_generated"] = total_masks_generated
            metadata["processing_info"]["objects_detected"] = list(objects_detected)
        
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
  
  # Use chunked processing for large videos (reduces memory)
  python process_annotations.py annotations.json video.mp4 --chunk-size 100
  
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
    parser.add_argument("--chunk-size", type=int, default=None, 
                       help="Process video in chunks of N frames to reduce memory usage (default: disabled)")
    
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
    
    print("=" * 60)
    print("SAM2 Annotation Processor")
    print("=" * 60)
    print(f"Annotation file: {args.annotation_file}")
    print(f"Video file: {args.video_file}")
    print(f"Output directory: {output_dir}")
    if args.chunk_size:
        print(f"Chunked processing: {args.chunk_size} frames per chunk")
    print()
    
    # Initialize processor
    try:
        if args.config:
            processor = SAM2Processor(config_file=args.config, checkpoint_file=args.checkpoint,
                                     chunk_size=args.chunk_size)
        else:
            processor = SAM2Processor(model_name=args.model, chunk_size=args.chunk_size)
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
            args.video_file, annotations_data
        )
        
        if masks_by_frame is None and not processor.use_chunking:
            print("ERROR: No masks generated")
            return 1
        
        if processor.use_chunking:
            print(f"\nOK: Generated masks (saved to {len(processor.chunk_files)} chunks)")
        else:
            print(f"\nOK: Generated masks for {len(masks_by_frame)} frames")
        
        # Export results
        processor.export_masks(masks_by_frame, args.video_file, object_names, output_dir)
        processor.export_video(masks_by_frame, args.video_file, object_names, object_colors, 
                             output_dir, args.fps, args.opacity)
        processor.export_metadata(annotations_data, masks_by_frame, output_dir, num_frames)
        
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
        
    finally:
        # Always clean up chunks
        if processor.use_chunking:
            processor._cleanup_chunks()

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
