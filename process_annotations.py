#!/usr/bin/env python3
"""
SAM2 Annotation Processor
========================

Takes annotation JSON from SAM2 Video UI and video file,
then exports segmented video and masks.

Usage:
    python process_annotations.py <annotation_file> <video_file> [output_dir]

Example:
    python process_annotations.py annotations.json video.mp4 results/
"""

import os
import sys
import json
import time
import argparse
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

class SAM2Processor:
    def __init__(self, config_path="configs/sam2.1/sam2.1_hiera_base_plus.pt"):
        self.config_path = config_path
        self.sam2_model = None
        self.video_predictor = None
        
    def load_model(self):
        """Load SAM2 model"""
        print(f"Loading SAM2 model from {self.config_path}...")
        try:
            self.sam2_model = build_sam2_video_predictor(self.config_path, device="cuda" if torch.cuda.is_available() else "cpu")
            print("OK: Model loaded successfully")
            return True
        except Exception as e:
            print(f"ERROR: Failed to load model: {e}")
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
    
    def load_video(self, video_path):
        """Load video and extract frames"""
        print(f"Loading video from {video_path}...")
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
            print(f"OK: Loaded {len(frames)} frames from video")
            return frames
        except Exception as e:
            print(f"ERROR: Failed to load video: {e}")
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
        
        # Initialize SAM2 state
        print("Initializing SAM2 state...")
        self.video_predictor.init_state(frames[0])
        
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
        print("Exporting segmented video...")
        
        video_path = Path(output_dir) / "segmented_video.mp4"
        
        if not frames:
            print("ERROR: No frames to process")
            return False
        
        # Get video dimensions
        height, width = frames[0].shape[:2]
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
        
        processed_frames = 0
        
        for frame_idx, frame in enumerate(frames):
            # Create overlay frame
            overlay_frame = frame.copy()
            
            if frame_idx in masks_by_frame:
                frame_masks = masks_by_frame[frame_idx]
                
                for obj_id, mask_data in frame_masks.items():
                    mask = mask_data['mask']
                    color = mask_data['color']
                    name = mask_data['name']
                    
                    # Create colored mask
                    colored_mask = np.zeros((height, width, 3), dtype=np.uint8)
                    colored_mask[mask] = color
                    
                    # Blend with original frame
                    overlay_frame = cv2.addWeighted(overlay_frame, 1-overlay_opacity, 
                                                  colored_mask, overlay_opacity, 0)
                    
                    # Add object name label
                    if mask.any():
                        # Find center of mask for label placement
                        y_coords, x_coords = np.where(mask)
                        if len(y_coords) > 0:
                            center_x = int(np.mean(x_coords))
                            center_y = int(np.mean(y_coords))
                            
                            # Add text label
                            cv2.putText(overlay_frame, name, (center_x, center_y),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Write frame to video
            out.write(overlay_frame)
            processed_frames += 1
            
            if processed_frames % 100 == 0:
                print(f"   Processed {processed_frames} frames...")
        
        out.release()
        print(f"OK: Exported segmented video to {video_path}")
        return True
    
    def export_metadata(self, annotations_data, masks_by_frame, output_dir):
        """Export processing metadata"""
        print("Exporting metadata...")
        
        metadata = {
            "processing_info": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_frames_processed": len(masks_by_frame),
                "total_masks_generated": sum(len(masks) for masks in masks_by_frame.values()),
                "objects_detected": list(set(
                    obj_id for masks in masks_by_frame.values() 
                    for obj_id in masks.keys()
                ))
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
    parser = argparse.ArgumentParser(description="Process SAM2 annotations and generate segmented output")
    parser.add_argument("annotation_file", help="Path to annotation JSON file from SAM2 Video UI")
    parser.add_argument("video_file", help="Path to input video file")
    parser.add_argument("--output_dir", default="sam2_output", help="Output directory (default: sam2_output)")
    parser.add_argument("--config", default="configs/sam2.1/sam2.1_hiera_b+.yaml", help="SAM2 model config path")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS (default: 30)")
    parser.add_argument("--opacity", type=float, default=0.4, help="Mask overlay opacity (default: 0.4)")
    
    args = parser.parse_args()
    
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
    print()
    
    # Initialize processor
    processor = SAM2Processor(args.config)
    
    # Load model
    if not processor.load_model():
        return 1
    
    # Load annotations
    annotations_data = processor.load_annotations(args.annotation_file)
    if not annotations_data:
        return 1
    
    # Load video
    frames = processor.load_video(args.video_file)
    if frames is None:
        return 1
    
    # Process segmentation
    masks_by_frame, object_names, object_colors = processor.process_segmentation(
        frames, annotations_data
    )
    
    if not masks_by_frame:
        print("ERROR: No masks generated")
        return 1
    
    print(f"OK: Generated masks for {len(masks_by_frame)} frames")
    
    # Export results
    processor.export_masks(masks_by_frame, frames, object_names, output_dir)
    processor.export_video(masks_by_frame, frames, object_names, object_colors, 
                         output_dir, args.fps, args.opacity)
    processor.export_metadata(annotations_data, masks_by_frame, output_dir)
    
    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE!")
    print("=" * 60)
    print(f"Output directory: {output_dir}")
    print(f"Frames processed: {len(masks_by_frame)}")
    print(f"Total masks: {sum(len(masks) for masks in masks_by_frame.values())}")
    print()
    print("Generated files:")
    print(f"  - Masks: {output_dir}/masks/")
    print(f"  - Video: {output_dir}/segmented_video.mp4")
    print(f"  - Metadata: {output_dir}/processing_metadata.json")
    print()
    
    return 0

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\nProcessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nProcessing failed: {e}")
        # sys.exit(1)
        raise
