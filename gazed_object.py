#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
import os
import glob
import pickle
import logging, sys
import argparse
from pathlib import Path
from typing import Optional

"""
This is an aligner script (published version) to find which object is being gazed at based on segmentation masks and gaze data.
This code includes functions to
    1) Load segmentation masks from SAM2 or 3,
    2) Load and align gaze data with video frames,
    3) Compute confidence scores for each gaze point against each object mask,
    4) Output the gazed object and confidence for each gaze point, as well as probabilities for all masks.

Authors: Yanbin Xu
Date: Jan.7 2026

Inputs:
    1) Subject ID(s) and Camera ID(s).
    Subject ID: e.g. 27 or 27,28 etc.
    Camera ID: e.g. child or parent or child,parent

    2) Gaze and World Camera data directory: /path/to/gaze_world_data
    Within this directory, the code expects
        a) Gaze data CSV files named as *{subject_id}_{camera}_gaze.csv*
        Rows: Each gaze recording ordered by timestamp.
        Columns(required/Column Name Case Specific):
            timestamp [ns]: Timestamp of the gaze point.
            gaze x [px]: Gaze x coordinate in pixels.
            gaze y [px]: Gaze y coordinate in pixels.
        b) World Camera frame timestamps CSV files named as *{subject_id}_{camera}_world_timestamps.csv* # These are timestamps of each frame in the egocentric video.
        Rows: Each frame in the egocentric video ordered by timestamp.
        Columns(required/Column Name Case Specific):
            timestamp [ns]: Timestamp of the frame.
        Note: these files can include more columns. Gaze recordings and frame rates do not need to be the same.

    3) Segmentation mask directory: /path/to/segmentation_masks
    Within this directory, the code expects
        a) folders named as *{subject_id}_{camera}/masks/*
        Each folder contains:
        Segmentation masks (.png) of multiple objects from SAM2/3 (numpy arrays).
        Mask name format: *mask_f{frame_id}_{mask_object_name}_{mask_id}*.png
        e.g. mask_f000000_ceiling_inside_id32.png

Outputs:
    1) Output directory: /path/to/output_directory
    Within this directory, the code will create new folders named as *{subject_id}_gazed_object/*

        Each folder contains:
        A csv file containing the gaze and its associated gazed object with confidence score.
        *{subject_id_temp}_{camera_temp}_gazed_object.csv*

        A pickle file containing the probabilities of all object masks for each gaze point. 
        *{subject_id_temp}_{camera_temp}_gaze_object_probabilities.pkl*

        if remove blink option is applied:
        A csv file containing the gaze data labeled within_blink.
        *{subject_id_temp}_{camera_temp}_gaze_blink_labeled.csv*

        A csv file containing gaze data removed all gazes in blink.
        *{subject_id_temp}_{camera_temp}_gaze_blink_removed.csv*

"""

def setup_logging(log_path: Path):
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_path, mode="w"),
            ],
            force=True,  # important if running in notebooks / reused envs
        )

class GazeObjectAligner:
    def __init__(self, subject_id: list, camera: str, gaze_world_dir: str, mask_dir: str, output_dir: str, blink_dir: Optional[str] = None, logger: Optional[logging.Logger] = None):
        self.subject_id = subject_id
        self.camera = camera
        self.gaze_world_dir = gaze_world_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.blink_dir = blink_dir
        self.logger = logger or logging.getLogger(__name__)

    def load_mask(self, frame_id, mask_dir):
        """Load segmentation mask of multiple objects from SAM2 for a given subject and camera.
        Args:
            frame_id (str): The ID of the frame.
            mask_path (str): The file path to the segmentation mask image.

        Returns:
            masks (dict): A dictionary containing the segmentation masks as numpy arrays (each array for one object).
        """
        frame_id_str = f"{frame_id:06d}"
        # print(frame_id_str)   # "003000"

        pattern = os.path.join(mask_dir, f"*mask_f{frame_id_str}*.png")
        mask_paths = sorted(glob.glob(pattern))
        # print(f"{mask_paths}")
        if not mask_paths:
            self.logger.warning(f"No masks found for frame {frame_id} in {mask_dir} at path: {pattern}.")
            return {}

        masks = {}

        for mask_path in mask_paths:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                continue

            # Convert to strict binary {0, 1}
            # Any non-zero pixel → 1
            binary_mask = np.where(mask > 0, 1, 0).astype(np.uint8)

            masks[os.path.basename(mask_path)] = binary_mask

        return masks

    def load_gaze_data(self, gaze_path, world_cam_path):
        """Load gaze data from a CSV file.
        Select gaze data within the cut video duration.
        Label each gaze with the corresponding frame id in the cut video. 

        Args:
            gaze_path (str): The file path to the gaze data CSV file.
            world_cam_path (str): The file path to the world camera timestamps CSV file.
        Returns:
            pd.DataFrame: A DataFrame containing the gaze data within the cut video duration.
        """
        gaze_dic = pd.read_csv(gaze_path)
        world_cam_dic = pd.read_csv(world_cam_path)
    

        # recreate two columns so that after merging we still have them
        world_cam_dic['frame_idx'] = world_cam_dic.index
        world_cam_dic['frame_timestamp'] = world_cam_dic['timestamp [ns]']

        aligned = pd.merge_asof(
            gaze_dic,
            world_cam_dic,
            left_on="timestamp [ns]",
            right_on="timestamp [ns]",
            direction="backward",        # <= gaze time (closest before)
            allow_exact_matches=True
        )
        aligned= aligned.dropna(axis = 0, subset=['section id_y'])  # drop rows where no matching world frame found
        return aligned 
    
    def label_blinks(self, blink_df, aligned_gaze_df):
        """Label each gaze point with blink information, whether they are in blink periods."""

        blink_starts = blink_df['start timestamp [ns]'].to_numpy()
        blink_ends   = blink_df['end timestamp [ns]'].to_numpy()
        gaze_ts      = aligned_gaze_df['timestamp [ns]'].to_numpy()

        pos = np.searchsorted(blink_starts, gaze_ts, side="right") - 1  # last blink start <= gaze
        valid = pos >= 0

        in_blink = np.zeros_like(gaze_ts, dtype=bool)
        in_blink[valid] = gaze_ts[valid] <= blink_ends[pos[valid]]
        aligned_gaze_df['in_blink'] = in_blink

        blink_id = -1 * np.ones_like(gaze_ts, dtype=int)
        blink_id[in_blink] = blink_df.loc[pos[valid][in_blink], 'blink id'].values
        aligned_gaze_df['blink id'] = blink_id

        return aligned_gaze_df


    def gaze_to_object_radius(self, mask, x, y, r=20):
        """
        Compute confidence that gaze hits any object (mask>0) within a radius r circle.

        Creates:
        1) a full-size (H,W) "circle_mask" (same size as `mask`)
        2) (optionally) a full-size "patch_mask" (same size as `mask`) containing only
            the local region used to build the circle (mostly zeros outside the bbox)

        Args:
            mask (np.ndarray): (H, W) mask with background=0, objects>0.
            x, y (float): gaze pixel coords (x=col, y=row).
            r (int): radius in pixels.

        Returns:
            confidence (float): (# pixels with mask>0 inside circle) / (circle area in pixels).
            circle_mask (np.ndarray, optional): (H,W) bool mask of the circle.
        """
        if mask.ndim != 2:
            raise ValueError(f"`mask` must be 2D (H,W). Got shape {mask.shape}.")

        H, W = mask.shape
        xi, yi = int(round(x)), int(round(y))

        # If gaze is outside image, there is no meaningful overlap.
        if xi < 0 or xi >= W or yi < 0 or yi >= H:
            return 0.0

        # Bounding box for circle (clipped)
        x0 = max(0, xi - r)
        x1 = min(W, xi + r + 1)
        y0 = max(0, yi - r)
        y1 = min(H, yi + r + 1)

        # ---- Local circle in bbox coordinates ----
        yy, xx = np.ogrid[y0:y1, x0:x1]          # yy: (h,1), xx: (1,w)
        circle_local = (xx - xi) ** 2 + (yy - yi) ** 2 <= r ** 2  # (h,w) boolean

        # ---- Chunk the whole mask into small region (same size as original mask) ----
        mask_patch = mask[y0:y1, x0:x1]
        circle_area = int(circle_local.sum())
        if circle_area == 0:
            return 0.0

        hit = (mask_patch[circle_local] > 0).sum()
        return float(hit) / circle_area
        # # Mostly zeros, only bbox region is filled.
        # patch_mask = np.zeros((H, W), dtype=bool)
        # patch_mask[y0:y1, x0:x1] = circle_local # You need to assign the local mask (the circle) to the corresponding location in the full-size patch_mask
        # circle_mask = patch_mask

        # # Compute confidence: fraction of circle pixels that land on any object (>0)
        # circle_area = int(circle_mask.sum())
        # if circle_area == 0:
        #     confidence = 0.0
        # else:
        #     confidence = float((mask[circle_mask] > 0).sum()) / circle_area

        # return confidence
    
    def process_subject(self, subject_id_temp, camera_temp):
        """Process gaze data for a specific subject and camera."""

        out_dir = Path(self.output_dir+f'/{subject_id_temp}_gazed_object/')
        out_dir.mkdir(parents=True, exist_ok=True)
        gaze_path = self.gaze_world_dir + "/" + subject_id_temp + "_" + camera_temp + "_gaze.csv"
        if not os.path.exists(gaze_path):
            self.logger.warning(f"No gaze data found for subject {subject_id_temp} and camera {camera_temp}. Skipping.")
            return
        
        world_cam_path = self.gaze_world_dir + "/" + subject_id_temp + "_" + camera_temp + "_world_timestamps.csv"
        if not os.path.exists(world_cam_path):
            self.logger.warning(f"No world camera timestamp data found for subject {subject_id_temp} and camera {camera_temp}. Skipping.")
            return
        gaze= self.load_gaze_data(gaze_path, world_cam_path)
    
        if self.blink_dir is None:
            self.logger.info("No blink directory provided, skipping blink labeling.")
            gaze_blink_removed = gaze
        else:
            if not os.path.exists(os.path.join(self.blink_dir, f"{subject_id_temp}_{camera_temp}_blinks.csv")):
                self.logger.warning(f"No blink data found for subject {subject_id_temp} and camera {camera_temp}. Skipping blink labeling.")
                gaze_blink_removed = gaze
                return
            blink_data = pd.read_csv(os.path.join(self.blink_dir, f"{subject_id_temp}_{camera_temp}_blinks.csv"))
            gaze_blink_labeled= self.label_blinks(blink_data, gaze)
            gaze_blink_labeled.to_csv(os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gaze_blink_labeled.csv"), index= False)
            self.logger.info(f"Saved blink labeled gaze data to {os.path.join(out_dir, f'{subject_id_temp}_{camera_temp}_gaze_blink_labeled.csv')}")
            # Create a copy of the gaze DataFrame to store results
            gaze_blink_removed = gaze_blink_labeled.loc[~gaze_blink_labeled['in_blink']].copy()
            gaze_blink_removed.to_csv(os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gaze_blink_removed.csv"), index= False)
            self.logger.info(f"Saved blink removed gaze data to {os.path.join(out_dir, f'{subject_id_temp}_{camera_temp}_gaze_blink_removed.csv')}")

        subject_gaze_probabilities= {}
        mask_subject= self.mask_dir + f'/{subject_id_temp}_{camera_temp}/masks/'
        if not os.path.exists(mask_subject):
            self.logger.warning(f"Mask directory {mask_subject} does not exist. Skipping subject {subject_id_temp} and camera {camera_temp}.")
            return
        self.logger.info(f"Loading masks from {mask_subject} for subject {subject_id_temp} and camera {camera_temp}.")

        for frame_idx, gdf in gaze_blink_removed.groupby('frame_idx', sort=False):
            masks_one_frame = self.load_mask(int(frame_idx), mask_subject)  # ONE disk read per frame
            # now score each gaze in that group against those masks
            for i, row in gdf.iterrows():
                subject_gaze_probabilities[i]= {}
                x = row['gaze x [px]']
                y = row['gaze y [px]']
                best_name, best_conf = None, 0.0
                for mask_name, mask in masks_one_frame.items():
                    conf = self.gaze_to_object_radius(mask, x, y, r=20)
                    subject_gaze_probabilities[i][mask_name]= conf
                    if conf > best_conf:
                        best_name, best_conf = mask_name, conf
                gaze_blink_removed.loc[i, 'gazed_object_id'] = best_name.split('.')[0].split('_')[-1] if best_name else None
                gaze_blink_removed.loc[i, 'gazed_object'] = '_'.join(best_name.split('.')[0].split('_')[2:-1]) if best_name else None
                gaze_blink_removed.loc[i, 'gazed_object_confidence'] = best_conf
            self.logger.info(f"Processed gaze index {i}: gazed object = {gaze_blink_removed.loc[i,'gazed_object']}, confidence = {gaze_blink_removed.loc[i,'gazed_object_confidence']:.4f}")
        output_path= os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gazed_object.csv")
        gaze_blink_removed.to_csv(output_path, index= False)
        self.logger.info(f"Saved gaze object results for blink removed data to {output_path}")
        with open(os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gaze_object_probabilities.pkl"), 'wb') as f:
            pickle.dump(subject_gaze_probabilities, f)
        self.logger.info(f"Saved probabilities of each mask for each eye gaze to {os.path.join(out_dir, f'{subject_id_temp}_{camera_temp}_gaze_object_probabilities.pkl')}")

    def process_multiple(self):
        """Process gaze data for multiple subjects."""
        for subject_id_temp in self.subject_id:
            for camera_temp in self.camera:
                self.logger.info(f"Processing subject {subject_id_temp} and camera {camera_temp}...")
                self.process_subject(subject_id_temp, camera_temp)

def main():
    parser = argparse.ArgumentParser(
    description='Detect gazed objects based on gaze data and SAM2 masks.',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    epilog=(
        'Example usage:\n'
        '# Process a single subject and camera\n'
        'python gazed_object.py 27 child '
        'gaze_dir /path/to/gaze '
        'mask_dir /path/to/masks '
        'output_dir /path/to/output '
        '--blink-dir /path/to/blinks(If remove gaze during blinks) '
        '--log-path /path/to/logfile.log\n\n'

        '# Process multiple subjects and cameras\n'
        'python gazed_object.py 27,28 child,parent'
        'gaze_dir /path/to/gaze '
        'mask_dir /path/to/masks '
        'output_dir /path/to/output '
        '--blink-dir /path/to/blinks(If remove gaze during blinks) '
        '--log-path /path/to/logfile.log\n\n'
        'This code will make new folders with the names '
        '[(subj_id)_gazed_object] in the specified output directory. '
        'Output files will be saved in that folder.'
    ))

    # Required positional arguments (already required by argparse)
    parser.add_argument('subj_id', help='Subject ID e.g. 27 or 27,28 etc.')
    parser.add_argument('camera_id', help='Camera ID e.g. child or parent or child,parent')
    parser.add_argument('gaze_world_dir', help='Path to the gaze timestamps files and Egocentric Video Frame timestamps files')
    parser.add_argument('mask_dir', help='Directory containing mask files')
    parser.add_argument('output_dir', help='Directory to save output files')
    parser.add_argument('--blink-dir', help='If remove gaze during blinks', required=False)
    parser.add_argument('--log-path', help='Path to the log file', required=False)
    args = parser.parse_args()

    if os.path.exists(args.gaze_world_dir) is False:
        raise FileNotFoundError(f"Gaze and world directory {args.gaze_world_dir} does not exist.")
    if os.path.exists(args.mask_dir) is False:
        raise FileNotFoundError(f"Mask directory {args.mask_dir} does not exist.")
    if args.blink_dir is not None and os.path.exists(args.blink_dir) is False:
        raise FileNotFoundError(f"Blink directory {args.blink_dir} does not exist.")
    if os.path.exists(args.log_path) is False and args.log_path is not None:
        raise FileNotFoundError(f"Log path directory {args.log_path} does not exist.")

    # Setup logging
    if args.log_path is None:
        args.log_path = os.path.join(args.output_dir, 'gaze_object.log')
    log_path = Path(args.log_path)
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_path.resolve()}")

    if ',' in args.subj_id: # if a list of subject ids is provided
        subj_ids = [s.strip() for s in args.subj_id.split(',')]
    else:
        subj_ids = [args.subj_id]

    if ',' in args.camera_id: # if a list of camera ids is provided
        camera_ids = [c.strip() for c in args.camera_id.split(',')] 
    else:
        camera_ids = [args.camera_id]
    
    if args.blink_dir is not None:
        gaze_aligner = GazeObjectAligner(
        subj_ids,
        camera_ids,
        args.gaze_world_dir,
        args.mask_dir,
        args.output_dir,
        blink_dir=args.blink_dir,
        logger=logger,
        )
        logger.info("Starting gaze object detection...")
        logger.info(f"Subjects: {subj_ids}, Cameras: {camera_ids}")
        logger.info(f"Gaze Directory: {args.gaze_world_dir}")
        logger.info(f"Mask Directory: {args.mask_dir}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"Blink Directory: {args.blink_dir}")

    else:
        gaze_aligner = GazeObjectAligner(
        subj_ids,
        camera_ids,
        args.gaze_world_dir,
        args.mask_dir,
        args.output_dir,
        logger=logger,
        )
        logger.info("Starting gaze object detection...")
        logger.info(f"Subjects: {subj_ids}, Cameras: {camera_ids}")
        logger.info(f"Gaze Directory: {args.gaze_world_dir}")
        logger.info(f"Mask Directory: {args.mask_dir}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"No blink directory provided, skipping blink labeling.")


    gaze_aligner.process_multiple()
    logger.info("Gaze object detection complete!")

if __name__ == "__main__":
    main()

   