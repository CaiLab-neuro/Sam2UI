#!/usr/bin/env python3
import cv2
import numpy as np
import pandas as pd
import os
import glob
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import logging, sys
import argparse
from pathlib import Path
from typing import Optional

"""
This is an aligner script (published version) to find which object is being gazed at based on segmentation masks and gaze data.
This code includes functions to
    1) Load segmentation masks from SAM2 or 3,
    2) Load gaze data and align gaze points with the corresponding video frames based on timestamps,
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

    4) (Optional) Blink data directory: /path/to/blink_data
    If provided, the code will label and remove gaze points that fall within blink periods and remove them from the analysis.
    Within this directory, the code expects
        a) Blink data CSV files named as *{subject_id}_{camera}_blinks.csv

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
    def __init__(
        self,
        gaze_world_dir: str,
        mask_dir: str,
        output_dir: str,
        blink_dir: Optional[str] = None,
        start_plot_time: Optional[float] = None,
        end_plot_time: Optional[float] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.gaze_world_dir = gaze_world_dir
        self.mask_dir = mask_dir
        self.output_dir = output_dir
        self.blink_dir = blink_dir
        self.start_plot_time = start_plot_time
        self.end_plot_time = end_plot_time
        self.logger = logger or logging.getLogger(__name__)

    def load_mask(self, frame_id: int, mask_dir: str) -> dict[str, np.ndarray]:
        """Load segmentation mask for one frame of multiple objects from SAM2 for a given subject and camera.
        Args:
            frame_id (str): The ID of the frame.
            mask_path (str): The file path to the segmentation mask image.

        Returns:
            masks (dict): A dictionary containing the segmentation masks for one frame as numpy arrays (each array for one object).
        """
        frame_id_str = f"{frame_id:06d}"
        # print(frame_id_str)   # "003000"

        pattern = os.path.join(mask_dir, f"*mask_f{frame_id_str}*.png")
        mask_paths = sorted(glob.glob(pattern))
        # print(f"{mask_paths}")
        if not mask_paths:
            self.logger.debug("No masks found for frame %s in %s", frame_id, mask_dir)
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

    @staticmethod
    def _mask_name_to_category(mask_name: str) -> str:
        """Convert a SAM mask filename into an object category name."""
        stem = Path(mask_name).stem
        parts = stem.split('_')
        if len(parts) >= 4 and parts[0] == "mask" and parts[1].startswith("f") and parts[-1].startswith("id"):
            category_parts = parts[2:-1]
        elif len(parts) > 1:
            category_parts = parts[:-1]
        else:
            category_parts = parts
        category = "_".join(category_parts).strip("_")
        return category if category else stem

    @staticmethod
    def _build_time_edges(time_s: np.ndarray, default_step: float = 1.0 / 30.0) -> np.ndarray:
        """Build bin edges from sample timestamps for bar and heatmap plotting."""
        if len(time_s) == 0:
            return np.array([0.0, default_step], dtype=float)
        if len(time_s) == 1:
            t0 = max(0.0, float(time_s[0]))
            return np.array([t0, t0 + default_step], dtype=float)

        midpoints = (time_s[:-1] + time_s[1:]) / 2.0
        left_edge = max(0.0, float(time_s[0] - (midpoints[0] - time_s[0])))
        right_edge = float(time_s[-1] + (time_s[-1] - midpoints[-1]))
        if right_edge <= midpoints[-1]:
            right_edge = float(midpoints[-1] + default_step)

        edges = np.empty(len(time_s) + 1, dtype=float)
        edges[0] = left_edge
        edges[1:-1] = midpoints
        edges[-1] = right_edge
        return edges

    def plot_method_figures(
        self,
        subject_id_temp: str,
        camera_temp: str,
        gaze_df: pd.DataFrame,
        subject_gaze_probabilities: dict[int, dict[str, float]],
        out_dir: Path,
    ) -> None:
        """Create method figures: trajectory trace and confidence heatmap."""
        if gaze_df.empty:
            self.logger.warning(
                "No gaze rows available for plotting for subject=%s camera=%s",
                subject_id_temp,
                camera_temp,
            )
            return

        if "timestamp [ns]" not in gaze_df.columns:
            self.logger.warning(
                "Cannot create method figures for subject=%s camera=%s because column 'timestamp [ns]' is missing",
                subject_id_temp,
                camera_temp,
            )
            return

        plot_df = gaze_df.sort_values("timestamp [ns]").copy()
        timestamp_ns = pd.to_numeric(plot_df["timestamp [ns]"], errors="coerce").to_numpy(dtype=float)
        valid_time = np.isfinite(timestamp_ns)
        if not valid_time.any():
            self.logger.warning(
                "Cannot create method figures for subject=%s camera=%s because all timestamps are invalid",
                subject_id_temp,
                camera_temp,
            )
            return
        plot_df = plot_df.loc[valid_time].copy()
        timestamp_ns = timestamp_ns[valid_time]
        full_time_s = (timestamp_ns - timestamp_ns[0]) / 1e9

        requested_start_s = 0.0 if self.start_plot_time is None else float(self.start_plot_time)
        requested_end_s = float(full_time_s[-1]) if self.end_plot_time is None else float(self.end_plot_time)

        if requested_start_s < 0.0:
            self.logger.warning(
                "start_plot_time=%.3f is negative for subject=%s camera=%s. Clamping to 0.0.",
                requested_start_s,
                subject_id_temp,
                camera_temp,
            )
            requested_start_s = 0.0
        if requested_end_s < 0.0:
            self.logger.warning(
                "end_plot_time=%.3f is negative for subject=%s camera=%s. Clamping to 0.0.",
                requested_end_s,
                subject_id_temp,
                camera_temp,
            )
            requested_end_s = 0.0
        if requested_end_s < requested_start_s:
            self.logger.warning(
                (
                    "Cannot create method figures for subject=%s camera=%s because "
                    "start_plot_time (%.3f) is greater than end_plot_time (%.3f)."
                ),
                subject_id_temp,
                camera_temp,
                requested_start_s,
                requested_end_s,
            )
            return

        segment_mask = (full_time_s >= requested_start_s) & (full_time_s <= requested_end_s)
        if not segment_mask.any():
            self.logger.warning(
                (
                    "No gaze samples in requested plot window [%.3f, %.3f] s for subject=%s camera=%s "
                    "(available range [0.000, %.3f] s)."
                ),
                requested_start_s,
                requested_end_s,
                subject_id_temp,
                camera_temp,
                float(full_time_s[-1]),
            )
            return

        plot_df = plot_df.loc[segment_mask].copy()
        time_s = full_time_s[segment_mask]
        self.logger.info(
            (
                "Plotting method figures for subject=%s camera=%s using time window [%.3f, %.3f] s "
                "(available [0.000, %.3f] s, points=%d)."
            ),
            subject_id_temp,
            camera_temp,
            requested_start_s,
            requested_end_s,
            float(full_time_s[-1]),
            len(plot_df),
        )
        time_edges = self._build_time_edges(time_s)

        if "gazed_object" in plot_df.columns:
            raw_categories = plot_df["gazed_object"].fillna("no_object").astype(str)
        else:
            raw_categories = pd.Series(["no_object"] * len(plot_df), index=plot_df.index)

        gaze_categories = []
        for category in raw_categories:
            cleaned = category.strip()
            if cleaned == "" or cleaned.lower() in {"none", "nan"}:
                cleaned = "no_object"
            gaze_categories.append(cleaned)

        category_order = list(dict.fromkeys(gaze_categories)) # remove duplicates while preserving order (like unique but preserves first occurrence order)
        category_to_y = {cat: i for i, cat in enumerate(category_order)} # This loops over the (index, category) pairs and builds a dictionary where: key = category, value = index

        figures_dir = Path(out_dir) / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        trajectory_height = float(np.clip(1.3 + 0.35 * len(category_order), 3.5, 18.0)) #np.clip: If value < 3.5 → return 3.5; If value > 18.0 → return 18.0
        trajectory_fig, trajectory_ax = plt.subplots(figsize=(12, trajectory_height))
        cmap = plt.get_cmap("tab20")
        y_values = np.array([category_to_y[cat] for cat in gaze_categories], dtype=float)
        point_colors = [cmap(category_to_y[cat] % cmap.N) for cat in gaze_categories]

        # Plot temporal trajectory using time-point samples (no bars).
        trajectory_ax.plot(
            time_s,
            y_values,
            color="#4a4a4a",
            linewidth=1.2,
            alpha=0.85,
            zorder=2,
        )
        trajectory_ax.scatter(
            time_s,
            y_values,
            c=point_colors,
            s=16,
            edgecolors="none",
            zorder=3,
        )
        trajectory_ax.set_yticks(np.arange(len(category_order)))
        trajectory_ax.set_yticklabels(category_order)
        trajectory_ax.set_xlabel("Time (s)")
        trajectory_ax.set_ylabel("Gazed Category")
        trajectory_ax.set_title(f"Gaze trajectory ({subject_id_temp}, {camera_temp})")
        trajectory_ax.set_xlim(float(time_edges[0]), float(time_edges[-1]))
        trajectory_ax.set_ylim(-0.5, len(category_order) - 0.5)
        trajectory_ax.grid(axis="x", linestyle="--", alpha=0.35)
        trajectory_fig.tight_layout()

        trajectory_png = figures_dir / f"{subject_id_temp}_{camera_temp}_trajectory_plot.png"
        trajectory_pdf = figures_dir / f"{subject_id_temp}_{camera_temp}_trajectory_plot.pdf"
        trajectory_fig.savefig(trajectory_png, dpi=300, bbox_inches="tight")
        trajectory_fig.savefig(trajectory_pdf, bbox_inches="tight")
        plt.close(trajectory_fig)
        self.logger.info("Saved trajectory plot to %s and %s", trajectory_png, trajectory_pdf)

        heatmap_categories = list(category_order)
        for probs_one_gaze in subject_gaze_probabilities.values():
            for mask_name in probs_one_gaze.keys():
                category = self._mask_name_to_category(mask_name)
                if category not in heatmap_categories:
                    heatmap_categories.append(category)
        heatmap_row = {cat: i for i, cat in enumerate(heatmap_categories)}
        confidence_matrix = np.zeros((len(heatmap_categories), len(plot_df)), dtype=float)

        for col_idx, gaze_index in enumerate(plot_df.index):
            probs_one_gaze = subject_gaze_probabilities.get(gaze_index, {})
            if probs_one_gaze:
                per_category_conf = {}
                for mask_name, conf in probs_one_gaze.items():
                    category = self._mask_name_to_category(mask_name)
                    conf_val = float(conf)
                    if category not in per_category_conf or conf_val > per_category_conf[category]:
                        per_category_conf[category] = conf_val
                for category, conf_val in per_category_conf.items():
                    confidence_matrix[heatmap_row[category], col_idx] = conf_val
            elif "gazed_object_confidence" in plot_df.columns:
                category = gaze_categories[col_idx]
                conf_val = float(plot_df.iloc[col_idx]["gazed_object_confidence"])
                if np.isfinite(conf_val):
                    confidence_matrix[heatmap_row[category], col_idx] = max(0.0, conf_val)

        heatmap_height = float(np.clip(1.3 + 0.35 * len(heatmap_categories), 3.5, 20.0))
        heatmap_fig, heatmap_ax = plt.subplots(figsize=(12, heatmap_height))
        y_edges = np.arange(len(heatmap_categories) + 1, dtype=float)
        mesh = heatmap_ax.pcolormesh(
            time_edges,
            y_edges,
            confidence_matrix,
            cmap="viridis",
            vmin=0.0,
            vmax=1.0,
            shading="auto",
        )
        heatmap_ax.set_yticks(np.arange(len(heatmap_categories)) + 0.5)
        heatmap_ax.set_yticklabels(heatmap_categories)
        heatmap_ax.set_xlabel("Time (s)")
        heatmap_ax.set_ylabel("Gazed Category")
        heatmap_ax.set_title(f"Gaze confidence heatmap ({subject_id_temp}, {camera_temp})")
        heatmap_ax.set_xlim(float(time_edges[0]), float(time_edges[-1]))
        cbar = heatmap_fig.colorbar(mesh, ax=heatmap_ax)
        cbar.set_label("Confidence")
        heatmap_fig.tight_layout()

        heatmap_png = figures_dir / f"{subject_id_temp}_{camera_temp}_confidence_heatmap.png"
        heatmap_pdf = figures_dir / f"{subject_id_temp}_{camera_temp}_confidence_heatmap.pdf"
        heatmap_fig.savefig(heatmap_png, dpi=300, bbox_inches="tight")
        heatmap_fig.savefig(heatmap_pdf, bbox_inches="tight")
        plt.close(heatmap_fig)
        self.logger.info("Saved confidence heatmap to %s and %s", heatmap_png, heatmap_pdf)

    def load_gaze_data(self, subj: str, camera: str) -> pd.DataFrame:
        """Load gaze data from a CSV file.
        Select gaze data within the cut video duration.
        Label each gaze with the corresponding frame id in the cut video. 

        Args:
            subj (str): The subject identifier.
            camera (str): The camera identifier.
        Returns:
            pd.DataFrame: A DataFrame containing the gaze data within the cut video duration.
        """
        gaze_path = Path(self.gaze_world_dir) / f"{subj}_{camera}_gaze.csv"
        gaze_dic = pd.read_csv(gaze_path)
        world_cam_path = Path(self.gaze_world_dir) / f"{subj}_{camera}_world_timestamps.csv"
        world_cam_dic = pd.read_csv(world_cam_path)
        self.logger.info(
            "Loaded gaze/world CSV: gaze_rows=%d world_rows=%d",
            len(gaze_dic),
            len(world_cam_dic),
        )

        required_cols = {"timestamp [ns]", "gaze x [px]", "gaze y [px]"}
        if not required_cols.issubset(gaze_dic.columns):
            missing = sorted(required_cols.difference(gaze_dic.columns))
            raise ValueError(f"Missing required gaze columns: {missing}")
        
        required_cols_world = {"timestamp [ns]"}
        if not required_cols_world.issubset(world_cam_dic.columns):
            missing = sorted(required_cols_world.difference(world_cam_dic.columns))
            raise ValueError(f"Missing required world camera timestamp columns: {missing}")
        self.logger.info(
            "Required columns verified")

        # Keep world-camera mapping columns authoritative from world CSV to avoid
        # merge suffix collisions when gaze CSV already contains these columns.
        mapping_cols = ["frame_idx", "frame_timestamp", "source_frame_idx"]
        gaze_dic = gaze_dic.drop(columns=[c for c in mapping_cols if c in gaze_dic.columns])

        # recreate two columns so that after merging we still have them
        world_cam_dic['frame_idx'] = world_cam_dic.index
        world_cam_dic['frame_timestamp'] = world_cam_dic['timestamp [ns]']
        gaze_dic = gaze_dic.sort_values("timestamp [ns]")
        world_cam_dic = world_cam_dic.sort_values("timestamp [ns]")
        aligned = pd.merge_asof(
            gaze_dic,
            world_cam_dic,
            left_on="timestamp [ns]",
            right_on="timestamp [ns]",
            direction="backward",        # <= gaze time (closest before)
            allow_exact_matches=True
        )
        aligned = aligned.dropna(subset=['frame_idx']) # drop rows where no matching world frame found
        return aligned

    def label_blinks(self, aligned_gaze_df: pd.DataFrame, subj: str, camera: str) -> pd.DataFrame:
        """Label each gaze point with blink information, whether they are in blink periods.
         Args:
            subj (str): The subject identifier.
            camera (str): The camera identifier.
        Returns:
            pd.DataFrame: A DataFrame containing the gaze data within the cut video duration and with blink removed.
        """
        blink_df = pd.read_csv(os.path.join(self.blink_dir, f"{subj}_{camera}_blinks.csv"))
        blink_df = blink_df.sort_values("start timestamp [ns]")
        blink_starts = blink_df['start timestamp [ns]'].to_numpy()
        blink_ends   = blink_df['end timestamp [ns]'].to_numpy()
        gaze_ts      = aligned_gaze_df['timestamp [ns]'].to_numpy()

        pos = np.searchsorted(blink_starts, gaze_ts, side="right") - 1  # last blink start <= gaze
        valid = pos >= 0

        in_blink = np.zeros_like(gaze_ts, dtype=bool)
        in_blink[valid] = gaze_ts[valid] <= blink_ends[pos[valid]]
        aligned_gaze_df['in_blink'] = in_blink

        blink_id = -1 * np.ones_like(gaze_ts, dtype=int)
        mask = valid & in_blink
        blink_id[mask] = blink_df.iloc[pos[mask]]['blink id'].values
        aligned_gaze_df['blink id'] = blink_id

        return aligned_gaze_df


    def gaze_to_object_radius(self, mask: np.ndarray, x: float, y: float, r: int = 20) -> float:
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

    def process_subject(self, subject_id_temp: str, camera_temp: str) -> None:
        """Process gaze data for a specific subject and camera."""
        out_dir= Path(self.output_dir) / f'{subject_id_temp}_gazed_object'
        out_dir.mkdir(parents=True, exist_ok=True)
        gaze= self.load_gaze_data(subject_id_temp, camera_temp)
        if len(gaze) == 0:
            self.logger.warning(f"No gaze data found for subject {subject_id_temp} and camera {camera_temp}. Skipping.")
            return

        if self.blink_dir is None:
            gaze_blink_removed = gaze
        else:
            gaze_blink_labeled= self.label_blinks(gaze, subject_id_temp, camera_temp)
            gaze_blink_labeled.to_csv(os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gaze_blink_labeled.csv"), index= False)
            self.logger.info(f"Saved blink labeled gaze data to {os.path.join(out_dir, f'{subject_id_temp}_{camera_temp}_gaze_blink_labeled.csv')}")
            # Create a copy of the gaze DataFrame to store results
            gaze_blink_removed = gaze_blink_labeled.loc[~gaze_blink_labeled['in_blink']].copy()
            gaze_blink_removed.to_csv(os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gaze_blink_removed.csv"), index= False)
            self.logger.info(f"Saved blink removed gaze data to {os.path.join(out_dir, f'{subject_id_temp}_{camera_temp}_gaze_blink_removed.csv')}")

        subject_gaze_probabilities = {}
        mask_subject = Path(self.mask_dir) / f"{subject_id_temp}_{camera_temp}" / "masks"
        self.logger.info(f"Loading masks from {mask_subject}")
        total_gaze_points = len(gaze_blink_removed)
        total_frames = int(gaze_blink_removed["frame_idx"].nunique())
        frames_with_masks = 0
        frames_without_masks = 0
        assigned_gaze_points = 0
        assigned_confidences = []

        gaze_blink_removed['gazed_object_id'] = None
        gaze_blink_removed['gazed_object'] = None
        gaze_blink_removed['gazed_object_confidence'] = 0.0
        for frame_idx, gdf in gaze_blink_removed.groupby('frame_idx', sort=False):
            masks_one_frame = self.load_mask(int(frame_idx), mask_subject)  # ONE disk read per frame
            if len(masks_one_frame) == 0:
                frames_without_masks += 1
                continue
            frames_with_masks += 1
            # now score each gaze in that group against those masks
            for i in gdf.index:
                row = gdf.loc[i]
                subject_gaze_probabilities[i]= {}
                x = row['gaze x [px]']
                y = row['gaze y [px]']
                best_name, best_conf = None, 0.0
                for mask_name, mask in masks_one_frame.items():
                    conf = self.gaze_to_object_radius(mask, x, y, r=20)
                    subject_gaze_probabilities[i][mask_name]= conf
                    if conf > best_conf:
                        best_name, best_conf = mask_name, conf
                if best_name:
                    gaze_blink_removed.loc[i, 'gazed_object_id'] = best_name.split('.')[0].split('_')[-1]
                    gaze_blink_removed.loc[i, 'gazed_object'] = '_'.join(best_name.split('.')[0].split('_')[2:-1])
                    gaze_blink_removed.loc[i, 'gazed_object_confidence'] = best_conf
                    assigned_gaze_points += 1
                    assigned_confidences.append(best_conf)
                    self.logger.debug(
                        "Processed gaze index %s: gazed object=%s confidence=%.4f",
                        i,
                        best_name,
                        best_conf,
                    )

        assignment_rate = (assigned_gaze_points / total_gaze_points * 100.0) if total_gaze_points > 0 else 0.0
        mean_conf = float(np.mean(assigned_confidences)) if assigned_confidences else 0.0
        median_conf = float(np.median(assigned_confidences)) if assigned_confidences else 0.0
        self.logger.info(
            (
                "Gaze-object assignment summary for subject=%s camera=%s: "
                "total_gaze_points=%d assigned=%d assignment_rate=%.2f%% "
                "frames_with_masks=%d/%d mean_conf=%.4f median_conf=%.4f"
            ),
            subject_id_temp,
            camera_temp,
            total_gaze_points,
            assigned_gaze_points,
            assignment_rate,
            frames_with_masks,
            total_frames,
            mean_conf,
            median_conf,
        )
        if frames_without_masks > 0:
            self.logger.info(
                "No segmentation masks were found for %d frame(s) for subject=%s camera=%s.",
                frames_without_masks,
                subject_id_temp,
                camera_temp,
            )
        output_path = os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gazed_object.csv")
        gaze_blink_removed.to_csv(output_path, index=False)
        self.logger.info(f"Saved gaze object results to {output_path}")
        with open(os.path.join(out_dir, f"{subject_id_temp}_{camera_temp}_gaze_object_probabilities.pkl"), 'wb') as f:
            pickle.dump(subject_gaze_probabilities, f)
        self.logger.info(f"Saved probabilities of each mask for each eye gaze to {os.path.join(out_dir, f'{subject_id_temp}_{camera_temp}_gaze_object_probabilities.pkl')}")
        self.plot_method_figures(
            subject_id_temp=subject_id_temp,
            camera_temp=camera_temp,
            gaze_df=gaze_blink_removed,
            subject_gaze_probabilities=subject_gaze_probabilities,
            out_dir=out_dir,
        )
        

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
    parser.add_argument('gaze_world_dir', help='Path to the gaze timestamps files and Egocentric Video Frame timestamps files')
    parser.add_argument('mask_dir', help='Directory containing mask files')
    parser.add_argument('output_dir', help='Directory to save output files')
    parser.add_argument('--subject-id', dest='subject_id', help='Subject ID e.g. 27 or 27,28')
    parser.add_argument('--camera-id', help='Camera ID e.g. child or child,parent')
    parser.add_argument('--blink-dir', help='If remove gaze during blinks', required=False)
    parser.add_argument('--log-path', help='Path to the log file', required=False)
    parser.add_argument(
        '--start-plot-time',
        type=float,
        default=None,
        help='Optional start time in seconds (from recording start) for method-figure plotting.',
    )
    parser.add_argument(
        '--end-plot-time',
        type=float,
        default=None,
        help='Optional end time in seconds (from recording start) for method-figure plotting.',
    )

    args = parser.parse_args()
    if args.start_plot_time is not None and args.start_plot_time < 0:
        parser.error("--start-plot-time must be >= 0.")
    if args.end_plot_time is not None and args.end_plot_time < 0:
        parser.error("--end-plot-time must be >= 0.")
    if (
        args.start_plot_time is not None
        and args.end_plot_time is not None
        and args.start_plot_time > args.end_plot_time
    ):
        parser.error("--start-plot-time must be <= --end-plot-time.")

    # Setup logging
    if args.log_path is None:
        args.log_path = os.path.join(args.output_dir, 'gaze_object.log')
    log_path = Path(args.log_path)
    setup_logging(log_path)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging to: {log_path.resolve()}")

    if args.subject_id:
        subj_ids = [int(id.strip()) for id in args.subject_id.split(',')]
    else:
        gaze_worldcam_dir = Path(args.gaze_world_dir)
        subj_ids = np.unique([
            int(p.stem.split('_')[0])
            for p in gaze_worldcam_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv"}
        ])

    if args.camera_id:
        camera_list = [i.strip() for i in args.camera_id.split(',')]
    else:
        gaze_worldcam_dir = Path(args.gaze_world_dir)
        camera_list = np.unique([
            p.stem.split('_')[1] for p in gaze_worldcam_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv"}
        ])
    
    if args.blink_dir is not None:
        gaze_aligner = GazeObjectAligner(
        args.gaze_world_dir,
        args.mask_dir,
        args.output_dir,
        blink_dir=args.blink_dir,
        start_plot_time=args.start_plot_time,
        end_plot_time=args.end_plot_time,
        logger=logger,
        )
        logger.info("------- Loaded Files and Directories -------")

        logger.info(f"Subjects: {subj_ids}, Cameras: {camera_list}")
        logger.info(f"Gaze Directory: {args.gaze_world_dir}")
        logger.info(f"Mask Directory: {args.mask_dir}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"Blink Directory: {args.blink_dir}")
        logger.info(f"Plot Start Time (s): {args.start_plot_time}")
        logger.info(f"Plot End Time (s): {args.end_plot_time}")

    else:
        gaze_aligner = GazeObjectAligner(
        args.gaze_world_dir,
        args.mask_dir,
        args.output_dir,
        start_plot_time=args.start_plot_time,
        end_plot_time=args.end_plot_time,
        logger=logger,
        )
        logger.info("------- Loaded Files and Directories -------")
        logger.info(f"Subjects: {subj_ids}, Cameras: {camera_list}")
        logger.info(f"Gaze Directory: {args.gaze_world_dir}")
        logger.info(f"Mask Directory: {args.mask_dir}")
        logger.info(f"Output Directory: {args.output_dir}")
        logger.info(f"Plot Start Time (s): {args.start_plot_time}")
        logger.info(f"Plot End Time (s): {args.end_plot_time}")
        logger.info(f"No blink directory provided, skipping blink labeling.")


    
    for subj in subj_ids:
        for cam in camera_list:
            logger.info("------- Start Gaze Mask Processing -------")
            logger.info(f"------- Processing subject {subj}, camera {cam} -------")
            gaze_aligner.process_subject(str(subj), cam)
            logger.info(f"------- Finished processing subject {subj}, camera {cam} -------")
    logger.info("Gaze object detection complete!")

if __name__ == "__main__":
    main()

   
