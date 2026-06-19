"""
SAM3 Pipeline - Core segmentation logic for hierarchical object detection.

This module implements the main processing pipeline for SAM3-based video segmentation,
including concept detection, instance management, and refinement operations.
"""

import os
# Must be set before any torch import so the CUDA allocator uses expandable segments,
# which avoids fragmentation-driven monotonic growth in nvidia-smi memory readings.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import json
import torch
import hashlib
import numpy as np
from collections import defaultdict
from datetime import datetime
from typing import List, Tuple, Optional, Callable
from pathlib import Path

from sam3_project import (
    SAM3Project, SAM3Concept, SAM3Instance, ConceptStatus,
    save_concept_state, load_concept_state
)
from sam3_utils import extract_frames_from_video


def _prune_non_cond_outputs(inner_state: dict, frame_idx: int, keep: int = 20) -> None:
    """Prune stale non_cond_frame_outputs from every tracker state.

    SAM3's multi-GPU structure stores maskmem tensors at:
      inner_state["tracker_inference_states"][gpu]["output_dict"]["non_cond_frame_outputs"]
    and per-object at:
      inner_state["tracker_inference_states"][gpu]["output_dict_per_obj"][obj]["non_cond_frame_outputs"]

    SAM3.1 multiplex stores the equivalent bucketized tracker states under
    inner_state["sam2_inference_states"] with the same output_dict key names.

    SAM3 attends to at most num_maskmem=7 previous frames, so keeping `keep` (default 20)
    frames of history is generous.  Call this once per yielded frame.

    Cond frames are intentionally NOT evicted here.  select_closest_cond_frames selects
    attending frames by temporal proximity (closest before + closest after + nearest by
    distance), so every user correction frame remains eligible as long as it stays in
    cond_frame_outputs.  Evicting old cond entries by index (as was done previously)
    would silently remove correction anchors that SAM3 would have attended to.
    """
    cutoff = frame_idx - keep
    tracker_states = (inner_state.get("tracker_inference_states", [])
                      + inner_state.get("sam2_inference_states", []))
    for ts in tracker_states:
        # Prune non-cond outputs (maskmem for non-conditioning frames)
        non_cond = ts.get("output_dict", {}).get("non_cond_frame_outputs", {})
        stale = [f for f in non_cond if f < cutoff]
        for f in stale:
            del non_cond[f]

        for obj_dict in ts.get("output_dict_per_obj", {}).values():
            non_cond = obj_dict.get("non_cond_frame_outputs", {})
            stale = [f for f in non_cond if f < cutoff]
            for f in stale:
                del non_cond[f]


def compute_period_peaks(periods: List[Tuple[int, int]],
                         frame_pixel_counts: "dict[int, int]",
                         total_pixels: int = 0) -> List[dict]:
    """For each (start, end) period return the peak frame and pixel-ratio stats.

    Returns a list of dicts:
        {"start": int, "end": int, "best_frame": int, "pixel_count": int,
         "avg_pixel_ratio": float, "min_pixel_ratio": float, "max_pixel_ratio": float}

    pixel_ratio fields are 0.0 when total_pixels is 0 or no counts are available.
    Intended to be computed once during propagation (while masks are in memory)
    and stored on SAM3Instance so later operations (delete/absorb anchor
    generation) can find the best frame per period without re-reading any PNGs.
    """
    peaks = []
    inv = 1.0 / total_pixels if total_pixels > 0 else 0.0
    for start, end in periods:
        best_frame = None
        best_count = 0
        total_count = 0
        min_count = None
        max_count = 0
        n_present = 0
        for f in range(start, end + 1):
            c = frame_pixel_counts.get(f, 0)
            if c > best_count:
                best_count = c
                best_frame = f
            if c > 0:
                total_count += c
                max_count = max(max_count, c)
                min_count = c if min_count is None else min(min_count, c)
                n_present += 1
        if best_frame is not None and best_count > 0:
            avg_ratio = (total_count * inv / n_present) if n_present > 0 else 0.0
            peaks.append({
                "start": start, "end": end,
                "best_frame": best_frame, "pixel_count": best_count,
                "avg_pixel_ratio": avg_ratio,
                "min_pixel_ratio": (min_count or 0) * inv,
                "max_pixel_ratio": max_count * inv,
            })
    return peaks


def compute_continuous_periods(mask_dir: str) -> List[Tuple[int, int]]:
    """
    Return maximal runs of consecutive frame indices that have a mask file.
    Result is a sorted list of (start_frame, end_frame) inclusive tuples.
    """
    if not os.path.exists(mask_dir):
        return []
    frame_nums = []
    for fname in os.listdir(mask_dir):
        try:
            frame_nums.append(int(os.path.splitext(fname)[0]))
        except ValueError:
            continue
    if not frame_nums:
        return []
    frame_nums.sort()
    periods: List[Tuple[int, int]] = []
    start = prev = frame_nums[0]
    for f in frame_nums[1:]:
        if f == prev + 1:
            prev = f
        else:
            periods.append((start, prev))
            start = prev = f
    periods.append((start, prev))
    return periods


def save_cond_frame_states(sam3_model, session_id: str, concept_dir: str) -> int:
    """
    Save maskmem cond frame states to disk after propagation.

    Must be called BEFORE close_session() while tracker states are still live.
    Saves the combined [B, ...] tensors per cond frame so that
    restore_cond_frame_states() can inject them back in a fresh session.

    tracker_inference_states is a list of "ranks" (SAM3's internal object-sharding
    structure), but in this codebase's usage it is always a single rank covering
    all objects — so ranks are concatenated together here rather than kept as
    separate per-rank files; "rank_sizes" in the index records how to split the
    combined batch back apart on restore if more than one rank is ever present.

    Returns number of cond frame files written.
    """
    inference_state = sam3_model._all_inference_states[session_id]["state"]
    tracker_states = inference_state.get("tracker_inference_states", [])
    tracker_metadata = inference_state.get("tracker_metadata", {})

    if not tracker_states:
        print("save_cond_frame_states: no tracker states found — nothing to save.")
        return 0

    cond_states_dir = os.path.join(concept_dir, "cond_states")
    os.makedirs(cond_states_dir, exist_ok=True)

    obj_ids_per_rank = tracker_metadata.get("obj_ids_per_gpu", [[]])

    # Group cond frame outputs across ranks by frame index.
    frames_data: dict = {}  # frame_t (int) -> [(rank, out), ...] in rank order
    for rank, tracker_state in enumerate(tracker_states):
        cond_outputs = tracker_state.get("output_dict", {}).get("cond_frame_outputs", {})
        for frame_t, out in cond_outputs.items():
            if out.get("maskmem_features") is None and out.get("obj_ptr") is None:
                continue
            frames_data.setdefault(int(frame_t), []).append((rank, out))

    def _cat(rank_outs, key, dtype):
        tensors = [ro[1].get(key) for ro in rank_outs]
        if any(t is None for t in tensors):
            return None
        return torch.cat([t.detach().cpu().to(dtype) for t in tensors], dim=0).numpy()

    maskmem_pos_enc_saved = False
    total_saved = 0
    saved_frames = []

    for frame_t, rank_outs in sorted(frames_data.items()):
        rank_outs.sort(key=lambda ro: ro[0])
        obj_ids = np.concatenate([
            np.array([int(x) for x in obj_ids_per_rank[r]], dtype=np.int64)
            for r, _ in rank_outs
        ]) if rank_outs else np.array([], dtype=np.int64)

        save_dict = {"obj_ids": obj_ids}
        mf = _cat(rank_outs, "maskmem_features", torch.float16)
        if mf is not None:
            save_dict["maskmem_features"] = mf
        obj_ptr = _cat(rank_outs, "obj_ptr", torch.float32)
        if obj_ptr is not None:
            save_dict["obj_ptr"] = obj_ptr
        pred_masks = _cat(rank_outs, "pred_masks", torch.float16)
        if pred_masks is not None:
            save_dict["pred_masks"] = pred_masks
        obj_score = _cat(rank_outs, "object_score_logits", torch.float32)
        if obj_score is not None:
            save_dict["object_score_logits"] = obj_score
        iou_score = _cat(rank_outs, "iou_score", torch.float32)
        if iou_score is not None:
            save_dict["iou_score"] = iou_score

        fname = os.path.join(cond_states_dir, f"frame{frame_t:06d}.npz")
        np.savez_compressed(fname, **save_dict)
        saved_frames.append(frame_t)
        total_saved += 1

        # maskmem_pos_enc is a model constant (same for all frames/objects); save once
        if not maskmem_pos_enc_saved:
            pos_enc = rank_outs[0][1].get("maskmem_pos_enc")
            if pos_enc is not None:
                torch.save(pos_enc, os.path.join(cond_states_dir, "maskmem_pos_enc.pt"))
                maskmem_pos_enc_saved = True

    index = {
        "saved_at": datetime.now().isoformat(),
        "rank_sizes": [len(ids) for ids in obj_ids_per_rank],
        "frames": sorted(saved_frames),
    }
    with open(os.path.join(cond_states_dir, "index.json"), "w") as f:
        json.dump(index, f, indent=2)

    print(f"Saved {total_saved} cond frame states → {cond_states_dir}")
    return total_saved


def restore_cond_frame_states(sam3_model, session_id: str, concept_dir: str, device: str) -> int:
    """
    Inject saved cond frame states into a fresh session's tracker states.

    Call AFTER add_prompt() (text/box detection creates tracker states) but BEFORE
    propagate_in_video().  Overwrites the freshly-computed cond frame maskmem with
    the saved round-1 values, giving the tracker the prior session's anchors.

    Matching: the saved total object count must equal the current session's total
    object count (same text prompt on same video → deterministic detection order →
    same total).  The combined batch is then split back across the current
    session's ranks using "rank_sizes" from the index, in order.  Mismatched totals
    are skipped with a warning.

    Returns number of cond frame files injected.
    """
    cond_states_dir = os.path.join(concept_dir, "cond_states")
    index_path = os.path.join(cond_states_dir, "index.json")

    if not os.path.exists(index_path):
        print(f"restore_cond_frame_states: no saved states at {cond_states_dir}, skipping.")
        return 0

    with open(index_path) as f:
        index = json.load(f)

    inference_state = sam3_model._all_inference_states[session_id]["state"]
    tracker_states = inference_state.get("tracker_inference_states", [])

    if not tracker_states:
        print("restore_cond_frame_states: no tracker states in session. Call add_prompt() first.")
        return 0

    # Load maskmem_pos_enc (model constant, same values for all frames and objects)
    maskmem_pos_enc = None
    pos_enc_path = os.path.join(cond_states_dir, "maskmem_pos_enc.pt")
    if os.path.exists(pos_enc_path):
        maskmem_pos_enc = torch.load(pos_enc_path, map_location=device, weights_only=False)

    cur_rank_sizes = [len(ts.get("obj_ids", [])) for ts in tracker_states]
    total_cur = sum(cur_rank_sizes)
    saved_frames = index.get("frames", [])
    offload_to_cpu = inference_state.get("offload_state_to_cpu", False)

    restored_count = 0

    for frame_t in saved_frames:
        fpath = os.path.join(cond_states_dir, f"frame{frame_t:06d}.npz")
        if not os.path.exists(fpath):
            print(f"  Missing: {fpath}")
            continue

        data = np.load(fpath, allow_pickle=False)
        b = int(data["obj_ids"].shape[0]) if "obj_ids" in data else total_cur
        if b != total_cur:
            print(f"  restore frame {frame_t}: obj count mismatch "
                  f"(saved={b}, current={total_cur}), skipping.")
            continue

        def _load(key, dtype):
            if key not in data:
                return None
            t = torch.from_numpy(data[key]).to(dtype)
            return t.cpu() if (offload_to_cpu and key == "maskmem_features") else t.to(device)

        iou_score = _load("iou_score", torch.float32)
        if iou_score is None:
            # Not saved in older format; default to zeros (shape [B, 1])
            iou_score = torch.zeros((b, 1), dtype=torch.float32, device=device)
        out = {
            "maskmem_features": _load("maskmem_features", torch.bfloat16),
            "obj_ptr":          _load("obj_ptr",          torch.float32),
            "pred_masks":       _load("pred_masks",       torch.bfloat16),
            "object_score_logits": _load("object_score_logits", torch.float32),
            "iou_score":        iou_score,
            "maskmem_pos_enc":  maskmem_pos_enc,
        }

        # Split the combined batch back across the current session's ranks.
        offset = 0
        for rank, tracker_state in enumerate(tracker_states):
            n = cur_rank_sizes[rank]
            if n == 0:
                continue
            combined_cond = tracker_state["output_dict"]["cond_frame_outputs"]
            consolidated_inds = tracker_state["consolidated_frame_inds"]["cond_frame_outputs"]
            cond_per_obj = tracker_state.get("output_dict_per_obj", {})

            rank_out = {}
            for key, val in out.items():
                if val is not None and torch.is_tensor(val) and val.shape[0] == b:
                    rank_out[key] = val[offset : offset + n]
                else:
                    rank_out[key] = val
            combined_cond[frame_t] = rank_out
            consolidated_inds.add(frame_t)

            # Update per-object slices (each B=1 view into the rank's batch)
            for obj_idx in range(n):
                if obj_idx not in cond_per_obj:
                    continue
                obj_out = {"maskmem_pos_enc": maskmem_pos_enc}
                for key in ("maskmem_features", "obj_ptr", "pred_masks", "object_score_logits", "iou_score"):
                    val = rank_out.get(key)
                    if val is not None and torch.is_tensor(val) and val.shape[0] == n:
                        obj_out[key] = val[obj_idx : obj_idx + 1]
                    else:
                        obj_out[key] = val
                cond_per_obj[obj_idx]["cond_frame_outputs"][frame_t] = obj_out

            offset += n

        restored_count += 1

    print(f"Restored {restored_count} cond frame states from {cond_states_dir}")
    return restored_count


def process_concept_detection(
    sam3_model,
    project: SAM3Project,
    concept: SAM3Concept,
    device: str = "cuda:0",
    progress_callback: Optional[Callable[[int, int], None]] = None,
    keep_session_alive: bool = False,
    save_cond_states: bool = False,
) -> SAM3Concept:
    """
    Run initial detection for a concept.

    Steps:
    1. Enable lazy loading (reuse sam_lazy_loader.py)
    2. Initialize inference state with CPU offloading (OOM mitigation)
    3. Add text prompt on detection frame
    4. Propagate (propagation_full = detector on every frame)
    5. Stream masks to disk incrementally (don't accumulate in memory)
    6. Save serializable inference state
    7. Create SAM3Instance objects with metadata
    8. Clear memory before next concept

    Args:
        sam3_model: SAM3 model instance
        project: SAM3 project
        concept: Concept to process
        device: Device to use
        progress_callback: Optional callback(frame_idx, num_frames)
        keep_session_alive: If True, do NOT close the session after processing.
            Caller is responsible for closing it. On success, session_id is stored
            in concept._live_session_id for retrieval.
        save_cond_states: If True, save maskmem cond frame states to disk for a
            future --restore-cond refinement round. Off by default since these
            states are only consumed by that opt-in path and can be sizeable.

    Returns:
        Updated concept with instances populated
    """
    from sam_lazy_loader import enable_lazy_loading

    # 1. Enable lazy loading
    print("Enabling lazy loading for SAM3...")
    enable_lazy_loading(cache_size=20, enable_sam3=True)

    # 2. Extract frames (reuse if already present)
    frames_dir = project.get_frames_dir()

    if not os.path.exists(frames_dir):
        print(f"Extracting frames to {frames_dir}...")
        extract_frames_from_video(project.video_path, frames_dir)
    else:
        print(f"Reusing existing frames in {frames_dir}")

    # 3. Start SAM3 session (SAM3 uses session-based API)
    print(f"Starting SAM3 session for concept '{concept.name}'...")
    concept.status = ConceptStatus.PROCESSING

    # Create unique session ID for this concept
    import uuid
    session_id = str(uuid.uuid4())

    # Apply per-concept instance cap. SAM3 stores max_num_objects on the model object;
    # -1 in our convention means no limit, which SAM3 represents internally as 10000.
    _inner_model = getattr(sam3_model, "model", sam3_model)
    _prev_max_num_objects = getattr(_inner_model, "max_num_objects", 10000)
    _cap = getattr(concept, "max_instances", -1)
    _inner_model.max_num_objects = _cap if _cap > 0 else 10000
    if _cap > 0:
        print(f"Instance cap for '{concept.name}': {_cap}")

    # offload_state_to_cpu: moves maskmem_features to CPU after each frame so GPU doesn't
    # accumulate the full video's worth of 648 KB/frame/instance tensors.
    sam3_model.start_session(
        resource_path=frames_dir,
        session_id=session_id,
        offload_state_to_cpu=True,
    )
    print(f"Session started: {session_id}")

    # Encode/write masks in a background thread so the GPU propagation loop
    # is not stalled on PNG/NPZ encoding and disk I/O each frame.
    from sam3_utils import AsyncMaskWriter
    mask_writer = AsyncMaskWriter(mask_format=project.mask_format)

    try:
        # 4. Add text prompt
        # The frame_idx here is just for showing initial results on that frame.
        # Text prompts apply to all frames, and propagation will scan the entire video.
        print(f"Adding text prompt: '{concept.text_prompt}' (initial results on frame {concept.detection_frame})")
        sam3_model.add_prompt(
            session_id=session_id,
            frame_idx=concept.detection_frame,
            text=concept.text_prompt
        )

        # 5. Propagate and stream masks to disk
        print("Propagating masks through video...")
        instance_masks = defaultdict(dict)  # instance_id -> {frame_idx -> mask_path}
        pixel_counts_by_obj: dict = defaultdict(dict)  # obj_id -> {frame_idx -> pixel_count}
        frame_count = 0
        mask_count = 0

        # Hoist per-loop constants: concept dir and total frame size for pixel-ratio.
        concept_dir = project.get_concept_dir(concept.name)
        _frame_w, _frame_h = project.frame_dimensions
        total_pixels = _frame_w * _frame_h
        output_dirs: dict = {}  # obj_id_int -> output_dir (cached per unique object)

        # Inner inference_state for cache eviction.
        # SAM3 accumulates full-resolution bool masks in cached_frame_outputs for every
        # yielded frame — ~2MB/obj/frame at 1080p.  Once masks are saved to disk the
        # entry is no longer needed (forward-only propagation never looks backwards).
        inner_state = sam3_model._all_inference_states[session_id]["state"]

        # Scan entire video forward from frame 0.
        # "both" with start_frame_idx=0 was a no-op for backward (range(0,-1,-1)=[0]),
        # so "forward" is correct and equivalent while being explicit.
        # Future improvement: keep the session alive across UI interactions so users
        # can add correction clicks on multiple frames and re-propagate once with all
        # cond-frame anchors — SAM3 attends to all cond frames simultaneously (no cap
        # with max_cond_frames_in_attn=-1), so multi-frame annotation before a single
        # propagate is both supported and more efficient than repeated single-frame runs.
        for out in sam3_model.propagate_in_video(
            session_id=session_id,
            propagation_direction="forward",
            start_frame_idx=0,
            max_frame_num_to_track=None
        ):
            frame_idx = out["frame_index"]
            outputs = out["outputs"]
            frame_count += 1

            # Skip frames with no outputs
            if outputs is None:
                # Still evict from cache even when output is empty
                inner_state["cached_frame_outputs"].pop(frame_idx, None)
                continue

            # SAM3 outputs: out_obj_ids (array), out_binary_masks (array)
            # These are parallel arrays, not a dictionary
            out_obj_ids = outputs.get("out_obj_ids", [])
            out_binary_masks = outputs.get("out_binary_masks", [])

            # Debug: show what we got on first few frames
            if frame_count <= 3:
                print(f"  Frame {frame_idx}: {len(out_obj_ids)} objects - {list(out_obj_ids)}")

            for i, (obj_id, mask) in enumerate(zip(out_obj_ids, out_binary_masks)):
                mask_count += 1
                # Convert obj_id to Python int
                obj_id_int = int(obj_id)

                # Convert mask to numpy if it's a tensor
                # (bool→uint8 conversion happens in the writer thread)
                if torch.is_tensor(mask):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = mask

                # Cache output_dir per unique obj_id (os.path.join called once per object,
                # not once per object per frame).
                if obj_id_int not in output_dirs:
                    output_dirs[obj_id_int] = os.path.join(
                        concept_dir, "instances", str(obj_id_int), "masks"
                    )
                output_dir = output_dirs[obj_id_int]

                mask_path = mask_writer.submit(mask_np, output_dir, frame_idx)
                instance_masks[obj_id_int][frame_idx] = mask_path
                pixel_counts_by_obj[obj_id_int][frame_idx] = int((mask_np > 0).sum())

            # Evict this frame's full-res masks from SAM3's internal cache now that
            # they're on disk — forward propagation never needs to revisit past frames.
            inner_state["cached_frame_outputs"].pop(frame_idx, None)

            # Prune stale non_cond_frame_outputs inside each tracker state.
            # (SAM3 stores maskmem tensors per-GPU-rank, not at top-level inner_state.)
            _prune_non_cond_outputs(inner_state, frame_idx)

            # Clear GPU cache periodically
            if frame_idx % 100 == 0:
                torch.cuda.empty_cache()

            # Progress callback
            if progress_callback:
                progress_callback(frame_idx, project.num_frames)

        # Drain the writer before reading mask dirs below (compute_continuous_periods);
        # re-raises any write error from the background thread.
        mask_writer.close()

        print(f"\nPropagation summary: {frame_count} frames, {mask_count} masks total")

        # 6. Save cond frame maskmem states for future refinement rounds (must be before close_session)
        if save_cond_states:
            print("Saving cond frame states for future refinement...")
            concept_dir = project.get_concept_dir(concept.name)
            save_cond_frame_states(sam3_model, session_id, concept_dir)

        # 7. Get inference state for serialization
        print("Saving inference state...")
        inference_state = sam3_model._all_inference_states[session_id]["state"]
        inference_state["video_path"] = project.video_path
        save_concept_state(concept, inference_state, project.project_dir,
                           project.get_frames_dir())

        # 7. Create SAM3Instance objects with distinct colors
        print(f"Creating instance metadata ({len(instance_masks)} instances detected)...")
        instances = []
        num_instances = len(instance_masks)

        from sam3_utils import generate_instance_color

        for idx, obj_id in enumerate(sorted(instance_masks.keys())):
            # Convert numpy int to Python int
            obj_id_int = int(obj_id)

            # Score = fraction of video frames where this instance was detected
            score = len(instance_masks[obj_id]) / max(frame_count, 1)

            # Generate distinct color for this instance
            # Use full color spectrum to maximize visual distinction
            instance_color = generate_instance_color(idx, num_instances)

            mask_dir = os.path.join(
                project.get_concept_dir(concept.name),
                "instances", str(obj_id_int), "masks"
            )
            periods = compute_continuous_periods(mask_dir)
            instance = SAM3Instance(
                sam3_obj_id=obj_id_int,
                user_name=f"{concept.name}_{obj_id_int}",  # Default name
                concept_name=concept.name,
                color_rgb=instance_color,  # Assign distinct color
                score=score,
                first_detection_frame=min(instance_masks[obj_id].keys()),
                last_detection_frame=max(instance_masks[obj_id].keys()),
                num_frames_with_mask=len(instance_masks[obj_id]),
                continuous_periods=periods,
                period_peaks=compute_period_peaks(
                    periods, pixel_counts_by_obj.get(obj_id_int, {}), total_pixels),
            )
            instances.append(instance)

        concept.instances = instances
        concept.status = ConceptStatus.COMPLETED

        print(f"Concept '{concept.name}' processing complete: {len(instances)} instances detected")

    except Exception as e:
        concept.status = ConceptStatus.ERROR
        print(f"Error processing concept '{concept.name}': {e}")
        raise

    finally:
        # Stop the writer thread if propagation failed mid-loop (idempotent;
        # errors already surfaced via the close() call in the happy path).
        try:
            mask_writer.close()
        except Exception:
            pass
        # 8. Close session and clear memory
        if keep_session_alive and concept.status == ConceptStatus.COMPLETED:
            # Caller owns this session; store session_id for retrieval
            concept._live_session_id = session_id
        else:
            try:
                sam3_model.close_session(session_id=session_id)
            except Exception:
                pass
        _inner_model.max_num_objects = _prev_max_num_objects
        torch.cuda.empty_cache()

    return concept



# TODO: dead code — verify no callers remain, then delete.
# sam3_ui.py imports this name but never calls it; the UI uses
# online_replay_concept_refinements (online path) and replay_concept_refinements
# (offline path) instead.  This single-instance version also has a latent bug:
# it passes both text= and points= to add_prompt in the same call, which the
# SAM3 session wrapper routes through add_tracker_new_points (points-only branch)
# while silently ignoring the text prompt — so text detection never runs.
def add_refinement_points(
    concept: SAM3Concept,
    instance: SAM3Instance,
    frame_idx: int,
    points: List[Tuple[float, float, bool]],  # (x, y, is_positive)
    resource_path: str,
    orig_width: int,
    orig_height: int,
    num_frames: int,
    sam3_model,
    project_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    restore_prior_cond_states: bool = False,
    device: str = "cuda:0",
    mask_format: str = "png",
):
    """
    Add point refinements to an instance and re-propagate.

    Starts a fresh session, re-adds the text prompt (which resets SAM3 state and
    runs detection), then optionally restores saved cond frame maskmem from the
    prior round so propagation benefits from the previous session's anchors.

    Args:
        concept: Concept containing the instance
        instance: Instance to refine
        frame_idx: Frame index where the user placed correction points
        points: List of (x, y, is_positive) tuples
        resource_path: Path to extracted frames directory
        orig_width: Video frame width in pixels
        orig_height: Video frame height in pixels
        num_frames: Total number of frames in video
        sam3_model: SAM3 model instance
        project_dir: Project directory
        progress_callback: Optional callback(frame_idx, num_frames)
        restore_prior_cond_states: If True, inject saved round-1 cond frame maskmem
            after text detection (before propagation) to reuse prior anchors.
        device: CUDA device string for tensor placement during restore
    """
    pts_normalized = [[x / orig_width, y / orig_height] for x, y, _ in points]
    pt_labels = [1 if is_positive else 0 for _, _, is_positive in points]

    # Start a new session (matches detection flow API)
    import uuid
    session_id = str(uuid.uuid4())
    print(f"Starting refinement session for instance {instance.sam3_obj_id} at frame {frame_idx}")
    sam3_model.start_session(
        resource_path=resource_path,
        session_id=session_id,
        offload_state_to_cpu=True,
    )

    # Re-add text prompt + refinement points.
    # add_prompt with text calls reset_state internally, then runs detection and
    # creates tracker_inference_states — required before restore_cond_frame_states.
    sam3_model.add_prompt(
        session_id=session_id,
        frame_idx=frame_idx,
        text=concept.text_prompt,
        points=pts_normalized,
        point_labels=pt_labels,
    )

    # Optionally overwrite the freshly-computed detection cond frame with the
    # saved round-1 maskmem, giving propagation the prior session's anchors.
    if restore_prior_cond_states:
        concept_dir = os.path.join(project_dir, "concepts", concept.name)
        n = restore_cond_frame_states(sam3_model, session_id, concept_dir, device)
        if n > 0:
            print(f"Using {n} restored cond frames from prior round.")

    # Re-propagate scanning entire video from frame 0
    print("Re-propagating with refinements...")
    from sam3_utils import AsyncMaskWriter
    mask_writer = AsyncMaskWriter(mask_format=mask_format)
    # output_dir is constant for this single-instance path — hoist before loop.
    output_dir = os.path.join(
        project_dir, "concepts", concept.name,
        "instances", str(instance.sam3_obj_id), "masks"
    )
    try:
        for out in sam3_model.propagate_in_video(
            session_id=session_id,
            start_frame_idx=0,
            propagation_direction="forward"
        ):
            out_frame_idx = out["frame_index"]
            outputs = out["outputs"]

            if outputs is None:
                continue

            out_obj_ids = outputs.get("out_obj_ids", [])
            out_binary_masks = outputs.get("out_binary_masks", [])

            for obj_id, mask in zip(out_obj_ids, out_binary_masks):
                if obj_id == instance.sam3_obj_id:
                    # Convert mask to numpy if it's a tensor
                    if torch.is_tensor(mask):
                        mask_np = mask.cpu().numpy()
                    else:
                        mask_np = mask

                    mask_writer.submit(mask_np, output_dir, out_frame_idx)

            _prune_non_cond_outputs(
                sam3_model._all_inference_states[session_id]["state"], out_frame_idx,
            )
            # Progress callback
            if progress_callback:
                progress_callback(out_frame_idx, num_frames)

            # Clear GPU cache periodically
            if out_frame_idx % 100 == 0:
                torch.cuda.empty_cache()
    finally:
        # Drain queued masks; re-raises any write error from the background thread.
        mask_writer.close()

    # Save refinement history to refinements.json
    refinements_path = os.path.join(
        project_dir, "concepts", concept.name,
        "instances", str(instance.sam3_obj_id), "refinements.json"
    )
    os.makedirs(os.path.dirname(refinements_path), exist_ok=True)

    import json
    from datetime import datetime

    # Load existing refinements
    refinements = []
    if os.path.exists(refinements_path):
        with open(refinements_path, 'r') as f:
            data = json.load(f)
            refinements = data.get("refinements", [])

    # Add new refinement
    refinements.append({
        "timestamp": datetime.now().isoformat(),
        "frame_idx": frame_idx,
        "points": [{"x": x, "y": y, "is_positive": is_pos} for x, y, is_pos in points],
        "propagated": True
    })

    # Save refinements
    with open(refinements_path, 'w') as f:
        json.dump({"refinements": refinements}, f, indent=2)

    print(f"Refinement complete. History saved to {refinements_path}")


def replay_concept_refinements(
    concept: SAM3Concept,
    instances_with_pending: List[Tuple["SAM3Instance", List[dict]]],
    resource_path: str,
    orig_width: int,
    orig_height: int,
    num_frames: int,
    sam3_model,
    project_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    device: str = "cuda:0",
    restore_cond_states: bool = False,
    save_cond_states: bool = False,
    mask_format: str = "png",
) -> bool:
    """
    Replay saved refinements for ALL instances of a concept in ONE session.

    All instances are tracked simultaneously so SAM3's non-overlapping constraints
    are applied across the full concept — the same way initial detection works.
    Running each instance in its own isolated session would break non-overlapping.

    After propagation, masks for EVERY detected instance are written to disk (not
    just the ones with new annotations), and saved cond states are updated so the
    next refinement round can restore the latest anchors.

    Args:
        concept: The concept whose instances are being refined.
        instances_with_pending: List of (instance, pending_entries) — every instance
            that has at least one entry with "propagated": false.  This acts as the
            TRIGGER only; the replay itself re-adds ALL saved annotations (historical
            and pending) of every non-deleted instance in the concept, so earlier
            rounds' corrections stay anchored without needing --restore-cond.
        resource_path: Path to extracted frames directory.
        orig_width / orig_height: Video dimensions in pixels.
        num_frames: Total frame count.
        sam3_model: Loaded SAM3 model.
        project_dir: Project root directory.
        progress_callback: Optional callback(frame_idx, num_frames).
        device: CUDA device string for tensor placement.

    Returns True if propagation ran; False if no pending entries found.
    """
    import uuid

    if not instances_with_pending:
        return False

    from sam_lazy_loader import enable_lazy_loading
    enable_lazy_loading(cache_size=20, enable_sam3=True)

    # Load refinements.json for EVERY non-deleted instance of the concept — not just
    # the ones with pending entries.  All historical clicks are replayed each round so
    # corrections from earlier rounds stay anchored in the fresh session (without this,
    # a round-2 propagation would silently undo round-1 fixes unless --restore-cond
    # was used).  Pending entries only act as the trigger; the replay set is everything.
    refinements_by_instance: dict = {}  # obj_id -> (path, all_entries, inst)
    for inst in concept.instances:
        if inst.deleted:
            continue
        rpath = os.path.join(
            project_dir, "concepts", concept.name,
            "instances", str(inst.sam3_obj_id), "refinements.json"
        )
        if not os.path.exists(rpath):
            continue
        with open(rpath) as f:
            all_entries = json.load(f).get("refinements", [])
        if all_entries:
            refinements_by_instance[inst.sam3_obj_id] = (rpath, all_entries, inst)

    # Build (obj_id, frame_idx) -> [points] from ALL entries in file order with
    # last-entry-wins per frame: each UI save writes a frame's complete point set as
    # one entry, so the newest entry supersedes older ones for that frame (and an
    # entry with no points clears the frame).  Merging entries instead would
    # resurrect points the user deleted in the UI.
    obj_frame_points: dict = defaultdict(dict)
    for obj_id, (rpath, all_entries, inst) in refinements_by_instance.items():
        for r in all_entries:
            frame_idx = r.get("frame_idx")
            if frame_idx is None:
                continue
            pts = [(p["x"], p["y"], p["is_positive"]) for p in r.get("points", [])]
            if pts:
                obj_frame_points[obj_id][frame_idx] = pts
            else:
                obj_frame_points[obj_id].pop(frame_idx, None)

    n_pending = sum(len(p) for _, p in instances_with_pending)
    total_frames_replayed = sum(len(v) for v in obj_frame_points.values())
    print(f"Replaying ALL saved annotations for concept '{concept.name}': "
          f"{total_frames_replayed} annotated frame(s) across "
          f"{len(obj_frame_points)} instance(s) "
          f"({n_pending} new pending entr{'y' if n_pending == 1 else 'ies'}) "
          f"in a single joint session.")

    session_id = str(uuid.uuid4())
    sam3_model.start_session(
        resource_path=resource_path,
        session_id=session_id,
        offload_state_to_cpu=True,
    )

    try:
        # Re-run text detection anchored to the same frame used during initial detection.
        # frame_idx only determines where initial results are displayed; the text prompt
        # applies to all frames.  Using concept.detection_frame (not hard-coded 0) keeps
        # the API call consistent with process_concept_detection, which matters if the
        # user explicitly set a non-zero detection_frame in their concepts JSON.
        sam3_model.add_prompt(
            session_id=session_id,
            frame_idx=concept.detection_frame,
            text=concept.text_prompt,
        )

        # Remove deleted instances immediately after detection — text-based detection is
        # deterministic so obj_ids match the stored sam3_obj_id values exactly.
        # Evicting them here prevents propagation from ever tracking or writing their masks.
        deleted_obj_ids = {inst.sam3_obj_id for inst in concept.instances if inst.deleted}
        for del_id in deleted_obj_ids:
            try:
                sam3_model.remove_object(session_id=session_id, obj_id=del_id)
                print(f"  Removed deleted instance obj_id={del_id} from session.")
            except Exception as e:
                print(f"  Warning: could not remove obj_id={del_id}: {e}")

        concept_dir = os.path.join(project_dir, "concepts", concept.name)
        if restore_cond_states:
            n = restore_cond_frame_states(sam3_model, session_id, concept_dir, device)
            if n > 0:
                print(f"  Restored {n} cond frame(s) from prior round.")
        else:
            print("  Skipping cond state restore (default); propagation uses text detection + correction points only.")

        # Raw per-frame backbone features (image + FPN) live in feature_cache, keyed by
        # frame_idx. run_backbone_and_detection() only evicts the *adjacent* frame_idx
        # (sam3_video_base.py), which assumes sequential propagation order. The replay
        # loops below visit frames out of order (each instance's own annotated frames,
        # sorted per-instance but not globally), so that eviction never matches and
        # entries pile up — one full backbone feature map per annotated frame, for the
        # entire replay set, all resident at once. propagate_in_video() recomputes this
        # cache unconditionally for every frame it visits (sam3_video_inference.py
        # _prepare_backbone_feats), so nothing here is reused once propagation starts —
        # evicting immediately after use is free and avoids the pileup.
        inner_state = sam3_model._all_inference_states[session_id]["state"]

        # Initialize manually-added instances (never text-detected) via ALL their pending
        # point sets in frame order.  The first add_prompt(obj_id=X, points=...) call
        # creates the tracker slot; subsequent calls add further cond anchors.
        # These are handled here in full so the normal refinement loop below can skip them.
        manually_added_ids = {inst.sam3_obj_id for inst in concept.instances
                              if getattr(inst, 'manually_added', False) and not inst.deleted}
        for obj_id in manually_added_ids:
            if obj_id not in obj_frame_points:
                continue  # no pending points → nothing to initialize
            for frame_idx in sorted(obj_frame_points[obj_id].keys()):
                pts = obj_frame_points[obj_id][frame_idx]
                pts_normalized = [[x / orig_width, y / orig_height] for x, y, _ in pts]
                pt_labels = [1 if is_positive else 0 for _, _, is_positive in pts]
                sam3_model.add_prompt(
                    session_id=session_id,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=pts_normalized,
                    point_labels=pt_labels,
                )
                inner_state["feature_cache"].pop(frame_idx, None)
            n_frames = len(obj_frame_points[obj_id])
            print(f"  Initialized manually-added instance obj_id={obj_id} "
                  f"via {n_frames} annotated frame(s).")

        # Add refinement prompts for text-detected instances on each annotated frame.
        # All calls use obj_id= (no text=) so reset_state is NOT triggered.
        # Manually-added instances were fully handled above — skip them here.
        for obj_id, frame_points in obj_frame_points.items():
            if obj_id in manually_added_ids:
                continue
            for frame_idx in sorted(frame_points.keys()):
                pts = frame_points[frame_idx]
                pts_normalized = [[x / orig_width, y / orig_height] for x, y, _ in pts]
                pt_labels = [1 if is_positive else 0 for _, _, is_positive in pts]
                sam3_model.add_prompt(
                    session_id=session_id,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    points=pts_normalized,
                    point_labels=pt_labels,
                )
                inner_state["feature_cache"].pop(frame_idx, None)

        # Single propagation: all instances in one batched forward pass.
        # Non-overlapping constraints are enforced across all instances jointly.
        # Write masks for ALL detected instances (not just the ones with new annotations).
        from sam3_utils import AsyncMaskWriter
        mask_writer = AsyncMaskWriter(mask_format=mask_format)
        pixel_counts_by_obj: dict = defaultdict(dict)
        total_pixels = orig_width * orig_height
        instances_root = os.path.join(project_dir, "concepts", concept.name, "instances")
        output_dirs: dict = {}  # obj_id_int -> output_dir (cached per unique object)
        try:
            for out in sam3_model.propagate_in_video(
                session_id=session_id,
                start_frame_idx=0,
                propagation_direction="forward",
            ):
                out_frame_idx = out["frame_index"]
                outputs = out["outputs"]
                if outputs is None:
                    inner_state["cached_frame_outputs"].pop(out_frame_idx, None)
                    continue

                out_obj_ids = outputs.get("out_obj_ids", [])
                out_binary_masks = outputs.get("out_binary_masks", [])

                for obj_id, mask in zip(out_obj_ids, out_binary_masks):
                    mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
                    if mask_np.ndim == 3 and mask_np.shape[0] == 1:
                        mask_np = mask_np[0]
                    obj_id_int = int(obj_id)
                    if obj_id_int not in output_dirs:
                        output_dirs[obj_id_int] = os.path.join(
                            instances_root, str(obj_id_int), "masks"
                        )
                    mask_writer.submit(mask_np, output_dirs[obj_id_int], out_frame_idx)
                    pixel_counts_by_obj[obj_id_int][out_frame_idx] = int((mask_np > 0).sum())

                # Evict after saving — forward propagation never revisits past frames
                inner_state["cached_frame_outputs"].pop(out_frame_idx, None)
                _prune_non_cond_outputs(inner_state, out_frame_idx)

                if progress_callback:
                    progress_callback(out_frame_idx, num_frames)
                if out_frame_idx % 100 == 0:
                    torch.cuda.empty_cache()
        finally:
            # Drain before the deleted-instance wipe below touches mask dirs;
            # re-raises any write error from the background thread.
            mask_writer.close()

        # Wipe mask files for deleted instances — they were excluded from propagation so
        # their on-disk masks are now stale and should not appear in any future compositing.
        import shutil
        for inst in concept.instances:
            if not inst.deleted:
                continue
            mask_dir = os.path.join(
                project_dir, "concepts", concept.name,
                "instances", str(inst.sam3_obj_id), "masks"
            )
            if os.path.isdir(mask_dir):
                shutil.rmtree(mask_dir)
                os.makedirs(mask_dir, exist_ok=True)
                print(f"  Cleared masks for deleted instance obj_id={inst.sam3_obj_id}.")

        # Report instance directories on disk that weren't covered by this refinement run.
        # This happens when a prior run detected more objects than this run (e.g., the text
        # prompt missed them, or they weren't visible at the new prompt frame).
        # We do NOT auto-delete — stale masks may still be valid from a prior round.
        instances_root = os.path.join(project_dir, "concepts", concept.name, "instances")
        if os.path.isdir(instances_root):
            on_disk = set()
            for d in os.listdir(instances_root):
                try:
                    on_disk.add(int(d))
                except ValueError:
                    pass
            # Manually-added instances that had no pending points this run were never in
            # the session, so their existing masks are valid — don't flag them as stale.
            manually_added_no_pending = {
                inst.sam3_obj_id for inst in concept.instances
                if getattr(inst, 'manually_added', False)
                and not inst.deleted
                and inst.sam3_obj_id not in obj_frame_points
            }
            stale = on_disk - set(pixel_counts_by_obj.keys()) - deleted_obj_ids - manually_added_no_pending
            if stale:
                print(f"  NOTE: {len(stale)} instance dir(s) on disk not updated by this "
                      f"refinement run: {sorted(stale)}")
                print(f"  Their existing masks are preserved. Delete manually if unwanted.")

        # Update cond states for next refinement round (multi-object tensor, all instances).
        if save_cond_states:
            save_cond_frame_states(sam3_model, session_id, concept_dir)

        # Recompute periods and frame-count metadata for every instance from actual mask files.
        # pixel_counts_by_obj has fresh counts for instances tracked in this run; stale instances
        # (not output by SAM3) still get their periods refreshed from disk, just with empty pixel
        # counts (so period_peaks may be incomplete for them — acceptable).
        for inst in concept.instances:
            if inst.deleted:
                continue
            mask_dir = os.path.join(
                instances_root, str(inst.sam3_obj_id), "masks"
            )
            periods = compute_continuous_periods(mask_dir)
            inst.continuous_periods = periods
            inst.period_peaks = compute_period_peaks(
                periods, pixel_counts_by_obj.get(inst.sam3_obj_id, {}), total_pixels)
            inst.num_frames_with_mask = sum(e - s + 1 for s, e in periods)
            if periods:
                inst.first_detection_frame = periods[0][0]
                inst.last_detection_frame = periods[-1][1]

    finally:
        try:
            sam3_model.close_session(session_id=session_id)
        except Exception:
            pass

    # Mark all pending entries as propagated
    for obj_id, (rpath, all_entries, inst) in refinements_by_instance.items():
        for r in all_entries:
            if not r.get("propagated", True):
                r["propagated"] = True
        with open(rpath, "w") as f:
            json.dump({"refinements": all_entries}, f, indent=2)

    print(f"  Concept-level replay complete for '{concept.name}'.")
    return True


def online_add_refinement_points(
    concept: SAM3Concept,
    instance: SAM3Instance,
    frame_idx: int,
    points: List[Tuple[float, float, bool]],  # (x, y, is_positive) in pixel coords
    orig_width: int,
    orig_height: int,
    num_frames: int,
    sam3_model,
    session_id: str,
    project_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    save_cond_states: bool = False,
    mask_format: str = "png",
):
    """
    Add correction points using a live (already-open) SAM3 session.

    Unlike add_refinement_points, this does NOT start a new session or re-run
    text detection.  It calls add_prompt(points=..., obj_id=X) which routes to
    add_tracker_new_points — no reset, no detection, just adds the clicked frame
    as a new cond anchor directly into the existing tracker state.

    Re-propagation then updates masks for ALL instances in the concept since the
    tracker processes all objects simultaneously.

    Args:
        concept: Concept containing the instance
        instance: Instance to refine (point anchor is attached to this obj_id)
        frame_idx: Frame where correction points were placed
        points: (x, y, is_positive) tuples in pixel coordinates
        orig_width/orig_height: Video frame dimensions (for normalization)
        num_frames: Total frames (for progress reporting)
        sam3_model: SAM3 model (session must be alive)
        session_id: Live session ID returned from process_concept_detection(keep_session_alive=True)
        project_dir: Project directory
        progress_callback: Optional callback(frame_idx, num_frames)
    """
    # SAM3 expects normalized [0,1] coordinates, shape [N, 2]
    pt_coords = torch.tensor(
        [[x / orig_width, y / orig_height] for x, y, _ in points],
        dtype=torch.float32
    )
    pt_labels = torch.tensor(
        [1 if is_pos else 0 for _, _, is_pos in points],
        dtype=torch.int32
    )

    print(f"Online refinement: adding {len(points)} point(s) at frame {frame_idx} "
          f"for obj {instance.sam3_obj_id} (session {session_id[:8]}...)")

    # Routes to add_tracker_new_points — does NOT call reset_state
    sam3_model.add_prompt(
        session_id=session_id,
        frame_idx=frame_idx,
        points=pt_coords,
        point_labels=pt_labels,
        obj_id=instance.sam3_obj_id
    )

    # Re-propagate with the new cond anchor included
    print("Re-propagating with new anchor...")
    from sam3_utils import AsyncMaskWriter
    mask_writer = AsyncMaskWriter(mask_format=mask_format)
    try:
        for out in sam3_model.propagate_in_video(
            session_id=session_id,
            start_frame_idx=0,
            propagation_direction="forward"
        ):
            out_frame_idx = out["frame_index"]
            outputs = out["outputs"]

            if outputs is None:
                continue

            out_obj_ids = outputs.get("out_obj_ids", [])
            out_binary_masks = outputs.get("out_binary_masks", [])

            # Update masks for ALL objects (propagation covers all simultaneously)
            for obj_id, mask in zip(out_obj_ids, out_binary_masks):
                obj_id_int = int(obj_id)
                if torch.is_tensor(mask):
                    mask_np = mask.cpu().numpy()
                else:
                    mask_np = mask

                output_dir = os.path.join(
                    project_dir, "concepts", concept.name,
                    "instances", str(obj_id_int), "masks"
                )
                mask_writer.submit(mask_np, output_dir, out_frame_idx)

            _prune_non_cond_outputs(
                sam3_model._all_inference_states[session_id]["state"], out_frame_idx,
            )
            if progress_callback:
                progress_callback(out_frame_idx, num_frames)
            if out_frame_idx % 100 == 0:
                torch.cuda.empty_cache()
    finally:
        # Drain queued masks; re-raises any write error from the background thread.
        mask_writer.close()

    # Save updated cond states for future offline rounds
    if save_cond_states:
        concept_dir = os.path.join(project_dir, "concepts", concept.name)
        save_cond_frame_states(sam3_model, session_id, concept_dir)

    # Save refinement history
    refinements_path = os.path.join(
        project_dir, "concepts", concept.name,
        "instances", str(instance.sam3_obj_id), "refinements.json"
    )
    os.makedirs(os.path.dirname(refinements_path), exist_ok=True)

    refinements = []
    if os.path.exists(refinements_path):
        with open(refinements_path) as f:
            refinements = json.load(f).get("refinements", [])

    refinements.append({
        "timestamp": datetime.now().isoformat(),
        "frame_idx": frame_idx,
        "points": [{"x": x, "y": y, "is_positive": is_pos} for x, y, is_pos in points],
        "propagated": True,
        "online": True,
    })

    with open(refinements_path, "w") as f:
        json.dump({"refinements": refinements}, f, indent=2)

    print(f"Online refinement complete. History saved to {refinements_path}")


def online_replay_concept_refinements(
    concept: SAM3Concept,
    instances_with_pending: List[Tuple["SAM3Instance", List[dict]]],
    orig_width: int,
    orig_height: int,
    num_frames: int,
    sam3_model,
    session_id: str,
    project_dir: str,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    save_cond_states: bool = False,
    mask_format: str = "png",
):
    """
    Apply pending refinements for multiple instances through a live session in one propagation pass.

    Unlike calling online_add_refinement_points per-instance (which re-propagates after
    each call and breaks non-overlapping constraints across instances), this function adds
    all correction anchors for all instances FIRST, then propagates ONCE.  SAM3's
    non-overlapping constraint is then enforced jointly — required for cases like swapping
    a region between two instances (positive on A + negative on B).

    Args:
        concept: The concept whose instances are being refined.
        instances_with_pending: List of (instance, [refinement_entry, ...]) where each
            entry is a dict with 'frame_idx' and 'points' ({'x', 'y', 'is_positive'}).
        orig_width / orig_height: Video frame dimensions for coordinate normalization.
        num_frames: Total frame count for progress reporting.
        sam3_model: SAM3 model (session must be alive).
        session_id: Live session ID.
        project_dir: Project root directory.
        progress_callback: Optional callback(frame_idx, num_frames).
    """
    if not instances_with_pending:
        return

    # Add correction anchors for ALL instances before propagating.
    # Each add_prompt(obj_id=X) call routes to add_tracker_new_points — no reset_state.
    for instance, pending_entries in instances_with_pending:
        for entry in pending_entries:
            frame_idx = entry["frame_idx"]
            raw_pts = entry.get("points", [])
            if not raw_pts:
                continue
            pt_coords = torch.tensor(
                [[p["x"] / orig_width, p["y"] / orig_height] for p in raw_pts],
                dtype=torch.float32,
            )
            pt_labels = torch.tensor(
                [1 if p["is_positive"] else 0 for p in raw_pts],
                dtype=torch.int32,
            )
            print(f"Online refinement: adding {len(raw_pts)} point(s) at frame {frame_idx} "
                  f"for obj {instance.sam3_obj_id}")
            sam3_model.add_prompt(
                session_id=session_id,
                frame_idx=frame_idx,
                points=pt_coords,
                point_labels=pt_labels,
                obj_id=instance.sam3_obj_id,
            )

    # Single propagation pass — non-overlapping applied across ALL instances simultaneously
    inst_names = [inst.user_name for inst, _ in instances_with_pending]
    print(f"Re-propagating with corrections for {len(instances_with_pending)} instance(s): {inst_names}")
    from sam3_utils import AsyncMaskWriter
    mask_writer = AsyncMaskWriter(mask_format=mask_format)
    pixel_counts_by_obj: dict = defaultdict(dict)
    total_pixels = orig_width * orig_height
    instances_root = os.path.join(project_dir, "concepts", concept.name, "instances")
    output_dirs: dict = {}  # obj_id_int -> output_dir (cached per unique object)
    try:
        for out in sam3_model.propagate_in_video(
            session_id=session_id,
            start_frame_idx=0,
            propagation_direction="forward",
        ):
            out_frame_idx = out["frame_index"]
            outputs = out["outputs"]
            if outputs is None:
                continue
            for obj_id, mask in zip(outputs.get("out_obj_ids", []), outputs.get("out_binary_masks", [])):
                obj_id_int = int(obj_id)
                mask_np = mask.cpu().numpy() if torch.is_tensor(mask) else np.asarray(mask)
                if obj_id_int not in output_dirs:
                    output_dirs[obj_id_int] = os.path.join(
                        instances_root, str(obj_id_int), "masks"
                    )
                mask_writer.submit(mask_np, output_dirs[obj_id_int], out_frame_idx)
                pixel_counts_by_obj[obj_id_int][out_frame_idx] = int((mask_np > 0).sum())
            _prune_non_cond_outputs(
                sam3_model._all_inference_states[session_id]["state"], out_frame_idx,
            )
            if progress_callback:
                progress_callback(out_frame_idx, num_frames)
            if out_frame_idx % 100 == 0:
                torch.cuda.empty_cache()
    finally:
        # Drain before compute_continuous_periods below reads the mask dirs;
        # re-raises any write error from the background thread.
        mask_writer.close()

    # Save updated cond states for the next refinement round
    if save_cond_states:
        concept_dir = os.path.join(project_dir, "concepts", concept.name)
        save_cond_frame_states(sam3_model, session_id, concept_dir)

    # Refresh metadata for ALL concept instances from actual mask files.
    # The propagation may have rewritten masks for instances beyond instances_with_pending,
    # so we scan the full concept rather than only the subset that had pending annotations.
    for inst in concept.instances:
        if inst.deleted:
            continue
        mask_dir = os.path.join(
            instances_root, str(inst.sam3_obj_id), "masks"
        )
        periods = compute_continuous_periods(mask_dir)
        inst.continuous_periods = periods
        inst.period_peaks = compute_period_peaks(
            periods, pixel_counts_by_obj.get(inst.sam3_obj_id, {}), total_pixels)
        inst.num_frames_with_mask = sum(e - s + 1 for s, e in periods)
        if periods:
            inst.first_detection_frame = periods[0][0]
            inst.last_detection_frame = periods[-1][1]

    # Mark all pending entries as propagated
    for instance, _ in instances_with_pending:
        refinements_path = os.path.join(
            project_dir, "concepts", concept.name,
            "instances", str(instance.sam3_obj_id), "refinements.json"
        )
        if not os.path.exists(refinements_path):
            continue
        with open(refinements_path) as f:
            data = json.load(f)
        for r in data.get("refinements", []):
            if not r.get("propagated", True):
                r["propagated"] = True
                r["online"] = True
        with open(refinements_path, "w") as f:
            json.dump(data, f, indent=2)

    print(f"Online concept-level refinement complete for '{concept.name}'.")


def load_sam3_model(model_name: str = "sam3", device: str = "cuda:0",
                    use_fa3: bool = False, max_cond_frames_in_attn: int = -1):
    """
    Load SAM3 model with proper configuration.

    Args:
        model_name: "sam3" (default) or "sam3.1" (Object Multiplex —
            bucketed joint multi-object tracking, much faster for many objects)
        device: Device to use
        use_fa3: Enable FlashAttention-3 fp8 kernels (SAM3.1 only). Requires a
            Hopper GPU (H100/H200) and the flash_attn_interface package;
            crashes on Ampere/Ada (e.g. L40S). Ignored for SAM3.
        max_cond_frames_in_attn: How many conditioning frames the tracker attends
            to per forward pass (default: -1 = no limit, matching SAM3's built-in
            default).  All user correction anchors are always attended to.
            Set to a small positive value (e.g. 4) only if attention compute is a
            bottleneck with many correction frames.

    Returns:
        SAM3 model instance (session-based predictor; same API for both versions)
    """
    import sys
    import os

    # Find project root and add SAM3 to path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sam3_path = os.path.join(project_root, "sam_models", "sam3")

    if not os.path.exists(sam3_path):
        raise RuntimeError(
            "SAM3 not found. Please run setup.py to install SAM3.\n"
            f"Expected path: {sam3_path}"
        )

    if sam3_path not in sys.path:
        sys.path.insert(0, sam3_path)

    use_multiplex = model_name in ("sam3.1", "3.1", "sam3_multiplex")
    if use_fa3 and not use_multiplex:
        print("WARNING: use_fa3 only applies to SAM3.1 (multiplex); ignored for SAM3.")

    # Load model
    ckpt_name = "sam3.1_multiplex.pt" if use_multiplex else "sam3.pt"
    checkpoint_path = os.path.join(sam3_path, "checkpoints", ckpt_name)
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            f"SAM3 checkpoint not found: {checkpoint_path}\n"
            "Please download checkpoint using setup.py"
        )

    print(f"Loading {'SAM3.1 multiplex' if use_multiplex else 'SAM3'} model "
          f"from {checkpoint_path} on {device}...")

    # Determine which GPU to use.
    # Sam3VideoPredictorMultiGPU accepts gpus_to_use=[int] to control placement.
    # Its __init__ calls torch.cuda.set_device(device) BEFORE super().__init__(),
    # so all internal .cuda() calls inside Sam3VideoPredictor land on the right GPU.
    if device.startswith("cuda"):
        # device may be "cuda:0" or a comma-separated list "cuda:0,cuda:1,cuda:2"
        parts = [d.strip() for d in device.split(",")]
        if any(":" in p for p in parts):
            gpus_to_use = [int(p.split(":")[1]) for p in parts if ":" in p]
        else:
            gpus_to_use = [torch.cuda.current_device()]
        gpu_id = gpus_to_use[0]
        # Set global current device so any module-level .cuda() calls also land here
        torch.cuda.set_device(gpu_id)
    else:
        # CPU: SAM3 requires CUDA; this will likely fail, but let SAM3 raise its own error
        print(f"WARNING: SAM3 requires CUDA. CPU mode ({device}) is not officially supported.")
        gpus_to_use = None

    if use_multiplex:
        from sam3.model_builder import build_sam3_multiplex_video_predictor

        # FlashAttention-3 runs fp8 kernels that only exist on Hopper (H100);
        # on Ampere/Ada (e.g. L40S) it must stay disabled so attention falls
        # back to PyTorch SDPA.
        # The builder hardcodes .cuda(), which lands on the device selected by
        # torch.cuda.set_device() above.
        if use_fa3:
            if gpus_to_use is None:
                raise RuntimeError("--use-fa3 requires a CUDA device.")
            cap = torch.cuda.get_device_capability(gpus_to_use[0])
            if cap[0] < 9:
                raise RuntimeError(
                    f"--use-fa3 requires a Hopper GPU (compute capability 9.0+), "
                    f"but {torch.cuda.get_device_name(gpus_to_use[0])} is sm_{cap[0]}{cap[1]}. "
                    "Run without --use-fa3 on this GPU."
                )
            print("FlashAttention-3 (fp8) enabled")
        predictor = build_sam3_multiplex_video_predictor(
            checkpoint_path=checkpoint_path,
            use_fa3=use_fa3,
            compile=False,
        )

        # Sam3BasePredictor.start_session always forwards offload_state_to_cpu to
        # init_state, but the multiplex init_state doesn't accept it (upstream
        # signature mismatch). Filter kwargs to what init_state actually takes.
        import inspect
        orig_init_state = predictor.model.init_state
        valid_params = set(inspect.signature(orig_init_state).parameters)

        def _filtered_init_state(**kwargs):
            return orig_init_state(
                **{k: v for k, v in kwargs.items() if k in valid_params}
            )

        predictor.model.init_state = _filtered_init_state
    else:
        from sam3.model_builder import build_sam3_video_predictor

        predictor = build_sam3_video_predictor(
            checkpoint_path=checkpoint_path,
            gpus_to_use=gpus_to_use,
        )

        # Offload per-frame maskmem_features to CPU so they don't accumulate on GPU.
        # predictor.model is Sam3VideoInferenceWithInstanceInteractivity — it does NOT have
        # this attribute.  The real flag lives on predictor.model.tracker (Sam3TrackerPredictor).
        # (Multiplex bounds GPU memory via _prune_non_cond_outputs instead; its bucketized
        # tracker states are not safe to offload through this flag.)
        tracker = getattr(getattr(predictor, "model", None), "tracker", None)
        if tracker is not None and hasattr(tracker, "offload_output_to_cpu_for_eval"):
            tracker.offload_output_to_cpu_for_eval = True
            print("SAM3: offload_output_to_cpu_for_eval=True on tracker")
        # Disable prev-mask-logit bias so user correction points segment from scratch
        # rather than refining the (possibly wrong) VG detector prediction at that frame.
        # With iter_use_prev_mask_pred=True (the SAM3 default), a correction at frame F
        # receives the VG mask as prev_sam_mask_logits, biasing the decoder toward the
        # wrong region even when the correction point is on a different object.
        # SAM3.1 multiplex already defaults to False, so this only affects SAM3.
        if tracker is not None and hasattr(tracker, "iter_use_prev_mask_pred"):
            tracker.iter_use_prev_mask_pred = False
            print("SAM3: iter_use_prev_mask_pred=False on tracker (correction points start fresh)")

    # Patch max_cond_frames_in_attn on the tracker (lives on SAM2Base which
    # Sam3TrackerPredictor inherits from). Works for both SAM3 and SAM3.1.
    _tracker = getattr(getattr(predictor, "model", None), "tracker", None)
    if _tracker is None:
        # SAM3.1 multiplex may expose tracker differently
        _tracker = getattr(predictor, "model", None)
    if _tracker is not None and hasattr(_tracker, "max_cond_frames_in_attn"):
        _tracker.max_cond_frames_in_attn = max_cond_frames_in_attn
        print(f"SAM3: max_cond_frames_in_attn={max_cond_frames_in_attn}")

    print(f"SAM3 model loaded on device: {device}")

    return predictor


def get_video_info(video_path: str) -> Tuple[int, Tuple[int, int], float]:
    """
    Get video metadata.

    Args:
        video_path: Path to video file

    Returns:
        (num_frames, (width, height), fps)
    """
    import cv2

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    cap.release()

    return num_frames, (width, height), fps
