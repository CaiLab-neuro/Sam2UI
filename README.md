# SAM2 Graphic User Interface and Gaze-Target Annotation (Gaze Target Annotator)

Gaze Target Annotator, as a part of GazeBehavior Annotation Toolkit (GBAT), provides a generic-purpose graphical user interface to annotate objects in a video with a few points, and utilize SAM2 or SAM3 to segment objects in the video. 
For eye-tracking research, it further includes utility to map eye-gaze coordinate in the scene video of head-mounted eye-tracker to categories of gaze target.

The toolkit includes three main components: an interactive segmentation UI, an offline processing script, and a script for using segmentation outputs and eye-tracker gaze coordiate data to generate time series of gaze target. For gaze coordinate data, we follow the format of Pupil Labs output.


## Prerequisites

**IMPORTANT**: Before running setup, create and activate a dedicated conda environment or virtualenv. This prevents dependency conflicts and ensures proper package installation.

### Recommended: Conda Environment

```bash
# Create conda environment with Python 3.10+ (3.12+ for SAM3)
conda create -n sam python=3.12 -y
conda activate sam
```

### Alternative: Virtual Environment (venv)

```bash
# Create virtual environment
python3 -m venv sam_env
# Activate it
source sam_env/bin/activate  # Linux/Mac
# OR
sam_env\Scripts\activate  # Windows
```

## 1. Setup Script (`install.py`)

**Purpose**: Automatically install SAM2, dependencies, and model checkpoints

**Usage** (activate your environment first, then run):
```bash
conda activate sam  # or: source sam_env/bin/activate
python install.py
```

**What it does**:
- Checks Python version (requires 3.10+ for SAM2, 3.12+ for SAM3)
- Installs Python packages (torch, opencv, numpy, etc.)
- **Clones and installs SAM2** into `sam_models/sam2/`
- **Interactive model selection** - choose which checkpoints to download
- Optionally installs SAM3 for text-based prompting (into `sam_models/sam3/`)
- Installs additional dependencies (`einops` for SAM3)
- Creates launcher scripts (run.bat/run.sh)
- Verifies installation

**Checkpoint Download**:
During setup, you'll be prompted to choose which model checkpoints to download:
- **Option B**: SAM2.1 Small + Base+ (~500 MB, lower memory usage)
- **Option A**: All SAM2.1 models (~1.5 GB)
- **Option C**: Custom selection — enter individual model numbers or a range (e.g., `2`, `2-4`, `2,4`)

Only models whose checkpoints are downloaded will be available at runtime.

## 2. Using the SAM2 Video UI

### Initial Setup
1. Load a video file
2. Create an object list — add each object you want to track and give it a name

### Annotation
1. Navigate to a frame where the object is clearly visible
2. Add annotation points on the object (positive and negative clicks)
3. Repeat for additional objects or frames as needed
4. Export annotations as a JSON file

### Segmentation
Select a model from the dropdown (larger models such as Base+ or Large generally produce better masks but are slower; only downloaded checkpoints appear), then either segment within the UI or use the processing script:
```bash
python process_annotations.py annotations.json video.mp4
```
For videos longer than a few hundred frames, using the processing script is recommended as the UI can be slow on long videos.

### Refinement
1. Import segmentation results (this also imports the original annotations)
2. Use quality metrics to identify segments in video where segmentation is poor
3. Add or adjust annotation points and re-segment the frame to verify
4. Re-export the updated annotations and re-run the processing script, or run segmentation in refinement mode for a range of frames within UI

## 3. Processing Script (`process_annotations.py`)

**Purpose**: Process annotation JSON from SAM2 Video UI to generate segmented video and masks

**Usage**:
```bash
# Basic usage (uses SAM2.1 Base+ by default)
python process_annotations.py annotations.json video.mp4

# Use SAM2.1 Large model
python process_annotations.py annotations.json video.mp4 --model sam2.1-large

# With custom output directory
python process_annotations.py annotations.json video.mp4 --output-dir results/

# With custom settings
python process_annotations.py annotations.json video.mp4 \
  --output-dir results/ \
  --fps 30 --opacity 0.4

# Re-render output video from existing masks (no re-segmentation)
python process_annotations.py annotations.json video.mp4 \
  --output-dir results/ --video-only --opacity 0.6

# Re-segment only objects that were updated in the annotation file
python process_annotations.py annotations.json video.mp4 \
  --output-dir results/ --only-updated
```

### `process_annotations.py` Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output-dir` | Output directory | `sam2_output` |
| `--model` | SAM2 model to use | auto (best for GPU) |
| `--fps` | Output video FPS | 30.0 |
| `--opacity` | Mask overlay opacity (0.0-1.0) | 0.4 |
| `--video-only` | Re-render video from existing masks, skip segmentation | — |
| `--only-updated` | Re-segment only objects marked as updated; reuse other masks | — |
| `--prev-results` | Directory with previous masks to reuse (with `--only-updated`) | output dir |
| `--offload-to-cpu` | Offload video frames to CPU to reduce GPU memory usage | — |
| `--frame-dir` | Persistent directory for extracted frames (avoids re-extraction) | temp dir |

## 4. Gaze-Target Annotation with Segmented Results (`process_gaze_mask_alignment.py`)

**Purpose**: Assign each gaze sample to the most likely segmented object by combining gaze coordinates, world-camera timestamps, and SAM2/SAM3 mask outputs.

**Current features**:
- Auto-discovers subjects and camera IDs from `{subject}_{camera}_gaze.csv` files when `--subject-id` or `--camera-id` is omitted.
- Supports both mask folder layouts: `{subject}/{camera}/masks/` and legacy `{subject}_{camera}/masks/`.
- Optionally labels blink periods and removes gaze points that fall within blinks before object assignment.
- Optionally excludes named objects or mask IDs with `--ignore-object-list`; filtered CSV/PKL outputs receive an `_excluding_ignored_objects` suffix.
- Can process subject-camera pairs in parallel with `--num-workers`.

**Expected CSV structure**:
- **`{subject_id}_{camera}_gaze.csv`**: one row per gaze sample, ordered by time. Required columns are `timestamp [ns]`, `gaze x [px]`, and `gaze y [px]`. Additional columns are preserved in the output.
- **`{subject_id}_{camera}_world_timestamps.csv`**: one row per world-camera frame, ordered by time. Required column is `timestamp [ns]`. Additional columns are allowed; the script rebuilds `frame_idx` and `frame_timestamp` during alignment and preserves `source_frame_idx` when present.
- **`{subject_id}_{camera}_blinks.csv`**: used only with `--blink-dir`. Required columns are `start timestamp [ns]`, `end timestamp [ns]`, and `blink id`.

**Minimum column examples**:
```csv
# {subject_id}_{camera}_gaze.csv
timestamp [ns],gaze x [px],gaze y [px]
1000000000,640.5,360.2
1000033333,642.1,361.0
```

```csv
# {subject_id}_{camera}_world_timestamps.csv
timestamp [ns]
999999000
1000030000
1000063000
```

```csv
# {subject_id}_{camera}_blinks.csv
blink id,start timestamp [ns],end timestamp [ns]
0,1000200000,1000400000
1,1001000000,1001200000
```

**Usage**:
```bash

# Process multiple subjects and cameras in parallel
python process_gaze_mask_alignment.py \
  /path/to/gaze_world_data \
  /path/to/segmentation_masks \
  /path/to/output_dir \
  --subject-id 27,28 \
  --camera-id child,parent \
  --num-workers 4

# Auto-discover all subjects and cameras from CSV filenames and process in parallel
python process_gaze_mask_alignment.py \
  /path/to/gaze_world_data \
  /path/to/segmentation_masks \
  /path/to/output_dir \
  --num-workers 4

# Remove gaze points during blinks
python process_gaze_mask_alignment.py \
  /path/to/gaze_world_data \
  /path/to/segmentation_masks \
  /path/to/output_dir \
  --subject-id 27 \
  --camera-id child \
  --blink-dir /path/to/blink_data

# Exclude objects from assignment, such as hands or calibration targets
python process_gaze_mask_alignment.py \
  /path/to/gaze_world_data \
  /path/to/segmentation_masks \
  /path/to/output_dir \
  --ignore-object-list /path/to/ignore_objects.txt

```

### `process_gaze_mask_alignment.py` Options

| Option | Description |
|--------|-------------|
| `gaze_world_dir` | Directory with `{subject}_{camera}_gaze.csv` and `{subject}_{camera}_world_timestamps.csv` |
| `mask_dir` | Directory with `{subject}/{camera}/masks/` or `{subject}_{camera}/masks/` subfolders |
| `output_dir` | Directory to save output files |
| `--subject-id` | Subject ID(s), e.g. `27` or `27,28`. Auto-discovered if omitted. |
| `--camera-id` | Camera ID(s), e.g. `child` or `child,parent`. Auto-discovered if omitted. |
| `--blink-dir` | Directory with `{subject}_{camera}_blinks.csv` files for blink labeling/removal |
| `--log-path` | Path for the log file; default is `{output_dir}/gaze_object.log` |
| `--ignore-object-list` | Text file of object labels, IDs, numeric IDs, or exact mask filenames to exclude from scoring |
| `--num-workers` | Number of subject-camera pairs to process in parallel; default is `1` |
| `--skip-figures` | Skip trajectory and confidence heatmap figure generation |
| `--start-plot-time` | Start time in seconds for method-figure plots |
| `--end-plot-time` | End time in seconds for method-figure plots |

**Output files**:
- **`output_dir/{subject_id}/{camera}/{subject_id}_{camera}_gazed_object.csv`** - Gaze samples with assigned object labels and confidence
- **`output_dir/{subject_id}/{camera}/{subject_id}_{camera}_gaze_object_probabilities.pkl`** - Per-gaze probabilities for all available masks
- **`output_dir/{subject_id}/{camera}/{subject_id}_{camera}_gaze_blink_labeled.csv`** - Blink-labeled gaze data when `--blink-dir` is used
- **`output_dir/{subject_id}/{camera}/{subject_id}_{camera}_gaze_blink_removed.csv`** - Blink-removed gaze data when `--blink-dir` is used
- **`output_dir/{subject_id}/{camera}/figures/{subject_id}_{camera}_trajectory_plot.png`** and **`.pdf`** - Gaze-object trajectory over time, unless `--skip-figures` is used
- **`output_dir/{subject_id}/{camera}/figures/{subject_id}_{camera}_confidence_heatmap.png`** and **`.pdf`** - Confidence heatmap by object over time, unless `--skip-figures` is used
- **`output_dir/gaze_object.log`** - Processing log, unless `--log-path` is provided

When `--ignore-object-list` is used, the main CSV and probability PKL filenames receive `_excluding_ignored_objects` before the extension.

The main output CSV keeps the original gaze columns and any extra gaze metadata, then adds the alignment/object-assignment fields below:
- `frame_idx`: world-camera frame index matched to the gaze sample
- `frame_timestamp`: timestamp of the matched world-camera frame
- `source_frame_idx`: preserved when present in the world-camera timestamp file
- `in_blink`: added only when `--blink-dir` is used
- `blink id`: added only when `--blink-dir` is used
- `gazed_object_id`: mask/object ID parsed from the exported mask filename
- `gazed_object`: object label parsed from the exported mask filename
- `gazed_object_confidence`: fraction of pixels inside the 20 px gaze-radius circle that overlap the winning object mask

## Output Files

After processing, you'll get:

- **`output_dir/masks/`** - Individual mask images (PNG files)
- **`output_dir/segmented_video.mp4`** - Video with colored mask overlays
- **`output_dir/processing_metadata.json`** - Processing statistics
- **`output_dir/{subject_id}_gazed_object/`** - Gaze-target annotation outputs generated from segmentation masks

## SAM3 Support (Optional)

SAM3 adds text-based prompting capabilities for object segmentation. So far, we have not integreated this feature. But users can still use the point-based prompt for SAM3. Empirically we found it may perform worse than SAM2 in this usage.

### Requirements

**System Requirements**:
- Python 3.12+
- PyTorch 2.7+
- **CUDA 12.6+** (for GPU acceleration)
- HuggingFace account with SAM3 access

**Note**: SAM3 has stricter requirements than SAM2. Consider creating a separate Python 3.12 environment if needed.

### Installation Steps

#### 1. During Setup
When running `install.py`, answer 'y' when prompted for SAM3 installation.

The installer will:
- Check Python version (≥3.12)
- Check PyTorch version (≥2.7)
- Check CUDA version (≥12.6)
- Clone SAM3 repository
- Install SAM3 package

#### 2. Request Checkpoint Access
Before you can download SAM3 checkpoints:

1. Visit https://huggingface.co/facebook/sam3
2. Click "Request Access"
3. Wait for approval (usually within 24-48 hours)

#### 3. Authenticate with HuggingFace

After access is granted:

```bash
# Install HuggingFace CLI (if not already installed)
pip install huggingface-hub

# Generate access token at: https://huggingface.co/settings/tokens
# Then authenticate
huggingface-cli login
# Paste your token when prompted
```

#### 4. Download Checkpoints

After authentication, download SAM3 checkpoints from HuggingFace and place them in the `sam_models/sam3/checkpoints/` directory:

```bash
# Create checkpoints directory
mkdir -p sam_models/sam3/checkpoints

# Download using Python (after huggingface-cli login)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/sam3',
    filename='sam3_hiera_l.pt',
    local_dir='sam_models/sam3/checkpoints'
)
"
```


**Expected checkpoint location**: `Sam2UI/sam_models/sam3/checkpoints/`

#### 5. Verify Installation

```bash
python -c "from sam3.model_builder import build_sam3_video_predictor; print('SAM3 OK')"
```

### Current Features

- Point-based prompts (compatible with SAM2 workflow)
- Model selection via UI (when SAM3 is detected)

### Coming Soon

- Combined text + point prompts for refinement

## Troubleshooting

### Setup Issues
- **Python version**: Requires Python 3.10+ for SAM2, 3.12+ for SAM3
- **Git not installed**: Download from https://git-scm.com/downloads
- **Failed to install some packages**: Try downgrading Python from the newest version, then rerun install.py. If it still fails, install the failed package with conda, then rerun install.py.
- **Failed to load pytorch_python dll**: Remove torch and torchvision, then let install.py reinstall them

### Processing Issues
- **Model not found**: Run `install.py` first to download models
- **Memory errors**: Use smaller model (tiny or small) or reduce video resolution
- **Path errors**: Ensure you're running from the Sam2UI directory

### SAM3 Issues
- **CUDA version too old**: SAM3 requires CUDA 12.6+. Check with `nvidia-smi` or upgrade CUDA toolkit
- **PyTorch too old**: Upgrade PyTorch: `pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
- **Access not granted**: Request access at https://huggingface.co/facebook/sam3 and wait for approval
- **Checkpoint not found**: Ensure checkpoints are in `sam_models/sam3/checkpoints/` after download
- **Import error**: Verify installation: `pip list | grep -i sam3`

### Common Solutions
1. **Re-run setup**: `python install.py`
2. **Check SAM2 installation**: Verify `sam_models/sam2/` directory exists with subdirectories
3. **Verify Python version**: `python --version` (must be 3.10+)
4. **Check file paths**: Ensure annotation and video files exist

## UI Video Loading Frame Inaccurate

### Issue

`sam2_ui.py` uses OpenCV lazy seeking when loading frames in the UI:

```python
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
ret, frame = cap.read()
```

For some inter-frame-compressed MP4 files, this seek is not frame-accurate: OpenCV can return a frame earlier than the requested `frame_idx`. This can make UI annotations appear to target the wrong frame even when the annotation data itself is valid.

`process_annotations.py` and `process_gaze_mask_alignment.py` rely on decoded video frames, so downstream processing should tolerate both the original inter-frame-compressed MP4s and intra-frame MJPEG AVI files. Reencoding is recommended when frame-accurate random access is needed in the UI or during quality checking.

### Solution

Reencode inter-frame-compressed MP4s into an intra-frame format such as MJPEG AVI. Intra-frame videos store each frame independently, which makes OpenCV random seeks much more reliable.

Use the seek-safe reencoding utility:

```bash
python reencode_seek_safe_videos.py --dry-run
python reencode_seek_safe_videos.py --verify --overwrite
```

To process selected subjects or cameras:

```bash
python reencode_seek_safe_videos.py --subjects 1,10,13 --camera child --verify --overwrite
python reencode_seek_safe_videos.py --camera child,parent --overwrite
```

Omit `--camera` to process all discovered cameras. Camera identifiers are general strings parsed from filenames like `{subject}_{camera}_...mp4`; they are not limited to `child` and `parent`.

### `reencode_seek_safe_videos.py` Options

| Option | Description |
|--------|-------------|
| `--source-dir` | Directory containing the original aligned MP4 files |
| `--output-dir` | Directory where seek-safe AVI files are written |
| `--manifest-path` | JSON summary path for the reencoding run |
| `--log-path` | Path for the run log; defaults to `{output_dir}/reencode_seek_safe_videos.log` |
| `--camera` | Camera ID(s) to process, e.g. `child` or `child,parent`; auto-discovers all cameras if omitted |
| `--subjects` | Subject ID(s), e.g. `1` or `1,10,13`; processes all discovered subjects if omitted |
| `--source-mode` | Use sequential MP4 decoding with `video`, or rebuild from PNG frames with `frames` |
| `--frames-root-dir` | Root containing non-legacy frame folders named `{camera}_frames` |
| `--child-frames-dir` | Legacy extracted PNG frame directory for child videos |
| `--parent-frames-dir` | Legacy extracted PNG frame directory for parent videos |
| `--suffix` | Suffix appended before `.avi`; default is `_seeksafe` |
| `--frame-limit` | Optional frame cap for smoke tests |
| `--progress-every` | Print progress every N encoded frames |
| `--verify` | Verify lazy random seeks against sequential reads on the output AVI |
| `--verify-frames` | Comma-separated frame indices to verify; auto-selected if omitted |
| `--overwrite` | Replace existing output AVI files |
| `--dry-run` | Print planned jobs without encoding videos |

### Output Files

By default, an input video like:

```text
aligned_video_YB_finalized_version/1_child_0_35662.mp4
```

is written as:

```text
videos/child/1_child_0_35662_seeksafe.avi
```

The reencoding run also writes:

- **`{output_dir}/{camera}/{source_stem}{suffix}.avi`** - seek-safe MJPEG AVI for each processed source video
- **`{output_dir}/reencode_seek_safe_videos.log`** - run log unless `--log-path` is provided
- **`last_run_manifest.json`** - JSON manifest unless `--manifest-path` is provided
