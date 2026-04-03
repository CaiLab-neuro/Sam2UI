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

## 4. Gaze-Target Annotation with Segmented Results (`gazed_object_published_version.py`)

**Purpose**: Use exported segmentation masks together with gaze and world-camera timestamps to assign each gaze sample to the most likely object.

This component is intended for a workflow where segmentation is completed first in Sam2UI, and the resulting masks are then matched against gaze coordinates frame by frame. For each gaze point, the script compares the gaze location to the available object masks in the corresponding frame and outputs the most likely gazed object together with a confidence score.

**Required inputs**:
- **Gaze/world-camera directory** containing files named like `{subject_id}_{camera}_gaze.csv` and `{subject_id}_{camera}_world_timestamps.csv`
- **Segmentation mask directory** containing folders named like `{subject_id}_{camera}/masks/`
- **Output directory** for gaze-target annotation results

**Optional input**:
- **Blink directory** containing `{subject_id}_{camera}_blinks.csv` files if you want to label or remove gaze points during blinks

**Expected CSV structure**:
- **`{subject_id}_{camera}_gaze.csv`**: one row per gaze sample, ordered by time. Required columns are `timestamp [ns]`, `gaze x [px]`, and `gaze y [px]`. Additional columns are allowed and are preserved in the merged/output tables.
- **`{subject_id}_{camera}_world_timestamps.csv`**: one row per world-camera frame, ordered by time. Required column is `timestamp [ns]`. Additional columns are allowed, but the script rebuilds `frame_idx` and `frame_timestamp` from this file during alignment.
- **`{subject_id}_{camera}_blinks.csv`**: used only with `--blink-dir`. The code expects `start timestamp [ns]`, `end timestamp [ns]`, and `blink id`, because those columns are used to mark whether a gaze sample falls inside a blink interval.

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
# Process one subject/camera pair
python gazed_object_published_version.py \
  /path/to/gaze_world_data \
  /path/to/segmentation_masks \
  /path/to/output_dir \
  --subject-id 27 \
  --camera-id child

# Remove gaze points during blinks
python gazed_object_published_version.py \
  /path/to/gaze_world_data \
  /path/to/segmentation_masks \
  /path/to/output_dir \
  --subject-id 27 \
  --camera-id child \
  --blink-dir /path/to/blink_data
```

**Output files**:
- **`output_dir/{subject_id}_gazed_object/{subject_id}_{camera}_gazed_object.csv`** - Gaze samples with assigned object labels and confidence
- **`output_dir/{subject_id}_gazed_object/{subject_id}_{camera}_gaze_object_probabilities.pkl`** - Per-gaze probabilities for all available masks
- **`output_dir/{subject_id}_gazed_object/{subject_id}_{camera}_gaze_blink_labeled.csv`** - Blink-labeled gaze data when `--blink-dir` is used
- **`output_dir/{subject_id}_gazed_object/{subject_id}_{camera}_gaze_blink_removed.csv`** - Blink-removed gaze data when `--blink-dir` is used

The main output CSV keeps the original gaze columns and any extra gaze metadata, then adds the alignment/object-assignment fields below:
- `frame_idx`: world-camera frame index matched to the gaze sample
- `frame_timestamp`: timestamp of the matched world-camera frame
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


