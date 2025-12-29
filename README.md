# SAM2 Video UI - Setup and Processing

Annotation tool for SAM2 video object segmentation with automatic setup and model management.

## 1. Setup Script (`setup.py`)

**Purpose**: Automatically install SAM2, dependencies, and model checkpoints

**Usage** (ideally in a virtual environment built for using this package by conda or venv etc.):
```bash
python setup.py
```

**What it does**:
- Checks Python version (requires 3.8+)
- Installs Python packages (torch, opencv, numpy, etc.)
- **Clones and installs SAM2** into `Sam2UI/sam2/`
- **Interactive model selection** - choose which checkpoints to download
- Optionally installs SAM3 for text-based prompting
- Creates launcher scripts (run.bat/run.sh)
- Verifies installation

**Interactive Model Selection**:
During setup, you'll be prompted to choose models:
- **Option B (Recommended)**: SAM2.1 Small + Base+ (~500 MB)
- **Option A**: All SAM2.1 models (~1.5 GB)
- **Options 1-8**: Individual models
- **Option C**: Custom selection

## 2. Model Selection

SAM2 UI supports multiple model variants with automatic selection:

**SAM2.1 Models (Recommended - Released Sept 2024)**:
- **Tiny** (156 MB): Fastest, good for real-time interaction
- **Small** (184 MB): Best balance of speed and quality
- **Base+** (324 MB): High quality segmentation
- **Large** (898 MB): Best quality, slower inference

**SAM2 Models (Legacy - Released July 2024)**:
- Available in same sizes as SAM2.1
- Use if you need compatibility with older workflows

**Model Selection Methods**:
1. **During Setup**: Choose which models to download
2. **In UI**: Use dropdown menu to select model before loading
3. **Auto Mode**: Automatically picks best model for your GPU memory:
   - <4GB VRAM: Tiny/Small models
   - 4-8GB VRAM: Small/Base+ models
   - ≥8GB VRAM: Large models (best quality)

**GPU Memory Requirements**:
- Tiny/Small: 2GB+ VRAM
- Base+: 4GB+ VRAM
- Large: 8GB+ VRAM

## 3. Processing Script (`process_annotations.py`)

**Purpose**: Process annotation JSON from SAM2 Video UI to generate segmented video and masks

**Usage**:
```bash
# Basic usage (uses SAM2.1 Base+ by default)
python process_annotations.py annotations.json video.mp4

# Use SAM2.1 Large model
python process_annotations.py annotations.json video.mp4 --model sam2.1-large

# With custom output directory
python process_annotations.py annotations.json video.mp4 --output_dir results/

# With custom settings
python process_annotations.py annotations.json video.mp4 \
  --output_dir results/ \
  --fps 30 --opacity 0.4
```

## Complete Workflow

### 1. Setup (First Time Only)
```bash
python setup.py
```

### 2. Use SAM2 Video UI
```bash
# Windows
run.bat

# Linux/Mac
./run.sh

# Manual
python sam2_ui.py
```

### 3. Create Annotations
- Load video in SAM2 Video UI
- Add click points on objects
- Export annotations as JSON

### 4. Process Annotations
```bash
python process_annotations.py annotations.json video.mp4
```

## Output Files

After processing, you'll get:

- **`output_dir/masks/`** - Individual mask images (PNG files)
- **`output_dir/segmented_video.mp4`** - Video with colored mask overlays
- **`output_dir/processing_metadata.json`** - Processing statistics

## Command Line Options

### `process_annotations.py` Options

| Option | Description | Default |
|--------|-------------|---------|
| `--output_dir` | Output directory | `sam2_output` |
| `--model` | SAM2 model to use | `sam2_hiera_base_plus.pt` |
| `--fps` | Output video FPS | 30.0 |
| `--opacity` | Mask overlay opacity (0.0-1.0) | 0.4 |

## Examples

### Basic Processing
```bash
python process_annotations.py my_annotations.json my_video.mp4
```

### Custom Output Directory
```bash
python process_annotations.py annotations.json video.mp4 --output_dir my_results/
```

### High Quality Processing
```bash
python process_annotations.py annotations.json video.mp4 \
  --model sam2.1-large \
  --fps 60 --opacity 0.6 \
  --output_dir high_quality_results/
```

## SAM3 Support (Optional)

SAM3 adds text-based prompting capabilities for object segmentation.

### Requirements

**System Requirements**:
- Python 3.12+
- PyTorch 2.7+
- **CUDA 12.6+** (for GPU acceleration)
- HuggingFace account with SAM3 access

**Note**: SAM3 has stricter requirements than SAM2. Consider creating a separate Python 3.12 environment if needed.

### Installation Steps

#### 1. During Setup
When running `setup.py`, answer 'y' when prompted for SAM3 installation.

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

After authentication, download SAM3 checkpoints from HuggingFace and place them in the `sam3/checkpoints/` directory:

```bash
# Create checkpoints directory
mkdir -p sam3/checkpoints

# Download using Python (after huggingface-cli login)
python -c "
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id='facebook/sam3',
    filename='sam3_hiera_l.pt',
    local_dir='sam3/checkpoints'
)
"
```

**Expected checkpoint location**: `Sam2UI/sam3/checkpoints/sam3_hiera_l.pt`

#### 5. Verify Installation

```bash
python -c "from sam3.model_builder import build_sam3_video_predictor; print('SAM3 OK')"
```

### Current Features

- Point-based prompts (compatible with SAM2 workflow)
- Model selection via UI (when SAM3 is detected)

### Coming Soon

- Text-based object prompts
- Combined text + point prompts for refinement
- Multi-modal prompting (text + points + boxes)

### Troubleshooting SAM3

- **CUDA version too old**: SAM3 requires CUDA 12.6+. Check with `nvidia-smi` or upgrade CUDA toolkit
- **PyTorch too old**: Upgrade PyTorch: `pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126`
- **Access not granted**: Request access at https://huggingface.co/facebook/sam3 and wait for approval
- **Checkpoint not found**: Ensure checkpoints are in `sam3/checkpoints/` after download
- **Import error**: Verify installation: `pip list | grep -i sam3`

## Requirements

**For SAM2**:
- Python 3.10 or higher
- PyTorch 2.5.1+
- torchvision 0.20.1+
- 4GB+ RAM (8GB+ recommended)
- NVIDIA GPU with CUDA (recommended)
- Git (for cloning SAM2 repository)
- Internet connection (for setup)

**For SAM3 (Optional)**:
- Python 3.12+
- PyTorch 2.7+
- HuggingFace account with SAM3 access

## Troubleshooting

### Setup Issues
- **Python version**: Requires Python 3.10+ for SAM2, 3.12+ for SAM3
- **Git not installed**: Download from https://git-scm.com/downloads
- **Internet connection**: Needed to download models
- **Permissions**: May need admin rights, or you can set up within your own virtual environment created by conda or venv
- **Failed to install some packages**: Try downgrading Python from the newest version, then rerun setup.py. If it still fails, install the failed package with conda, then rerun setup.py.
- **Failed to install pycocotools-windows**: Modify setup.py to install pycocotools instead.
- **Failed to load pytorch_python dll**: Remove torch and torchvision, then let setup.py reinstall them
- **SAM2 import error**: Check that `sam2/sam2/` directory exists with `__init__.py`
- **Model not found**: Ensure checkpoints are in `sam2/checkpoints/`
- **Config not found**: Verify `sam2/configs/` directory exists

### Processing Issues
- **Model not found**: Run `setup.py` first to download models
- **Memory errors**: Use smaller model (tiny or small) or reduce video resolution
- **Import errors**: Verify SAM2 package is installed: `pip list | grep -i sam`
- **Path errors**: Ensure you're running from the Sam2UI directory

### Common Solutions
1. **Re-run setup**: `python setup.py`
2. **Check SAM2 installation**: Verify `sam2/` directory exists with subdirectories
3. **Verify Python version**: `python --version` (must be 3.10+)
4. **Check file paths**: Ensure annotation and video files exist
5. **Verify formats**: Use JSON files exported from SAM2 Video UI
6. **Try different model**: Use the dropdown to select a different model variant


