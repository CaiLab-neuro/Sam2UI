# SAM2 Video UI - Setup and Processing

Two scripts for SAM2 Video UI:

## 1. Setup Script (`setup.py`)

**Purpose**: Install everything needed for SAM2 Video UI

**Usage**:
```bash
python setup.py
```

**What it does**:
- Installs Python packages (torch, opencv, numpy, etc.)
- Downloads SAM2 model checkpoints
- Creates necessary directories
- Creates launcher scripts (run.bat/run.sh)
- Verifies installation

## 2. Processing Script (`process_annotations.py`)

**Purpose**: Process annotation JSON from SAM2 Video UI to generate segmented video and masks

**Usage**:
```bash
# Basic usage
python process_annotations.py annotations.json video.mp4

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
  --model checkpoints/sam2_hiera_large.pt \
  --fps 60 --opacity 0.6 \
  --output_dir high_quality_results/
```

## Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended)
- NVIDIA GPU with CUDA (recommended)
- Internet connection (for setup)

## Troubleshooting

### Setup Issues
- **Python version**: Requires Python 3.8+
- **Internet connection**: Needed to download models
- **Permissions**: May need admin rights, or you can set up within your own virtual environment created by conda

### Processing Issues
- **Model not found**: Run `setup.py` first
- **Memory errors**: Use smaller model or reduce video resolution
- **Import errors**: Reinstall dependencies

### Common Solutions
1. **Re-run setup**: `python setup.py`
2. **Check file paths**: Ensure annotation and video files exist
3. **Verify formats**: Use JSON files exported from SAM2 Video UI

4. **Check dependencies**: Ensure all packages are installed
