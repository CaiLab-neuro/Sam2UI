#!/bin/bash
echo "Starting SAM2 Video UI..."
echo ""

# Activate conda environment (detected during setup)
CONDA_BASE="/opt/miniconda3"
CONDA_ENV="SAM2"

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    echo "Activating conda environment '$CONDA_ENV' from $CONDA_BASE..."
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
    python3 sam2_ui.py
    exit $?
fi

# Fallback: try conda run if conda is in PATH
if command -v conda &> /dev/null; then
    echo "Using 'conda run' to launch with environment '$CONDA_ENV'..."
    conda run -n "$CONDA_ENV" python3 sam2_ui.py
    exit $?
fi

# Final fallback
echo ""
echo "WARNING: Could not activate conda environment '$CONDA_ENV'"
echo "Please activate manually with: conda activate $CONDA_ENV"
echo ""
read -p "Press Enter to try anyway..."
python3 sam2_ui.py
