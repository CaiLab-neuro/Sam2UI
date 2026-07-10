# Contributing to Gaze Target Annotator

Thank you for your interest in improving this toolkit! Contributions of all kinds are welcome — bug reports, feature suggestions, documentation improvements, and code.

## Ways to Contribute

- **Report a bug or problem** — open an issue on the [Issues tab](../../issues)
- **Suggest a feature** — open an issue describing your use case and what you would like the tool to do
- **Improve the documentation** — fix unclear instructions, add examples, or report where you got stuck
- **Contribute code** — fix a bug or implement a feature via a pull request

## Reporting Issues

When reporting a bug, please include as much of the following as applies:

1. **What you did** — the exact command you ran or UI steps you took
2. **What happened** vs. **what you expected**
3. **Error messages** — copy the full traceback from the terminal (both UIs print errors to the terminal they were launched from)
4. **Your environment**:
   - Operating system
   - Python version (`python --version`)
   - GPU model and CUDA version (`nvidia-smi`)
   - PyTorch version (`python -c "import torch; print(torch.__version__)"`)
   - Which pipeline (SAM2 / SAM3) and which model checkpoint
5. **Video characteristics** if relevant — resolution, number of frames, codec

If the problem is about segmentation *quality* (rather than a crash), you can email the author (mingbo[dot]cai[at]miami[dot]edu) — a screenshot of the problematic frame with the mask overlay helps a lot.

## Contributing Code

### Workflow

We use the standard GitHub fork-and-pull-request workflow:

```bash
# 1. Fork the repository on GitHub (button in the top-right corner), then:
git clone git@github.com:<your-username>/SegmentVideo4Gaze.git
cd SegmentVideo4Gaze
git remote add upstream git@github.com:CaiLab-neuro/SegmentVideo4Gaze.git

# 2. Create a branch for your change
git checkout -b fix-some-bug

# 3. Make your changes, commit with a clear message
git commit -am "Fix X when Y"

# 4. Keep your branch up to date with upstream
git fetch upstream
git rebase upstream/main

# 5. Push and open a pull request on GitHub
git push origin fix-some-bug
```

If your work is not ready for review yet, open it as a **draft pull request**.

### Development Setup

Follow the normal installation in the [README](README.md) (dedicated conda environment + `python install.py`). SAM2 and SAM3 are installed in editable mode inside `sam_models/`, so the toolkit's own code (`*.py` in the repository root) can be edited and run directly.

### Coding Standards

- **Python style**: follow [PEP 8](https://peps.python.org/pep-0008/); write docstrings for public functions
- **Prefer monkey-patching over forking**: we patch SAM2/SAM3 behavior from our own modules (see `sam_lazy_loader.py`) instead of modifying the cloned upstream repositories, as they are not shipped by this repo but downloaded directly from github.
- **Memory discipline**: this toolkit's main value is handling long videos on modest hardware. Never accumulate all frames or masks in memory — stream masks to disk as they are produced, and call `enable_lazy_loading()` *before* creating a predictor
- **UI responsiveness**: any operation longer than a fraction of a second must run in a background thread, communicating with Tkinter only via `root.after()` (never touch Tk widgets from a worker thread)
- **Schema compatibility**: `sam3_project.py` defines the on-disk `project.json` schema, and the SAM3↔SAM2 handoff fields (`sam2_covered_ids`, `sam3_sub_ids`) are read by both pipelines. Changes to these must remain backward-compatible with projects users already have on disk
- **Logging**: print user-facing progress to the console; include enough context in error messages to debug from a user's report

### Testing

There is no formal pytest suite yet, and we may develop them later. Before opening a pull request:

1. **Run the validation scripts** relevant to your change if you find such scripts in the repo (TODO) (e.g., `test_sam3_infrastructure.py`, `test_sam3_prompts.py` for the SAM3 pipeline) — they are standalone scripts, run directly with `python`
2. **Exercise the change end-to-end** with a short video (a few hundred frames): run the affected UI or CLI workflow and confirm the output masks/video look correct
3. **Check the other pipeline is unaffected** if you touched shared modules (`segment.py`, `utils.py`, `sam_lazy_loader.py`)

Contributions that add proper automated tests are especially welcome.

### Documentation

If your change affects how users operate the tool, update the relevant page:

- `README.md` — landing page (overview, installation)
- `docs/sam3_guide.md` — SAM3 pipeline usage
- `docs/sam2_guide.md` — SAM2 pipeline usage
- `docs/gaze_alignment.md` — gaze-target alignment

## Questions

Not sure whether something is a bug, or whether a feature fits the project? Open an issue and ask — we are happy to discuss before you invest time in a pull request.