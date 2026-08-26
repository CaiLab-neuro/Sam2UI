#!/usr/bin/env python3
"""
sam3_sync.py — push local annotation edits to a remote project directory.

Copies only the small metadata/annotation files needed for --refine:
  project.json, concept_metadata.json, refinements.json, redetect sentinels

Skips large server-generated data:
  masks/, cond_states/, inference_state.pkl, video files, image files

Usage:
    python sam3_sync.py <local_project_dir> <remote_dest>
    python sam3_sync.py --dry-run <local_project_dir> <remote_dest>

Examples:
    python sam3_sync.py ./test_sam3_concept user@server:/data/projects/test_sam3_concept
    python sam3_sync.py --dry-run ./my_project user@192.168.1.10:/home/user/projects/my_project

Windows notes:
  rsync is not built into Windows. This script tries, in order:
    1. rsync         (available with Git for Windows if rsync was selected, or Cygwin)
    2. rsync-win.exe (standalone Windows rsync port, if present on PATH or in known install dirs)
    3. wsl rsync     (available if WSL is installed: wsl --install, then: wsl apt install rsync)
  If none are found, install one of the above or use WinSCP / MobaXterm as a GUI alternative.
"""

import argparse
import os
import shutil
import subprocess
import sys


EXCLUDES = [
    "*/masks/",
    "*/cond_states/",
    "inference_state.pkl",
    "*.mp4",
    "*.avi",
    "*.jpg",
    "*.jpeg",
    "*.png",
]


# Common Windows installation paths for standalone rsync ports (cwRsync, DeltaCopy, etc.),
# plus rsync-win.exe (https://github.com/nheinemann/rsync-win / similar standalone builds).
# These are only searched when rsync is not on PATH.
_WIN_RSYNC_SEARCH_PATHS = [
    r"C:\Program Files\cwRsync\bin\rsync.exe",
    r"C:\Program Files (x86)\cwRsync\bin\rsync.exe",
    r"C:\cwRsync\bin\rsync.exe",
    r"C:\Program Files\DeltaCopy\rsync.exe",
    r"C:\Program Files (x86)\DeltaCopy\rsync.exe",
    r"C:\Tools\rsync\rsync.exe",
]

_WIN_RSYNC_WIN_SEARCH_PATHS = [
    r"C:\Tools\rsync-win\rsync-win.exe",
    r"C:\Program Files\rsync-win\rsync-win.exe",
    r"C:\Program Files (x86)\rsync-win\rsync-win.exe",
]


def find_rsync():
    """Return the rsync invocation to use, or None if not found.

    Search order on Windows:
      1. rsync on PATH (Git for Windows, Cygwin, or any port already configured)
      2. Known installation paths for cwRsync / DeltaCopy standalone ports
      3. rsync-win.exe on PATH, or in known install dirs (fallback standalone port)
      4. wsl rsync (WSL must be installed and have rsync)
    On Linux/Mac only step 1 is tried.
    """
    if shutil.which("rsync"):
        return ["rsync"]
    if sys.platform == "win32":
        for candidate in _WIN_RSYNC_SEARCH_PATHS:
            if os.path.isfile(candidate):
                return [candidate]
        rsync_win = shutil.which("rsync-win.exe") or shutil.which("rsync-win")
        if rsync_win:
            return [rsync_win]
        for candidate in _WIN_RSYNC_WIN_SEARCH_PATHS:
            if os.path.isfile(candidate):
                return [candidate]
        if shutil.which("wsl") and _wsl_has_rsync():
            return ["wsl", "rsync"]
    return None


def _wsl_has_rsync():
    try:
        result = subprocess.run(["wsl", "which", "rsync"],
                                capture_output=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def _to_wsl_path(path: str) -> str:
    """Convert a Windows path to its WSL mount path (e.g. C:\foo -> /mnt/c/foo)."""
    try:
        result = subprocess.run(
            ["wsl", "wslpath", path.replace("\\", "/")],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return path


def main():
    parser = argparse.ArgumentParser(
        description="Push local SAM3 annotation edits to a remote project directory.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-n", "--dry-run", action="store_true",
                        help="Show what would be transferred without copying anything.")
    parser.add_argument("local_project_dir",
                        help="Path to the local SAM3 project folder.")
    parser.add_argument("remote_dest",
                        help="rsync destination, e.g. user@host:/path/to/project")
    args = parser.parse_args()

    rsync_cmd = find_rsync()
    if rsync_cmd is None:
        print("ERROR: rsync not found.")
        if sys.platform == "win32":
            print(
                "\nOn Windows, install one of:\n"
                "  - Git for Windows (enable rsync during install)\n"
                "  - rsync-win.exe standalone port (place it on PATH or in one of the\n"
                "    paths listed below)\n"
                "  - WSL:  wsl --install  then  wsl apt install rsync\n"
                "  - cwRsync standalone: https://itefix.net/cwrsync\n"
                "  - DeltaCopy (bundles rsync.exe)\n"
                "  - Cygwin with the rsync package\n"
                "\nAlso searched these paths and did not find rsync.exe:\n"
                + "\n".join(f"  {p}" for p in _WIN_RSYNC_SEARCH_PATHS) +
                "\n\nAnd these paths and did not find rsync-win.exe:\n"
                + "\n".join(f"  {p}" for p in _WIN_RSYNC_WIN_SEARCH_PATHS) +
                "\n\nOr use WinSCP / MobaXterm for a graphical alternative."
            )
        sys.exit(1)

    # Trailing slash on source: rsync copies the *contents*, not the directory itself
    src = args.local_project_dir.rstrip("/\\") + "/"
    dst = args.remote_dest

    # WSL rsync needs the local path in Linux form
    if rsync_cmd == ["wsl", "rsync"]:
        src = _to_wsl_path(src)

    # Force group-readable/writable permissions on transferred files and dirs.
    # rsync -a preserves the source file's mode bits by default, but on Windows
    # there is no real umask: MSYS/Cygwin/WSL rsync ports synthesize a Unix mode
    # from the Windows read-only attribute, which commonly comes out owner-only
    # (e.g. 600). That mode then overwrites whatever shared/group permissions
    # the file had on a multi-user destination. --chmod here adds group rw
    # (files) / rwx (dirs) and other r/rx on top of the transferred mode, so
    # collaborators can still read and edit synced files regardless of what
    # permissions the source machine reported.
    cmd = rsync_cmd + ["-av", "--update", "--chmod=Fg+rw,Fo+r,Dg+rwx,Do+rx"]
    if args.dry_run:
        cmd.append("--dry-run")
    for pattern in EXCLUDES:
        cmd += ["--exclude", pattern]
    cmd += [src, dst]

    print(f"Source : {args.local_project_dir}")
    print(f"Dest   : {dst}")
    if args.dry_run:
        print("(dry run — no files will be transferred)")
    print(f"Command: {' '.join(cmd)}")
    print()

    sys.exit(subprocess.run(cmd).returncode)


if __name__ == "__main__":
    main()
