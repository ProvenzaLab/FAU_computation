#!/usr/bin/env python3
"""
Batch wrapper for demo.py.
"""

import argparse
import glob
import pathlib
import subprocess
import sys
from multiprocessing import Pool
from typing import List
import tempfile, shutil  
from pathlib import Path    

def _run_one(video: pathlib.Path, opts):
    cmd = [
        sys.executable, "./demo.py",
        "--video",
        "--input", str(video),
        "--gpu_ids", str(opts.gpu_ids),
        "--arc", opts.arc,
        "--resume", opts.resume,
        "--batch_size", "15",
        "--num_workers", "5",
    ]
    subprocess.run(cmd, check=True)

VIDEO_EXTS = {".mp4"}

def _collect_sources(src_args):
    videos, seen = [], set()
    for spec in src_args:
        p = Path(spec).expanduser()
        if p.is_dir():
            it = p.rglob("*")
        else:                           # file or glob
            it = Path().glob(spec)
        for f in it:
            if f.suffix.lower() in VIDEO_EXTS and f.is_file() and f not in seen:
                seen.add(f)
                videos.append(f)        # keep as Path
    return videos

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, nargs="+",
                    help="Video paths or glob patterns (absolute or relative).")
    ap.add_argument("--resume", default="/Users/bezolge1/Downloads/BCM-Collaboration/OpenGraphAU/checkpoints/OpenGprahAU-ResNet18_second_stage.pth")
    ap.add_argument("--arc", default="resnet18")
    ap.add_argument("--gpu_ids", default="-1")
    ap.add_argument("--parallel", action="store_true")
    opts = ap.parse_args()

    videos = _collect_sources(opts.src)
    if not videos:
        sys.exit("No matching input files.")

    if opts.parallel:
        with Pool() as pool:
            pool.starmap(_run_one, [(v, opts) for v in videos])
    else:
        for v in videos:
            print(f"Runnning video: {v}")
            _run_one(v, opts)

if __name__ == "__main__":
    main()
