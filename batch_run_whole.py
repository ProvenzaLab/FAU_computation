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
import demo
import os

def _run_one(opts):
    demo.main(
        opts.num_main_classes,
        opts.num_sub_classes,
        opts.arc,
        opts.neighbor_num,
        opts.metric,
        opts.input,
        opts.out_path,
    )

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

def main(PATH_VIDEO):
    ap = argparse.ArgumentParser()
    #ap.add_argument("--src", required=True, nargs="+",
    #                help="Video paths or glob patterns (absolute or relative).")
    #ap.add_argument("--resume", default="/Users/bezolge1/Downloads/BCM-Collaboration/OpenGraphAU/checkpoints/OpenGprahAU-ResNet18_second_stage.pth")
    ap.add_argument("--arc", default="resnet18")
    ap.add_argument("--gpu_ids", default="-1")
    ap.add_argument("--parallel", action="store_true")
    ap.add_argument("--num_main_classes", type=int, default=27)
    ap.add_argument("--num_sub_classes", type=int, default=14)
    ap.add_argument("--neighbor_num", type=int, default=4)
    ap.add_argument("--metric", type=str, default='dots', help="metric for graph top-K nearest neighbors selection")
    opts = ap.parse_args()
    #opts.src = [PATH_VIDEO]  # Directly set the source path for testing
    opts.input = PATH_VIDEO
    opts.out_path = os.path.dirname(PATH_VIDEO)
    _run_one(opts)

#if __name__ == "__main__":
#    PATH_VIDEO = "/Users/Timon/Documents/Houston/video_features/extracting_FAUs/outpath/GH010383.MOV"
#    main(PATH_VIDEO)
