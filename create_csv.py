#!/usr/bin/env python3

import argparse
import pathlib
import re
import cv2
import numpy as np
import pandas as pd
import os

PRED_GLOB = "au_predictions.npy"
COMPANION_SUFFIX = "filtered_results.csv"

ROOT_DEFAULT = pathlib.Path("numpy_results")
VIDEO_DIR_DEFAULT = pathlib.Path("videos012")

IDENT_RE = re.compile(r"(?P<id>.+?)\au_predictions\.npy")

def sec_to_hms(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def process(video_path: pathlib.Path) -> None:
    
    AU_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22',
           '23', '24', '25', '26', '27', '32', '38', '39', 'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14']
    
    video_dir = os.path.dirname(video_path)

    pred_path = os.path.join(video_dir, PRED_GLOB)
    comp_path = os.path.join(video_dir, COMPANION_SUFFIX)
    
    arr = np.load(pred_path)  

    if arr.ndim != 2:
        raise ValueError(f"{pred_path}: {arr.shape} is not 2‑D")
    frames = arr.shape[0]

    
    extra = (
        pd.read_csv(comp_path, header=None, usecols=[0])
        .squeeze("columns")
        .to_numpy()
    )

    extra = extra[:-1]

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()

    time_hms = [sec_to_hms(i / fps) for i in range(frames)]

    df = pd.DataFrame(
        {
            "frame_time_seconds": time_hms,
            "frame_idx": np.arange(frames),
            **{f"AU_{AU_ids[i]}": arr[:, i] for i in range(arr.shape[1])},
            "face_detected": extra,
        }
    )
    return df

def main(video_path) -> None:

    # e.g. video_path = "/Users/Timon/Documents/Houston/video_features/extracting_FAUs/outpath/GH010349.MP4"
    video_dir = os.path.dirname(video_path)
    out_csv = f"{video_dir}/full_au_results.csv"
    full = process(video_path)
    full.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}  rows={len(full)}")

#if __name__ == "__main__":
#    PATH_VIDEO = "/Users/Timon/Documents/Houston/video_features/extracting_FAUs/outpath/GH010349.MP4"
#    main(PATH_VIDEO)
