#!/usr/bin/env python3

import argparse
import pathlib
import re
import cv2
import numpy as np
import pandas as pd

PRED_GLOB = "*_au_predictions.npy"
COMPANION_SUFFIX = "_filtered_results.csv"

ROOT_DEFAULT = pathlib.Path("numpy_results")
VIDEO_DIR_DEFAULT = pathlib.Path("videos012")

IDENT_RE = re.compile(r"(?P<id>.+?)\.mp4_au_predictions\.npy")

def sec_to_hms(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"

def process(video_dir: pathlib.Path) -> None:
    
    AU_ids = ['1', '2', '4', '5', '6', '7', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '22',
           '23', '24', '25', '26', '27', '32', '38', '39', 'L1', 'R1', 'L2', 'R2', 'L4', 'R4', 'L6', 'R6', 'L10', 'R10', 'L12', 'R12', 'L14', 'R14']
    
    rows = []

    for pred_path in sorted(video_dir.glob(PRED_GLOB)):

        m = IDENT_RE.fullmatch(pred_path.name)
        if not m:
            print(f"Error finding {pred_path}")
            break
        
        ident = m.group("id")
        arr = np.load(pred_path)  

        if arr.ndim != 2:
            raise ValueError(f"{pred_path}: {arr.shape} is not 2‑D")
        frames = arr.shape[0]

        comp_path = video_dir / f"{ident}{COMPANION_SUFFIX}"
        extra = (
            pd.read_csv(comp_path, header=None, usecols=[0])
            .squeeze("columns")
            .to_numpy()
        )

        extra = extra[:-1]

        if extra.shape != (frames,):
            raise ValueError(f"{comp_path}: {extra.shape} rows, expected {frames}")

        video_path = video_dir/ f"cropped_{ident}.mp4"
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        time_hms = [sec_to_hms(i / fps) for i in range(frames)]

        df = pd.DataFrame(
            {
                "identifier": ident,
                "frame_time_seconds": time_hms,
                "frame_idx": np.arange(frames),
                **{f"AU_{AU_ids[i]}": arr[:, i] for i in range(arr.shape[1])},
                "face_detected": extra,
            }
        )
        rows.append(df)

    return pd.concat(rows, ignore_index=True)

def main() -> None:
    p = argparse.ArgumentParser(
        description="Merge AU predictions and face‑detection flags into one CSV."
    )
    p.add_argument("--video_dir", type=pathlib.Path, default="videos012")
    p.add_argument("--out_csv", type=pathlib.Path, default=".")
    args = p.parse_args()
    
    full = process(args.video_dir)
    full.to_csv(args.out_csv, index=False)
    print(f"wrote {args.out_csv}  rows={len(full)}")

if __name__ == "__main__":
    main()
