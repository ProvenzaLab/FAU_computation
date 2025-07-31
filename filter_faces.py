#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import pandas as pd

CASCADE_PATH = "./haarcascade_frontalface_default.xml"

def _detect_cpu(cascade):
    def f(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return int(len(faces) == 1)
    return f

def _detect_gpu(cascade):
    gpu_mat = cv2.cuda_GpuMat()
    def f(frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gpu_mat.upload(gray)
        faces_gpu = cascade.detectMultiScale(gpu_mat)
        faces = faces_gpu.download()
        return int(len(faces) == 1)
    return f

def main(base: str, video_dir:str):

    filtered_output_csv = f"{video_dir}/{base}_filtered_results.csv"
    video_path = f"{video_dir}/{base}.mp4"
    au_path = f"{video_path}_au_predictions.npy"

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    au_df = pd.DataFrame(np.load(au_path))

    use_gpu = cv2.cuda.getCudaEnabledDeviceCount() > 0
    if use_gpu:
        face_cascade = cv2.cuda.CascadeClassifier.create(CASCADE_PATH)
        face_cascade.setScaleFactor(1.1)
        face_cascade.setMinNeighbors(5)
        face_cascade.setMinObjectSize((60, 60))
        detect = _detect_gpu(face_cascade)
    else:
        face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        detect = _detect_cpu(face_cascade)

    cap = cv2.VideoCapture(video_path)
    is_valid = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= len(au_df):
            break
        is_valid.append(detect(frame))
        frame_idx += 1

    cap.release()

    # --- Save filtered AU results ---
    filtered_df = pd.DataFrame(is_valid)
    filtered_df.to_csv(filtered_output_csv, index=False)

    print(f"Filtered AU results saved to {filtered_output_csv} with {len(filtered_df)} valid frames.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter AU results keeping frames with exactly one face.")
    parser.add_argument("--video_dir", help="video directory")
    parser.add_argument("--base", help="Base filename without extension")
    args = parser.parse_args()
    main(args.base, args.video_dir)