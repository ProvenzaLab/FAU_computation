#!/bin/bash

source ~/.bashrc
source .venv/bin/activate

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CKPT="${SCRIPT_DIR}/checkpoints/OpenGraphAU-ResNet18_second_stage.pth"
VIDEO_DIR=videoTest

echo $CUDA_VISIBLE_DEVICES
nvidia-smi || echo "nvidia-smi failed"

for VID in "$VIDEO_DIR"/*.mp4; do

  uv run python batch_run_whole.py \
        --src "$VID" \
        --gpu_ids 0 \
        --arc resnet18 \
        --resume "$CKPT" \

  base=$(basename "$VID" .mp4)
  uv run python filter_faces.py --base "$base" --video_dir "$VIDEO_DIR"
  
done

uv run python create_csv.py --video_dir "$VIDEO_DIR" --out_csv "${VIDEO_DIR}_FAU.csv"