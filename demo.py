import os
import numpy as np
import torch
import torch.nn as nn
import logging
#from extracting_FAUs.dataset import pil_loader
from model.ANFL import MEFARG
from utils import *
from conf import get_config,set_logger,set_outdir,set_env
import torch
import cv2
from PIL import Image

def extract_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_ids = []
    idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        frames.append(pil_img)
        frame_ids.append(idx)
        idx += 1
    cap.release()
    return frames, frame_ids


def main(num_main_classes, num_sub_classes, arc, neighbor_num, metric, input, out_path):

    #dataset_info = hybrid_prediction_infolist
    img_transform = image_eval()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")   # for Apple Silicon
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load model once
    net = MEFARG(
        num_main_classes=num_main_classes,
        num_sub_classes=num_sub_classes,
        backbone=arc,
        neighbor_num=neighbor_num,
        metric= metric
    )
    net = net.to(device)

    #if conf.resume != '':
    #    net = load_state_dict(net, conf.resume)

    net.eval()

    # Open video
    cap = cv2.VideoCapture(input)
    idx = 0
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        img_tensor = img_transform(pil_img).unsqueeze(0).to(device)
        #img_tensor = pil_img.unsqueeze(0).to(device)

        with torch.no_grad():
            pred = net(img_tensor).squeeze().cpu().numpy()

        predictions.append(pred)
        idx += 1

        # Optional: cleanup
        del frame, frame_rgb, pil_img, img_tensor, pred

    cap.release()

    predictions = np.stack(predictions)
    np.save(f"{out_path}/au_predictions.npy", predictions)
    print(f"Saved to {out_path}")
