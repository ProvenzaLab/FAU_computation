import os
import numpy as np
import torch
import torch.nn as nn
import logging
from extracting_FAUs.dataset import pil_loader
from model.ANFL import MEFARG
from extracting_FAUs.utils import *
from extracting_FAUs.conf import get_config,set_logger,set_outdir,set_env
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


def main(conf):

    #dataset_info = hybrid_prediction_infolist
    img_transform = image_eval()
    device = torch.device("cuda")  # Still safe for now

    # Load model once
    net = MEFARG(
        num_main_classes=conf.num_main_classes,
        num_sub_classes=conf.num_sub_classes,
        backbone=conf.arc,
        neighbor_num=conf.neighbor_num,
        metric=conf.metric
    )
    net = net.to(device)

    if conf.resume != '':
        net = load_state_dict(net, conf.resume)

    net.eval()

    # Open video
    cap = cv2.VideoCapture(conf.input)
    idx = 0
    predictions = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)

        img_tensor = img_transform(pil_img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred = net(img_tensor).squeeze().cpu().numpy()

        predictions.append(pred)
        idx += 1

        # Optional: cleanup
        del frame, frame_rgb, pil_img, img_tensor, pred

    cap.release()

    predictions = np.stack(predictions)
    np.save(f"{conf.input}_au_predictions.npy", predictions)
    print(f"Saved to {conf.input}")

# ---------------------------------------------------------------------------------

if __name__=="__main__":

    # python ./demo.py --video --input /Users/bezolge1/Downloads/BCM-Collaboration/OpenGraphAU/perfect-004-13-43-48.mov --gpu_ids -1 --arc resnet18 --resume /Users/bezolge1/Downloads/BCM-Collaboration/OpenGraphAU/checkpoints/OpenGprahAU-ResNet18_second_stage.pth

    conf = get_config()
    conf.evaluate = True
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

