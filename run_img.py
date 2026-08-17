import demo
from model.ANFL import MEFARG
from PIL import Image
import torch
import cv2
from utils import *

arc = "resnet18"
num_main_classes = 27
num_sub_classes = 14
neighbor_num = 4
metric = 'dots'

# Load model once
net = MEFARG(
    num_main_classes=num_main_classes,
    num_sub_classes=num_sub_classes,
    backbone=arc,
    neighbor_num=neighbor_num,
    metric= metric
)

device = "cpu"
net = net.to(device)
net.eval()

img_transform = image_eval()

#frame = cv2.imread('/Users/Timon/Downloads/compressed/Screenshot 2025-12-28 at 06.48.01.jpg')
frame = cv2.imread('/Users/Timon/Downloads/compressed/Screenshot 2025-12-28 at 06.56.56.jpg')

frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
pil_img = Image.fromarray(frame_rgb)

img_tensor = img_transform(pil_img).unsqueeze(0).to(device)
#img_tensor = pil_img.unsqueeze(0).to(device)

with torch.no_grad():
    pred = net(img_tensor).squeeze().cpu().numpy()

#np.save("au_predictions_image.npy", pred)
np.save("au_predictions_image_2.npy", pred)