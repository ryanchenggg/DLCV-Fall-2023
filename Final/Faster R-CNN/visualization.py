import torch
import torchvision
from torch import nn
from torchvision import transforms as T
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
import cv2

# 檢查gt的bounding box是否正確

file_path = "/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/images/validation_images"
file_name = "d2de06de-81ec-47f8-88d2-0f0d5182c215_69.jpg"
img_path = os.path.join(file_path, file_name)
image = Image.open(img_path).convert("RGB")

draw = ImageDraw.Draw(image)
box = [28, 164, 48, 190]

draw.rectangle(box, outline='green', width=2)

image.save(f'./gt_test.jpg')