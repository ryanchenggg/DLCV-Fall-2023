# -*- coding: utf-8 -*-
# Reference: https://speech.ee.ntu.edu.tw/~hylee/ml/2023-spring.php Hw3 sample code
import numpy as np
import pandas as pd
import torch
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import torchvision
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead

import mean_iou_evaluate as m
import imageio

from torch.utils.data import  DataLoader, Dataset
from p3_dataloader import p3_dataset
import torchvision.models as models
import torchvision.transforms.functional as F
from tqdm.auto import tqdm
import random

myseed = 1000  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

#Reference: https://github.com/pha123661/NTU-2022Fall-DLCV/blob/master/HW1/P2_B_training.py function code
class FocalLoss(nn.Module):
    def __init__(
        self,
        alpha=0.25,
        gamma=2,
        ignore_index=6,
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.CE = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, logits, labels):
        log_pt = -self.CE(logits, labels)
        loss = -((1 - torch.exp(log_pt)) ** self.gamma) * self.alpha * log_pt
        return loss
    
def deeplabv3_resnet50(num_classes=7, pretrained=True):
    # model = torch.hub.load('pytorch/vision:v0.10.0','deeplabv3_resnet50', pretrained=True)
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DeepLabV3_ResNet50_Weights.DEFAULT')
    model.classifier = DeepLabHead(2048, num_classes)
    model.aux_classifier = FCNHead(1024, num_classes)
    return model

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"
model = deeplabv3_resnet50(7)
model = model.to(device)

batch_size = 12
n_epochs = 20
mean = [0.485, 0.456, 0.406]  # imagenet
std = [0.229, 0.224, 0.225]

train_set = p3_dataset("./hw1_data/p3_data/train", task="train")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_set = p3_dataset("./hw1_data/p3_data/validation", task="test")
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

#criterion = FocalLoss()
criterion = nn.CrossEntropyLoss() #the performance is better 
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)

# Learning rate scheduler
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-4, steps_per_epoch=len(train_loader), epochs=n_epochs)

ckpt_path = 'p3_B_checkpoint'
if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)
bast_model_path = 'p3_bestmodel_B.pth'
# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_loss = np.inf
best_miou = -1

save_epochs = [0, 9, 19]  # 早期、中期和後期

#reference: GPT
for epoch in range(n_epochs):

    model.train()
    train_loss = []

    for batch in tqdm(train_loader):
        imgs, mask = batch
        logits = model(imgs.to(device))
        loss = criterion(logits['out'], mask.to(device))
        optimizer.zero_grad()
        loss.backward()
        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)
        optimizer.step()
        train_loss.append(loss.item())

    train_loss = sum(train_loss) / len(train_loss)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}")

    # ---------- Validation ----------
    valid_loss = []
    all_preds = []
    all_gt = []
    
    model.eval()
    for batch in tqdm(valid_loader):
        imgs, mask = batch
        with torch.no_grad():
            logits = model(imgs.to(device))
        loss = criterion(logits['out'], mask.to(device))
        valid_loss.append(loss.item())

        pred = torch.argmax(logits['out'], dim=1).detach().cpu().numpy().astype(np.int64)
        y = mask.detach().cpu().numpy().astype(np.int64)
        all_preds.append(pred)
        all_gt.append(y)

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    mIoU = m.mean_iou_score(np.concatenate(all_preds, axis=0), np.concatenate(all_gt, axis=0))
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, mIoU = {mIoU:.4f}")

    # save models
    if mIoU > best_miou:
        print(f"Successfully save the best model found at epoch {epoch}")
        torch.save(model.state_dict(), os.path.join(ckpt_path, f"{bast_model_path}")) # only save best to prevent output memory exceed error
        best_miou = mIoU
        stale = 0
        
    if epoch in save_epochs:
        torch.save(model.state_dict(), os.path.join(ckpt_path, f"model_{epoch}.pth"))
    print(f'best mIoU so far:{best_miou:.4f}')
    
    scheduler.step()
