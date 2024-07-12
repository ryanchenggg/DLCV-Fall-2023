import os

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as trns
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm
from torchvision.models import VGG16_Weights, vgg16
from p3_dataloader import p3_dataset

def mean_iou(pred_mask, mask):
    pred_mask = F.softmax(pred_mask, dim=1)
    pred_mask = torch.argmax(pred_mask, dim=1)
    pred_mask = pred_mask.cpu().numpy()
    #mask = mask.cpu().numpy()
    mean_iou = 0
    for i in range(6):  # Adjust range if you have a different number of classes
        tp_fp = np.sum(pred_mask == i)
        tp_fn = np.sum(mask == i)
        tp = np.sum((pred_mask == i) * (mask == i))
        if (tp_fp + tp_fn) == 0:
            mean_iou += 1/6
            continue
        iou = tp / (tp_fp + tp_fn - tp)
        mean_iou += iou / 6
    return mean_iou

# load data
mean = [0.485, 0.456, 0.406]  # imagenet
std = [0.229, 0.224, 0.225]

train_set = p3_dataset("./hw1_data/p3_data/train", task="train")
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
valid_set = p3_dataset("./hw1_data/p3_data/validation", task="test")
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

device = torch.device('cuda')
batch_size = 8
epochs = 100
best_loss = 5.0
ckpt_path = f'./p3_A_checkpoint'
best_val_iou = -1

# Define the FCN32s model
class FCN32s(nn.Module):
    def __init__(self, n_classes):
        super(FCN32s, self).__init__()
        vgg16 = models.vgg16(weights=VGG16_Weights.DEFAULT)
        features = list(vgg16.features.children())
        self.features = nn.Sequential(*features[:23])
        self.classifier = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=7),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Conv2d(4096, n_classes, kernel_size=1)
        )
        # ... (other layers) ...
        self.deconv = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=64, stride=32, padding=16, bias=False)
        self.upsample = nn.Upsample(size=(512, 512), mode='bilinear', align_corners=True)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = self.upsample(x) # Upsample the output to match target size
        #x = self.deconv(x)
        return x
    
# model
n_classes = 7
net = FCN32s(n_classes=n_classes).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9)
best_model_path = '3_model_A.pth'

if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

# Training loop
for epoch in range(1, epochs + 1):
    net.train()
    # Training step
    for x, y in tqdm(train_loader, desc=f"Training Epoch {epoch}", unit="batch"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = net(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

    # Validation step
    net.eval()
    va_loss = 0
    ACCs = []
    val_miou = []
    with torch.no_grad():
        for x, y in tqdm(valid_loader, desc=f"Validating Epoch {epoch}", unit="batch"):
            x, y = x.to(device), y.to(device)
            out = net(x)
            va_loss += F.cross_entropy(out, y).item()
            pred = out.argmax(dim=1).cpu().numpy().astype(np.int64)
            y = y.cpu().numpy().astype(np.int64)
            ACCs.append(np.mean(pred == y))
            val_miou.append(mean_iou(out, y))
        va_loss /= len(valid_loader)
        acc = np.mean(ACCs)
        miou = np.mean(val_miou)

    # Print and save results
    print(f"Epoch {epoch}, Acc = {acc:.4f}, Val Loss = {va_loss:.4f}, mIoU = {miou:.4f}")
    if miou > best_val_iou:
        best_val_iou = miou
        torch.save(net.state_dict(), best_model_path)
        print(f"Improved mIoU! Saving model to {best_model_path}.")
    print(f"Best mIoU so far: {best_val_iou:.4f}")

    # Save checkpoints periodically
    if epoch % 10 == 0 or epoch == 1:
        checkpoint_path = os.path.join(ckpt_path, f'{epoch}_model.pth')
        torch.save(net.state_dict(), checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}.")