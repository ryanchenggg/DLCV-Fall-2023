import sys
import torch
from torchvision import models
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import csv
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip the header
            for row in reader:
                self.data_frame.append(row)
        self.img_dir = img_dir
        self.transform = transform
    def __len__(self):
        return len(self.data_frame)
    def __getitem__(self, idx):
        img_name = str(self.data_frame[idx][1])
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            return None, None
        label = int(self.data_frame[idx][2])
        if self.transform:
            image = self.transform(image)
        img_filename = os.path.basename(img_name)
        return image, img_filename, label
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(1000, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 65)
        )
    def forward(self, x):
        x = self.linear(x)
        return x
def test(model, test_loader, device):
    model.eval()
    prediction = []
    for i, (imgs, img_filenames, labels) in enumerate(test_loader):
        imgs = imgs.to(device)
        with torch.no_grad():
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            for img_filename, pred, label in zip(img_filenames, preds.cpu().numpy(), labels):
                prediction.append((img_filename, pred, label))
    return prediction
def calculate_accuracy(predictions):
    correct = sum(1 for img_filename, pred, label in predictions if pred == label)
    total = len(predictions)
    accuracy = correct / total * 100.0
    return accuracy
########################################################
val_csv = sys.argv[1]
val_root = sys.argv[2]
output = sys.argv[3]
########################################################
weight_backbone = './p2_best_model_C.pth'
weight_classifier = './p2_best_classifier_C.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
#validation setting
val_dataset = CustomDataset(csv_file=val_csv, img_dir=val_root, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

#model setting
model = models.resnet50(pretrained=False)
checkpoint = torch.load(weight_backbone)
model.load_state_dict(checkpoint)
model.to(device)
classifier = Classifier()
checkpoint = torch.load(weight_classifier)
classifier.load_state_dict(checkpoint)
classifier.to(device)
#testing
predictions = test(model, val_loader, device)
with open(output, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['id', 'filename', 'label'])
    for i, (img_name, pred, label) in enumerate(predictions):
        img_name_str = os.path.basename(str(img_name))
        writer.writerow([i, img_name_str, pred.item()])
val_accuracy = calculate_accuracy(predictions)
#print(f"Validation Accuracy: {val_accuracy:.2f}%")
print('Done!')