import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.models as models
from torchvision.models import ResNet101_Weights
from PIL import Image
import matplotlib.pyplot as plt
import csv
import torch.nn.functional as F
class Dataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        super(Dataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.ids = list(os.listdir(root))
    def __getitem__(self, index):
        imgs = os.listdir(self.root)
        img = Image.open(os.path.join(self.root, imgs[index]))
        label = int(imgs[index].split('_')[0])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label
    def __len__(self):
        return len(self.ids)
def test(model, test_loader, device):
    model.eval()
    predictions = []
    for batch in test_loader:
        images, _ = batch
        images = images.to(device)
        with torch.no_grad():
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds.cpu().numpy())
    return predictions

def get_pretrained_model(num_classes):
    model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)    
    return model

    
def calculate_accuracy(predictions, true_labels):
    correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
    total = len(true_labels)
    accuracy = correct / total * 100.0
    return accuracy

#############################################################
val_root = sys.argv[1]
output_root = sys.argv[2]
weight = './p1_best_model_B.pth'  
#############################################################

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
])

print('Loading data...')
val_loader = torch.utils.data.DataLoader(Dataset(val_root, val_transform), batch_size=64, shuffle=False)
model = get_pretrained_model(50)
model.load_state_dict(torch.load(weight))
model.to(device)

print('Start predicting...')
predictions = test(model, val_loader, device)

print('Writing predictions to', output_root)
with open(output_root, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['image_id', 'label'])
    for i, pred in zip(os.listdir(val_root), predictions):
        writer.writerow([i, pred])

true_labels = [int(img.split('_')[0]) for img in os.listdir(val_root)]
val_accuracy = calculate_accuracy(predictions, true_labels)
print(f"Validation Accuracy: {val_accuracy:.3f}%")