import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
import torchvision
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR


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
        img_name = self.data_frame[idx][1]
        img_path = os.path.join(self.img_dir, img_name)
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"File not found: {img_path}")
            return None, None
        label = int(self.data_frame[idx][2])
        if self.transform:
            image = self.transform(image)
        return image, label

# Set device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Set transforms
transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Instantiate datasets and loaders
train_dataset = CustomDataset(csv_file='/home/ryan/dlcv/dlcv-fall-2023-hw1-ryanchenggg-main/hw1_data/p2_data/office/train.csv', img_dir='/home/ryan/dlcv/dlcv-fall-2023-hw1-ryanchenggg-main/hw1_data/p2_data/office/train', transform=transform)
val_dataset = CustomDataset(csv_file='/home/ryan/dlcv/dlcv-fall-2023-hw1-ryanchenggg-main/hw1_data/p2_data/office/val.csv', img_dir='/home/ryan/dlcv/dlcv-fall-2023-hw1-ryanchenggg-main/hw1_data/p2_data/office/val', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

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

# Load model
model = torchvision.models.resnet50(weights=None)
# model.fc = nn.Sequential(
#     nn.Linear(2048, 1024),
#     nn.ReLU(),
#     nn.Dropout(),
#     nn.Linear(1024, 512),
#     nn.ReLU(),
#     nn.Dropout(),
#     nn.Linear(512, 256),
#     nn.ReLU(),
#     nn.Dropout(),
#     nn.Linear(256, 65),
# )
model = model.to(device) 
classifier = Classifier()
classifier.to(device)
#The last step must syncronize to device


# Set loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=1e-3)
#scheduler = StepLR(optimizer, step_size=110, gamma=0.1)

# Set up arrays to store loss values
train_loss_values = []
val_loss_values = []
train_acc = []
val_acc = []
# Training loop
num_epochs = 500
best_val_loss = float('inf')
patience_counter = 0
best_acc = -1
#patience_threshold = 50  # Set this to your desired patience
print(classifier)
for epoch in range(num_epochs):
    model.train()
    classifier.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    # Training
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", unit="batch"):
        if images is None or labels is None:
            continue  # Skip batch if there was an error loading images
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        optimizer_classifier.zero_grad()
        outputs = model(images)
        outputs = classifier(outputs)

        loss = criterion(outputs, labels)
        loss.backward()

        optimizer.step()
        optimizer_classifier.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    #scheduler.step()
    #print(f"Epoch [{epoch+1}/{num_epochs}], Current Learning Rate: {scheduler.get_last_lr()[0]}")
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total   
    # Validation
    val_loss, val_correct, val_total = 0.0, 0, 0
    model.eval()
    classifier.eval()

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} - Validation", unit="batch"):
            if images is None or labels is None:
                continue  # Skip batch if there was an error loading images
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            outputs = classifier(outputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total  
    
    # # # Check if the current validation loss is lower than the best known
    # if avg_val_loss < best_val_loss:
    #     best_val_loss = avg_val_loss
    #     # patience_counter = 0  # Reset the counter
    #     # # Optionally, save the model at this point
    #     # torch.save(model.state_dict(), 'best_model.pth')
    #     print(best_val_loss)
    # else:
    #     patience_counter += 1  # Increment the counter if no improvement in validation loss
    
    # # Check if patience is exhausted
    # if patience_counter >= patience_threshold:
    #     print(f"Early stopping triggered: patience limit reached after {epoch+1} epochs")
    #     break  # Exit the loop to stop training

    if val_accuracy > best_acc:
        best_acc = val_accuracy
        print(f"New model successfully get better accuracy at epoch {epoch+1} with Validation Loss: {avg_val_loss:.4f} and Accuracy: {val_accuracy:.2f}%!")

    # Append loss values for this epoch
    train_loss_values.append(avg_train_loss)
    val_loss_values.append(avg_val_loss)
    train_acc.append(train_accuracy)
    val_acc.append(val_accuracy)
    # Print Summary
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

plot_accuracy(train_acc, val_acc, num_epochs, "ResNet50")
plot_loss(train_loss_values, val_loss_values, num_epochs, "ResNet50")