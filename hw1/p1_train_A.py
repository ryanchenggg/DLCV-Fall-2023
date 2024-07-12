import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
import matplotlib.pyplot as plt


from PIL import Image
import glob
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(root_dir, "*.png"))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        label = int(os.path.basename(img_path).split('_')[0])

        if self.transform:
            image = self.transform(image)

        return image, label

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], last_lr: {:.5f}, train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch, result['lrs'][-1], result['train_loss'], result['val_loss'], result['val_acc']))

def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1), 
              nn.BatchNorm2d(out_channels), 
              nn.ReLU(inplace=True)]
    if pool: layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)

class ResNet9(ImageClassificationBase):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
        
        self.classifier = nn.Sequential(nn.MaxPool2d(4), 
                                        nn.Flatten(), 
                                        nn.Dropout(0.2),
                                        nn.Linear(512, num_classes))
        
    def forward(self, xb):
        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) +out
        out = self.classifier(out)
        return out
    
    def get_embedding(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        return out.view(out.size(0), -1)  # This should be the output of your second last layer

def extract_features(loader, model, device):
    features_list = []
    labels_list = []
    
    # Set the model to evaluation mode
    model.eval()
    
    # Iterate over the data in the loader
    with torch.no_grad():
        for images, labels in loader:
            # Move the images and labels to the device the model is on
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass to get the features
            features = model.extract_features(images)
            
            # Append the features and labels to the lists
            features_list.append(features.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            
    # Convert the lists of features and labels to numpy arrays
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.concatenate(labels_list, axis=0)
    
    return features_array, labels_array

mean, std = [0.5077, 0.4813, 0.4312], [0.2000, 0.1986, 0.2034]
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomRotation(30),
    # transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean, std)
])

#dataset
train_dataset = CustomDataset(root_dir="hw1_data/p1_data/train_50/", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

val_dataset = CustomDataset(root_dir="hw1_data/p1_data/val_50/", transform=transform)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
epochs = 300
plot_epochs = {100 * i for i in range(1, 4)} | {1}
best_acc = 0.0


# Instantiate the model
model = ResNet9(in_channels=3, num_classes=50).to(device)
print(model)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
model_name = 'ResNet9'
ckpt_path = f'./p1_A_checkpoint_{model_name}'

if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists("./p1_plots"):
    os.makedirs("./p1_plots")


# Training loop
num_epochs = 300
model.train()
global_step = 0
for epoch in range(num_epochs):
    
    # Initialize metrics for training
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for images, labels in tqdm(train_loader, total=len(train_loader), desc="Training", leave=True):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Accumulate training loss
        train_loss += loss.item()

        # Calculate training accuracy
        predicted = torch.argmax(outputs.data, dim=1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    # Calculate average training loss and accuracy
    avg_train_loss = train_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total   

    # Initialize metrics for validation
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Validation loop
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(val_loader, total=len(val_loader), desc="Validation", leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            predicted = torch.argmax(outputs.data, dim=1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()

    # Calculate average validation loss and accuracy
    avg_val_loss = val_loss / len(val_loader)
    val_accuracy = 100 * val_correct / val_total

    model.train()
    # plot
    if epoch in plot_epochs:
        with torch.no_grad():
            all_x = None
            all_y = None
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model.get_embedding(x)  # calling the second last layer
                if all_x is None:
                    all_x = out.detach().cpu().numpy()
                    all_y = y.detach().cpu().numpy().flatten()
                else:
                    out = out.detach().cpu().numpy()
                    y = y.detach().cpu().numpy().flatten()
                    all_x = np.vstack((all_x, out))
                    all_y = np.concatenate((all_y, y))

        # all_x: (2500, 192, 8, 8), all_y: (2500,)
        all_x = all_x.reshape(all_x.shape[0], -1)
        # plot PCA
        pca = PCA(n_components=2)
        d_x = pca.fit_transform(all_x)
        plt.figure()
        plt.title(f"PCA figure for epoch {epoch}")
        plt.scatter(d_x[:, 0], d_x[:, 1], c=all_y)
        plt.savefig(f"./p1_plots/{model_name}_PCA_{epoch}")

        # plot t-SNE
        tsne = TSNE(n_components=2)
        d_x = tsne.fit_transform(all_x)
        plt.figure()
        plt.title(f"t-SNE figure for epoch {epoch}")
        plt.scatter(d_x[:, 0], d_x[:, 1], c=all_y)
        plt.savefig(f"./p1_plots/{model_name}_TSNE_{epoch}")
    
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        torch.save(optimizer.state_dict(), os.path.join(
            ckpt_path, 'best_optimizer_A.pth'))
        torch.save(model.state_dict(), os.path.join(ckpt_path, 'best_model_A.pth'))
        print("new model saved  sucessfully!")

    print(f"best validation acc: {best_acc}%")
    # Print epoch-level summary
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")
    print('====================================================================')