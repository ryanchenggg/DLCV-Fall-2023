import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torchvision.models as models
from torchvision.models import ResNet101_Weights
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

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

        return img, label, imgs[index]# Return filename as well

    def __len__(self):
        return len(self.ids)

def plot_loss(train_loss, val_loss, epoch):
    plt.clf()
    num = range(epoch)
    
    plt.plot(num,train_loss, label='training loss')
    plt.plot(num, val_loss, label='validation loss')
    plt.legend()
    plt.title('loss')
    plt.savefig('B_loss.png')

def plot_acc(train_acc, val_acc, epoch):
    plt.clf()
    num = range(epoch)
    
    plt.plot(num,train_acc, label='training acc')
    plt.plot(num, val_acc, label='validation acc')
    plt.legend()
    plt.title('acc')
    plt.savefig('B_acc.png')

def get_pretrained_model(num_classes):
    model = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)    
    return model


#data root
train_data = 'hw1_data/p1_data/train_50'
val_data = 'hw1_data/p1_data/val_50'

num_classes = 50
batch_size = 64
n_epochs = 50
print_every = 352
learning_rate = 0.0001

#mean, std = [0.5077, 0.4813, 0.4312], [0.2000, 0.1986, 0.2034]
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.RandomRotation(degrees=25),
    torchvision.transforms.RandomHorizontalFlip(p=0.8),
    torchvision.transforms.ColorJitter(contrast=(1, 1.5)),
    torchvision.transforms.ToTensor(), 
    #torchvision.transforms.Normalize(mean, std)
])

val_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(), 
])

#data loading
train_dataset = Dataset(root=train_data,transforms=transform)
val_dataset = Dataset(root=val_data,transforms=val_transform)
train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=12)
val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,shuffle=False,num_workers=12)

#model = CustomCNN(num_classes)
model = get_pretrained_model(num_classes)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f'Device:{device}')
model = model.to(device)

model_name = 'ResNet101(True)'
ckpt_path = f'./p1_B_checkpoint_{model_name}'

if not os.path.isdir(ckpt_path):
    os.makedirs(ckpt_path)


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#parameters
n_epochs = 100  # Set the number of epochs
best_acc = 0.0  # Initialize the best accuracy
train_loss = []  # List to store training loss per epoch
train_acc = []   # List to store training accuracy per epoch
val_loss = []    # List to store validation loss per epoch
val_acc = []     # List to store validation accuracy per epoch
early_stop = 0   # Counter to keep track of number of epochs with no improvement


# Training Loop
for epoch in range(n_epochs):
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    print(f'Epoch {epoch+1}/{n_epochs}')
    print('-' * 10)

    # Set the model to training mode
    model.train()
    for batch_idx, (data, target, _) in enumerate(tqdm(train_loader, desc="Training", unit="batch")):
        # Move data and target to the device
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        # Update running loss
        running_loss += loss.item()
        
        # Update correct predictions count
        _, pred = torch.max(outputs, dim=1)
        correct_train += (pred == target).sum().item()
        total_train += target.size(0)

    print(f'Epoch [{epoch+1}/{n_epochs}], Step [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    # Calculate and store average training loss and accuracy for the epoch
    train_loss.append(running_loss / len(train_loader))
    train_acc.append(100 * correct_train / total_train)
    print(f'Train Loss: {train_loss[-1]:.4f}, Train Accuracy: {train_acc[-1]:.4f}%')

    # Validation
    model.eval()
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    val_results = []  # List to store validation results (filename, predicted label)
    with torch.no_grad():
        for data_val, target_val , filenames in tqdm(val_loader, desc="Validation", unit="batch"):
            data_val, target_val = data_val.to(device), target_val.to(device)
            outputs_val = model(data_val)
            loss_val = criterion(outputs_val, target_val)
            running_val_loss += loss_val.item()
            _, pred_val = torch.max(outputs_val, dim=1)
            correct_val += (pred_val == target_val).sum().item()
            total_val += target_val.size(0)
            
            # Append filenames and predicted labels to val_results
            pred_labels = pred_val.cpu().numpy()
            for i, filename in enumerate(filenames):
                val_results.append((filename, pred_labels[i]))

    # Calculate and store average validation loss and accuracy for the epoch
    val_loss.append(running_val_loss / len(val_loader))
    val_acc.append(100 * correct_val / total_val)
    print(f'Validation Loss: {val_loss[-1]:.4f}, Validation Accuracy: {val_acc[-1]:.4f}%')

    # Save the model if validation accuracy improves
    if val_acc[-1] > best_acc:
        best_acc = val_acc[-1]
        torch.save(optimizer.state_dict(), os.path.join(
            ckpt_path, 'best_optimizer_B.pth'))
        torch.save(model.state_dict(), os.path.join(ckpt_path, 'best_model_B.pth'))
        print(f'Detected network improvement, Successfully saving current model with Validation Accuracy: {best_acc:.4f}%\n')
        
        # # Write validation results to CSV
        # with open('/home/ryan/dlcv/dlcv-fall-2023-hw1-ryanchenggg-main/p1/validation_results.csv', mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(['image_id', 'label'])
        #     writer.writerows(val_results)

    print(f'Best Validation Accuracy: {best_acc:.4f}%')
    print('====================================================================')

    if epoch > 0 and val_acc[-1] < best_acc:
        early_stop += 1
        if early_stop == 5:
            print('Early stopping')
            break
# Plotting loss and accuracy
plot_loss(train_loss, val_loss, n_epochs)
plot_acc(train_acc, val_acc, n_epochs)
print("Training complete!")
