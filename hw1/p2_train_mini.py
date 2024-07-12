import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
from byol_pytorch_t import BYOL
from torchvision import models
import torch_optimizer as optim
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import warmup_schedular

##backbone training refer to the warmup method from:
#github: https://github.com/pha123661/NTU-2022Fall-DLCV/tree/master/HW4

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess the Mini-ImageNet dataset
transform = transforms.Compose([
    transforms.RandomResizedCrop(128),
    # transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(128),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

full_dataset = datasets.ImageFolder('/home/ryan/dlcv/dlcv-fall-2023-hw1-ryanchenggg-main/hw1_data/p2_data/mini', transform=transform)

# Split the dataset into training and validation sets
train_size = int(1 * len(full_dataset))  # 100% for training
val_size = len(full_dataset) - train_size  # 20% for validation
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

# Create data loaders for training and validation
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=4)

# Initialize ResNet50 with no pretrained weights and BYOL learner
resnet = models.resnet50(weights=None).to(device)
learner = BYOL(
    resnet, 
    image_size=128, 
    hidden_layer='avgpool',
    ).to(device)

# Set optimizer
opt = torch.optim.Adam(learner.parameters(), lr=3e-4)
# opt = optim.SWATS(
#     resnet.parameters(),
#     lr=1e-1,
#     betas=(0.9, 0.999),
#     eps=1e-3,
#     weight_decay= 0.0,
#     amsgrad=False,
#     nesterov=False,
# )
# Set scheduler with warm-up and cosine annealing
warmup_epochs = 10  # Set your desired number of warm-up epochs
total_epochs = 300  # Total number of epochs
# scheduler = warmup_schedular.GradualWarmupScheduler(
#     opt,
#     multiplier=1,
#     total_epoch=warmup_epochs,
#     after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
#         opt,
#         T_max=total_epochs * len(train_loader) - warmup_epochs * len(train_loader)
#     )
# )

best_loss = float('inf')
patience = 30  # Set your desired patience level
counter = 0

# Training loop
for epoch in range(total_epochs):  # Number of epochs
    epoch_loss = 0.0
    for images, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch"):
        images = images.to(device)
        loss = learner(images)  # Compute loss for the current batch
        epoch_loss += loss.item()  # Accumulate loss for the epoch
        opt.zero_grad()  # Zero the parameter gradients
        loss.backward()  # Backward pass
        opt.step()  # Optimize
        learner.update_moving_average()  # Update moving average of target encoder

    #scheduler.step()  # Step the scheduler at each epoch end

    # Print average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Loss: {avg_epoch_loss:.4f}")

    # Early stopping logic
    if avg_epoch_loss < best_loss:
        best_loss = avg_epoch_loss
        print("Best model saving")
        torch.save(resnet.state_dict(), './p2_backbone.pt')
        counter = 0  # Reset counter when new best loss is found
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break  # Terminate training loop
    
