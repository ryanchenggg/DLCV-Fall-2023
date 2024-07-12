import glob
import os
import random
from copy import deepcopy

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import hflip, vflip


# ... [Dataset and Configuration parts remain unchanged]
class p3_dataset(Dataset):

    def __init__(self, path, task="test"):
        super(Dataset, self).__init__()

        self.img_files = sorted(glob.glob(os.path.join(path, "*_sat.jpg")))
        self.mask_files = sorted(glob.glob(os.path.join(path, "*_mask.png")))
        self.task = task


    def __len__(self):
        return len(self.img_files)
    
    def training_transform(self, img, mask):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if random.random() > 0.5:
            img, mask = hflip(img), hflip(mask)
        if random.random() > 0.5:
            img, mask = vflip(img), vflip(mask)

        # Convert to Tensor
        # img, mask = F.to_tensor(img), F.to_tensor(mask)
        if not isinstance(img, torch.Tensor):
            img = F.to_tensor(img)
        if not isinstance(mask, torch.Tensor):
            mask = F.to_tensor(mask)

        # Normalize (for ResNet)
        normalize = transforms.Normalize(mean, std)
        img = normalize(img)

        return img, mask
    
    def validation_transform(self, img, mask):
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # Convert to Tensor
        img, mask = F.to_tensor(img), F.to_tensor(mask)

        # Normalize (for ResNet)
        normalize = transforms.Normalize(mean, std)
        img = normalize(img)

        return img, mask


    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        mask_path = self.mask_files[idx]

        img = Image.open(img_path)
        mask = Image.open(mask_path)
        mask = np.array(mask)
        mask = (mask >= 128).astype(int)
        mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
        raw_mask = mask.copy()

        mask[raw_mask == 3] = 0  # (Cyan: 011) Urban land
        mask[raw_mask == 6] = 1  # (Yellow: 110) Agriculture land
        mask[raw_mask == 5] = 2  # (Purple: 101) Rangeland
        mask[raw_mask == 2] = 3  # (Green: 010) Forest land
        mask[raw_mask == 1] = 4  # (Blue: 001) Water
        mask[raw_mask == 7] = 5  # (White: 111) Barren land
        mask[raw_mask == 0] = 6  # (Black: 000) Unknown
        mask = torch.tensor(mask)

        if self.task == "train":
            img, mask = self.training_transform(img, mask)
        else:
            img, mask = self.training_transform(img, mask)

        return img, mask



if __name__ == '__main__':
    import torch
    from torchvision import transforms
    dst = P3_Dataset('/home/ryan/dlcv/dlcv-fall-2023-hw1-ryanchenggg-main/hw1_data/p3_data/train',
                     transform=transforms.ToTensor(), train=True, augmentation=True)
    