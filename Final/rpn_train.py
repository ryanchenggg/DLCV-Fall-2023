import os
import json
from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms

# import RPN related modules
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation, transform=None):
        self.root_dir = root_dir
        self.transform = transform or transforms.Compose([
            transforms.Resize((800, 800)),
            transforms.ToTensor()
        ])
        self.annotation = annotation
        
        self.counter = 0
        # load annotations
        with open(annotation) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data['images'])
    
    def __getitem__(self, idx):

        img_information = self.data['images'][idx]
        img_path =  os.path.join(self.root_dir, img_information['file_name'])
        image = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        img_id = img_information['id']
        img_targets = [target for target in self.data['annotations'] if target['image_id'] == img_id]

        target = {
            'boxes': torch.tensor([x['bbox'] for x in img_targets], dtype=torch.float32)
        }
        self.counter += 1
        print(f"counter: {self.counter}")
        return image, target

class CustomRPN(nn.Module):
    def __init__(self):
        super(CustomRPN, self).__init__()
        # load backbone
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # remove the last two layers of resnet50
        modules = list(backbone.children())[:-2]
        self.backbone = nn.Sequential(*modules)

        # ResNet-50 last channel will be 2048
        backbone_out_channel = 2048
        
        # RPN setup(anchor_generator, head)
        anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                           aspect_ratios=((0.5, 1.0, 2.0),) * 5)
        
        rpn_head = RPNHead(backbone_out_channel, anchor_generator.num_anchors_per_location()[0])
        
        self.rpn = RegionProposalNetwork(anchor_generator, rpn_head, 
                                         fg_iou_thresh=0.7, bg_iou_thresh=0.3,
                                         batch_size_per_image=256, positive_fraction=0.5,
                                         pre_nms_top_n={'training': 2000, 'testing': 1000},
                                         post_nms_top_n={'training': 2000, 'testing': 1000},
                                         nms_thresh=0.7)
    
    def forward(self, images):
        features = self.backbone(images)
        proposals, _ = self.rpn(images, features)
        return proposals

def train_one_epoch(model, data_loader, optimizer, device):
    
    model.train()
    for images, targets in data_loader:
        
        images = list(img.to(device) for img in images)

        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        # 計算RPN的輸出和損失
        proposals, losses = model(images, targets)

        # 總損失
        loss = losses['loss_objectness'] + losses['loss_rpn_box_reg']

        # 反向傳播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # training loss
        print(f"train_loss: {loss.item()}")

train_root = 'DLCV_vq2d_data/images/training_images'
valid_root = 'DLCV_vq2d_data/images/validation_images'
train_dataset = CustomDataset(train_root, 'DLCV_vq2d_data/objects_train_fixed.json')
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# init model
model = CustomRPN().to(device)
model.train()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

num_epoch = 10
for epoch in range(num_epoch):
    for imgs in tqdm(train_dataloader):
        # imgs = imgs.to(device)
        # optimizer.zero_grad()
        # proposals = model(imgs)
        # loss = sum(loss for loss in proposals.values())
        # loss.backward()
        # optimizer.step()
        train_one_epoch(model, train_dataloader, optimizer, device)
        print(f"Epoch {epoch} completed")