import os
import json
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as T
import torchvision.transforms.functional as TF

# import RPN related modules
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.rpn import RPNHead, RegionProposalNetwork
from torchvision.models.detection.image_list import ImageList
from torchvision.ops import box_iou

import numpy as np
from collections import OrderedDict

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation):
        self.root_dir = root_dir
        self.annotation = annotation
        self.transform = T.Compose([
                            T.Resize((800, 800), interpolation=T.InterpolationMode.BILINEAR),
                            T.ToTensor()
                        ])
        # load annotations
        with open(annotation) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data['images'])
    
    def __getitem__(self, idx):
        img_information = self.data['images'][idx]
        img_path = os.path.join(self.root_dir, img_information['file_name'])
        image = Image.open(img_path).convert("RGB")
        image_tensor = self.transform(image)

        img_id = img_information['id']
        img_targets = [target for target in self.data['annotations'] if target['image_id'] == img_id]

        targets = {"boxes": torch.tensor([t['bbox'] for t in img_targets], dtype=torch.float32)}
        return image, image_tensor, targets
    
def collate_fn(batch):
    images = [item[0] for item in batch]
    images_tensor = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    return images,  torch.stack(images_tensor, dim=0), targets

class CustomRPN(nn.Module):
    def __init__(self):
        super(CustomRPN, self).__init__()
        self.model = models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(
            weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT            , 
            weights_backbone = models.ResNet50_Weights.DEFAULT,
            rpn_pre_nms_top_n_train = 2000,
            rpn_pre_nms_top_n_test = 1000,
            rpn_post_nms_top_n_train = 20,
            rpn_post_nms_top_n_test = 5,
            rpn_score_thresh = 0.9
        )

        self.transform = self.model.transform
        self.backbone = self.model.backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.rpn = self.model.rpn


    def forward(self, images, targets=None, visualize=False):
        # images, targets = self.transform(images, targets)
        features = self.backbone(images)

        image_list = ImageList(images, [image.shape[-2:] for image in images])
        if visualize:
            self.visualize_feature_map(images, features)

        # print(f"features.shape: {features.shape}")
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])

        boxes, losses = self.rpn(image_list, features, targets)
        return boxes, losses
        
    def visualize_feature_map(self, images, features):
        feature_to_visualize = features[0]  # choose the first image

        # choose the channel to visualize
        channel_to_show = 0
        feature_map = feature_to_visualize[channel_to_show].cpu().detach()
        feature_map = feature_map.unsqueeze(0).unsqueeze(0)
        # print(f"feature_map.shape: {feature_map.shape}" )
        
        # matching the feature map size to the input image size
        resized_feature_map = TF.resize(feature_map, images.shape[-2:])
        resized_feature_map = resized_feature_map.squeeze(0)
        # print(f"resized_feature_map.shape: {resized_feature_map.shape}" )

        # transform the feature map to a color map
        feature_map_colored = plt.get_cmap('hot')(resized_feature_map.numpy())[..., :3]

        # transform the input image to numpy array
        original_image = images[0].permute(1, 2, 0).cpu().numpy()
        # print(f"original_image.shape: {original_image.shape}" )
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())
        # sprint(f"original_image.shape: {original_image.shape}")

        # overlay the feature map color on the input image
        overlayed_image = 0.6 * original_image + 0.4 * feature_map_colored
        overlayed_image = overlayed_image.squeeze(0)

        # save the overlayed image
        plt.imshow(overlayed_image)
        plt.axis('off')
        plt.savefig('overlayed_image.png')
        plt.close()

def load_model(ckpt_path):
    model = CustomRPN()
    model.load_state_dict(torch.load(ckpt_path))
    return model

def main():
    ckpt_save_path = './ckpt_rpn'
    if not os.path.exists(ckpt_save_path):
        os.makedirs(ckpt_save_path)

    # path
    train_root = 'DLCV_vq2d_data/images/training_images'
    valid_root = 'DLCV_vq2d_data/images/validation_images'
    train_json = 'DLCV_vq2d_data/objects_train_fixed.json'
    val_json = 'DLCV_vq2d_data/objects_val_fixed.json'
    # dataset
    train_dataset = CustomDataset(train_root, train_json)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_dataset = CustomDataset(valid_root, val_json)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, num_workers=0, collate_fn=collate_fn)

    # init model
    load = False
    if load:
        ckpt_path = './ckpt_rpn/epoch2_iou0.00499950535595417.pth'
        model = load_model(ckpt_path).to(device)
    else:
        model = CustomRPN().to(device)
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)

    num_epoch = 10
    min_iou = 0
    for epoch in range(1, num_epoch+1):
        total_loss = 0
        model.train()
        pbar = tqdm(train_dataloader)
        pbar.set_description(f"Epoch {epoch}")
        for _, images, targets in pbar:
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]   

            boxes, losses = model(images, targets)
            # calculate loss
            # print(f"boxes: {boxes}")
            loss_objectness = losses['loss_objectness']
            loss_rpn_box_reg = losses['loss_rpn_box_reg']
            losses = loss_objectness + loss_rpn_box_reg

            # backward propagation
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # training loss
            pbar.set_postfix(loss=losses.item())
            total_loss += losses.item()

        average_loss = total_loss / len(train_dataloader)
        print(f'Epoch {epoch} average training loss: {average_loss:.4f}')

        # validation
        model.eval()
        ious = []
        pbar = tqdm(val_dataloader)
        pbar.set_description(f"Validation")
        with torch.no_grad():
            for _, images, targets in pbar:
                images = images.to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]   

                boxes, _ = model(images, targets) # losses is none
                # val loss

                # in theory, we should calculate the iou after extracting features from the query image
                for box, target in zip(boxes, targets):
                    gt_boxes = target['boxes'].to(device)
                    iou = box_iou(box, gt_boxes)
                    ious.append(iou)
        
        avg_iou = torch.cat(ious).mean().item()
        print(f"Validation IOU: {avg_iou}")
        if avg_iou > min_iou:
            min_iou = avg_iou
            torch.save(model.state_dict(), os.path.join(ckpt_save_path, f'epoch{epoch}_iou{avg_iou}.pth'))
            print(f"Save model at epoch {epoch} with iou {avg_iou:.4f}")

if __name__ == '__main__':
    main()