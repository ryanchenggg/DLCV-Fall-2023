import torch
import torchvision
from torch import nn
from torchvision import transforms as T
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
import json
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw
import cv2

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation):
        self.root_dir = root_dir
        self.annotation = annotation
        
        # load annotations
        with open(annotation) as f:
            self.data = json.load(f)

    def __len__(self):
        return len(self.data['images'])
    
    def __getitem__(self, idx):
        img_information = self.data['images'][idx]
        img_path = os.path.join(self.root_dir, img_information['file_name'])
        image = Image.open(img_path).convert("RGB")
        image_tensor = T.ToTensor()(image)

        img_id = img_information['id']
        img_targets = [target for target in self.data['annotations'] if target['image_id'] == img_id]

        targets = {"boxes": torch.tensor([t['bbox'] for t in img_targets], dtype=torch.float32)}
        return img_id, image, image_tensor, targets
    
def collate_fn(batch):
    img_id = [item[0] for item in batch]
    images = [item[1] for item in batch]
    images_tensor = [item[2] for item in batch]
    targets = [item[3] for item in batch]
    return img_id, images, torch.stack(images_tensor, dim=0), targets

class backbone_n_rpn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
            rpn_pre_nms_top_n_train = 2000,
            rpn_pre_nms_top_n_test = 1000,
            rpn_post_nms_top_n_train = 500,
            rpn_post_nms_top_n_test = 5,
            rpn_score_thresh = 0.7
        )

        self.transform = self.model.transform
        self.backbone = self.model.backbone
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.rpn = self.model.rpn

    def forward(self, image, targets=None):
        image, targets = self.transform(image, targets) #targets is None
        features = self.backbone(image.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(image, features, targets)
        return proposals, proposal_losses

def validation(model):
    model.eval()
    writer = SummaryWriter()
    #path
    image_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/images/validation_images'
    json_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/objects_val_fixed.json'

    #dataset
    val_dataset = CustomDataset(image_path, json_path)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

    #validation
    iou = 0
    pbar = tqdm(val_dataloader)
    pbar.set_description('Validation')
    for i, (img_id, img, img_tensor, target) in enumerate(pbar):
        print(img_id)
        img_tensor = img_tensor.to(device)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        result = model(img_tensor)

        for i in range(result[0]['boxes'].shape[0]):
            iou += torchvision.ops.box_iou(result[0]['boxes'][i].unsqueeze(0), target[0]['boxes']).item()

            # visualize
            draw = ImageDraw.Draw(img[0])
            box = result[0]['boxes'][i]
            box.to('cpu').tolist()
            box = [int(b) for b in box]

            gt = target[0]['boxes'][0]
            gt.to('cpu').tolist()
            gt = [int(b) for b in gt]

            draw.rectangle(box, outline='red', width=2)
            draw.rectangle(gt, outline='green', width=2)
            img[0].save(f'./{i}_test.jpg')
        exit()

        writer.add_scalar('Validation IOU', iou, i)

    avg_iou = iou / len(val_dataloader)
    print(f"avg iou: {avg_iou}")

    writer.close()

def main():
    model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
            rpn_pre_nms_top_n_train = 2000,
            rpn_pre_nms_top_n_test = 1000,
            rpn_post_nms_top_n_train = 500,
            rpn_post_nms_top_n_test = 10,
            rpn_score_thresh = 0.7
        )
    model = model.to(device)
    validation(model)

if __name__ == "__main__":
    main()