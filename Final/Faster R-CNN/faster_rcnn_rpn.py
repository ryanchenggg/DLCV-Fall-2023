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
        return image, image_tensor, targets
    
def collate_fn(batch):
    images = [item[0] for item in batch]
    images_tensor = [item[1] for item in batch]
    targets = [item[2] for item in batch]
    return images, torch.stack(images_tensor, dim=0), targets

class backbone_n_rpn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
            rpn_pre_nms_top_n_train = 2000,
            rpn_pre_nms_top_n_test = 1000,
            rpn_post_nms_top_n_train = 500,
            rpn_post_nms_top_n_test = 10,
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

def train(model, optimizer, epochs=10):
        model.train()
        writer = SummaryWriter()
        #path
        train_image_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/images/training_images'
        train_json_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/objects_train_fixed.json'
        val_image_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/images/validation_images'
        val_json_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/objects_val_fixed.json'

        #dataset
        train_dataset = CustomDataset(train_image_path, train_json_path)
        train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)
        val_dataset = CustomDataset(val_image_path, val_json_path)
        val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

        best_iou = 0

        #train
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            pbar = tqdm(train_dataloader)
            pbar.set_description('Epoch %i' % epoch)
            for i, (img, img_tensor, target) in enumerate(pbar):
                img_tensor = img_tensor.to(device)
                target = [{k: v.to(device) for k, v in t.items()} for t in target]
                proposals, proposal_losses = model(img_tensor, target)

                #visualize
                # draw = ImageDraw.Draw(img[0])
                # for box in proposals[0][:10]:
                #     box.to('cpu').tolist()
                #     box = [int(b) for b in box]
                #     draw.rectangle(box, outline='red', width=2)

                # img[0].save('./test.jpg')

                loss_objectness = proposal_losses['loss_objectness']
                loss_rpn_box_reg = proposal_losses['loss_rpn_box_reg']
                loss = loss_objectness + loss_rpn_box_reg
                pbar.set_postfix(loss=loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss

                writer.add_scalar('Training Loss', loss, i)

            avg_loss = total_loss / len(train_dataloader)
            print(f"epoch: {epoch}, training loss: {avg_loss}")

            #validation
            model.eval()
            with torch.no_grad():
                iou = 0
                pbar = tqdm(val_dataloader)
                pbar.set_description('Validation')
                for i, (img, img_tensor, target) in enumerate(pbar):
                    img_tensor = img_tensor.to(device)
                    target = [{k: v.to(device) for k, v in t.items()} for t in target]
                    proposals, _ = model(img_tensor)

                    for i in range(proposals[0].shape[0]):
                        iou += torchvision.ops.box_iou(proposals[0][i].unsqueeze(0), target[0]['boxes']).item()
                    writer.add_scalar('Validation IOU', iou, i)
                    pbar.set_postfix(iou=iou)

                if iou > best_iou:
                    print(f"iou improved from {best_iou} to {iou}")
                    print("saving model...")
                    torch.save(model.state_dict(), './ckpt/rpn_best_model.pth')
                    best_iou = iou

                avg_iou = iou / len(val_dataloader)
                print(f"avg iou: {avg_iou}")

        writer.close()

def validation(model):
    model.eval()
    writer = SummaryWriter()
    #path
    image_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/images/validation_images'
    json_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/objects_val_fixed.json'

    #dataset
    val_dataset = CustomDataset(image_path, json_path)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=collate_fn)

    #validation
    iou = 0
    pbar = tqdm(val_dataloader)
    pbar.set_description('Validation')
    for i, (img, img_tensor, target) in enumerate(pbar):
        img_tensor = img_tensor.to(device)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        proposals, _ = model(img_tensor)

        for i in range(proposals[0].shape[0]):
            iou += torchvision.ops.box_iou(proposals[0][i].unsqueeze(0), target[0]['boxes']).item()

            # visualize
            draw = ImageDraw.Draw(img[0])
            box = proposals[0][i]
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
    # #train
    rpn = backbone_n_rpn().to(device)
    optimizer = torch.optim.SGD(rpn.parameters(), lr=0.005)
    train(rpn, optimizer)

    #test
    # rpn = backbone_n_rpn().to(device)
    # validation(rpn)

if __name__ == "__main__":
    main()