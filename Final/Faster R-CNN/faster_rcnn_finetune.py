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
    def __init__(self, image_path, json_path):
            self.image_path = image_path
            self.json_path = json_path
            with open(json_path) as f:
                self.json_data = json.load(f)
            self.annotations = self.json_data['annotations']
            self.images = self.json_data['images']

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_name = self.images[idx]['file_name']
        annotation = self.annotations[idx]
        image = Image.open(os.path.join(self.image_path, image_name)).convert("RGB")
        image_tensor = T.ToTensor()(image)
        gt = annotation['bbox']
        gt = {"boxes": torch.tensor([gt], dtype=torch.int32), "labels": torch.tensor([3], dtype=torch.int64)}

        return image, image_tensor, gt

def validation_collate_fn(batch):
    images = [item[0] for item in batch]
    images_tensor = [item[1] for item in batch]
    gt = [item[2] for item in batch]
    return images, torch.stack(images_tensor, dim=0), gt

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
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4, collate_fn=validation_collate_fn)
    val_dataset = CustomDataset(val_image_path, val_json_path)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=validation_collate_fn)

    best_iou = 0

    #train
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_dataloader)
        pbar.set_description('Epoch %i' % epoch)
        
        # training
        for i, (image, image_tensor, target) in enumerate(pbar):
            optimizer.zero_grad()
            image_tensor = image_tensor.to(device)

            #data裡有一些bounding box的長寬為0，這些資料不要訓練
            next_or_not = False
            for t in target:
                if t['boxes'][0][2] == t['boxes'][0][0] or t['boxes'][0][3] == t['boxes'][0][1]:
                    next_or_not = True
                    break
            if next_or_not:
                continue
            
            target = [{k: v.to(device) for k, v in t.items()} for t in target]
            losses = model(image_tensor, target)
            losses = sum(loss for loss in losses.values())
            pbar.set_postfix(loss=losses.item())
            losses.backward()
            optimizer.step()
            total_loss += losses.item()
            writer.add_scalar('Training Loss', losses.item(), i)
        avg_loss = total_loss / len(train_dataloader)
        pbar.set_postfix(avg_loss=avg_loss)
        print(f"epoch: {epoch}, training loss: {avg_loss}")

        #validation
        model.eval()
        with torch.no_grad():
            iou = 0
            val_pbar = tqdm(val_dataloader)
            val_pbar.set_description('Validation')
            for i, (img, img_tensor, gt) in enumerate(val_pbar):
                gt = gt[0]['boxes'].tolist()[0]
                img_tensor = img_tensor.to(device)
                result = model(img_tensor)
                predict_boxes = [result[0]['boxes'][i].tolist() for i, score in enumerate(result[0]['scores'])]
                for i in range(len(predict_boxes)):
                    iou += torchvision.ops.box_iou(torch.tensor(predict_boxes[i]).unsqueeze(0), torch.tensor(gt).unsqueeze(0)).item()
                writer.add_scalar('Validation IOU', iou, i)
                val_pbar.set_postfix(iou=iou)
                if i % 1000 == 0:
                    draw = ImageDraw.Draw(img[0])
                    box = [[int(value) for value in predict_box] for predict_box in predict_boxes]

                    if len(box) > 0:
                        for b in box:
                            draw.rectangle(b, outline='red', width=2)
                    draw.rectangle(gt, outline='green', width=2)
                    img[0].save(f'./{i}_test.jpg')

            if iou > best_iou:
                print(f"iou improved from {best_iou} to {iou}")
                print("saving model...")
                name = int(iou)
                torch.save(model.state_dict(), f'./ckpt/faster_rcnn_{name}_model.pth')
                best_iou = iou

def testing(model):
    writer = SummaryWriter()
    #path
    testing_image_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/images/validation_images'
    testing_json_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/objects_val_fixed.json'

    #dataset
    testing_dataset = CustomDataset(testing_image_path, testing_json_path)
    testing_dataloader = DataLoader(testing_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=validation_collate_fn)

    model.eval()
    with torch.no_grad():
        iou = 0
        test_pbar = tqdm(testing_dataloader)
        test_pbar.set_description('Testing')
        for i, (img, img_tensor, gt) in enumerate(test_pbar):
            if i == 564:
                break
            gt = gt[0]['boxes'].tolist()[0]
            img_tensor = img_tensor.to(device)
            result = model(img_tensor)
            predict_boxes = [result[0]['boxes'][i].tolist() for i, score in enumerate(result[0]['scores'])]
            for j in range(len(predict_boxes)):
                iou += torchvision.ops.box_iou(torch.tensor(predict_boxes[j]).unsqueeze(0), torch.tensor(gt).unsqueeze(0)).item()
            writer.add_scalar('Testing IOU', iou, i)
            test_pbar.set_postfix(Total_IOU = iou)
            if i % 2 == 0:
                print(f"saved image {i}.jpg")
                draw = ImageDraw.Draw(img[0])
                box = [[int(value) for value in predict_box] for predict_box in predict_boxes]

                if len(box) > 0:
                    for b in box:
                        draw.rectangle(b, outline='red', width=2)
                draw.rectangle(gt, outline='green', width=2)
                img[0].save(f'./baseline_images/{i}_testing.jpg')


def main():
    # ----model----
    model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
            rpn_pre_nms_top_n_train = 2000,
            rpn_pre_nms_top_n_test = 1000,
            rpn_post_nms_top_n_train = 1000,
            rpn_post_nms_top_n_test = 20,
            box_positive_fraction = 0.5,
            rpn_nms_thresh = 0.15,
        )
    # freeze the backbone's parameters of the trainable model
    # for param in model.backbone.parameters():
    #     param.requires_grad = False
    model = model.to(device)


    # ----train----
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.005)
    # train(model, optimizer)


    # ----testing----
    # model.load_state_dict(torch.load('./ckpt/faster_rcnn_6135_model.pth'))
    testing(model)

if __name__ == "__main__":
    main()