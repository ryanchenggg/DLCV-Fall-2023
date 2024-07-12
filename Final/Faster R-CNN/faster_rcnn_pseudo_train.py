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

class TrainingDataset(Dataset):
    def __init__(self, video_files_path, annotation_files_path):

        self.video_files_path = video_files_path
        self.qannotation_files_path = annotation_files_path
        self.video_name_list = os.listdir(annotation_files_path)
        
    def __len__(self):
        return len(self.video_name_list)
    
    def __getitem__(self, idx):
        json_file_name = self.video_name_list[idx]
        video_file_name = json_file_name.split('.')[0] + '.mp4'
        with open(os.path.join(self.qannotation_files_path, json_file_name)) as f:
            annotations = json.load(f)
        
        return video_file_name, annotations
    
class ValidationDataset(Dataset):
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
    video_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/clips'
    train_json_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/video_with_pesudo_label/'
    val_image_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/images/validation_images'
    val_json_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/objects_val_fixed.json'

    #dataset
    train_dataset = TrainingDataset(video_path, train_json_path)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_dataset = ValidationDataset(val_image_path, val_json_path)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=validation_collate_fn)

    best_iou = 0

    #train
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_dataloader)
        pbar.set_description('Epoch %i' % epoch)
        
        # 取得此video的所有frame
        for i, (video_name, frame_n_bbox) in enumerate(pbar):
            if i == 1:
                break
            count = 0
            image_list = {}
            video = cv2.VideoCapture(os.path.join(video_path, video_name[0]))
            if not video.isOpened():
                print("Error opening video stream or file")
            while video.isOpened():
                success, frame = video.read()
                if success:
                    image_tensor = T.ToTensor()(frame)
                    image_tensor = image_tensor.unsqueeze(0)
                    image_tensor = image_tensor.to(device)
                    image_list[str(count)] = image_tensor
                    count += 1
                else:
                    break
            video.release()

            train_pbar = tqdm(range(count))
            train_pbar.set_description('Training')
            for j in train_pbar:
                optimizer.zero_grad()
                image_tensor = image_list[str(j)]
                target = frame_n_bbox[str(j)]
                if len(target) != 0:
                    target = [
                        {
                            'boxes': torch.tensor([[tensor.item() for tensor in inner_list]], dtype=torch.float32).to(device), 
                            'labels': torch.tensor([3], dtype=torch.int64).to(device) #使用"hongfa"這個conda env, 我把classification loss拔掉了，所以這邊的label沒有用處
                        } for inner_list in target
                    ]
                else:
                    target = [
                        {
                            "boxes": torch.tensor([[1, 1, 2, 2]], dtype=torch.float32).to(device), #暫時用1,1,2,2來代替沒有bbox的frame，但應該要改裡面的loss的計算方式  
                            'labels': torch.tensor([3], dtype=torch.int64).to(device) #使用"hongfa"這個conda env, 我把classification loss拔掉了，所以這邊的label沒有用處
                        }
                    ] 
                losses = model(image_tensor, target)
                losses = sum(loss for loss in losses.values())
                train_pbar.set_postfix(loss=losses.item())
                losses.backward()
                optimizer.step()
                total_loss += losses.item()
                writer.add_scalar('Training Loss', losses.item(), j)
        avg_loss = total_loss / (len(train_dataloader) * 1502)
        pbar.set_postfix(avg_loss=avg_loss)
        print(f"epoch: {epoch}, training loss: {avg_loss}")

        #validation
        model.eval()
        with torch.no_grad():
            iou = 0
            val_pbar = tqdm(val_dataloader)
            val_pbar.set_description('Validation')
            for i, (img, img_tensor, gt) in enumerate(val_pbar):
                img_tensor = img_tensor.to(device)
                result = model(img_tensor)
                predict_boxes = [result[0]['boxes'][i].tolist() for i, score in enumerate(result[0]['scores'])]
                for i in range(len(predict_boxes)):
                    iou += torchvision.ops.box_iou(torch.tensor(predict_boxes[i]).unsqueeze(0), torch.tensor(gt[0]).unsqueeze(0)).item()
                writer.add_scalar('Validation IOU', iou, i)
                val_pbar.set_postfix(iou=iou)
                if i % 1000 == 0:
                    draw = ImageDraw.Draw(img[0])
                    box = [[int(value) for value in predict_box] for predict_box in predict_boxes]

                    gt = gt[0]
                    gt = [int(b) for b in gt]

                    if len(box) > 0:
                        for b in box:
                            draw.rectangle(b, outline='red', width=2)
                    draw.rectangle(gt, outline='green', width=2)
                    img[0].save(f'./{i}_test.jpg')

            if iou > best_iou:
                print(f"iou improved from {best_iou} to {iou}")
                print("saving model...")
                name = int(iou)
                torch.save(model.state_dict(), f'./ckpt/faster_rcnn_pseudo_{name}_model.pth')
                best_iou = iou
        

            

# def validation(model):
#     model.eval()
#     writer = SummaryWriter()
#     #path
#     video_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/DLCV_vq2d_data/clips'
#     json_path = '/home/ryan/dlcv/DLCV-Fall-2023-Final-2-boss-sdog/faster_rcnn_rpn/frame_with_gt_train.json'

#     #dataset
#     val_dataset = CustomDataset(video_path, json_path)
#     val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False)
#     for i, (video_name, gt_frame_n_bbox) in enumerate(val_dataloader):
#         count = 0
#         image_list = {}
#         target_list = gt_frame_n_bbox
#         video = cv2.VideoCapture(os.path.join(video_path, video_name[0]))
#         if not video.isOpened():
#             print("Error opening video stream or file")
#         while video.isOpened():
#             success, frame = video.read()
#             if success:
#                 image_tensor = T.ToTensor()(frame)
#                 image_tensor = image_tensor.unsqueeze(0)
#                 image_tensor = image_tensor.to(device)
#                 image_list[str(count)] = image_tensor
#                 result = model(image_tensor)
#                 pesudo_bboxs = [result[0]['boxes'][i].tolist() for i, score in enumerate(result[0]['scores']) if score > 0.8]
#                 pesudo_bboxs = [[int(x) for x in box] for box in pesudo_bboxs] # [[x1, y1, x2, y2], ...]]
#                 if str(count) not in target_list.keys() and len(pesudo_bboxs) > 0:
#                     target_list[str(count)] = pesudo_bboxs
#                 elif str(count) in target_list.keys() and len(pesudo_bboxs) > 0:
#                     for pesudo_bbox in pesudo_bboxs:
#                         target_list[str(count)].append(pesudo_bbox)
#                 elif str(count) not in target_list.keys() and len(pesudo_bboxs) == 0:
#                     target_list[str(count)] = [[0, 0, 0, 0]]
#                 count += 1
#             else:
#                 break
#         video.release()

            


#     #validation
#     iou = 0
#     pbar = tqdm(val_dataloader)
#     pbar.set_description('Validation')
#     for i, (img, img_tensor, target) in enumerate(pbar):
#         img_tensor = img_tensor.to(device)
#         target = [{k: v.to(device) for k, v in t.items()} for t in target]
#         proposals, _ = model(img_tensor)

#         for i in range(proposals[0].shape[0]):
#             iou += torchvision.ops.box_iou(proposals[0][i].unsqueeze(0), target[0]['boxes']).item()

#             # visualize
#             draw = ImageDraw.Draw(img[0])
#             box = proposals[0][i]
#             box.to('cpu').tolist()
#             box = [int(b) for b in box]

#             gt = target[0]['boxes'][0]
#             gt.to('cpu').tolist()
#             gt = [int(b) for b in gt]

#             draw.rectangle(box, outline='red', width=2)
#             draw.rectangle(gt, outline='green', width=2)
#             img[0].save(f'./{i}_test.jpg')
#         exit()

#         writer.add_scalar('Validation IOU', iou, i)

#     avg_iou = iou / len(val_dataloader)
#     print(f"avg iou: {avg_iou}")

#     writer.close()

def main():
    # #train
    model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
            weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
            rpn_pre_nms_top_n_train = 2000,
            rpn_pre_nms_top_n_test = 1000,
            rpn_post_nms_top_n_train = 500,
            rpn_post_nms_top_n_test = 10,
            rpn_score_thresh = 0.7
        )

    # freeze the backbone's parameters of the trainable model
    for param in model.backbone.parameters():
        param.requires_grad = False

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005)
    train(model, optimizer)

    #test
    # model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(
    #         weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, 
    #         weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT,
    #         rpn_pre_nms_top_n_train = 2000,
    #         rpn_pre_nms_top_n_test = 1000,
    #         rpn_post_nms_top_n_train = 500,
    #         rpn_post_nms_top_n_test = 10,
    #         rpn_score_thresh = 0.7
    #     )
    # model = model.to(device)
    # validation(model)

if __name__ == "__main__":
    main()