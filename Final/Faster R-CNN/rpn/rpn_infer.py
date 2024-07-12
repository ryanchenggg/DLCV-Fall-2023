import os
import json
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
import torch
import torchvision.models as models
from torchvision import transforms as T
from rpn_train import CustomRPN

from torchvision.ops import box_iou
# Import RPN and other necessary modules as before

class CustomDataset(Dataset):
    def __init__(self, root_dir, annotation, indices=None):
        self.root_dir = root_dir
        self.annotation = annotation
        self.transform = T.Compose([
                            T.Resize((800, 800), interpolation=T.InterpolationMode.BILINEAR),
                            T.ToTensor()
                        ])
        # load annotations
        with open(annotation) as f:
            self.data = json.load(f)
            
        if indices is not None:
            self.data['images'] = [self.data['images'][i] for i in indices]
            self.data['annotations'] = [self.data['annotations'][i] for i in indices]

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

def load_model(ckpt_path):
    model = CustomRPN()
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()
    return model

def infer(model, dataloader, device):
    # validation
    model.eval()
    ious = []
    pbar = tqdm(dataloader)
    pbar.set_description(f"Validation")
    with torch.no_grad():
        idx = 0
        for img, images, targets in pbar: # images is a tensor of shape (batch_size, 3, 800, 800)
            results = []
            images = images.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]   

            boxes, _ = model(images, targets) # losses is none
            results.append((img, boxes))
            visualize_results(results, idx)
            
            idx+=1
            # val loss
            for box, target in zip(boxes, targets):
                gt_boxes = target['boxes'].to(device)
                iou = box_iou(box, gt_boxes)
                ious.append(iou)
    avg_iou = torch.cat(ious).mean().item()
    print(f"Validation IOU: {avg_iou}")

    return results

def visualize_results(results, idx):
    for images, boxes in results:
        for img, bxs in zip(images, boxes):
            plt.imshow(img)  # 'img' is already a PIL Image, no need for .cpu()
            ax = plt.gca()
            for box in bxs.cpu():  # Move each box tensor to CPU
                rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], 
                                         linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
            plt.savefig(f'result{idx}.png')
            plt.show()
            plt.close()
def main():
    # Set device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # Load model
    # ckpt_path = './ckpt_rpn/epoch5_iou0.016235657036304474.pth' # Update this path
    # model = load_model(ckpt_path).to(device)
    model = CustomRPN().to(device)

    # Prepare dataset for inference
    inference_path = './DLCV_vq2d_data/images/training_images'
    json_path = './DLCV_vq2d_data/objects_train_fixed.json' # useless for inference but lazy to change
    
    selected_indices = [0, 1, 2, 3, 4, 5]
    inference_dataset = CustomDataset(inference_path, json_path, indices=selected_indices) 
    inference_dataloader = DataLoader(inference_dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    # Run inference
    results = infer(model, inference_dataloader, device)

if __name__ == '__main__':
    main()
