import os
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torchvision
import torch.nn.functional as F
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead
from PIL import Image
import sys
import matplotlib.pyplot as plt

def deeplabv3_resnet50(num_classes=7, pretrained=True):
    # model = torch.hub.load('pytorch/vision:v0.10.0','deeplabv3_resnet50', pretrained=True)
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights='DeepLabV3_ResNet50_Weights.DEFAULT')
    model.classifier = DeepLabHead(2048, num_classes)
    model.aux_classifier = FCNHead(1024, num_classes)
    return model


def test(imgPath, model, device):
    
    cls_color = {
        0:  [0, 255, 255],
        1:  [255, 255, 0],
        2:  [255, 0, 255],
        3:  [0, 255, 0],
        4:  [0, 0, 255],
        5:  [255, 255, 255],
        6: [0, 0, 0],
        }

    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    custom_transforms.append(torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    test_transform = torchvision.transforms.Compose(custom_transforms)

    img = Image.open(imgPath).convert('RGB')
    img_tensor = test_transform(img).float()
    img_tensor = img_tensor.unsqueeze_(0).to(device)
    
    outputs = model(img_tensor)
    output = F.softmax(outputs['out'], dim=1)
    output = torch.argmax(output, dim=1).cpu().numpy()[0]

    # class to mask color
    mask = np.empty((512,512,3))
    for label in range(7):
        mask[output == label] = cls_color[label]
    # print(mask)

    return mask

def main():
    #############################################################
    test_root = sys.argv[1]
    output = sys.argv[2]
    weight = './p3_bestmodel_B.pth'
    #############################################################
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Device:{device}')

    model = deeplabv3_resnet50(num_classes=7, pretrained=True)
    model.to(device)
    model.load_state_dict(torch.load(weight))
    model.eval()
 
    # test_img = list(sorted(f for f in os.listdir(test_root) if f.endswith('sat.jpg')))
    test_img = list(sorted(f for f in os.listdir(test_root)))

    for i, img in enumerate(test_img):
        print(f'Processing {img} ...')
        imgPath = os.path.join(test_root, img)
        
        with torch.no_grad():
            mask = test(imgPath, model, device)
            mask = Image.fromarray(np.uint8(mask))
            mask.save(f'{output}/{img[:4]}.png')
    
if __name__ == "__main__":
    main()
