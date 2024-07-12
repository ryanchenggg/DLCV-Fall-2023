import torch
import torchvision
from torch import nn
from torchvision import transforms as T
from collections import OrderedDict

from PIL import Image
import cv2

class backbone_n_rpn(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = torchvision.models.detection.faster_rcnn.fasterrcnn_resnet50_fpn_v2(weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT, weights_backbone=torchvision.models.ResNet50_Weights.DEFAULT)
        self.model.eval()
        self.transform = self.model.transform
        self.backbone = self.model.backbone
        self.rpn = self.model.rpn
        self.roi_heads = self.model.roi_heads

    def forward(self, image):
        image, targets = self.transform(image) #targets is None
        features = self.backbone(image.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        proposals, proposal_losses = self.rpn(image, features, targets)
        return proposals, proposal_losses

def main():
    rpn = backbone_n_rpn()
    img = Image.open('./bus.jpg')
    transform = T.ToTensor()
    img = transform(img)
    img = img.unsqueeze(0)
    with torch.no_grad():
        proposals, proposal_losses = rpn(img)
    print(proposals[0].shape)
    print(proposal_losses)

if __name__ == "__main__":
    main()