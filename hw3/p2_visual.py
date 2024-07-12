import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data as Dataset
from tqdm import tqdm

import clip
from utils import ImageCaptionDataset, ValidationDataset, validation_dataset
import utils
from decoder_v import *
# import loralib as lora
import tokenizer
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.transforms.functional import to_pil_image


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_json = './encoder.json'
vocab_bpe = './vocab.bpe'
encoding = tokenizer.BPETokenizer(encoder_json, vocab_bpe)
print('Using device:', device)

# 1. Load a pre-trained Vit model from OpenAI CLIP
class VitEncoder(nn.Module):
    def __init__(self):
        super(VitEncoder, self).__init__()
        self.model, _ = clip.load('ViT-L/14', device=device)
        self.model.eval()  # freeze the model
        
    def forward(self, x):
        with torch.no_grad():
            x = self.model.encode_image(x)
        return x  # (batch_size, embedding_size=768)
 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711]),
])

cfg = Config(checkpoint='./hw3_data/p2_data/decoder_model.bin') 
vit_encoder = VitEncoder().to(device) # load the pre-trained Vit model
for param in vit_encoder.parameters():
    param.requires_grad = False

text_decoder = Decoder(cfg).to(device)
# lora.mark_only_lora_as_trainable(text_decoder)
checkpoint_path = './p2_model_best.bin'
checkpoint = torch.load(checkpoint_path)
text_decoder.load_state_dict(checkpoint, strict=False)
print(f'Load model from {checkpoint_path}')
text_decoder.freeze_pretrained()

print("Total params:", sum(p.numel() for p in text_decoder.parameters() if p.requires_grad))
# generate predictions into a  json file with key and value being ‘filename’ and ‘predicted_caption’
# Load and preprocess the image
image_path = 'hw3_data/p3_data/images/bike.jpg'
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])
image = transform(image).unsqueeze(0).to(device)


def visualize_attention(image, attention_weights, tokens, encoding):
    # Decode tokens using the encoding dictionary
    decoded_tokens = [encoding.decode([tk]) if tk != 50256 else '<end>' for tk in tokens]

    # Prepare the image for plotting
    image = to_pil_image(image.squeeze(0))
    image_np = np.array(image)
    
    fig, axs = plt.subplots(1, len(decoded_tokens), figsize=(20, 5))
    axs[0].imshow(image_np)
    axs[0].set_title("<start>")
    axs[0].axis('off')

    for i, (token_str, att_map) in enumerate(zip(decoded_tokens, attention_weights)):
        ax = axs[i]
        ax.set_title(token_str)

        # Assuming the original image size is 224x224 and each patch is 16x16
        # We calculate the grid size from the attention map shape
        grid_size = int(np.sqrt(att_map.shape[-1]))

        # Reshape attention map to (grid_size, grid_size)
        att_map_reshaped = att_map.view(grid_size, grid_size)

        # Resize attention map to match the image size
        att_map_resized = F.interpolate(att_map_reshaped.unsqueeze(0).unsqueeze(0),
                                        size=image_np.shape[:2],
                                        mode='bilinear',
                                        align_corners=False)
        att_map_resized = att_map_resized.squeeze().detach().cpu().numpy()

        # Overlay the attention map on the image
        ax.imshow(image_np, alpha=0.5)
        ax.imshow(att_map_resized, cmap='hot', alpha=0.5)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# Prediction and attention map generation
with torch.no_grad():
    start_token = 50256
    end_token = 50256
    max_len = 70
    
    # Prepare inputs for the model
    captions = torch.full((1, 1), start_token, dtype=torch.long, device=device)
    image_features = vit_encoder(image).float().to(device)
    
    attention_weights = []

    for _ in range(max_len):
        logits, last_cross_attention_weights = text_decoder(captions, image_features)
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        captions = torch.cat([captions, next_token], dim=-1)
        
        if last_cross_attention_weights is not None:
            # head 0
            attention_weights.append(last_cross_attention_weights[:, 0, :, :].squeeze(0))

        if next_token.item() == end_token:
            break
    # Process the output tokens to get the caption
    pred_tokens = captions.squeeze(0).cpu().numpy().tolist()

    # print(attention_weights_per_step)
    visualize_attention(image, attention_weights, pred_tokens, encoding)

