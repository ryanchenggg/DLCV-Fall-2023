import os
import json
import sys

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
from decoder import *
# import loralib as lora
import tokenizer


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

def get_val_loader(root_dir, transform, batch_size=32, shuffle=False, num_workers=0):
    dataset = ValidationDataset(root_dir, transform=transform)
    data_loader = Dataset.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711]),
])

#################### Arguments ####################
path_to_test_img_dir = sys.argv[1]
path_to_the_output_json_file = sys.argv[2]
decoder_weight_path = sys.argv[3]
####################################################

val_loader = get_val_loader(path_to_test_img_dir, transform=transform, shuffle=False,num_workers=0)

cfg = Config(checkpoint=decoder_weight_path) 
vit_encoder = VitEncoder().to(device) # load the pre-trained Vit model
for param in vit_encoder.parameters():
    param.requires_grad = False

text_decoder = Decoder(cfg).to(device)
# lora.mark_only_lora_as_trainable(text_decoder)
checkpoint_path = './p2_model_best.bin'
checkpoint = torch.load(checkpoint_path, map_location=device)
text_decoder.load_state_dict(checkpoint, strict=False)
print(f'Load model from {checkpoint_path}')
text_decoder.freeze_pretrained()

print("Total params:", sum(p.numel() for p in text_decoder.parameters() if p.requires_grad))
# generate predictions into a  json file with key and value being ‘filename’ and ‘predicted_caption’
with torch.no_grad():
    text_decoder.eval()  # Make sure the model is in evaluation mode
    predictions = {} # the dict to store predictions with key and value being ‘filename’ and ‘predicted_caption’
    start_token = 50256
    end_token = 50256
    max_len = 70  # Set the maximum length of the generated captions
    for images, filenames in tqdm(val_loader):  # Assuming img_ids are the filenames
        images = images.to(device)
        image_features = vit_encoder(images).float().to(device)
        
        # Iterate over each image in the batch
        for i in range(image_features.size(0)):
            # Start with a tensor containing the start_token
            captions = torch.full((1, 1), start_token, dtype=torch.long, device=device)
            cur_image_features = image_features[i].unsqueeze(0)  # Process one image at a time

            # Generate tokens until we reach max_len or end_token
            for _ in range(max_len):
                logits = text_decoder(captions, cur_image_features)  # Generate logits
                next_token_logits = logits[:, -1, :]  # Only need the last token's logits
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1) # Get the most likely next token
                captions = torch.cat([captions, next_token], dim=-1)  # Append the next token

                # Break if we've reached the end_token
                if next_token.item() == end_token:
                    break

            # Convert the generated tokens to text
            pred_tokens = captions.squeeze(0).cpu().numpy().tolist()
            sentence = encoding.decode(pred_tokens)
            sentence = sentence.replace('<|endoftext|>', '').strip()  
            # Remove any additional end tokens and padding
            sentence = sentence.split(encoding.decode([end_token]))[0]
            filename = filenames[i].split('.')[0]
    
            predictions[str(filename)] = sentence
        
    with open(path_to_the_output_json_file, 'w') as f:          
        json.dump(predictions, f)
