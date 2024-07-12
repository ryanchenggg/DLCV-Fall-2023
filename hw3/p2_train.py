import os
import json
import subprocess

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
# from decoder_c import *
from decoder import *
import loralib as lora
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

def get_train_loader(root_dir, json_file, transform, batch_size=32, shuffle=True, num_workers=4):
    dataset = ImageCaptionDataset(root_dir, json_file, transform=transform)
    data_loader = Dataset.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader

def get_val_loader(root_dir, json_file, transform, batch_size=32, shuffle=False, num_workers=0):
    dataset = validation_dataset(root_dir, json_file, transform=transform)
    data_loader = Dataset.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return data_loader
 
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],std=[0.26862954, 0.26130258, 0.27577711]),
])

train_loader = get_train_loader('hw3_data/p2_data/images/train', 'hw3_data/p2_data/train.json', transform=transform)
val_loader = get_val_loader('hw3_data/p2_data/images/val', 'hw3_data/p2_data/val.json', transform=transform, shuffle=False,num_workers=0)

cfg = Config(checkpoint='./hw3_data/p2_data/decoder_model.bin') 
vit_encoder = VitEncoder().to(device) # load the pre-trained Vit model
for param in vit_encoder.parameters():
    param.requires_grad = False

text_decoder = Decoder(cfg).to(device)
# lora.mark_only_lora_as_trainable(text_decoder)
# checkpoint_path = './p2_model_adapter_210.bin' 
# checkpoint = torch.load(checkpoint_path)
# text_decoder.load_state_dict(checkpoint)
# print(f'Load model from: {checkpoint_path}')
text_decoder.freeze_pretrained()

# optimizer = torch.optim.Adam(text_decoder.parameters(), lr=1e-4)
optimizer = torch.optim.AdamW(text_decoder.parameters(), lr=5e-5, weight_decay=1e-5)
criterion = nn.CrossEntropyLoss(ignore_index=-100)

# Training loop
num_epochs = 10
best_loss = 1000000
print("Total params:", sum(p.numel() for p in text_decoder.parameters() if p.requires_grad))
for epoch in range(num_epochs):
    text_decoder.train()
    total_loss = 0.0
    average_loss = 0.0
    total_batches = 0
    for images, captions, _ in tqdm(train_loader):
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()

        image_features = vit_encoder(images).to(device)
        image_features = image_features.float()
        image_features = image_features.to(device)

        outputs = text_decoder(captions, image_features).to(device)
        outputs = outputs.reshape(-1, outputs.size(-1))


        # Process captions
        gt_tokens = utils.gt_pad(captions, device)
        gt_tokens = gt_tokens.view(-1)

        loss = criterion(outputs, gt_tokens)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        total_batches += 1

    average_loss = total_loss / total_batches
    print(f"Epoch [{epoch+1}/{num_epochs}], Training Loss: {average_loss:.4f}")

    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(text_decoder.state_dict(), f'./ckpt/p2_model_adapter_{epoch}.bin')
        print('Successfully save model with loss:', best_loss)

    # generate predictions into a  json file with key and value being ‘filename’ and ‘predicted_caption’
    with torch.no_grad():
        text_decoder.eval()  # Make sure the model is in evaluation mode
        predictions = {} # the dict to store predictions with key and value being ‘filename’ and ‘predicted_caption’
        start_token = 50256
        end_token = 50256
        max_len = 70  # Set the maximum length of the generated captions
        for images, _, filenames in tqdm(val_loader):  # Assuming img_ids are the filenames
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
            
        with open('p2_result.json', 'w') as f:          
            json.dump(predictions, f)
        
        pred_file = 'p2_result.json'
        # Call the evaluation script using subprocess.run
        eval_command = ['python', 'p2_evaluate.py', '--pred_file', pred_file]
        subprocess.run(eval_command, check=True)
