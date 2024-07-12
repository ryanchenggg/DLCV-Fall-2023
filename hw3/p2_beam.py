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
from decoder import *
import loralib as lora
import tokenizer
import numpy as np

myseed = 777  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

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
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# train_loader = get_train_loader('hw3_data/p2_data/images/train', 'hw3_data/p2_data/train.json', transform=transform)
val_loader = get_val_loader('hw3_data/p2_data/images/val', 'hw3_data/p2_data/val.json', transform=transform, shuffle=False,num_workers=0)

cfg = Config(checkpoint='./hw3_data/p2_data/decoder_model.bin') 
vit_encoder = VitEncoder().to(device) # load the pre-trained Vit model
for param in vit_encoder.parameters():
    param.requires_grad = False

text_decoder = Decoder(cfg).to(device)
# lora.mark_only_lora_as_trainable(text_decoder)
checkpoint_path = './p2_model_adapter_try.bin'
checkpoint = torch.load(checkpoint_path, map_location=device)
text_decoder.load_state_dict(checkpoint, strict=False)
print(f'Load model from {checkpoint_path}')
text_decoder.freeze_pretrained()

# Generate predictions on the validation set
print("Total params:", sum(p.numel() for p in text_decoder.parameters() if p.requires_grad))

# generate predictions into a  json file with key and value being ‘filename’ and ‘predicted_caption’
beam_width = 5
with torch.no_grad():
    text_decoder.eval()  # Make sure the model is in evaluation mode
    predictions = {}  # Dictionary to store predictions

    start_token = 50256
    end_token = 50256
    max_len = 70  # Maximum length of the generated captions

    for images, _, filenames in tqdm(val_loader):  # Iterate over validation data
        # print('filenames', filenames)
        images = images.to(device)
        image_features = vit_encoder(images).float().to(device)

        for i in range(image_features.size(0)):
            cur_image_features = image_features[i].unsqueeze(0)
            beam_candidates = [(torch.full((1, 1), start_token, dtype=torch.long, device=device), 0.0)]  # Initial beam candidates

            completed_sequences = []  # Store completed sequences here

            for step in range(max_len):
                all_candidates = []

                # Expand each candidate in beam
                for seq, score in beam_candidates:
                    if seq[0, -1].item() == start_token and step == 0:
                        # Skip the check at the first step since start_token == end_token
                        pass
                    elif seq[0, -1].item() == end_token:
                        completed_sequences.append((seq, score))  # Sequence is complete
                        continue

                    # Generate logits for the last token in sequence
                    logits = text_decoder(seq, cur_image_features)
                    log_probs = F.log_softmax(logits[:, -1, :], dim=-1)
                    # print('log_probs', log_probs)

                    # Get top beam_width tokens
                    top_log_probs, top_tokens = torch.topk(log_probs, beam_width)

                    # Expand the sequence with the new tokens
                    for j in range(beam_width):
                        next_token = top_tokens[:, j].unsqueeze(1)
                        new_score = score + top_log_probs[:, j].item()
                        new_seq = torch.cat([seq, next_token], dim=-1)
                        all_candidates.append((new_seq, new_score))

                # Select top beam_width candidates based on score
                beam_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_width]

            # Choose the best sequence from completed_sequences
            if completed_sequences:
                best_seq, best_score = max(completed_sequences, key=lambda x: x[1])
            else:
                # If there are no completed sequences, choose the best from the beam candidates
                best_seq, best_score = max(beam_candidates, key=lambda x: x[1])

            # Convert tokens to text
            pred_tokens = best_seq.squeeze(0).cpu().numpy().tolist()
            sentence = encoding.decode(pred_tokens).replace('<|endoftext|>', '').strip()
            sentence = sentence.split(encoding.decode([end_token]))[0]
            # print(f'filename: {filenames[i]}, sentence: {sentence}')
            filename = filenames[i].split('.')[0]

            # Store the predicted caption
            predictions[str(filename)] = sentence

# Save predictions to a JSON file
with open('p2_result.json', 'w') as f:
    json.dump(predictions, f)
