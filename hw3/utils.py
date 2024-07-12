# dependences
import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import tokenizer

from p2_evaluate import *

encoder_json = './encoder.json'
vocab_bpe = './vocab.bpe'
encoding = tokenizer.BPETokenizer(encoder_json, vocab_bpe)

# dataset def
class ImageCaptionDataset(Dataset):
    def __init__(self, root_dir, json_file, transform=None, tokenizer=encoding, max_len=59):
        self.root_dir = root_dir
        self.transform = transform
        self.tokenizer = tokenizer
        self.max_len = max_len
        
        # Load json file
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        # Create a mapping from image_id to image_path
        self.image_map = {item['id']: item['file_name'] for item in self.data['images']}
        self.annotations = self.data['annotations']
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_id = annotation['image_id']
        filename = self.image_map[img_id]
        img_path = os.path.join(self.root_dir, filename)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        
        # Tokenize captions
        captions = annotation['caption']
        # print('------------------')
        # print(f'img_id: {img_id}')
        # print('img_path: ', img_path)
        # print(f'captions: {captions}')
        
        # Tokenize and Pad captions
        tokenized_captions = self.tokenizer.encode(captions)
        padded_tokens = [50256] + tokenized_captions + [50256] * (self.max_len - len(tokenized_captions))
        padded_tokens = padded_tokens[:self.max_len] # Truncate to max_len
        # print(f'padded_tokens: {padded_tokens}')
        # print('------------------')
        return img, torch.tensor(padded_tokens, dtype=torch.long), filename


def gt_pad(captions, device):
    # Process captions
    processed_captions = []
    original_len = captions.size(1)  # Save original length for padding later
    for caption in captions:
        # Remove the first element
        caption = caption[1:]
        caption = caption.to(device)

        # Remove all but one 50256 from the end
        idx = len(caption) - 1
        while idx > 0 and caption[idx] == 50256:
            idx -= 1
        caption = caption[:idx + 2]  # Keep one 50256

        # Pad to original length with -100
        pad_len = original_len - len(caption)
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), -100, dtype=torch.long, device=device)
            caption = torch.cat([caption, pad_tensor], dim=0)

        processed_captions.append(caption)

    # Convert list to a tensor
    processed_captions = torch.stack(processed_captions, dim=0)
    return processed_captions

def greedy_decode(decoder, image_features, max_len, start_token=50256, end_token=50256):
    device = image_features.device
    B = image_features.size(0)  # Batch size 

    # 初始化文本输入仅包含开始标记
    captions = torch.full((B, 1), start_token, dtype=torch.long, device=device)

    # 循环生成文本
    for i in range(max_len):
        logits = decoder(captions, image_features)  # 生成预测
        next_token_logits = logits[:, -1, :]  # 只取最后一个时间步的预测
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        captions = torch.cat([captions, next_token], dim=-1)  # 更新输入

        # 检查是否到达结束标记
        if torch.all(next_token == end_token):
            break
        
        decode_text = [encoding.decode([tokens]) for tokens in captions.squeeze(0).cpu().numpy()]
        sentence = ''.join(decode_text)

    return sentence

# # pevit
class validation_dataset(Dataset):
    def __init__(self, images_dir, annotations_file, tokenizer=encoding, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.tokenizer = tokenizer
        with open(annotations_file, 'r') as f:
            data = json.load(f)
        self.annotations = data['annotations']
        images_info = data['images']
        self.image_id_to_file_name = {image['id']: image['file_name'] for image in images_info}
        self.image_captions = {ann['image_id']: [] for ann in self.annotations}
        max_length = 0
        for ann in self.annotations:
            caption = ann['caption']
            self.image_captions[ann['image_id']].append(caption)
            # Calculate the length of the caption with the start token
            length_with_start_token = len(self.tokenizer.encode(caption)) + 1
            if length_with_start_token > max_length:
                max_length = length_with_start_token
        self.image_ids = list(self.image_captions.keys())
        self.max_length = max_length
        self.pad_value = 50256  # Define pad value here
    def __len__(self):
        return len(self.image_ids)
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        file_name = self.image_id_to_file_name[image_id]
        image_path = os.path.join(self.images_dir, file_name)
        if not os.path.exists(image_path):
            print(f'Warning: Image file {image_path} not found. Skipping.')
            return None
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        captions = self.image_captions[image_id]
        caption = captions[0]
        # Add start token
        encoded_caption = [self.pad_value] + self.tokenizer.encode(caption)
       
        return image, image_id, file_name

class ValidationDataset(Dataset):
    def __init__(self, images_dir, transform=None):
        self.images_dir = images_dir
        self.transform = transform
        self.image_files = os.listdir(images_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        file_name = self.image_files[idx]
        image_path = os.path.join(self.images_dir, file_name)
        if not os.path.exists(image_path):
            print(f'Warning: Image file {image_path} not found. Skipping.')
            return None
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return image, file_name