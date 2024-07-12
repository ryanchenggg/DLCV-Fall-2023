import os
import json
import random
import torch
import clip
from PIL import Image
import matplotlib.pyplot as plt

def plot_top5_predictions(image_path, top_labels, top_probs, true_label_name):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))  
    img = Image.open(image_path)
    axes[0].imshow(img)
    axes[0].axis('off') 
    axes[0].text(0.5, 0.05, f'True label: {true_label_name}', ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
    if true_label_name in top_labels:
        probability = top_probs[top_labels.index(true_label_name)]
        axes[0].text(0.5, 0.02, f'Probability: {probability:.2f}', ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
    else:
        axes[0].text(0.5, 0.02, 'Probability: Not in top 5', ha='center', va='center', transform=axes[0].transAxes, fontsize=12)
    axes[1].barh(range(5), top_probs, color='skyblue')
    axes[1].set_xlabel('Probability')
    axes[1].set_title(f'Top-5 Predictions for {os.path.basename(image_path)}')
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(top_labels)
    axes[1].invert_yaxis()  

    plt.tight_layout()  
    plt.savefig(f'./{os.path.basename(image_path)}_top5.png')
    plt.close() 

img_data = './hw3_data/p1_data/val'
label_data = './hw3_data/p1_data/id2label.json'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, preprocess = clip.load('ViT-L/14', device)

with open(label_data, 'r') as f:
    id2label = json.load(f)
    id2label = {int(k): v for k, v in id2label.items()}

# Function to determine whether to use "a" or "an"
def article_for_label(label):
    if label[0].lower() in 'aeiou':
        return 'An'
    else:
        return 'A'
    
text_prompt = [f"{article_for_label(label)} {label} in the image" for label in id2label.values()]
image_paths = random.sample([os.path.join(img_data, f) for f in os.listdir(img_data) if f.endswith('.png')], 3)

for image_path in image_paths:
    numeric_id = int(os.path.basename(image_path).split("_")[0]) 
    true_label_name = id2label.get(numeric_id, "Unknown label")

    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text_tokens = clip.tokenize(text_prompt).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        similarities = (image_features @ text_features.T).squeeze(0).softmax(dim=0)
        top_probs, top_labels_indices = similarities.topk(5)
        top_probs = top_probs.cpu().numpy()
        top_labels = [id2label[label_idx] for label_idx in top_labels_indices.cpu().numpy()]
        plot_top5_predictions(image_path, top_labels, top_probs, true_label_name)
