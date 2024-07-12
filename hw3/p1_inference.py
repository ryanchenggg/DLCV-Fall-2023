import os
import sys
import torch
import clip
import json
from PIL import Image

# print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

# Load arguments
test_image_path = sys.argv[1]
id2label_path = sys.argv[2]
output_csv_path = sys.argv[3]

# Load validation data
with open(id2label_path, 'r') as f:
    id2label = json.load(f)

# Function to determine whether to use "a" or "an"
def article_for_label(label):
    if label[0].lower() in 'aeiou':
        return 'An'
    else:
        return 'A'

# Generate labels with correct article
text_prompt = [f"{article_for_label(label)} {label} in the image" for label in id2label.values()]
text_tokens = clip.tokenize(text_prompt).to(device)
results = []

for filename in os.listdir(test_image_path):
    image = preprocess(Image.open(os.path.join(test_image_path, filename))).unsqueeze(0).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text_tokens)
        
        # Compute the similarity between the image and the text features 
        similarities = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        predicted_label_id = similarities.argmax(dim=-1).item()
    
    # Append the result
    results.append([filename, predicted_label_id])


# Convert results to a DataFrame and save as CSV
import pandas as pd
df = pd.DataFrame(results, columns=['filename', 'label'])
df.to_csv(output_csv_path, index=False)

# Calculate accuracy
correct = 0
total = len(df)
for _, row in df.iterrows():
    actual_label_id = row['filename'].split('_')[0]
    if row['label'] == int(actual_label_id):
        correct += 1
accuracy = correct / total
print(f"Accuracy: {accuracy}")
# print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
