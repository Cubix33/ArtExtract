import os
import torch
import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from tqdm import tqdm

# --- CONFIG ---
CSV_PATH = "./" 
OUTPUT_FILE = "nga_full_embeddings.npy"
BATCH_SIZE = 128 

# 1. Load and Merge everything
obj = pd.read_csv(f"{CSV_PATH}/objects.csv", low_memory=False)
img = pd.read_csv(f"{CSV_PATH}/published_images.csv", low_memory=False)
df = pd.merge(obj, img, left_on='objectid', right_on='depictstmsobjectid')
df = df.dropna(subset=['iiifurl']).reset_index(drop=True)

# 2. Model Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(device)
model.eval()

from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

import concurrent.futures

# 3. Extraction Loop
all_features = []
print(f"🚀 Processing {len(df)} artworks...")

# Create a helper function to fetch a single image
def fetch_image(url):
    try:
        # Request the thumbnail
        r = requests.get(f"{url}/full/!224,224/0/default.jpg", timeout=5)
        img_data = Image.open(BytesIO(r.content)).convert('RGB')
        return transform(img_data)
    except:
        # If the URL is dead or times out, return a blank black image
        return torch.zeros(3, 224, 224)

for i in tqdm(range(0, len(df), BATCH_SIZE)):
    batch = df.iloc[i:i+BATCH_SIZE]
    urls = batch['iiifurl'].tolist()
    
    # 🌟 MULTITHREADING: Download up to 32 images simultaneously
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        tensors = list(executor.map(fetch_image, urls))
        
    # Now that all 128 images are downloaded, send them to the GPU
    batch_tensor = torch.stack(tensors).to(device)
    
    with torch.no_grad():
        out = model(batch_tensor)
        out = torch.nn.functional.normalize(out, dim=1)
        all_features.append(out.cpu().numpy())

# 4. Save the "Brain" of your search engine
np.save(OUTPUT_FILE, np.vstack(all_features))
print("✅ Done. Transfer this .npy file to your Google Drive for the Notebook.")