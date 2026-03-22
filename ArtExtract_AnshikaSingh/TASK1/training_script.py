import os
import gc
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- CONFIGURATION & PATHS ---
DATA_ROOT = "/workspace/art/dataset/wikiart/" 
LABEL_DIR = "/workspace/art/dataset/" 
CHECKPOINT_DIR = "./checkpoints_v4"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

Image.MAX_IMAGE_PIXELS = None 
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. INVESTIGATION LAYER ---
def investigate_data():
    print("\n--- [INVESTIGATION] ANALYZING DATASET ---")
    tasks = ['style', 'artist', 'genre']
    dfs = {}
    
    for t in tasks:
        train_path = os.path.join(LABEL_DIR, f"{t}_train.csv")
        val_path = os.path.join(LABEL_DIR, f"{t}_val.csv")
        
        train = pd.read_csv(train_path, names=['path', t])
        val = pd.read_csv(val_path, names=['path', t])
        dfs[t] = pd.concat([train, val]).drop_duplicates('path')

    merged = dfs['style'].merge(dfs['artist'], on='path', how='outer').merge(dfs['genre'], on='path', how='outer')
    
    num_styles = int(dfs['style']['style'].max() + 1)
    num_artists = int(dfs['artist']['artist'].max() + 1)
    num_genres = int(dfs['genre']['genre'].max() + 1)
    
    print(f"Class Counts: Style={num_styles}, Artist={num_artists}, Genre={num_genres}")
    return merged, num_styles, num_artists, num_genres

# --- 2. DATASET & TRANSFORMS ---
train_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05), # Art-specific
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(), # No flipping or color jittering during evaluation!
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ArtMultiTaskDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(DATA_ROOT, row['path'])
        
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (448, 448), (0,0,0))

        labels = {
            'style': torch.tensor(row['style'] if pd.notna(row['style']) else -1, dtype=torch.long),
            'artist': torch.tensor(row['artist'] if pd.notna(row['artist']) else -1, dtype=torch.long),
            'genre': torch.tensor(row['genre'] if pd.notna(row['genre']) else -1, dtype=torch.long)
        }
        
        if self.transform:
            image = self.transform(image)
        return image, labels

# --- 3. MODEL ARCHITECTURE (V4) ---
class ArtExtract_V4(nn.Module):
    def __init__(self, n_styles, n_artists, n_genres):
        super().__init__()
        resnet = models.resnet50(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        
        # Dedicated capacity for each task to prevent interference
        def make_neck():
            return nn.Sequential(
                nn.Linear(2048 * 2, 512),
                nn.BatchNorm1d(512),
                nn.SiLU(), 
                nn.Dropout(0.5) 
            )
            
        self.style_neck = make_neck()
        self.artist_neck = make_neck()
        self.genre_neck = make_neck()
        
        self.style_head = nn.Linear(512, n_styles)
        self.artist_head = nn.Linear(512, n_artists)
        self.genre_head = nn.Linear(512, n_genres)

    def forward(self, x):
        features = self.backbone(x)
        combined = torch.cat([self.avgpool(features).view(x.size(0), -1), 
                              self.maxpool(features).view(x.size(0), -1)], dim=1)
        
        return {
            'style': self.style_head(self.style_neck(combined)), 
            'artist': self.artist_head(self.artist_neck(combined)), 
            'genre': self.genre_head(self.genre_neck(combined))
        }

# --- 4. TRAINING & EVALUATION ---
def run_epoch(model, loader, optimizer, criterion, is_train=True):
    model.train() if is_train else model.eval()
    running_loss = 0.0
    correct = {'style': 0, 'artist': 0, 'genre': 0}
    totals = {'style': 0, 'artist': 0, 'genre': 0}

    pbar = tqdm(loader, desc="Training" if is_train else "Evaluating")
    for images, labels in pbar:
        images = images.to(DEVICE)
        labels = {k: v.to(DEVICE) for k, v in labels.items()}

        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            
            losses = [criterion(outputs[k], labels[k]) for k in labels]
            losses = [torch.nan_to_num(l, nan=0.0) for l in losses]
            total_loss = (1.0 * losses[0]) + (1.5 * losses[1]) + (0.8 * losses[2])

            if is_train:
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        running_loss += total_loss.item()
        
        for k in correct.keys():
            mask = labels[k] != -1
            if mask.sum() > 0:
                pred = outputs[k].argmax(1)
                correct[k] += (pred[mask] == labels[k][mask]).sum().item()
                totals[k] += mask.sum().item()

    accs = {k: (100 * correct[k] / totals[k]) if totals[k] > 0 else 0 for k in correct}
    return running_loss / len(loader), accs

# --- 5. EXECUTION ---
if __name__ == "__main__":
    full_df, n_s, n_a, n_g = investigate_data()
    
    train_df, val_df = train_test_split(full_df, test_size=0.15, random_state=42)
    
    # Note: Passing the separated transforms here
    train_loader = DataLoader(ArtMultiTaskDataset(train_df, train_transforms), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(ArtMultiTaskDataset(val_df, val_transforms), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

    model = ArtExtract_V4(n_s, n_a, n_g).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

    best_acc = 0
    for epoch in range(15):
        print(f"\nEpoch {epoch+1}/15")
        t_loss, _ = run_epoch(model, train_loader, optimizer, criterion, True)
        v_loss, v_accs = run_epoch(model, val_loader, None, criterion, False)
        
        print(f"Val Loss: {v_loss:.4f} | Style: {v_accs['style']:.2f}% | Artist: {v_accs['artist']:.2f}% | Genre: {v_accs['genre']:.2f}%")
        
        scheduler.step(v_accs['style'])
        
        if v_accs['style'] > best_acc:
            best_acc = v_accs['style']
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_model_v4.pth"))
            print("🌟 Checkpoint Saved!")
            
        gc.collect()
        torch.cuda.empty_cache()