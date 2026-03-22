import os
import gc
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from PIL import Image, ImageFile
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from sklearn.model_selection import KFold
from tqdm import tqdm

# --- CONFIGURATION & PATHS ---
DATA_ROOT = "/workspace/art/dataset/wikiart/" 
LABEL_DIR = "/workspace/art/dataset/" 
CHECKPOINT_DIR = "./checkpoints_v4_cv"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

Image.MAX_IMAGE_PIXELS = None 
ImageFile.LOAD_TRUNCATED_IMAGES = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. INVESTIGATION LAYER ---
def investigate_data():
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
    return merged, num_styles, num_artists, num_genres

# --- 2. DATASET & TRANSFORMS ---
train_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
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
        if self.transform: image = self.transform(image)
        return image, labels

# --- 3. MODEL ARCHITECTURE (V4) ---
class ArtExtract_V4(nn.Module):
    def __init__(self, n_styles, n_artists, n_genres):
        super().__init__()
        resnet = models.resnet50(weights='DEFAULT')
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))
        def make_neck():
            return nn.Sequential(nn.Linear(2048 * 2, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.5))
        self.style_neck = make_neck()
        self.artist_neck = make_neck()
        self.genre_neck = make_neck()
        self.style_head = nn.Linear(512, n_styles)
        self.artist_head = nn.Linear(512, n_artists)
        self.genre_head = nn.Linear(512, n_genres)

    def forward(self, x):
        feat = self.backbone(x)
        flat = torch.cat([self.avgpool(feat).view(x.size(0), -1), self.maxpool(feat).view(x.size(0), -1)], dim=1)
        return {'style': self.style_head(self.style_neck(flat)), 'artist': self.artist_head(self.artist_neck(flat)), 'genre': self.genre_head(self.genre_neck(flat))}

# --- 4. TRAINING LOGIC ---
def run_epoch(model, loader, optimizer, criterion, is_train=True):
    model.train() if is_train else model.eval()
    running_loss, correct, totals = 0.0, {k:0 for k in ['style','artist','genre']}, {k:0 for k in ['style','artist','genre']}
    
    for images, labels in tqdm(loader, leave=False, desc="Training" if is_train else "Eval"):
        images, labels = images.to(DEVICE), {k: v.to(DEVICE) for k, v in labels.items()}
        with torch.set_grad_enabled(is_train):
            outputs = model(images)
            losses = [criterion(outputs[k], labels[k]) for k in ['style', 'artist', 'genre']]
            total_loss = (1.0 * losses[0]) + (1.5 * losses[1]) + (0.8 * losses[2])
            if is_train:
                optimizer.zero_grad(); total_loss.backward(); optimizer.step()
        
        running_loss += total_loss.item()
        for k in correct.keys():
            mask = labels[k] != -1
            if mask.sum() > 0:
                correct[k] += (outputs[k].argmax(1)[mask] == labels[k][mask]).sum().item()
                totals[k] += mask.sum().item()
    
    return running_loss/len(loader), {k: (100*correct[k]/totals[k]) if totals[k]>0 else 0 for k in correct}

# --- 5. CROSS VALIDATION EXECUTION ---
if __name__ == "__main__":
    full_df, n_s, n_a, n_g = investigate_data()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Store final best results for each fold
    cv_stats = {'style': [], 'artist': [], 'genre': []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_df)):
        print(f"\n{'='*25} FOLD {fold+1}/5 {'='*25}")
        train_df, val_df = full_df.iloc[train_idx], full_df.iloc[val_idx]
        
        train_loader = DataLoader(ArtMultiTaskDataset(train_df, train_transforms), batch_size=32, shuffle=True, num_workers=2, pin_memory=True)
        val_loader = DataLoader(ArtMultiTaskDataset(val_df, val_transforms), batch_size=128, shuffle=False, num_workers=2, pin_memory=True)

        model = ArtExtract_V4(n_s, n_a, n_g).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(), lr=1e-4)
        criterion = nn.CrossEntropyLoss(ignore_index=-1)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)

        # Track bests for THIS fold
        best_fold_style = 0
        best_fold_metrics = {'style': 0, 'artist': 0, 'genre': 0}

        for epoch in range(10): # Reduced to 12 as discussed
            t_loss, _ = run_epoch(model, train_loader, optimizer, criterion, True)
            v_loss, v_accs = run_epoch(model, val_loader, None, criterion, False)
            
            print(f"Fold {fold+1} E{epoch+1:02d} | Loss: {v_loss:.3f} | S: {v_accs['style']:.2f}% | A: {v_accs['artist']:.2f}% | G: {v_accs['genre']:.2f}%")
            
            scheduler.step(v_accs['style'])
            
            # Save based on Style improvement
            if v_accs['style'] > best_fold_style:
                best_fold_style = v_accs['style']
                best_fold_metrics = v_accs
                torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_model_fold_{fold+1}.pth"))
                print(f"   🌟 New Best Fold {fold+1} Style: {v_accs['style']:.2f}%")
        
        # After fold is done, log the best metrics from this fold
        for k in cv_stats:
            cv_stats[k].append(best_fold_metrics[k])
            
        del model, optimizer; gc.collect(); torch.cuda.empty_cache()

    # --- FINAL CROSS-VAL SUMMARY ---
    print(f"\n{'-'*30}")
    print(f"🏆 5-FOLD CROSS-VALIDATION SUMMARY")
    print(f"{'-'*30}")
    
    results_table = []
    for k in ['style', 'artist', 'genre']:
        mean_val = np.mean(cv_stats[k])
        std_val = np.std(cv_stats[k])
        results_table.append([k.capitalize(), f"{mean_val:.2f}%", f"±{std_val:.2f}"])
        print(f"{k.capitalize():<8}: Mean Accuracy = {mean_val:.2f}% | Std Dev = {std_val:.2f}")
    
    print(f"{'-'*30}")