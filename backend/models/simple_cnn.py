# ============================================================
# SIMPLE CNN FOR MEL SPECTROGRAMS
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
from snowflake.snowpark import Session

# ============================================================
# CONFIG
# ============================================================
NUM_CLASSES = 5
BATCH_SIZE = 16
LR = 1e-3
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOCAL_CACHE = "./backend/models/cache_mel"
OUTPUT_DIR = "./backend/models/simple_cnn"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['asthma', 'bronchial', 'copd', 'healthy', 'pneumonia']

# ============================================================
# SETUP DEVICE
# ============================================================
print(f"🖥️ Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"✅ GPU: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB\n")
else:
    print("⚠️  Using CPU\n")

# ============================================================
# LOAD DATA FROM SNOWFLAKE
# ============================================================
load_dotenv()

connection_params = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_TOKEN"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": "HACKATHON_WH",
    "database": "M2_ISD_EQUIPE_1_DB",
    "schema": "PROCESSED"
}

session = Session.builder.configs(connection_params).create()
print("✅ Connected to Snowflake")

# Load data
TRAINING_VIEW = "M2_ISD_EQUIPE_1_DB.PROCESSED.TRAINING_DATA_V"
df = session.sql(f"""
    SELECT FILE_NAME, CLASS, FEAT_MEL
    FROM {TRAINING_VIEW}
    WHERE FEAT_MEL IS NOT NULL
""").to_pandas()

print(f"📊 Loaded {len(df)} rows\n")

# ============================================================
# LABEL ENCODING
# ============================================================
label_map = {cls: i for i, cls in enumerate(CLASS_NAMES)}
df["CLASS"] = df["CLASS"].str.strip().str.lower()
df["LABEL"] = df["CLASS"].map(label_map)
df = df[df["LABEL"].notnull()]
df["LABEL"] = df["LABEL"].astype(int)

# ============================================================
# DOWNLOAD MEL FILES
# ============================================================
def download_from_stage(row):
    feat_path = row['FEAT_MEL']
    if pd.isna(feat_path):
        return None
    
    filename = feat_path.split("/")[-1]
    cls = row["CLASS"]
    local_path = os.path.join(LOCAL_CACHE, filename)

    possible_paths = [
        f"@STG_RESPIRATORY_FEATURES/{cls}/{filename}",
        f"@STG_RESPIRATORY_FEATURES_AUG/{cls}/{filename}",
    ]

    if not os.path.exists(local_path):
        for full_path in possible_paths:
            try:
                session.file.get(full_path, LOCAL_CACHE)
                return local_path
            except:
                continue
        return None

    return local_path

print("📥 Downloading mel files...")
df["LOCAL_MEL"] = df.apply(download_from_stage, axis=1)
df = df[df["LOCAL_MEL"].notnull()]
df = df.reset_index(drop=True)
print(f"✅ Downloaded {len(df)} mel files\n")

# ============================================================
# SPLIT DATA
# ============================================================
from sklearn.model_selection import train_test_split

# Avoid data leakage
df["BASE_FILE"] = df["FILE_NAME"].str.replace("_aug.*", "", regex=True)
file_labels = df.groupby("BASE_FILE")["LABEL"].first()

train_files, temp_files = train_test_split(
    file_labels.index,
    test_size=0.3,
    stratify=file_labels.values,
    random_state=42
)

temp_labels = file_labels.loc[temp_files]
val_files, test_files = train_test_split(
    temp_files,
    test_size=0.5,
    stratify=temp_labels.values,
    random_state=42
)

train_df = df[df["BASE_FILE"].isin(train_files)]
val_df = df[df["BASE_FILE"].isin(val_files)]
test_df = df[df["BASE_FILE"].isin(test_files)]

print(f"📊 Data split:")
print(f"   Train: {len(train_df)}")
print(f"   Val: {len(val_df)}")
print(f"   Test: {len(test_df)}\n")

# ============================================================
# DATASET
# ============================================================
class MelDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mel_data = np.load(row["LOCAL_MEL"])
        
        # Convert to tensor and normalize
        mel_tensor = torch.tensor(mel_data, dtype=torch.float32)
        mel_tensor = (mel_tensor - mel_tensor.min()) / (mel_tensor.max() - mel_tensor.min() + 1e-8)
        
        # Ensure 2D: (freq, time)
        if mel_tensor.dim() == 1:
            mel_tensor = mel_tensor.unsqueeze(0)
        elif mel_tensor.dim() == 3:
            mel_tensor = mel_tensor.view(mel_tensor.shape[0], -1)
        
        # Pad to standard size (128, 128)
        h, w = mel_tensor.shape
        if h < 128:
            mel_tensor = F.pad(mel_tensor, (0, 0, 0, 128-h))
        else:
            mel_tensor = mel_tensor[:128, :]
        
        if w < 128:
            mel_tensor = F.pad(mel_tensor, (0, 128-w))
        else:
            mel_tensor = mel_tensor[:, :128]
        
        # Add channel dimension: (1, 128, 128)
        mel_tensor = mel_tensor.unsqueeze(0)
        
        label = int(row["LABEL"])
        return mel_tensor, label

    def __len__(self):
        return len(self.df)

# ============================================================
# DATALOADERS
# ============================================================
class_counts = train_df["LABEL"].value_counts().to_dict()
sample_weights = train_df["LABEL"].map(lambda x: 1.0 / class_counts[x]).values

train_loader = DataLoader(
    MelDataset(train_df, train=True),
    batch_size=BATCH_SIZE,
    sampler=WeightedRandomSampler(sample_weights, len(sample_weights)),
)
val_loader = DataLoader(MelDataset(val_df, train=False), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(MelDataset(test_df, train=False), batch_size=BATCH_SIZE, shuffle=False)

# ============================================================
# SIMPLE CNN MODEL
# ============================================================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # FC layers
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        # Block 1: (B, 1, 128, 128) -> (B, 32, 64, 64)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2: (B, 32, 64, 64) -> (B, 64, 32, 32)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3: (B, 64, 32, 32) -> (B, 128, 16, 16)
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # FC
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# ============================================================
# TRAINING
# ============================================================
model = SimpleCNN(num_classes=NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

print(f"\n🎯 Model Architecture:")
print(f"   Input: (1, 128, 128)")
print(f"   Output: {NUM_CLASSES} classes")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

best_val_f1 = 0.0
epochs_no_improve = 0
early_stopping_patience = 10

train_losses = []
val_losses = []
train_accs = []
val_accs = []

for epoch in range(EPOCHS):
    # Training
    model.train()
    train_loss = 0.0
    train_preds, train_labels = [], []
    
    for mel, label in train_loader:
        mel, label = mel.to(DEVICE), label.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(mel)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_preds.extend(outputs.argmax(dim=1).cpu().detach().numpy())
        train_labels.extend(label.cpu().numpy())
    
    train_loss /= len(train_loader)
    train_acc = accuracy_score(train_labels, train_preds)
    train_f1 = f1_score(train_labels, train_preds, average='macro', zero_division=0)
    
    train_losses.append(train_loss)
    train_accs.append(train_acc)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []
    
    with torch.no_grad():
        for mel, label in val_loader:
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            outputs = model(mel)
            loss = criterion(outputs, label)
            
            val_loss += loss.item()
            val_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            val_labels.extend(label.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average='macro', zero_division=0)
    
    val_losses.append(val_loss)
    val_accs.append(val_acc)
    
    # Early stopping
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
        print(f"✅ Epoch {epoch+1}/{EPOCHS} | Val F1: {val_f1:.4f} (NEW BEST)")
    else:
        epochs_no_improve += 1
        print(f"   Epoch {epoch+1}/{EPOCHS} | Val F1: {val_f1:.4f}")
        
        if epochs_no_improve >= early_stopping_patience:
            print(f"\n🛑 Early stopping at epoch {epoch+1}")
            break
    
    if (epoch + 1) % 5 == 0:
        print(f"   Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train F1: {train_f1:.4f}")
        print(f"   Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

# ============================================================
# TESTING
# ============================================================
print("\n" + "="*60)
print("🧪 FINAL TEST RESULTS")
print("="*60)

model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "best_model.pth")))
model.eval()

test_preds, test_labels = [], []
with torch.no_grad():
    for mel, label in test_loader:
        mel = mel.to(DEVICE)
        outputs = model(mel)
        test_preds.extend(outputs.argmax(dim=1).cpu().numpy())
        test_labels.extend(label.numpy())

test_accuracy = accuracy_score(test_labels, test_preds)
test_f1 = f1_score(test_labels, test_preds, average='macro')
cm = confusion_matrix(test_labels, test_preds)

print(f"\n📊 METRICS")
print(f"   Accuracy: {test_accuracy:.4f}")
print(f"   F1-Score (macro): {test_f1:.4f}")

print(f"\n📋 Classification Report:")
print(classification_report(test_labels, test_preds, target_names=CLASS_NAMES, zero_division=0))

# ============================================================
# VISUALIZATIONS
# ============================================================
print("\n📈 Generating visualizations...")

# Create figure with subplots
fig = plt.figure(figsize=(16, 12))

# 1. Training curves
ax1 = plt.subplot(2, 3, 1)
ax1.plot(train_losses, label='Train Loss', linewidth=2)
ax1.plot(val_losses, label='Val Loss', linewidth=2)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Accuracy curves
ax2 = plt.subplot(2, 3, 2)
ax2.plot(train_accs, label='Train Accuracy', linewidth=2)
ax2.plot(val_accs, label='Val Accuracy', linewidth=2)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training & Validation Accuracy', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Confusion Matrix
ax3 = plt.subplot(2, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'},
            ax=ax3)
ax3.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
ax3.set_ylabel('True Label')
ax3.set_xlabel('Predicted Label')

# 4. Per-class accuracy
ax4 = plt.subplot(2, 3, 4)
per_class_acc = []
for i in range(NUM_CLASSES):
    mask = np.array(test_labels) == i
    if mask.sum() > 0:
        acc = (np.array(test_preds)[mask] == i).sum() / mask.sum()
        per_class_acc.append(acc)
    else:
        per_class_acc.append(0)

colors = plt.cm.Set3(np.linspace(0, 1, NUM_CLASSES))
ax4.bar(CLASS_NAMES, per_class_acc, color=colors)
ax4.set_ylabel('Accuracy')
ax4.set_title('Per-Class Accuracy', fontsize=12, fontweight='bold')
ax4.set_ylim([0, 1.1])
for i, v in enumerate(per_class_acc):
    ax4.text(i, v + 0.03, f'{v:.2%}', ha='center', fontweight='bold')

# 5. Class distribution
ax5 = plt.subplot(2, 3, 5)
unique, counts = np.unique(test_labels, return_counts=True)
ax5.bar([CLASS_NAMES[i] for i in unique], counts, color=colors)
ax5.set_ylabel('Count')
ax5.set_title('Test Set Class Distribution', fontsize=12, fontweight='bold')
for i, (u, c) in enumerate(zip(unique, counts)):
    ax5.text(i, c + 1, str(c), ha='center', fontweight='bold')

# 6. Prediction distribution
ax6 = plt.subplot(2, 3, 6)
unique_pred, counts_pred = np.unique(test_preds, return_counts=True)
ax6.bar([CLASS_NAMES[i] for i in unique_pred], counts_pred, color=colors)
ax6.set_ylabel('Count')
ax6.set_title('Predicted Class Distribution', fontsize=12, fontweight='bold')
for i, (u, c) in enumerate(zip(unique_pred, counts_pred)):
    ax6.text(i, c + 1, str(c), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'results.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/results.png")
plt.show()

# Save detailed confusion matrix
fig2, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'},
            ax=ax, annot_kws={'size': 12})
ax.set_title('Confusion Matrix - Detailed', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
print(f"✅ Saved: {OUTPUT_DIR}/confusion_matrix.png")
plt.show()

print("\n" + "="*60)
print("✅ ALL DONE!")
print("="*60)
