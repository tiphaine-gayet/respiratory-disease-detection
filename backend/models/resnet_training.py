# ============================================================
# IMPORTS
# ============================================================
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import random
import pickle

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import resnet34, ResNet34_Weights

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score

from dotenv import load_dotenv
from snowflake.snowpark import Session

# ============================================================
# CONFIG
# ============================================================
NUM_CLASSES = 5
BATCH_SIZE = 8
LR = 1e-4
EPOCHS = 40
IMG_SIZE = (224, 224)
ACCUMULATION_STEPS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOCAL_CACHE = "./cache_mel"
os.makedirs(LOCAL_CACHE, exist_ok=True)

TRAINING_VIEW = "M2_ISD_EQUIPE_1_DB.PROCESSED.TRAINING_DATA_V"
CLASS_NAMES = ['asthma', 'bronchial', 'copd', 'healthy', 'pneumonia']

MODEL_PATH = "best_model.pth"
META_PATH  = "model_metadata.pkl"

# ============================================================
# SNOWFLAKE CONNECTION
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

# ============================================================
# LOAD DATA
# ============================================================
df = session.sql(f"""
    SELECT FILE_NAME, CLASS, CLASS_WEIGHT, FEAT_MEL
    FROM {TRAINING_VIEW}
    WHERE FEAT_MEL IS NOT NULL
""").to_pandas()

print(f"Loaded {len(df)} rows")

# ============================================================
# LABEL ENCODING
# ============================================================
label_map = {cls: i for i, cls in enumerate(CLASS_NAMES)}
df["CLASS"] = df["CLASS"].str.strip().str.lower()
df["LABEL"] = df["CLASS"].map(label_map)

df = df[df["LABEL"].notnull()]
df["LABEL"] = df["LABEL"].astype(int)

# ============================================================
# DOWNLOAD FILES
# ============================================================
def download_from_stage(row):
    filename = row["FEAT_MEL"].split("/")[-1]
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
        print(f"❌ Could not download: {filename}")
        return None

    return local_path

print("📥 Downloading MEL files...")
df["LOCAL_MEL"] = df.apply(download_from_stage, axis=1)
df = df[df["LOCAL_MEL"].notnull()].reset_index(drop=True)

# ============================================================
# BASE FILE (LEAKAGE FIX)
# ============================================================
df["BASE_FILE"] = df["FILE_NAME"].str.replace("_aug.*", "", regex=True)

# ============================================================
# SPLIT (GROUP + STRATIFIED)
# ============================================================
from sklearn.model_selection import train_test_split

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
val_df   = df[df["BASE_FILE"].isin(val_files)]
test_df  = df[df["BASE_FILE"].isin(test_files)]

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")

# ============================================================
# NORMALIZATION
# The mel features were extracted with:
#   librosa.power_to_db(..., ref=np.max)
# This produces values in dB relative to the peak, range roughly [-80, 0].
# Since all audio was peak-normalized to 0.95 before feature extraction,
# inter-sample amplitude is already consistent.
# We simply rescale the known dB range to [0, 1] — no per-sample stats needed.
# ============================================================
MEL_DB_MIN = -80.0
MEL_DB_MAX =   0.0

def normalize_mel(mel: torch.Tensor) -> torch.Tensor:
    """Rescale power_to_db output from [-80, 0] dB to [0, 1]."""
    mel = mel.clamp(MEL_DB_MIN, MEL_DB_MAX)
    return (mel - MEL_DB_MIN) / (MEL_DB_MAX - MEL_DB_MIN)

# ============================================================
# DATA AUGMENTATION  (applied in feature space, after dB rescaling)
# ============================================================
def augment(mel: torch.Tensor) -> torch.Tensor:
    # Gaussian noise — small relative to [0,1] range
    if random.random() < 0.5:
        mel = mel + 0.005 * torch.randn_like(mel)

    # Time shift with zero-padding (no wrap-around)
    if random.random() < 0.5:
        shift = random.randint(-10, 10)
        mel = torch.roll(mel, shifts=shift, dims=2)
        if shift > 0:
            mel[:, :, :shift] = 0.0
        elif shift < 0:
            mel[:, :, shift:] = 0.0

    return mel

# ============================================================
# DATASET
# ============================================================
class RespiratoryDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        mel = np.load(row["LOCAL_MEL"])            # shape: (128, T), dB values
        mel = torch.tensor(mel).unsqueeze(0).float()  # → (1, 128, T)
        mel = normalize_mel(mel)                   # → [0, 1], preserves inter-sample relationships

        if self.train:
            mel = augment(mel)

        mel = F.interpolate(
            mel.unsqueeze(0),
            size=IMG_SIZE,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)                               # → (1, 224, 224)

        return mel, int(row["LABEL"])

    def __len__(self):
        return len(self.df)

# ============================================================
# LOADERS
# ============================================================
class_counts = train_df["LABEL"].value_counts().to_dict()
sample_weights = train_df["LABEL"].map(lambda x: 1.0 / class_counts[x]).values

train_loader = DataLoader(
    RespiratoryDataset(train_df, train=True),
    batch_size=BATCH_SIZE,
    sampler=WeightedRandomSampler(sample_weights, len(sample_weights)),
)
val_loader  = DataLoader(RespiratoryDataset(val_df,  train=False), batch_size=BATCH_SIZE)
test_loader = DataLoader(RespiratoryDataset(test_df, train=False), batch_size=BATCH_SIZE)

# ============================================================
# MODEL
# ============================================================
class ResNetAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = resnet34(weights=ResNet34_Weights.DEFAULT)

        pretrained_weight = self.model.conv1.weight.data
        self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.model.conv1.weight.data = pretrained_weight.mean(dim=1, keepdim=True)

        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)

    def forward(self, x):
        return self.model(x)

# ============================================================
# METRICS
# ============================================================
def compute_metrics(y_true, y_pred, y_probs):
    try:
        auc = roc_auc_score(y_true, y_probs, multi_class="ovr")
    except Exception:
        auc = None

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "auc": auc,
        "confusion_matrix": confusion_matrix(y_true, y_pred)
    }

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for mel, label in loader:
            mel, label = mel.to(DEVICE), label.to(DEVICE)
            probs = torch.softmax(model(mel), dim=1)
            preds = probs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return compute_metrics(all_labels, all_preds, all_probs)

def predict_with_confidence(model, mel):
    model.eval()
    with torch.no_grad():
        probs = torch.softmax(model(mel.to(DEVICE)), dim=1).cpu().numpy()[0]
    pred_class = np.argmax(probs)
    return {
        "probabilities": {CLASS_NAMES[i]: float(probs[i]) for i in range(NUM_CLASSES)},
        "prediction": CLASS_NAMES[pred_class],
        "confidence": float(probs[pred_class])
    }

# ============================================================
# TRAINING
# ============================================================
model     = ResNetAudio().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss()

best_f1 = 0.0
optimizer.zero_grad()

for epoch in range(EPOCHS):
    model.train()
    all_preds, all_labels, all_probs = [], [], []

    for step, (mel, label) in enumerate(train_loader):
        mel, label = mel.to(DEVICE), label.to(DEVICE)

        logits = model(mel)
        loss   = criterion(logits, label) / ACCUMULATION_STEPS
        probs  = torch.softmax(logits, dim=1)

        loss.backward()
        if (step + 1) % ACCUMULATION_STEPS == 0:
            optimizer.step()
            optimizer.zero_grad()

        preds = probs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(label.cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    # Flush remaining gradients
    if (step + 1) % ACCUMULATION_STEPS != 0:
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()

    train_metrics = compute_metrics(all_labels, all_preds, all_probs)
    val_metrics   = evaluate(model, val_loader)

    if val_metrics["macro_f1"] > best_f1:
        best_f1 = val_metrics["macro_f1"]
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names":      CLASS_NAMES,
            "img_size":         IMG_SIZE,
            "num_classes":      NUM_CLASSES,
            "val_f1":           best_f1,
            "mel_db_min":       MEL_DB_MIN,
            "mel_db_max":       MEL_DB_MAX,
        }, MODEL_PATH)

        with open(META_PATH, "wb") as f:
            pickle.dump({
                "class_names": CLASS_NAMES,
                "img_size":    IMG_SIZE,
                "mel_db_min":  MEL_DB_MIN,
                "mel_db_max":  MEL_DB_MAX,
            }, f)

    print(f"\nEpoch {epoch+1}/{EPOCHS}  lr={scheduler.get_last_lr()[0]:.2e}")
    print(f"  Train F1: {train_metrics['macro_f1']:.4f}  |  Val F1: {val_metrics['macro_f1']:.4f}")

# ============================================================
# TEST
# ============================================================
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint["model_state_dict"])

test_metrics = evaluate(model, test_loader)

print("\n🧪 FINAL TEST RESULTS")
print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
print(f"  Macro F1 : {test_metrics['macro_f1']:.4f}")
print(f"  AUC      : {test_metrics['auc']}")
print(f"\nConfusion matrix:\n{test_metrics['confusion_matrix']}")