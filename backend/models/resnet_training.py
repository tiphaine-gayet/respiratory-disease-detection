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

from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score, roc_curve, auc, recall_score
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
LR = 1e-4
EPOCHS = 40
IMG_SIZE = (224, 224)
ACCUMULATION_STEPS = 4
# Initialized after regularization selection
EARLY_STOPPING_PATIENCE = 8  # Défaut - sera modifié par menu de régularisation
DROPOUT_RATE = 0.3  # Défaut - sera modifié par menu de régularisation
AUGMENTATION_NOISE_PROB = 0.5  # Défaut - sera modifié par menu de régularisation
AUGMENTATION_NOISE_AMPLITUDE = 0.005  # Défaut - sera modifié par menu de régularisation
AUGMENTATION_SHIFT_PROB = 0.5  # Défaut - sera modifié par menu de régularisation
AUGMENTATION_SHIFT_MAX = 10  # Défaut - sera modifié par menu de régularisation
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# GPU Optimization
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    print(f"✅ GPU disponible: {torch.cuda.get_device_name(0)}")
    print(f"   VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️  GPU non disponible, utilisation du CPU")

print(f"🖥️  Device utilisé: {DEVICE}\n")

LOCAL_CACHE = "./backend/models/cache_mel"
OUTPUT_DIR = "./backend/models/resnet"
os.makedirs(LOCAL_CACHE, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAINING_VIEW = "M2_ISD_EQUIPE_1_DB.PROCESSED.TRAINING_DATA_V"
CLASS_NAMES = ['asthma', 'bronchial', 'copd', 'healthy', 'pneumonia']

MODEL_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
META_PATH = os.path.join(OUTPUT_DIR, "model_metadata.pkl")

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
AVAILABLE_FEATURES = ['FEAT_MEL', 'FEAT_BANDWIDTH', 'FEAT_CHROMA', 'FEAT_CENTROID', 'FEAT_MFCC', 'FEAT_ZCR']

# Interactive feature selection
print("\n🎯 AVAILABLE FEATURES:")
for i, feat in enumerate(AVAILABLE_FEATURES, 1):
    print(f"   {i}. {feat}")

print("\n📝 Entrez les numéros des features à utiliser (séparés par des virgules)")
print("   Exemple: 1,2,3 ou laissez vide pour utiliser toutes les features")
user_input = input(">>> ").strip()

if user_input:
    try:
        selected_indices = [int(x.strip()) - 1 for x in user_input.split(",")]
        FEATURES = [AVAILABLE_FEATURES[i] for i in selected_indices if 0 <= i < len(AVAILABLE_FEATURES)]
        if not FEATURES:
            print("⚠️  Sélection invalide, utilisation de toutes les features")
            FEATURES = AVAILABLE_FEATURES
    except:
        print("⚠️  Erreur de parsing, utilisation de toutes les features")
        FEATURES = AVAILABLE_FEATURES
else:
    FEATURES = AVAILABLE_FEATURES

print(f"\n✅ Features sélectionnées: {FEATURES}\n")

feature_cols = ", ".join(FEATURES)

df = session.sql(f"""
    SELECT FILE_NAME, CLASS, CLASS_WEIGHT, {feature_cols}
    FROM {TRAINING_VIEW}
    WHERE FEAT_MEL IS NOT NULL AND FEAT_MFCC IS NOT NULL
""").to_pandas()

print(f"Loaded {len(df)} rows with features: {FEATURES}")

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
def download_from_stage(row, feature_name):
    """Download a single feature file from Snowflake stage."""
    feat_path = row[feature_name]
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
        print(f"❌ Could not download: {filename}")
        return None

    return local_path

print("📥 Downloading feature files...")
for feat in FEATURES:
    print(f"  Downloading {feat}...")
    df[f"LOCAL_{feat}"] = df.apply(lambda row: download_from_stage(row, feat), axis=1)

# Filter rows where all features were downloaded successfully
for feat in FEATURES:
    df = df[df[f"LOCAL_{feat}"].notnull()]

df = df.reset_index(drop=True)
print(f"✅ Successfully downloaded {len(df)} samples with all features")

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
# COMPUTE CLASS WEIGHTS FOR LOSS
# ============================================================
# Use CLASS_WEIGHT from Snowflake table
class_weights_dict = train_df.groupby("LABEL")["CLASS_WEIGHT"].mean().to_dict()
class_weights = torch.tensor([class_weights_dict.get(i, 1.0) for i in range(NUM_CLASSES)], dtype=torch.float32)

# Sensitivity adjustment: increase weight to reduce false negatives (increase recall)
print("⚠️  Ajustement de sensibilité pour réduire les faux négatifs:")
print("   1. Normal (pas d'ajustement)")
print("   2. Sensibilité haute (+50%)")
print("   3. Sensibilité très haute (+100%)")
sensitivity_input = input("Choisissez (1-3, défaut: 1): ").strip()

sensitivity_factor = 1.0
if sensitivity_input == "2":
    sensitivity_factor = 1.5
    print("📈 Sensibilité haute activée (+50%)")
elif sensitivity_input == "3":
    sensitivity_factor = 2.0
    print("📈 Sensibilité très haute activée (+100%)")
else:
    print("Sensibilité normale")

class_weights = class_weights * sensitivity_factor
class_weights = class_weights / class_weights.sum() * NUM_CLASSES  # Normalize
print(f"Class Weights: {class_weights.numpy()}\n")

# Metric selection: Choose what to optimize during training
print("📊 Choix de métrique d'optimisation:")
print("   1. Recall (minimiser les faux négatifs) - RECOMMANDÉ pour diagnostic médical")
print("   2. F1-Score (équilibre precision/recall)")
metric_input = input("Choisissez (1-2, défaut: 1): ").strip()

OPTIMIZATION_METRIC = "recall"  # default
if metric_input == "2":
    OPTIMIZATION_METRIC = "f1"
    print("📈 Optimisation sur F1-Score")
else:
    print("📈 Optimisation sur Recall")

# ============================================================
# REGULARIZATION SELECTION
# ============================================================
# Regularization profiles: (dropout_rate, early_stopping_patience, noise_prob, noise_amp, shift_prob, shift_max)
REGULARIZATION_PROFILES = {
    "1": {  # Light
        "dropout": 0.1,
        "early_stopping": 10,
        "noise_prob": 0.3,
        "noise_amplitude": 0.002,
        "shift_prob": 0.3,
        "shift_max": 5,
        "description": "🟢 Light (peu d'overfitting, apprentissage plus rapide)"
    },
    "2": {  # Normal (défaut)
        "dropout": 0.3,
        "early_stopping": 8,
        "noise_prob": 0.5,
        "noise_amplitude": 0.005,
        "shift_prob": 0.5,
        "shift_max": 10,
        "description": "🟡 Normal (équilibre)"
    },
    "3": {  # Heavy
        "dropout": 0.5,
        "early_stopping": 5,
        "noise_prob": 0.7,
        "noise_amplitude": 0.01,
        "shift_prob": 0.7,
        "shift_max": 15,
        "description": "🔴 Heavy (forte régularisation, risque d'underfitting)"
    }
}

print("\n🎯 Choix du profil de régularisation:")
for key, profile in REGULARIZATION_PROFILES.items():
    print(f"   {key}. {profile['description']}")
reg_input = input("Choisissez (1-3, défaut: 2): ").strip()

selected_profile = REGULARIZATION_PROFILES.get(reg_input, REGULARIZATION_PROFILES["2"])
DROPOUT_RATE = selected_profile["dropout"]
EARLY_STOPPING_PATIENCE = selected_profile["early_stopping"]
AUGMENTATION_NOISE_PROB = selected_profile["noise_prob"]
AUGMENTATION_NOISE_AMPLITUDE = selected_profile["noise_amplitude"]
AUGMENTATION_SHIFT_PROB = selected_profile["shift_prob"]
AUGMENTATION_SHIFT_MAX = selected_profile["shift_max"]

print(f"\n✅ Profil de régularisation: {selected_profile['description']}")
print(f"   • Dropout: {DROPOUT_RATE}")
print(f"   • Early stopping patience: {EARLY_STOPPING_PATIENCE}")
print(f"   • Augmentation - Noise: prob={AUGMENTATION_NOISE_PROB}, amp={AUGMENTATION_NOISE_AMPLITUDE}")
print(f"   • Augmentation - Shift: prob={AUGMENTATION_SHIFT_PROB}, max={AUGMENTATION_SHIFT_MAX}\n")

# ============================================================
# NORMALIZATION FOR EACH FEATURE
# ============================================================
# Each feature has different ranges - we'll normalize each individually
FEATURE_RANGES = {
    'FEAT_MEL': (-80.0, 0.0),      # dB scale
    'FEAT_BANDWIDTH': (0.0, 1.0),  # typically normalized
    'FEAT_CHROMA': (0.0, 1.0),     # normalized
    'FEAT_CENTROID': (0.0, 1.0),   # normalized
    'FEAT_MFCC': (-20.0, 20.0),    # typically in this range
    'FEAT_ZCR': (0.0, 1.0),        # zero crossing rate [0,1]
}

# Target dimension for features (to stack them as separate channels)
FEATURE_FREQ_DIM = 64

MEL_DB_MIN = -80.0
MEL_DB_MAX = 0.0

def normalize_feature(feat: torch.Tensor, feat_name: str) -> torch.Tensor:
    """Normalize a feature to [0, 1] range based on its type."""
    feat_min, feat_max = FEATURE_RANGES[feat_name]
    feat = feat.clamp(feat_min, feat_max)
    return (feat - feat_min) / (feat_max - feat_min + 1e-8)

# ============================================================
# DATA AUGMENTATION (applied in feature space, after normalization)
# ============================================================
def augment(feat: torch.Tensor) -> torch.Tensor:
    """Augment features with noise and time shift. Input shape: (NUM_FEATURES, 64, T)"""
    # Gaussian noise - uses global AUGMENTATION_NOISE_PROB and AUGMENTATION_NOISE_AMPLITUDE
    if random.random() < AUGMENTATION_NOISE_PROB:
        feat = feat + AUGMENTATION_NOISE_AMPLITUDE * torch.randn_like(feat)

    # Time shift with zero-padding - uses global AUGMENTATION_SHIFT_PROB and AUGMENTATION_SHIFT_MAX
    if random.random() < AUGMENTATION_SHIFT_PROB:
        shift = random.randint(-AUGMENTATION_SHIFT_MAX, AUGMENTATION_SHIFT_MAX)
        feat = torch.roll(feat, shifts=shift, dims=2)  # dims=2 is time dimension
        if shift > 0:
            feat[:, :, :shift] = 0.0
        elif shift < 0:
            feat[:, :, shift:] = 0.0

    return feat

# ============================================================
# DATASET
# ============================================================
class RespiratoryDataset(Dataset):
    def __init__(self, df, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Load and normalize all features
        features_list = []
        time_dims = []
        
        for feat_name in FEATURES:
            local_path = row[f"LOCAL_{feat_name}"]
            feat_data = np.load(local_path)  # Load numpy array
            
            # Convert to tensor
            feat_tensor = torch.tensor(feat_data).float()
            
            # Ensure 2D shape (freq/channels, time)
            if feat_tensor.dim() == 1:
                # If 1D, expand to 2D by repeating
                feat_tensor = feat_tensor.unsqueeze(0)  # (1, T)
            elif feat_tensor.dim() == 3:
                # If 3D, take first dimension as is
                feat_tensor = feat_tensor.reshape(feat_tensor.shape[0], -1)  # (F, T)
            
            # Normalize the feature
            feat_tensor = normalize_feature(feat_tensor, feat_name)
            features_list.append(feat_tensor)
            time_dims.append(feat_tensor.shape[1])
        
        # Ensure all features have the same time dimension (take max and pad)
        max_time = max(time_dims)
        padded_features = []
        
        for feat_tensor in features_list:
            if feat_tensor.shape[1] < max_time:
                pad_size = max_time - feat_tensor.shape[1]
                feat_tensor = F.pad(feat_tensor, (0, pad_size), mode='constant', value=0.0)
            
            # Resize feature freq dimension to FEATURE_FREQ_DIM
            # Shape: (F, T) -> (FEATURE_FREQ_DIM, T)
            if feat_tensor.shape[0] != FEATURE_FREQ_DIM:
                feat_tensor_reshaped = feat_tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, F, T)
                feat_tensor_reshaped = F.interpolate(
                    feat_tensor_reshaped,
                    size=(FEATURE_FREQ_DIM, feat_tensor.shape[1]),
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0).squeeze(0)  # (FEATURE_FREQ_DIM, T)
                feat_tensor = feat_tensor_reshaped
            
            padded_features.append(feat_tensor)
        
        # All features now have shape (FEATURE_FREQ_DIM, T) - stack them as channels
        # Stack along new dimension to create (NUM_FEATURES, FEATURE_FREQ_DIM, T)
        combined = torch.stack(padded_features, dim=0)  # (6, 64, T)
        
        if self.train:
            combined = augment(combined)
        
        # Add batch dimension for interpolation: (1, 6, 64, T)
        combined = combined.unsqueeze(0)
        
        # Interpolate to standard size: (1, 6, 224, 224)
        combined = F.interpolate(
            combined,
            size=IMG_SIZE,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # (6, 224, 224)
        
        return combined, int(row["LABEL"])

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
# Calculate number of input channels based on features
NUM_CHANNELS = len(FEATURES)

class ResNetAudio(nn.Module):
    def __init__(self, num_channels=NUM_CHANNELS, dropout_rate=0.3):
        super().__init__()
        self.model = resnet34(weights=ResNet34_Weights.DEFAULT)

        # Adapt first conv layer to accept multiple feature channels
        pretrained_weight = self.model.conv1.weight.data  # (64, 3, 7, 7)
        self.model.conv1 = nn.Conv2d(num_channels, 64, 7, 2, 3, bias=False)
        
        # Average pretrained weights across color channels and replicate for each feature channel
        averaged_weight = pretrained_weight.mean(dim=1, keepdim=True)  # (64, 1, 7, 7)
        self.model.conv1.weight.data = averaged_weight.repeat(1, num_channels, 1, 1) / num_channels

        # Ajouter Dropout avant la couche FC
        self.dropout = nn.Dropout(p=dropout_rate)
        self.model.fc = nn.Linear(self.model.fc.in_features, NUM_CLASSES)

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.model.fc(x)
        return x

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
        "macro_recall": recall_score(y_true, y_pred, average="macro"),
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
model     = ResNetAudio(num_channels=NUM_CHANNELS, dropout_rate=DROPOUT_RATE).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))

print(f"\n🎯 Model Configuration:")
print(f"   Input Channels: {NUM_CHANNELS} ({', '.join(FEATURES)})")
print(f"   Output Classes: {NUM_CLASSES}")
print(f"   Dropout Rate: {model.dropout.p}")
print(f"   Loss Weight: CrossEntropyLoss with class weights\n")

best_f1 = 0.0
early_stopping_counter = 0
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

    # Libérer la cache GPU
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Choose metric for early stopping based on user selection
    best_metric_key = "macro_recall" if OPTIMIZATION_METRIC == "recall" else "macro_f1"
    current_metric = val_metrics[best_metric_key]

    if current_metric > best_f1:
        best_f1 = current_metric
        early_stopping_counter = 0  # Réinitialiser le compteur
        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names":      CLASS_NAMES,
            "img_size":         IMG_SIZE,
            "num_classes":      NUM_CLASSES,
            "num_channels":     NUM_CHANNELS,
            "features":         FEATURES,
            "val_f1":           best_f1,
            "feature_ranges":   FEATURE_RANGES,
            "dropout_rate":     DROPOUT_RATE,
            "early_stopping_patience": EARLY_STOPPING_PATIENCE,
        }, MODEL_PATH)

        with open(META_PATH, "wb") as f:
            pickle.dump({
                "class_names": CLASS_NAMES,
                "img_size":    IMG_SIZE,
                "num_channels": NUM_CHANNELS,
                "features":    FEATURES,
                "feature_ranges": FEATURE_RANGES,
                "dropout_rate": DROPOUT_RATE,
                "early_stopping_patience": EARLY_STOPPING_PATIENCE,
                "optimization_metric": OPTIMIZATION_METRIC,
                "augmentation_noise_prob": AUGMENTATION_NOISE_PROB,
                "augmentation_noise_amplitude": AUGMENTATION_NOISE_AMPLITUDE,
                "augmentation_shift_prob": AUGMENTATION_SHIFT_PROB,
                "augmentation_shift_max": AUGMENTATION_SHIFT_MAX,
            }, f)
        print(f"✅ Meilleur modèle sauvegardé ({best_metric_key}: {best_f1:.4f})")
    else:
        early_stopping_counter += 1
        print(f"⚠️  Pas d'amélioration ({early_stopping_counter}/{EARLY_STOPPING_PATIENCE})")

    print(f"\nEpoch {epoch+1}/{EPOCHS}  lr={scheduler.get_last_lr()[0]:.2e}")
    print(f"  Train Acc: {train_metrics['accuracy']:.4f}  |  Val Acc: {val_metrics['accuracy']:.4f}")
    print(f"  Train F1: {train_metrics['macro_f1']:.4f}  |  Val F1: {val_metrics['macro_f1']:.4f}")
    print(f"  Train Recall: {train_metrics['macro_recall']:.4f}  |  Val Recall: {val_metrics['macro_recall']:.4f}")
    
    # Custom stopping criterion: Train accuracy in [91%-96%] with train-val gap of 4-7% AND Val accuracy >= 91%
    train_acc = train_metrics['accuracy']
    val_acc = val_metrics['accuracy']
    acc_gap = train_acc - val_acc
    
    if 0.91 <= train_acc <= 0.96 and 0.04 <= acc_gap <= 0.07 and val_acc >= 0.91:
        print(f"\n✅ Critère de convergence atteint!")
        print(f"   Train Acc: {train_acc:.4f} ✓ (dans [91%, 96%])")
        print(f"   Val Acc: {val_acc:.4f} ✓ (>= 91%)")
        print(f"   Gap Train-Val: {acc_gap:.4f} ✓ (dans [4%, 7%])")
        print(f"🛑 Arrêt à l'époque {epoch+1} - Modèle bien régularisé et convergé")
        break
    
    # Early stopping
    if early_stopping_counter >= EARLY_STOPPING_PATIENCE:
        print(f"\n🛑 Early stopping activé à l'époque {epoch+1} - Pas d'amélioration depuis {EARLY_STOPPING_PATIENCE} épochs")
        break

# ============================================================
# TEST
# ============================================================
checkpoint = torch.load(MODEL_PATH)
model.load_state_dict(checkpoint["model_state_dict"])

test_metrics = evaluate(model, test_loader)

print("\n🧪 FINAL TEST RESULTS")
print(f"  Accuracy : {test_metrics['accuracy']:.4f}")
print(f"  Macro F1 : {test_metrics['macro_f1']:.4f}")
print(f"  Macro Recall : {test_metrics['macro_recall']:.4f}")
print(f"  AUC      : {test_metrics['auc']}")
print(f"\nConfusion matrix:\n{test_metrics['confusion_matrix']}")

# ============================================================
# VISUALISATIONS
# ============================================================
print("\n📊 Génération des graphiques...")

# Récupérer toutes les prédictions du test set
model.eval()
all_test_preds, all_test_labels, all_test_probs = [], [], []

with torch.no_grad():
    for mel, label in test_loader:
        mel, label = mel.to(DEVICE), label.to(DEVICE)
        probs = torch.softmax(model(mel), dim=1)
        
        all_test_preds.extend(probs.argmax(dim=1).cpu().numpy())
        all_test_labels.extend(label.cpu().numpy())
        all_test_probs.extend(probs.cpu().numpy())

all_test_probs = np.array(all_test_probs)

# Créer une figure avec 4 sous-graphiques
fig = plt.figure(figsize=(16, 12))

# 1. Matrice de confusion
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(test_metrics['confusion_matrix'], 
            annot=True, fmt='d', cmap='Blues',
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES,
            cbar_kws={'label': 'Count'},
            ax=ax1)
ax1.set_title('Matrice de Confusion', fontsize=14, fontweight='bold')
ax1.set_ylabel('True Label')
ax1.set_xlabel('Predicted Label')

# 2. Distribution des prédictions par classe
ax2 = plt.subplot(2, 2, 2)
pred_counts = pd.Series(all_test_preds).value_counts().reindex(range(NUM_CLASSES))
colors = plt.cm.Set3(range(NUM_CLASSES))
ax2.bar(CLASS_NAMES, pred_counts.values, color=colors)
ax2.set_title('Distribution des Prédictions', fontsize=14, fontweight='bold')
ax2.set_ylabel('Nombre de prédictions')
ax2.set_xlabel('Classe')
for i, v in enumerate(pred_counts.values):
    ax2.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

# 3. Courbes ROC (One-vs-Rest)
ax3 = plt.subplot(2, 2, 3)
colors_roc = plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))

all_test_labels_bin = label_binarize(all_test_labels, classes=range(NUM_CLASSES))

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(all_test_labels_bin[:, i], all_test_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax3.plot(fpr, tpr, color=colors_roc[i], lw=2, 
             label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.3f})')

ax3.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random')
ax3.set_xlim([0.0, 1.0])
ax3.set_ylim([0.0, 1.05])
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.set_title('Courbes ROC (One-vs-Rest)', fontsize=14, fontweight='bold')
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(alpha=0.3)

# 4. Probabilités prédites (heatmap)
ax4 = plt.subplot(2, 2, 4)
sample_indices = np.random.choice(len(all_test_probs), min(30, len(all_test_probs)), replace=False)
sample_indices = sorted(sample_indices)
probs_sample = all_test_probs[sample_indices]
labels_sample = [CLASS_NAMES[l] for l in all_test_labels_bin[sample_indices].argmax(axis=1)]

im = ax4.imshow(probs_sample, cmap='YlOrRd', aspect='auto')
ax4.set_xticks(range(NUM_CLASSES))
ax4.set_xticklabels(CLASS_NAMES, rotation=45)
ax4.set_ylabel('Échantillons de test')
ax4.set_title('Probabilités Prédites (30 échantillons)', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax4, label='Probabilité')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'test_results_visualization.png'), dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé: outputs/test_results_visualization.png")
plt.show()

# Graphique ROC supplémentaire (plus grand)
fig2, ax = plt.subplots(figsize=(10, 8))

for i in range(NUM_CLASSES):
    fpr, tpr, _ = roc_curve(all_test_labels_bin[:, i], all_test_probs[:, i])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2.5, 
            label=f'{CLASS_NAMES[i]} (AUC = {roc_auc:.4f})')

ax.plot([0, 1], [0, 1], 'k--', lw=2, alpha=0.5, label='Random Classifier')
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
ax.set_xlabel('Taux de Faux Positifs', fontsize=12)
ax.set_ylabel('Taux de Vrais Positifs', fontsize=12)
ax.set_title('Courbes ROC Détaillées (One-vs-Rest)', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'roc_curve_detailed.png'), dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé: outputs/roc_curve_detailed.png")
plt.show()

print("\n✅ Toutes les visualisations ont été générées!")