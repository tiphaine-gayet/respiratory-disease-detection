# ============================================================
# train_test.py
# Pipeline complet : Snowflake → Train → Eval
# ============================================================

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    f1_score, classification_report,
    confusion_matrix, roc_auc_score
)
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns

from snowflake.snowpark import Session
from Model_CNN import RespiratoryClassifier

import os
from dotenv import load_dotenv


# ─────────────────────────────────────────────────────────────
# 1. CONNEXION & CHARGEMENT DEPUIS SNOWFLAKE
# ─────────────────────────────────────────────────────────────




load_dotenv()

connection_params = {
    "account":   os.getenv("SNOWFLAKE_ACCOUNT"),
    "user":      os.getenv("SNOWFLAKE_USER"),
    "password":  os.getenv("SNOWFLAKE_TOKEN"),
    "warehouse": os.getenv("SNOWFLAKE_WAREHOUSE", "COMPUTE_WH"),
    "database":  os.getenv("SNOWFLAKE_DATABASE", "M2_ISD_EQUIPE_1_DB"),
    "schema":    os.getenv("SNOWFLAKE_SCHEMA", "RAW")
}

session = Session.builder.configs(connection_params).create()
# ...reste du code...

# Chargement de la table complète
df = session.table("RESPIRATORY_SOUNDS_METADATA").to_pandas()
df.columns = df.columns.str.lower()

print(f"✅ Données chargées : {df.shape}")
print(df.dtypes)
print(df.head())
print(f"\n📊 Distribution des classes :\n{df['class'].value_counts()}")


# ─────────────────────────────────────────────────────────────
# 2. PRÉPARATION DES FEATURES
# ─────────────────────────────────────────────────────────────

# Features disponibles dans la BDD
FEATURE_COLS = [
    "sample_rate",
    "duration_s",
    "n_samples",
    "amplitude_max",
    "rms"
]
TARGET_COL = "class"

# Suppression des lignes avec valeurs manquantes
df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
print(f"\n📦 Après nettoyage : {len(df)} lignes")

# Encodage des labels
le = LabelEncoder()
df["label"] = le.fit_transform(df[TARGET_COL])

print(f"\n🏷️  Classes encodées : {dict(zip(le.classes_, le.transform(le.classes_)))}")

X = df[FEATURE_COLS].values.astype(np.float32)   # (N, 5)
y = df["label"].values.astype(np.int64)           # (N,)

# Normalisation (StandardScaler)
scaler = StandardScaler()
X = scaler.fit_transform(X).astype(np.float32)

# Sauvegarde du scaler pour l'inférence
import pickle
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

print(f"\n✅ Features shape : {X.shape}")
print(f"✅ Labels shape   : {y.shape}")
print(f"✅ Classes        : {list(le.classes_)}")


# ─────────────────────────────────────────────────────────────
# 3. SPLIT STRATIFIÉ 70 / 15 / 15
# ─────────────────────────────────────────────────────────────

# Train vs (Val + Test)
sss1 = StratifiedShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
train_idx, temp_idx = next(sss1.split(X, y))

# Val vs Test
sss2 = StratifiedShuffleSplit(n_splits=1, test_size=0.50, random_state=42)
val_idx, test_idx = next(sss2.split(X[temp_idx], y[temp_idx]))
val_idx  = temp_idx[val_idx]
test_idx = temp_idx[test_idx]

X_train, y_train = X[train_idx], y[train_idx]
X_val,   y_val   = X[val_idx],   y[val_idx]
X_test,  y_test  = X[test_idx],  y[test_idx]

print(f"\n📊 Split des données :")
print(f"   Train : {len(X_train)} ({np.bincount(y_train)})")
print(f"   Val   : {len(X_val)}   ({np.bincount(y_val)})")
print(f"   Test  : {len(X_test)}  ({np.bincount(y_test)})")


# ─────────────────────────────────────────────────────────────
# 4. DATASET PYTORCH
# ─────────────────────────────────────────────────────────────

class RespiratoryDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# WeightedRandomSampler pour compenser le déséquilibre
class_counts  = np.bincount(y_train)
class_weights = 1.0 / (class_counts + 1e-8)
sample_weights = class_weights[y_train]

sampler = WeightedRandomSampler(
    weights     = torch.FloatTensor(sample_weights),
    num_samples = len(y_train),
    replacement = True
)

BATCH_SIZE = 32

train_loader = DataLoader(
    RespiratoryDataset(X_train, y_train),
    batch_size = BATCH_SIZE,
    sampler    = sampler        # remplace shuffle=True
)
val_loader = DataLoader(
    RespiratoryDataset(X_val, y_val),
    batch_size = BATCH_SIZE,
    shuffle    = False
)
test_loader = DataLoader(
    RespiratoryDataset(X_test, y_test),
    batch_size = BATCH_SIZE,
    shuffle    = False
)


# ─────────────────────────────────────────────────────────────
# 5. ENTRAÎNEMENT
# ─────────────────────────────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Device : {device}")

# Poids pour la loss (gestion du déséquilibre)
loss_weights = torch.FloatTensor(class_weights / class_weights.sum()).to(device)

model     = RespiratoryClassifier(input_dim=5, num_classes=5, dropout=0.3).to(device)
criterion = nn.CrossEntropyLoss(weight=loss_weights)
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)

EPOCHS       = 300
best_val_f1  = 0.0
patience     = 15        # early stopping
no_improve   = 0

history = {
    "train_loss": [], "val_loss": [],
    "val_acc":    [], "val_f1":   []
}

print(f"\n🚀 Début de l'entraînement ({EPOCHS} epochs)...\n")

for epoch in range(1, EPOCHS + 1):

    # ── TRAIN ──────────────────────────────────────────────
    model.train()
    train_losses = []

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss   = criterion(logits, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_losses.append(loss.item())

    # ── VALIDATION ─────────────────────────────────────────
    model.eval()
    val_losses, val_preds, val_true = [], [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss   = criterion(logits, y_batch)
            val_losses.append(loss.item())

            preds = logits.argmax(dim=1).cpu().numpy()
            val_preds.extend(preds)
            val_true.extend(y_batch.cpu().numpy())

    scheduler.step()

    # ── MÉTRIQUES ──────────────────────────────────────────
    val_f1  = f1_score(val_true, val_preds, average='macro', zero_division=0)
    val_acc = np.mean(np.array(val_preds) == np.array(val_true))
    t_loss  = np.mean(train_losses)
    v_loss  = np.mean(val_losses)

    history["train_loss"].append(t_loss)
    history["val_loss"].append(v_loss)
    history["val_f1"].append(val_f1)
    history["val_acc"].append(val_acc)

    # ── EARLY STOPPING ─────────────────────────────────────
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        no_improve  = 0
        torch.save(model.state_dict(), "best_model.pth")
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"⏹️  Early stopping à l'epoch {epoch}")
            break

    # Log toutes les 10 epochs
    if epoch % 10 == 0 or epoch == 1:
        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"Train Loss: {t_loss:.4f} | "
            f"Val Loss: {v_loss:.4f} | "
            f"Val F1: {val_f1:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

print(f"\n🏆 Meilleur Val F1 macro : {best_val_f1:.4f}")


# ─────────────────────────────────────────────────────────────
# 6. ÉVALUATION SUR LE TEST SET
# ─────────────────────────────────────────────────────────────

print("\n" + "="*55)
print("📋 ÉVALUATION FINALE — TEST SET")
print("="*55)

# Chargement du meilleur modèle
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

test_preds, test_true, test_probas = [], [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        probas  = model.predict_proba(X_batch)
        preds   = probas.argmax(dim=1).cpu().numpy()

        test_preds.extend(preds)
        test_true.extend(y_batch.numpy())
        test_probas.extend(probas.cpu().numpy())

test_probas = np.array(test_probas)
class_names = list(le.classes_)

# ── Rapport de classification ───────────────────────────────
print("\n📊 Classification Report :")
print(classification_report(
    test_true, test_preds,
    target_names = class_names,
    zero_division = 0
))

# ── AUC-ROC ─────────────────────────────────────────────────
y_test_bin = label_binarize(test_true, classes=list(range(len(class_names))))
auc_macro  = roc_auc_score(
    y_test_bin, test_probas,
    average='macro', multi_class='ovr'
)
f1_macro   = f1_score(test_true, test_preds, average='macro', zero_division=0)
acc        = np.mean(np.array(test_preds) == np.array(test_true))

print(f"🎯 Accuracy      : {acc:.4f}")
print(f"🎯 F1 macro      : {f1_macro:.4f}")
print(f"🎯 AUC-ROC macro : {auc_macro:.4f}")


# ─────────────────────────────────────────────────────────────
# 7. VISUALISATIONS
# ─────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# ── Courbes d'apprentissage ──────────────────────────────────
axes[0].plot(history["train_loss"], label="Train Loss", color="steelblue")
axes[0].plot(history["val_loss"],   label="Val Loss",   color="coral")
axes[0].set_title("Courbes de Loss",       fontsize=13, fontweight='bold')
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history["val_f1"],  label="Val F1 macro", color="green")
axes[1].plot(history["val_acc"], label="Val Accuracy",  color="purple")
axes[1].set_title("Métriques de validation", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Epoch")
axes[1].legend()
axes[1].grid(alpha=0.3)

# ── Matrice de confusion ─────────────────────────────────────
cm = confusion_matrix(test_true, test_preds)
sns.heatmap(
    cm,
    annot        = True,
    fmt          = 'd',
    cmap         = 'Blues',
    xticklabels  = class_names,
    yticklabels  = class_names,
    ax           = axes[2]
)
axes[2].set_title("Matrice de confusion — Test Set", fontsize=13, fontweight='bold')
axes[2].set_ylabel("Vraie classe")
axes[2].set_xlabel("Classe prédite")
axes[2].tick_params(axis='x', rotation=45)

plt.suptitle(
    f"Résultats — Accuracy: {acc:.3f} | F1 macro: {f1_macro:.3f} | AUC: {auc_macro:.3f}",
    fontsize=14, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig("results.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n💾 Graphiques sauvegardés dans results.png")
print("💾 Modèle sauvegardé dans best_model.pth")
