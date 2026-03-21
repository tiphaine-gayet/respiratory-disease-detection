# ============================================================
# model.py
# CNN basé sur les features de RESPIRATORY_SOUNDS_METADATA
# Colonnes : FILE_NAME, CLASS, SAMPLE_RATE, DURATION_S, 
#            N_SAMPLES, AMPLITUDE_MAX, RMS
# ============================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class RespiratoryClassifier(nn.Module):
    """
    Réseau entièrement dense (MLP) basé sur les 5 features
    numériques déjà présentes dans Snowflake :
        - SAMPLE_RATE
        - DURATION_S
        - N_SAMPLES
        - AMPLITUDE_MAX
        - RMS
    
    Entrée  : (B, 5)  — vecteur de features
    Sortie  : (B, 5)  — logits pour 5 classes
    """

    def __init__(self, 
                 input_dim:   int = 5,
                 num_classes: int = 5,
                 dropout:     float = 0.0):
        super().__init__()

        self.network = nn.Sequential(

            # Bloc 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Bloc 2
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            # Bloc 3
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout / 2),

            # Bloc 4
            nn.Linear(64, 32),
            nn.ReLU(),

            # Sortie
            nn.Linear(32, num_classes)
        )

        # Initialisation des poids
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Retourne les probabilités softmax."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


if __name__ == "__main__":
    model = RespiratoryClassifier()
    print(model)

    # Test forward pass
    x = torch.randn(8, 5)          # batch=8, features=5
    out   = model(x)
    proba = model.predict_proba(x)

    print(f"\n✅ Input  shape : {x.shape}")
    print(f"✅ Output shape : {out.shape}")
    print(f"✅ Proba  shape : {proba.shape}")
    print(f"✅ Somme probas : {proba.sum(dim=1)}")

    total = sum(p.numel() for p in model.parameters())
    print(f"\n📐 Paramètres totaux : {total:,}")
