CREATE OR REPLACE PROCEDURE M2_ISD_EQUIPE_1_DB.MODEL.RUN_INFERENCE(mel_npy_filename VARCHAR)
RETURNS VARIANT
LANGUAGE PYTHON
RUNTIME_VERSION = '3.9'
PACKAGES = ('snowflake-snowpark-python', 'pytorch', 'torchvision', 'numpy')
IMPORTS = (
    '@"M2_ISD_EQUIPE_1_DB"."MODEL"."STG_MODEL"/v0/best_model.pth',
    '@"M2_ISD_EQUIPE_1_DB"."MODEL"."STG_MODEL"/v0/model_metadata.pkl'
)
HANDLER = 'run_inference'
AS
$$
import sys, os, pickle, torch, numpy as np, tempfile
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

def run_inference(session, mel_npy_filename):
    import_dir = sys._xoptions.get("snowflake_import_directory", "/tmp")

    # ── 1. Charger metadata & modèle ──
    with open(os.path.join(import_dir, "model_metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)

    class_names    = metadata["class_names"]
    num_channels   = metadata["num_channels"]
    num_classes    = len(class_names)
    dropout_rate   = metadata["dropout_rate"]
    img_size       = metadata["img_size"]
    feature_ranges = metadata["feature_ranges"]
    model_version  = metadata.get("version", "1.0.0")

    class ResNetAudio(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = resnet34(weights=None)
            self.model.conv1 = nn.Conv2d(num_channels, 64, 7, 2, 3, bias=False)
            self.dropout = nn.Dropout(p=dropout_rate)
            self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        def forward(self, x):
            x = self.model.conv1(x); x = self.model.bn1(x)
            x = self.model.relu(x);  x = self.model.maxpool(x)
            x = self.model.layer1(x); x = self.model.layer2(x)
            x = self.model.layer3(x); x = self.model.layer4(x)
            x = self.model.avgpool(x); x = torch.flatten(x, 1)
            x = self.dropout(x); x = self.model.fc(x)
            return x

    model = ResNetAudio()
    ckpt  = torch.load(os.path.join(import_dir, "best_model.pth"), map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # ── 2. Télécharger le .npy et prédire ──
    def predict_from_npy(file_path):
        with tempfile.TemporaryDirectory() as tmp_dir:
            session.file.get(file_path, tmp_dir)
            local_file = os.path.join(tmp_dir, os.path.basename(file_path))
            feat_data  = np.load(local_file)

        t = torch.tensor(feat_data).float()
        if t.dim() == 1:   t = t.unsqueeze(0)
        elif t.dim() == 3: t = t.reshape(t.shape[0], -1)

        feat_min, feat_max = feature_ranges["FEAT_MEL"]
        t = (t.clamp(feat_min, feat_max) - feat_min) / (feat_max - feat_min + 1e-8)

        if t.shape[0] != 64:
            t = F.interpolate(
                t.unsqueeze(0).unsqueeze(0),
                size=(64, t.shape[1]),
                mode="bilinear", align_corners=False
            ).squeeze(0).squeeze(0)

        t = F.interpolate(
            t.unsqueeze(0).unsqueeze(0),
            size=img_size,
            mode="bilinear", align_corners=False
        ).squeeze(0)
        t = t.unsqueeze(0)  # batch dim

        with torch.no_grad():
            probs = torch.softmax(model(t), dim=1)[0].numpy()

        return probs

    stage_path = f"@M2_ISD_EQUIPE_1_DB.INGESTED.STG_MEL_NPY/{mel_npy_filename}"
    probs      = predict_from_npy(stage_path)
    prob_map   = {cls: round(float(probs[i]) * 100, 2) for i, cls in enumerate(class_names)}

    return {
        "pct_asthma":    prob_map.get("Asthma",    0.0),
        "pct_copd":      prob_map.get("COPD",      0.0),
        "pct_bronchial": prob_map.get("Bronchial", 0.0),
        "pct_pneumonia": prob_map.get("Pneumonia", 0.0),
        "pct_healthy":   prob_map.get("Healthy",   0.0),
        "pct_confiance": round(float(np.max(probs)) * 100, 2),
        "model_version": model_version,
    }
$$;
