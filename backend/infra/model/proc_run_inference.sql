CREATE OR REPLACE PROCEDURE M2_ISD_EQUIPE_1_DB.MODEL.RUN_INFERENCE()
RETURNS VARCHAR
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
import sys, os, pickle, torch, numpy as np, tempfile, uuid
from datetime import datetime
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

def run_inference(session):
    import_dir = sys._xoptions.get("snowflake_import_directory", "/tmp")

    # ── 1. Charger metadata & modèle (identique à TEST_SINGLE_FILE) ──
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

    # ── 2. Helper preprocessing (identique à TEST_SINGLE_FILE) ──
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

    def get_action(pred_class: str, confidence: float) -> tuple[str, str]:
        
        # Returns (action, detailed_action)
        # action: "RAS" | "surveillance_7j" | "surveillance_48h" | "consultation_24h" | "urgent_6h"
        c = confidence  # 0.0 → 1.0

        ACTIONS = {
            "Healthy": [
                (0.60, "RAS",
                "Aucune anomalie détectée. Suivi préventif annuel suffisant. "
                "Consulter si dyspnée ou toux persistante > 3 semaines."),
                (0.00, "surveillance_7j",
                "Profil sain probable mais certitude faible. Consultation de contrôle conseillée si symptômes persistants."),
            ],
            "Asthma": [
                (0.85, "consultation_24h",
                "Asthme très probable. Consultation sous 24h, traitement de fond (CSI ± LABA) à évaluer. "
                "Vérifier disponibilité d'un bronchodilatateur d'urgence (salbutamol). "
                "⚠ SpO₂ < 92% ou crise nocturne → SAMU (15)."),
                (0.60, "surveillance_48h",
                "Profil asthmatique probable. Suivi médical 48–72h, spirométrie pour confirmer. "
                "⚠ Sibilances ou dyspnée aiguë → consultation urgente."),
                (0.00, "surveillance_7j",
                "Asthme possible mais incertain. Téléconsultation ou spirométrie recommandée. "
                "Éviter déclencheurs connus (froid, poussière, effort)."),
            ],
            "Pneumonia": [
                (0.85, "urgent_6h",
                "Pneumonie très probable. Prise en charge sous 6h, score CRB-65 à évaluer pour hospitalisation. "
                "⚠ SpO₂ < 90% ou confusion → SAMU (15) immédiat."),
                (0.60, "consultation_24h",
                "Pneumonie probable. Consultation dans les 24h, antibiothérapie probable (amoxicilline 1ère intention). "
                "⚠ SpO₂ < 94% ou FR > 25/min → urgences."),
                (0.00, "surveillance_48h",
                "Pneumonie possible. Radio thoracique + bilan biologique (NFS, CRP) nécessaires. "
                "Surveiller température, fréquence respiratoire et état général."),
            ],
            "COPD": [
                (0.85, "consultation_24h",
                "BPCO très probable. Consultation pneumologique sous 48h, stadification GOLD indispensable. "
                "⚠ SpO₂ < 88% → oxygénothérapie + urgences ; exacerbation sévère → hospitalisation."),
                (0.60, "surveillance_7j",
                "BPCO probable. Consultation pneumologique sous 7 jours, arrêt du tabac prioritaire. "
                "Évaluation bronchodilatateurs longue durée (LAMA/LABA), vaccination grippe + pneumocoque. "
                "⚠ Exacerbation (↑ dyspnée + ↑ expectoration) → consultation sous 24h."),
                (0.00, "surveillance_7j",
                "BPCO possible. Spirométrie nécessaire pour confirmer (VEMS/CVF < 0.70 post-bronchodilatateur). "
                "Évaluer tabagisme et expositions professionnelles."),
            ],
            "Bronchitis": [
                (0.85, "surveillance_48h",
                "Bronchite aiguë très probable. Consultation sous 48h pour écarter surinfection. "
                "⚠ FR > 25/min ou SpO₂ < 94% → éliminer pneumonie en urgence."),
                (0.60, "surveillance_7j",
                "Bronchite aiguë probable. Généralement virale, résolution en 7–14j. "
                "Pas d'antibiotiques en 1ère intention sauf expectoration purulente > 10j ou terrain fragile."),
                (0.00, "surveillance_7j",
                "Bronchite possible. Traitement symptomatique (repos, hydratation). "
                "Consulter si fièvre > 38.5°C au-delà de 5 jours."),
            ],
        }

        levels = ACTIONS.get(pred_class)
        if not levels:
            return "surveillance_7j", f"Classe non reconnue : {pred_class}. Téléconsultation conseillée."

        for threshold, action, detail in levels:
            if c >= threshold:
                pct = f"{c:.0%}"
                return action, f"Détection à {pct} : {detail}"

        return "surveillance_7j", f"Résultat incertain ({pred_class}, {c:.0%}). Téléconsultation conseillée."
        
    # ── 3. Lire la table : fichiers pas encore prédits ──
    rows = session.sql("""
        SELECT
            m.FILE_NAME,
            m.MEL_NPY_FILENAME,
            m.PATIENT_ID,
            m.PHARMACIE_ID
        FROM M2_ISD_EQUIPE_1_DB.INGESTED.PROCESSED_SOUNDS_METADATA m
        LEFT JOIN M2_ISD_EQUIPE_1_DB.APP.PREDICTIONS p
            ON p.AUDIO_FILE_NAME = m.FILE_NAME
        WHERE p.AUDIO_FILE_NAME IS NULL
    """).collect()

    if not rows:
        return "ℹ️ Aucun nouveau fichier à traiter."

    # ── 4. Inférence fichier par fichier ──
    ok, errors = 0, []

    for row in rows:
        file_name        = row["FILE_NAME"]
        mel_npy_filename = row["MEL_NPY_FILENAME"]
        patient_id       = row["PATIENT_ID"]
        pharmacie_id     = row["PHARMACIE_ID"]

        # Construire le chemin stage vers le .npy
        stage_path = f"@M2_ISD_EQUIPE_1_DB.INGESTED.STG_MEL_NPY/{mel_npy_filename}"

        try:
            probs      = predict_from_npy(stage_path)
            prob_map   = {cls: float(probs[i]) for i, cls in enumerate(class_names)}
            pred_class = class_names[int(np.argmax(probs))]
            confidence = float(np.max(probs))

            def esc(v):
                if v is None: return "NULL"
                if isinstance(v, str): return "'" + v.replace("'", "''") + "'"
                return str(v)

            session.sql(f"""
                INSERT INTO M2_ISD_EQUIPE_1_DB.APP.PREDICTIONS (
                    PREDICTION_ID, PREDICTED_AT,
                    PATIENT_ID, PHARMACIE_ID, AUDIO_FILE_NAME,
                    PCT_ASTHMA, PCT_COPD, PCT_BRONCHIAL, PCT_PNEUMONIA, PCT_HEALTHY,
                    PCT_CONFIANCE, ACTION, MODEL_VERSION
                ) VALUES (
                    UUID_STRING(), CURRENT_TIMESTAMP(),
                    {esc(patient_id)}, {esc(pharmacie_id)}, {esc(file_name)},
                    {round(prob_map.get('Asthma',    0.0) * 100, 2)},
                    {round(prob_map.get('COPD',      0.0) * 100, 2)},
                    {round(prob_map.get('Bronchial', 0.0) * 100, 2)},
                    {round(prob_map.get('Pneumonia', 0.0) * 100, 2)},
                    {round(prob_map.get('Healthy',   0.0) * 100, 2)},
                    {round(confidence * 100, 2)},
                    {esc(get_action(pred_class, confidence))},
                    {esc(model_version)}
                )
            """).collect()
            ok += 1

        except Exception as e:
            errors.append(f"{file_name} → {str(e)}")

    # ── 5. Résumé ──
    summary = f"✅ {ok} fichier(s) traité(s).\n❌ {len(errors)} erreur(s)."
    if errors:
        summary += "\nDétail :\n" + "\n".join(errors[:20])
    return summary
$$;
