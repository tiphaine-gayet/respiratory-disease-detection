# ============================================================
# CREATE INFERENCE UDF & PIPELINE
# ============================================================
import os
import pickle
from dotenv import load_dotenv
from snowflake.snowpark import Session

load_dotenv()

connection_params = {
    "account": os.getenv("SNOWFLAKE_ACCOUNT"),
    "user": os.getenv("SNOWFLAKE_USER"),
    "password": os.getenv("SNOWFLAKE_TOKEN"),
    "role": os.getenv("SNOWFLAKE_ROLE"),
    "warehouse": "HACKATHON_WH",
    "database": "M2_ISD_EQUIPE_1_DB",
    "schema": "APP"
}

print("🔌 Connexion à Snowflake...")
session = Session.builder.configs(connection_params).create()
print("✅ Connecté\n")

# ============================================================
# 1. READ MODEL CONFIG
# ============================================================
print("📖 Lecture de la configuration du modèle...")
MODEL_DIR = "./backend/models/resnet"
META_PATH = os.path.join(MODEL_DIR, "model_metadata.pkl")

with open(META_PATH, "rb") as f:
    config = pickle.load(f)

CLASS_NAMES = config["class_names"]
NUM_CLASSES = len(CLASS_NAMES)

print(f"✅ Chargé: {NUM_CLASSES} classes\n")
view_code = """
CREATE OR REPLACE VIEW INGESTED.ingested_inference_data AS
SELECT
    FILE_NAME as SAMPLE_ID,
    FEAT_MEL
FROM M2_ISD_EQUIPE_1_DB.PROCESSED.TRAINING_DATA_V
WHERE FEAT_MEL IS NOT NULL
"""

try:
    session.sql(view_code).collect()
    print("   ✅ Vue créée\n")
except Exception as e:
    print(f"   ⚠️  Erreur: {str(e)[:50]}\n")

# ============================================================
# 2. CREATE UDF: udf_respiratory_predict
# ============================================================
print("⚙️  Création de la UDF udf_respiratory_predict...")

udf_code = f'''
import torch
import torch.nn as nn
import numpy as np
import json
from torchvision.models import resnet34

CLASS_NAMES = {CLASS_NAMES}
NUM_CLASSES = {NUM_CLASSES}

_model = None
_device = None

def get_model():
    global _model, _device
    
    if _model is not None:
        return _model
    
    try:
        _device = torch.device('cpu')
        
        class ResNetAudio(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = resnet34(weights=None)
                self.model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
                self.dropout = nn.Dropout(p=0.3)
                self.model.fc = nn.Linear(512, NUM_CLASSES)
            
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
        
        _model = ResNetAudio().to(_device)
        _model.load_state_dict(torch.load('best_model.pth', map_location=_device))
        _model.eval()
    except Exception as e:
        _model = None
        raise Exception(f"Model loading failed: {{str(e)}}")
    
    return _model

def udf_respiratory_predict(feat_mel: bytes):
    try:
        if feat_mel is None:
            return json.dumps({{"error": "Missing FEAT_MEL"}})
        
        # Deserialize
        feat_array = np.frombuffer(feat_mel, dtype=np.float32)
        if feat_array.size != 224 * 224:
            return json.dumps({{"error": f"Wrong size: {{feat_array.size}}, expected {{224*224}}"}})
        
        feat_array = feat_array.reshape(1, 224, 224)
        tensor = torch.tensor(feat_array, dtype=torch.float32).unsqueeze(0)
        
        # Predict
        model = get_model()
        with torch.no_grad():
            logits = model(tensor.to(_device))
            probs = torch.softmax(logits, dim=1)
        
        # Results
        pred_idx = int(torch.argmax(probs, dim=1).cpu().numpy()[0])
        confidence = float(probs[0, pred_idx].cpu().numpy())
        pred_class = CLASS_NAMES[pred_idx]
        
        probs_dict = {{}}
        for i in range(NUM_CLASSES):
            probs_dict[CLASS_NAMES[i]] = float(probs[0, i].cpu().numpy())
        
        return json.dumps({{
            "prediction": pred_class,
            "confidence": confidence,
            "probabilities": probs_dict
        }})
    
    except Exception as e:
        return json.dumps({{"error": f"Prediction failed: {{str(e)[:200]}}"}})
'''

try:
    session.sql("DROP FUNCTION IF NOT EXISTS APP.udf_respiratory_predict(BINARY)").collect()
except:
    pass

try:
    session.sql(f"""
    CREATE FUNCTION APP.udf_respiratory_predict(feat_mel BINARY)
    RETURNS STRING
    LANGUAGE PYTHON
    RUNTIME_VERSION = '3.11'
    PACKAGES = ('pytorch', 'torchvision', 'numpy')
    IMPORTS = ('@STG_MODELS/respiratory_disease_model/best_model.pth')
    HANDLER = 'udf_respiratory_predict'
    AS
    $$
    {udf_code}
    $$
    """).collect()
    print("   ✅ UDF créée\n")
except Exception as e:
    print(f"   ⚠️  Erreur UDF:\n{str(e)}\n")

# ============================================================
# 3. CREATE PROCEDURE: sp_run_inference
# ============================================================
print("🔄 Création de la procédure sp_run_inference...")

sp_code = """
CREATE OR REPLACE PROCEDURE APP.sp_run_inference()
RETURNS VARCHAR
LANGUAGE SQL
AS
$$
BEGIN
    INSERT INTO APP.PREDICTION (SAMPLE_ID, PREDICTION, CONFIDENCE, PROBABILITIES)
    SELECT
        SAMPLE_ID,
        PARSE_JSON(result):prediction::VARCHAR,
        PARSE_JSON(result):confidence::FLOAT,
        PARSE_JSON(result):probabilities
    FROM (
        SELECT
            SAMPLE_ID,
            APP.udf_respiratory_predict(FEAT_MEL) as result
        FROM INGESTED.ingested_inference_data
    );
    
    RETURN 'Inférence complétée';
END
$$
"""

try:
    session.sql("DROP PROCEDURE IF NOT EXISTS APP.sp_run_inference(INT)").collect()
except:
    pass

try:
    session.sql(sp_code).collect()
    print("   ✅ Procédure créée\n")
except Exception as e:
    print(f"   ⚠️  Erreur Procédure:\n{str(e)}\n")

# ============================================================
# 7. SUMMARY
# ============================================================
print("=" * 70)
print("✅ PIPELINE D'INFÉRENCE CRÉÉE")
print("=" * 70)
print("""
📁 Composants créés:

1️⃣  VUE: INGESTED.ingested_inference_data
    └─ Lit: FILE_NAME (as SAMPLE_ID), FEAT_MEL

2️⃣  UDF: APP.udf_respiratory_predict(feat_mel BINARY)
    └─ Entrée: 1 feature BINARY (224x224)
    └─ Sortie: JSON {prediction, confidence, probabilities}

3️⃣  PROCÉDURE: APP.sp_run_inference(limit_rows INT)
    └─ Orchestre: VIEW → UDF → TABLE APP.PREDICTION

4️⃣  TABLE: APP.PREDICTION
    └─ Colonnes: prediction_id, sample_id, prediction, confidence, probabilities, created_at

🚀 UTILISATION:

   -- Exécuter l'inférence (100 premiers samples)
   CALL APP.sp_run_inference(100);
   
   -- Changer le nombre de samples
   CALL APP.sp_run_inference(50);
   
   -- Voir les résultats
   SELECT * FROM APP.PREDICTION ORDER BY CREATED_AT DESC;
   
   -- Stats par classe
   SELECT 
       PREDICTION, 
       COUNT(*) as count,
       ROUND(AVG(CONFIDENCE), 4) as avg_conf
   FROM APP.PREDICTION
   GROUP BY PREDICTION;
""")

session.close()
print("\n✅ Configuration complète!\n")
