from snowflake.snowpark import Session
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import io

def train_respiratory_model(session: Session) -> str:

    return f"✅ Modèle Random Forest entraîné et sauvegardé sur @STG_RESPIRATORY_SOUNDS. Accuracy: {model.score(X_test, y_test):.2f}"