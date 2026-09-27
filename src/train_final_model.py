from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestClassifier
from load_data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIRECTORY = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIRECTORY / "wall_follower.joblib"

def train_final_model():
    x, y = load_data(sensor_count=2)
    model = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=42, n_jobs=-1)
    model.fit(x, y)
    artifact = {"model": model, "feature_names": list(x.columns), "sensor_count": 2}

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)
    print(f"Модель обучена на {len(x)} объектах.")
    print(f"Модель сохранена: {MODEL_PATH}")

if __name__ == "__main__":
    train_final_model()