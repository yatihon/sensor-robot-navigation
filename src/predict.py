from pathlib import Path
import pandas as pd
import joblib
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "wall_follower.joblib"

def parse_arguments():
    parser = argparse.ArgumentParser(description="Предсказание команды движения робота")
    parser.add_argument("--front", type=float, required=True, help="Агреггированое состояние спереди")
    parser.add_argument("--left", type=float, required=True, help="Агреггированое состояние слева")
    return parser.parse_args()

def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Модель не найдена. Сначала запустите: python src/train_final_model.py")
    return joblib.load(MODEL_PATH)

def predict_command(front, left):
    if front < 0 or left < 0:
        raise ValueError("Расстояние не может быть отрицательным")
    artifact = load_model()
    model = artifact["model"]
    feature_names = artifact["feature_names"]
    features = pd.DataFrame([[front, left]], columns=feature_names)

    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    print(f"Предсказанная команда: {prediction}")
    print("\nВероятности классов:")
    probability_pairs = zip(model.classes_, probabilities)

    for class_name, probability in sorted(probability_pairs, key=lambda pair : pair[1], reverse=True):
        print(f"  {class_name:20} {probability:.2%}")
    return prediction

if __name__ == "__main__":
    args = parse_arguments()
    predict_command(front=args.front, left=args.left)


