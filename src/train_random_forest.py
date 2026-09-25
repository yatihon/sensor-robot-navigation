from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from load_data import load_data
from split_data import sequential_split

def evaulate_random_forest(sensor_count):
    x, y = load_data(sensor_count)
    xtr, xvalid, _, ytr, yvalid, _ = sequential_split(x, y)
    model = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=1, random_state=42, n_jobs=-1)
    model.fit(xtr, ytr)
    train_predictions = model.predict(xtr)
    validation_predictions = model.predict(xvalid)

    train_accuracy = accuracy_score(ytr, train_predictions)
    train_f1 = f1_score(ytr, train_predictions, average="macro")
    validation_accuracy = accuracy_score(yvalid, validation_predictions)
    validation_f1 = f1_score(yvalid, validation_predictions, average="macro")

    print(f"\nПризнаков: {sensor_count}")
    print(
        f"Train      | accuracy={train_accuracy:.4f} "
        f"| macro-F1={train_f1:.4f}"
    )
    print(
        f"Validation | accuracy={validation_accuracy:.4f} "
        f"| macro-F1={validation_f1:.4f}"
    )
    print(
        f"Разрыв по macro-F1: "
        f"{train_f1 - validation_f1:.4f}"
    )

if __name__ == "__main__":
    for sensor_count in (24, 4, 2):
        evaulate_random_forest(sensor_count)