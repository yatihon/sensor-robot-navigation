from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from load_data import load_data
from split_data import sequential_split

def evaulate_logistic_regression(sensor_count, class_weight):
    x, y = load_data(sensor_count)
    (xtr, xvalid, _, ytr, yvalid, _) = sequential_split(x, y)

    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight=class_weight, random_state=42))
    model.fit(xtr, ytr)
    predictions = model.predict(xvalid)
    accuracy = accuracy_score(yvalid, predictions)
    macro_f1 = f1_score(yvalid, predictions, average="macro")
    return accuracy, macro_f1

for sensor_count in (24, 4, 2):
    for class_weight in (None, "balanced"):
        accuracy, macro_f1 = evaulate_logistic_regression(sensor_count, class_weight)
        weight_name = ("Обычные веса" if class_weight is None else "balanced")
        print(
            f"{sensor_count:2d} признаков | "
            f"{weight_name:13s} | "
            f"accuracy={accuracy:.4f} | "
            f"macro-F1={macro_f1:.4f}"
        )