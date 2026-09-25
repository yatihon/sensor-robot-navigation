from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from load_data import load_data
from split_data import sequential_split

x, y = load_data(sensor_count=2)
xtr, xvalid, xtest, ytr, yvalid, ytest = sequential_split(x, y)

x_train_final = pd.concat([xtr, xvalid], ignore_index=True)
y_train_final = pd.concat([ytr, yvalid], ignore_index=True)
model = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=1, random_state=42, n_jobs=-1)
model.fit(x_train_final, y_train_final)

test_predictions = model.predict(xtest)
test_accuracy = accuracy_score(ytest, test_predictions)
test_f1 = f1_score(ytest, test_predictions, average="macro")

print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test macro-F1: {test_f1:.4f}")
print("\nОтчёт по классам:")
print(classification_report(ytest, test_predictions, digits=4))

project_root = Path(__file__).resolve().parents[1]
figure_directory = project_root / "reports" / "figures"
figure_directory.mkdir(parents=True, exist_ok=True)

ConfusionMatrixDisplay.from_predictions(ytest, test_predictions, cmap="Blues", xticks_rotation=30)
plt.title("Random forest: confusion matrix on test")
plt.tight_layout()

figure_path = figure_directory / "final_confusion_matrix.png"
plt.savefig(figure_path, dpi=150)
plt.close()

print(f"\nМатрица ошибок сохранена: {figure_path}")
