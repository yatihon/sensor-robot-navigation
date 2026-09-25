from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from load_data import load_data
from split_data import sequential_split

x, y = load_data(sensor_count=2)
(xtr, xvalid, xtest, ytr, yvalid, ytest) = sequential_split(x, y)
x_train_final = pd.concat([xtr, xvalid], ignore_index=True)
y_train_final = pd.concat([ytr, yvalid], ignore_index=True)
model = RandomForestClassifier(n_estimators=300, max_depth=4, min_samples_leaf=1, random_state=42, n_jobs=-1)
model.fit(x_train_final, y_train_final)

front_values = np.linspace(x["front"].min() - 0.1, x["front"].max() + 0.1, 400)
left_values = np.linspace(x["left"].min() - 0.1, x["left"].max() + 0.1, 400)
front_grid, left_grid = np.meshgrid(front_values, left_values)
grid = pd.DataFrame({"front": front_grid.ravel(), "left": left_grid.ravel()})
grid_predictions = model.predict(grid)

classes = list(model.classes_)
class_to_number = {class_name: num for num, class_name in enumerate(classes)}
prediction_numbers = np.array([class_to_number[pred] for pred in grid_predictions]).reshape(front_grid.shape)

colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756"]
color_map = ListedColormap(colors)
plt.figure(figsize=(10,7))
plt.contourf(front_grid, left_grid, prediction_numbers, alpha=0.25, cmap=color_map, levels=np.arange(len(classes) + 1) - 0.5)

for class_name in classes: 
    class_mask = ytest == class_name
    plt.scatter(xtest.loc[class_mask, "front"], xtest.loc[class_mask, "left"],
        label=class_name, s=20, alpha=0.8, edgecolors="black", linewidths=0.2, color=colors[class_to_number[class_name]])

plt.xlabel("Расстояние спереди")
plt.ylabel("Расстояние слева")
plt.title("Random Forest: границы выбора команды")
plt.legend()
plt.tight_layout()

project_root = Path(__file__).resolve().parents[1]
figure_directory = project_root / "reports" / "figures"
figure_directory.mkdir(parents=True, exist_ok=True)
figure_path = figure_directory / "decision_boundary.png"
plt.savefig(figure_path, dpi=150)
plt.close()

print(f'График сохранен: {figure_path}')

