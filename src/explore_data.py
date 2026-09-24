from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from load_data import load_data

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIGURES_DIRECTORY = PROJECT_ROOT / "reports" / "figures"
FIGURES_DIRECTORY.mkdir(parents=True, exist_ok=True)

x_24, y_24 = load_data(24)
x_4, y_4 = load_data(4)
x_2, y_2 = load_data(2)


print("Первые пять строк набора с 24 датчиками:")
print(x_24.head())

print("\nТипы данных:")
print(x_24.dtypes.value_counts())

print("\nКоличество пропусков:")
print(x_24.isna().sum().sum())

print("\nОсновная статистика:")
print(x_24.describe())

print("\nРаспределение классов, %:")
class_percentages = (y_24.value_counts(normalize=True).mul(100).round(2))
print(class_percentages)

print("\nМетки во всех наборах совпадают:")
print(y_24.equals(y_4) and y_24.equals(y_2))

class_order = y_24.value_counts().index

plt.figure(figsize=(9, 5))

sns.countplot(
    x=y_24,
    order=class_order,
    hue=y_24,
    palette="viridis",
    legend=False,
)

plt.title("Распределение команд движения")
plt.xlabel("Команда")
plt.ylabel("Количество наблюдений")
plt.xticks(rotation=15)
plt.tight_layout()

figure_path = FIGURES_DIRECTORY / "class_distribution.png"
plt.savefig(figure_path, dpi=150)
plt.close()

print(f"\nГрафик сохранён: {figure_path}")