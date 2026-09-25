import pandas as pd
from load_data import load_data
from split_data import sequential_split

def audit(sensor_count):
    x, y = load_data(sensor_count)
    xtr, xvalid, xtest, _, _, _ = sequential_split(x, y)
    x_train_final = pd.concat([xtr, xvalid], ignore_index=True)
    train_rows = set(map(tuple, x_train_final.to_numpy()))
    test_overlap = sum(tuple(row) in train_rows for row in xtest.to_numpy())

    data = x.copy()
    data["target"] = y.to_numpy()
    feature_columns = list(x.columns)
    labels_per_features = data.groupby(feature_columns)["target"].nunique()
    conflicting_combos = (labels_per_features > 1).sum()
    unique_rows = len(x.drop_duplicates())
    duplicate_rows = len(x) - unique_rows

    print(f"\n=== {sensor_count} признаков ===")
    print(f"Названия признаков: {feature_columns}")
    print(f"Всего объектов: {len(x)}")
    print(f"Уникальных комбинаций: {unique_rows}")
    print(f"Повторяющихся строк: {duplicate_rows}")
    print(
        f"Объектов test, уже встречавшихся в train/validation: "
        f"{test_overlap} из {len(xtest)} "
        f"({test_overlap / len(xtest):.2%})"
    )
    print("Комбинаций признаков с разными правильными ответами:", conflicting_combos)

if __name__ == "__main__":
    for sensor_count in (24, 4, 2):
        audit(sensor_count)

