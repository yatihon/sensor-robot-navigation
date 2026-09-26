import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from load_data import load_data

x, y = load_data(sensor_count=2)
development_size = int(len(x) * 0.85)
x_development = x.iloc[:development_size]
y_development = y.iloc[:development_size]
cross_validation = TimeSeriesSplit(n_splits=5, gap=9)

for ifold, (train_i, valid_i) in enumerate(cross_validation.split(x_development), start=1):
    ytr = y_development.iloc[train_i]
    yvalid = y_development.iloc[valid_i]
    distribution = pd.DataFrame({"train": ytr.value_counts(), 
                                 "validation": yvalid.value_counts()}).fillna(0).astype(int)
    print(f"\n=== Fold {ifold} ===")
    print(
        f"Train: {len(train_i)} объектов, "
        f"индексы {train_i[0]}–{train_i[-1]}"
    )
    print(
        f"Validation: {len(valid_i)} объектов, "
        f"индексы {valid_i[0]}–{valid_i[-1]}"
    )
    print(distribution)