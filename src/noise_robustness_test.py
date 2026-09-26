from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import pandas as pd
from load_data import load_data
from split_data import sequential_split

CONFIGS = [{"sensor_count": 2, "max_depth": 4}, 
                  {"sensor_count": 24, "max_depth": None}]
NOISE_LVLS = [0.00, 0.01, 0.05, 0.10, 0.20]
REPETITIONS = 20

def evaulate_config(sensor_count, max_depth):
    x, y = load_data(sensor_count)
    xtr, xvalid, xtest, ytr, yvalid, ytest = sequential_split(x, y)
    x_train_final = pd.concat([xtr, xvalid], ignore_index=True)
    y_train_final = pd.concat([ytr, yvalid], ignore_index=True)
    model = RandomForestClassifier(n_estimators=300, max_depth=max_depth, random_state=42, n_jobs=-1)
    model.fit(x_train_final, y_train_final)

    feature_std = x_train_final.std(ddof=0).to_numpy()
    rndm_generator = np.random.default_rng(42)
    print(f"\n--- Модель с {sensor_count} признаками ---")

    for lvl in NOISE_LVLS:
        accuracy_scores = []
        f1_scores = []

        for _ in range(REPETITIONS):
            noise = rndm_generator.normal(loc=0.0, scale=lvl, size=xtest.shape)
            noisy_values = xtest.to_numpy() + noise * feature_std
            noisy_values = np.clip(noisy_values, a_min=0.0, a_max=None)
            xtest_noisy = pd.DataFrame(noisy_values, columns=xtest.columns, index=xtest.index)

            predictions = model.predict(xtest_noisy)
            accuracy_scores.append(accuracy_score(ytest, predictions))
            f1_scores.append(f1_score(ytest, predictions, average="macro"))

        print(
            f"Шум {lvl:>5.0%} | "
            f"accuracy={np.mean(accuracy_scores):.4f} "
            f"± {np.std(accuracy_scores):.4f} | "
            f"macro-F1={np.mean(f1_scores):.4f} "
            f"± {np.std(f1_scores):.4f}"
        )

if __name__ == "__main__":
    for config in CONFIGS:
        evaulate_config(**config)