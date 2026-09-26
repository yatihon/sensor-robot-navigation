import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from load_data import load_data
from split_data import sequential_split

def evaulate_model(model_name, sensor_count, model):
    x, y = load_data(sensor_count)
    xtr, xvalid, _, ytr, yvalid, _ = sequential_split(x, y)
    x_development = pd.concat([xtr, xvalid], ignore_index=True)
    y_development = pd.concat([ytr, yvalid], ignore_index=True)

    cross_validation = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(model, x_development, y_development, cv=cross_validation,
                            scoring={"accuracy" : "accuracy", "macro_f1" : "f1_macro"},
                            return_train_score=True, n_jobs=-1)
    print(f"\n{model_name}")
    print(f"Количество признаков: {sensor_count}")
    print(
        "Train macro-F1: "
        f"{scores['train_macro_f1'].mean():.4f} "
        f"± {scores['train_macro_f1'].std():.4f}"
    )

    print(
        "CV accuracy:     "
        f"{scores['test_accuracy'].mean():.4f} "
        f"± {scores['test_accuracy'].std():.4f}"
    )

    print(
        "CV macro-F1:     "
        f"{scores['test_macro_f1'].mean():.4f} "
        f"± {scores['test_macro_f1'].std():.4f}"
    )

if __name__ == "__main__":
    evaulate_model(model_name="Logistic regression", sensor_count=2, model=make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
    ))
    evaulate_model(model_name="Random Forest, 2 features", sensor_count=2, model=RandomForestClassifier(
        n_estimators=300, max_depth=4, random_state=42, n_jobs=-1
    ))
    evaulate_model(model_name="Random Forest, 24 features", sensor_count=24, model=RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=-1
        ))
    