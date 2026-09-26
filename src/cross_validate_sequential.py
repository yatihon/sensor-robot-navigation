import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from load_data import load_data

def evaulate_model(model_name, sensor_count, model):
    x, y = load_data(sensor_count)
    development_size = int(len(x) * 0.85)
    x_development = x.iloc[:development_size]
    y_development = y.iloc[:development_size]
    classes = sorted(y.unique())

    macro_f1_scorer = make_scorer(f1_score, average="macro", labels=classes, zero_division=0)
    cross_validation = TimeSeriesSplit(n_splits=5, gap=9)
    scores = cross_validate(model, x_development, y_development, cv=cross_validation,
                                scoring={"accuracy" : "accuracy", "macro_f1" : macro_f1_scorer},
                                return_train_score=True, n_jobs=-1)
    print(f"\n{model_name}")
    print(f"Количество признаков: {sensor_count}")
    print(
        "Train macro-F1: "
        f"{scores['train_macro_f1'].mean():.4f} "
        f"± {scores['train_macro_f1'].std():.4f}"
    )
    print(
        "Sequential CV accuracy: "
        f"{scores['test_accuracy'].mean():.4f} "
        f"± {scores['test_accuracy'].std():.4f}"
    )
    print(
        "Sequential CV macro-F1: "
        f"{scores['test_macro_f1'].mean():.4f} "
        f"± {scores['test_macro_f1'].std():.4f}"
    )
    print(
        "Macro-F1 по фолдам:",
        [round(score, 4) for score in scores["test_macro_f1"]],
    )

if __name__ == "__main__":
    evaulate_model(model_name="Logistic regression", sensor_count=2, model=make_pipeline(
        StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
    ))
    evaulate_model(model_name="Random Forest, 2 features", sensor_count=2, model=RandomForestClassifier(
        n_estimators=300, max_depth=4, random_state=42, n_jobs=1
    ))
    evaulate_model(model_name="Random Forest, 24 features", sensor_count=24, model=RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=42, n_jobs=1
        ))

