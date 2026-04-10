# app.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE


def generate_dummy_data(n_samples: int = 1000) -> pd.DataFrame:
    np.random.seed(42)
    return pd.DataFrame({
        "risk score": np.random.gamma(2, 2, n_samples) * 10,
        "mp count per l": np.random.exponential(100, n_samples),
        "risk level": np.random.choice(["Low", "Medium", "High"], n_samples, p=[0.5, 0.3, 0.2]),
        "Risk_Type": np.random.choice(["Type A", "Type B"], n_samples, p=[0.8, 0.2]),
        "Polymer Type": np.random.choice(["PET", "PE", "PP", "PVC", "PS"], n_samples),
    })


def preprocess(df: pd.DataFrame):
    skewed_cols = ["risk score", "mp count per l"]

    pt = PowerTransformer(method="yeo-johnson")
    df[skewed_cols] = pt.fit_transform(df[skewed_cols])

    robust_scaler = RobustScaler()
    df[skewed_cols] = robust_scaler.fit_transform(df[skewed_cols])

    df = pd.get_dummies(df, columns=["risk level", "Polymer Type"], drop_first=True)

    encoder = LabelEncoder()
    df["Risk_Type"] = encoder.fit_transform(df["Risk_Type"])

    return df, encoder


def feature_selection(X, y):
    selector = SelectKBest(score_func=mutual_info_classif, k="all")
    selector.fit(X, y)

    scores = pd.DataFrame({
        "Feature": X.columns,
        "Score": selector.scores_
    }).sort_values(by="Score", ascending=False)

    return scores


def train_models(X_train, y_train):
    models = {
        "logreg": LogisticRegression(max_iter=1000, random_state=42),
        "rf": RandomForestClassifier(random_state=42),
        "gb": GradientBoostingClassifier(random_state=42),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def evaluate(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }

    return pd.DataFrame(results).T


def tune_logistic(X_train, y_train):
    grid = {
        "C": [0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
    }

    search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        grid,
        cv=5,
        scoring="roc_auc",
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_


def plot_confusion_matrix(y_test, y_pred, labels):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.show()


def plot_roc(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_prob):.2f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.legend()
    plt.title("ROC Curve")
    plt.show()


def main():
    df = generate_dummy_data()

    df, encoder = preprocess(df)

    X = df.drop("Risk_Type", axis=1)
    y = df["Risk_Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    models = train_models(X_train, y_train)

    print("\nModel Comparison:")
    print(evaluate(models, X_test, y_test))

    best_model, params = tune_logistic(X_train, y_train)

    y_pred = best_model.predict(X_test)
    y_prob = best_model.predict_proba(X_test)[:, 1]

    print("\nBest Logistic Params:", params)
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    plot_confusion_matrix(y_test, y_pred, encoder.classes_)
    plot_roc(y_test, y_prob)


if __name__ == "__main__":
    main()
