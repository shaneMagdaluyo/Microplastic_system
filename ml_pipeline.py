import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score


def load_data(file):
    return pd.read_csv(file)


def preprocess_data(df, target):

    df = df.copy()

    if df[target].dtype == "object":
        df[target] = LabelEncoder().fit_transform(df[target].astype(str))

    y = df[target]
    X = df.drop(columns=[target])

    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    X = X.apply(pd.to_numeric, errors="coerce")

    X = pd.DataFrame(SimpleImputer(strategy="mean").fit_transform(X), columns=X.columns)

    X = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns)

    return X, y


def train_models(df, target):

    X, y = preprocess_data(df, target)

    stratify = y if y.value_counts().min() >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}
    best_model = None
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = model.score(X_test, y_test)

        results[name] = {"accuracy": acc}

        if best_model is None or acc > results[best_name]["accuracy"]:
            best_model = model
            best_name = name

    return results, best_name, best_model, X
