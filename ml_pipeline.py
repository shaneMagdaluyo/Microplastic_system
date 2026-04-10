import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score


# =========================
# CLEAN DATA SAFE VERSION
# =========================
def preprocess_data(df, target):

    df = df.copy()

    # Encode target if needed
    if df[target].dtype == "object":
        df[target] = LabelEncoder().fit_transform(df[target].astype(str))

    y = df[target]
    X = df.drop(columns=[target])

    # Encode categorical features safely
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    # Convert everything to numeric safely
    X = X.apply(pd.to_numeric, errors="coerce")

    # Fill missing values safely
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y


# =========================
# TRAIN MODELS SAFE VERSION
# =========================
def train_models(df, target):

    X, y = preprocess_data(df, target)

    # SAFE SPLIT (avoid stratify crash)
    if len(np.unique(y)) > 1:
        stratify = y
    else:
        stratify = None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}
    best_model = None
    best_score = 0
    best_name = ""

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        results[name] = {"accuracy": acc}

        if acc > best_score:
            best_score = acc
            best_model = model
            best_name = name

    return results, best_name, best_model, X_test, y_test, X
