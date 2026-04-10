import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report


# =========================
# LOAD DATA
# =========================
def load_data(file):
    df = pd.read_csv(file)
    return df


# =========================
# CLEAN DATA (SAFE VERSION)
# =========================
def clean_data(df, target):

    df = df.copy()

    # drop empty rows
    df = df.dropna(axis=0, how="all")

    y = df[target]
    X = df.drop(columns=[target])

    # encode target if needed
    if y.dtype == "object":
        le = LabelEncoder()
        y = le.fit_transform(y)

    # keep only numeric columns safely
    for col in X.columns:
        if X[col].dtype == "object":
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))

    # fill missing values safely
    imputer = SimpleImputer(strategy="mean")
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    return X, y


# =========================
# TRAIN MODELS (FIXED STRATIFY ERROR)
# =========================
def train_models(df, target):

    X, y = clean_data(df, target)

    # check class distribution
    y_series = pd.Series(y)
    class_counts = y_series.value_counts()

    # SAFE STRATIFY LOGIC
    if len(class_counts) < 2:
        raise ValueError("❌ Need at least 2 classes in target column")

    if class_counts.min() < 2:
        stratify_value = None
    else:
        stratify_value = y

    # split dataset safely
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=stratify_value
    )

    # models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}

    best_model = None
    best_name = ""
    best_acc = 0

    # training loop
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds)

        results[name] = {
            "accuracy": acc,
            "report": report
        }

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name

    return results, best_name, best_model


# =========================
# SAVE MODEL
# =========================
def save_model(model):
    joblib.dump(model, "best_model.pkl")
