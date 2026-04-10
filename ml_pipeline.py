import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE


# =========================
# LOAD DATA
# =========================
def load_data(file):
    return pd.read_csv(file)


# =========================
# PREPROCESSING
# =========================
def build_preprocessor(X):
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", StandardScaler(), num_cols)
    ])

    return preprocessor


# =========================
# MODELS
# =========================
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=2000),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "SVM": SVC(probability=True)
    }


# =========================
# TRAIN PIPELINE
# =========================
def train_models(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    preprocessor = build_preprocessor(X)

    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_enc, y_train)

    models = get_models()

    results = {}
    best_model = None
    best_name = ""
    best_acc = 0

    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        preds = model.predict(X_test_enc)

        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)
        report = classification_report(y_test, preds)

        results[name] = {
            "accuracy": acc,
            "confusion_matrix": cm,
            "report": report,
            "model": model
        }

        if acc > best_acc:
            best_acc = acc
            best_name = name
            best_model = model

    return results, best_name, best_model


# =========================
# SAVE MODEL
# =========================
def save_model(model, filename="best_model.pkl"):
    joblib.dump(model, filename)
