import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ----------------------------
# Load Data
# ----------------------------
def load_data(path, encoding="utf-8"):
    df = pd.read_csv(path, encoding=encoding)
    return df


# ----------------------------
# Preprocessing
# ----------------------------
def build_preprocessor(X):
    categorical_cols = X.select_dtypes(include=["object"]).columns
    numeric_cols = X.select_dtypes(exclude=["object"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numeric_cols),
        ]
    )

    return preprocessor


# ----------------------------
# Models
# ----------------------------
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "SVM": SVC(probability=True)
    }


# ----------------------------
# Evaluation
# ----------------------------
def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)

    print("\n==============================")
    print(f"📊 Model: {name}")
    print("==============================")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:\n", classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(f"Confusion Matrix - {name}")
    plt.show()


# ----------------------------
# Feature Importance (Tree model only)
# ----------------------------
def feature_importance(model, preprocessor, X):
    try:
        feature_names = preprocessor.get_feature_names_out()
        importances = model.feature_importances_

        feat_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        print("\n🔥 Top Features:")
        print(feat_df.head(10))
    except:
        print("Feature importance not available for this model.")


# ----------------------------
# Hyperparameter tuning
# ----------------------------
def tune_logistic(X_train, y_train):
    pipe = Pipeline([
        ("clf", LogisticRegression(max_iter=1000))
    ])

    params = {
        "clf__C": [0.01, 0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["lbfgs"]
    }

    grid = GridSearchCV(pipe, params, cv=3, scoring="accuracy")
    grid.fit(X_train, y_train)

    print("\n🏆 Best Logistic Regression Params:", grid.best_params_)
    return grid.best_estimator_


# ----------------------------
# Main Pipeline
# ----------------------------
def main(data_path, target):
    df = load_data(data_path)

    print("\n📊 Data Loaded:", df.shape)
    print("\nClass Distribution:\n", df[target].value_counts())

    X = df.drop(columns=[target])
    y = df[target]

    preprocessor = build_preprocessor(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------------- SMOTE ----------------
    print("\n⚖️ Applying SMOTE...")
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_encoded, y_train)

    # ---------------- Models ----------------
    models = get_models()
    results = {}

    for name, model in models.items():
        print(f"\n🚀 Training {name}...")

        model.fit(X_train_res, y_train_res)
        results[name] = model

        evaluate_model(name, model, X_test_encoded, y_test)

    # ---------------- Compare Models ----------------
    print("\n📈 Model Comparison:")
    for name, model in results.items():
        acc = accuracy_score(y_test, model.predict(X_test_encoded))
        print(f"{name}: {acc:.4f}")

    # ---------------- Hyperparameter Tuning ----------------
    print("\n⚙️ Hyperparameter Tuning Logistic Regression...")
    best_model = tune_logistic(X_train_res, y_train_res)

    evaluate_model("Tuned Logistic Regression", best_model, X_test_encoded, y_test)

    print("\n✅ Pipeline Complete!")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument("--target", required=True, help="Target column name")

    args = parser.parse_args()

    main(args.data, args.target)
