import argparse
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

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


# =========================
# LOAD DATA
# =========================
def load_data(path, encoding="utf-8"):
    print("\n📥 Loading data...")
    df = pd.read_csv(path, encoding=encoding)
    print("Shape:", df.shape)
    return df


# =========================
# PREPROCESSOR
# =========================
def preprocess(X):
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    print("\n🔧 Categorical columns:", list(cat_cols))
    print("🔧 Numerical columns:", list(num_cols))

    transformer = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])

    return transformer


# =========================
# MODELS
# =========================
def get_models():
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=200),
        "SVM": SVC(probability=True)
    }


# =========================
# EVALUATION
# =========================
def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)

    print(f"\n📊 {name}")
    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))

    cm = confusion_matrix(y_test, preds)
    ConfusionMatrixDisplay(cm).plot()
    plt.title(name)
    plt.show()


# =========================
# MAIN PIPELINE
# =========================
def main(data_path, target):
    try:
        df = load_data(data_path)

        if target not in df.columns:
            raise Exception(f"Target column '{target}' not found!")

        print("\n📊 Class distribution:")
        print(df[target].value_counts())

        X = df.drop(columns=[target])
        y = df[target]

        preprocessor = preprocess(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        # Encode
        print("\n⚙️ Encoding data...")
        X_train_enc = preprocessor.fit_transform(X_train)
        X_test_enc = preprocessor.transform(X_test)

        # SMOTE
        print("\n⚖️ Applying SMOTE...")
        smote = SMOTE(random_state=42)

        X_train_bal, y_train_bal = smote.fit_resample(X_train_enc, y_train)

        # Models
        models = get_models()
        trained_models = {}

        for name, model in models.items():
            print(f"\n🚀 Training {name}...")
            model.fit(X_train_bal, y_train_bal)
            trained_models[name] = model
            evaluate(model, X_test_enc, y_test, name)

        # Comparison
        print("\n📈 FINAL COMPARISON:")
        for name, model in trained_models.items():
            acc = accuracy_score(y_test, model.predict(X_test_enc))
            print(f"{name}: {acc:.4f}")

        print("\n✅ DONE SUCCESSFULLY!")

    except Exception as e:
        print("\n❌ CRASH DETECTED")
        print(str(e))
        input("\nPress Enter to exit...")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True, help="CSV file path")
    parser.add_argument("--target", required=True, help="Target column")

    args = parser.parse_args()

    main(args.data, args.target)
