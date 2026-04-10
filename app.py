import argparse
import traceback

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE


print("\n🚀 ML PIPELINE STARTED")


# =========================
# LOAD DATA
# =========================
def load_data(path):
    print("\n📥 Loading data...")
    df = pd.read_csv(path)
    print("Shape:", df.shape)
    return df


# =========================
# PREPROCESS
# =========================
def preprocess(X):
    cat_cols = X.select_dtypes(include=["object"]).columns
    num_cols = X.select_dtypes(exclude=["object"]).columns

    print("\n🔧 Categorical:", list(cat_cols))
    print("🔧 Numerical:", list(num_cols))

    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ("num", "passthrough", num_cols)
    ])


# =========================
# MODELS
# =========================
def get_models():
    return {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=200),
        "SVM": SVC()
    }


# =========================
# MAIN PIPELINE
# =========================
def main(data_path, target):
    try:
        df = load_data(data_path)

        print("\n📊 Columns:", list(df.columns))

        if target not in df.columns:
            raise Exception(f"Target '{target}' not found in dataset!")

        X = df.drop(columns=[target])
        y = df[target]

        print("\n📊 Class distribution:")
        print(y.value_counts())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        preprocessor = preprocess(X)

        print("\n⚙️ Encoding...")
        X_train_enc = preprocessor.fit_transform(X_train)
        X_test_enc = preprocessor.transform(X_test)

        print("\n⚖️ Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X_train_bal, y_train_bal = smote.fit_resample(X_train_enc, y_train)

        models = get_models()

        best_acc = 0
        best_model_name = ""
        best_model = None

        for name, model in models.items():
            print(f"\n🤖 Training {name}...")

            model.fit(X_train_bal, y_train_bal)
            preds = model.predict(X_test_enc)

            acc = accuracy_score(y_test, preds)

            print(f"📊 {name} Accuracy: {acc:.4f}")
            print(classification_report(y_test, preds))

            if acc > best_acc:
                best_acc = acc
                best_model_name = name
                best_model = model

        print("\n🏆 BEST MODEL:", best_model_name)
        print("🏆 BEST ACCURACY:", best_acc)

        print("\n✅ PIPELINE COMPLETE SUCCESSFULLY")

    except Exception as e:
        print("\n❌ ERROR OCCURRED")
        print(str(e))
        traceback.print_exc()
        input("\nPress Enter to exit...")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--target", required=True)

    args = parser.parse_args()

    main(args.data, args.target)
