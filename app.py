from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

import joblib
import os

app = Flask(__name__)

MODEL_PATH = "model.pkl"

# -----------------------------
# LOAD & PREPROCESS FUNCTION
# -----------------------------
def preprocess_data(df):
    # Example column names (adjust based on your dataset)
    target = "Risk_Type"

    X = df.drop(columns=[target])
    y = df[target]

    # Separate column types
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X.select_dtypes(include=['object']).columns

    # Pipelines
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return X, y, preprocessor


# -----------------------------
# TRAIN MODEL
# -----------------------------
@app.route("/train", methods=["POST"])
def train_model():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        X, y, preprocessor = preprocess_data(df)

        # Handle imbalance
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Model options
        models = {
            "logistic": LogisticRegression(max_iter=1000),
            "random_forest": RandomForestClassifier(),
            "svm": SVC()
        }

        results = {}

        best_model = None
        best_score = 0

        for name, model in models.items():
            pipe = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            X_train, X_test, y_train, y_test = train_test_split(
                X_resampled, y_resampled, test_size=0.2, random_state=42
            )

            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            acc = accuracy_score(y_test, preds)
            results[name] = acc

            if acc > best_score:
                best_score = acc
                best_model = pipe

        # Save best model
        joblib.dump(best_model, MODEL_PATH)

        return jsonify({
            "message": "Training complete",
            "results": results,
            "best_score": best_score
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -----------------------------
# PREDICT
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])

        model = joblib.load(MODEL_PATH)
        prediction = model.predict(df)

        return jsonify({
            "prediction": prediction[0]
        })

    except Exception as e:
        return jsonify({"error": str(e)})


# -----------------------------
# FEATURE IMPORTANCE
# -----------------------------
@app.route("/feature-importance", methods=["GET"])
def feature_importance():
    try:
        model = joblib.load(MODEL_PATH)

        if hasattr(model.named_steps["model"], "feature_importances_"):
            importance = model.named_steps["model"].feature_importances_
            return jsonify({"importance": importance.tolist()})
        else:
            return jsonify({"message": "Model has no feature importance"})

    except Exception as e:
        return jsonify({"error": str(e)})


# -----------------------------
# RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
