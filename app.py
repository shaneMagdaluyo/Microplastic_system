import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# -----------------------------
# APP TITLE
# -----------------------------
st.title("🌊 Microplastic Risk Prediction System")

# -----------------------------
# FILE UPLOAD
# -----------------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # BASIC INFO
    # -----------------------------
    st.subheader("📌 Dataset Info")
    st.write(df.describe())

    # -----------------------------
    # HANDLE MISSING VALUES
    # -----------------------------
    df = df.fillna(df.median(numeric_only=True))

    # -----------------------------
    # ENCODE CATEGORICAL
    # -----------------------------
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # -----------------------------
    # OUTLIER REMOVAL (IQR)
    # -----------------------------
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    st.subheader("🧹 Cleaned Data Shape")
    st.write(df.shape)

    # -----------------------------
    # SELECT TARGET
    # -----------------------------
    target = st.selectbox("Select Target Column (Risk_Type)", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # EDA SECTION
    # -----------------------------
    st.subheader("📈 Risk Score Distribution")

    if "Risk_Score" in df.columns:
        fig = plt.figure()
        plt.hist(df["Risk_Score"], bins=20)
        plt.title("Risk Score Distribution")
        st.pyplot(fig)

    # -----------------------------
    # TRAIN MODELS
    # -----------------------------
    if st.button("🚀 Train Models"):
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "SVM": SVC()
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc

        st.subheader("📊 Model Performance")
        st.write(results)

        # Save best model
        best_model_name = max(results, key=results.get)
        best_model = models[best_model_name]
        joblib.dump(best_model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        st.success(f"Best Model: {best_model_name}")

    # -----------------------------
    # PREDICTION SECTION
    # -----------------------------
    st.subheader("🔮 Make Prediction")

    input_data = {}

    for col in X.columns:
        val = st.number_input(f"{col}", value=float(df[col].mean()))
        input_data[col] = val

    if st.button("Predict"):
        try:
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            pred = model.predict(input_scaled)
            st.success(f"Prediction: {pred[0]}")

        except Exception as e:
            st.error(f"Error: {e}")

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.subheader("📌 Feature Importance")

    try:
        model = joblib.load("model.pkl")

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            fig = plt.figure()
            plt.bar(X.columns, importance)
            plt.xticks(rotation=45)
            plt.title("Feature Importance")
            st.pyplot(fig)
        else:
            st.info("Model does not support feature importance")

    except:
        st.info("Train model first to see importance")
