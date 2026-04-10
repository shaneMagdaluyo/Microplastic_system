import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.multiclass import type_of_target

from imblearn.over_sampling import SMOTE
from collections import Counter

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Microplastic Risk System", layout="wide")
st.title("🌊 Microplastic Risk Prediction System (Production Ready)")

# -----------------------------
# UPLOAD DATA
# -----------------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # -----------------------------
    # CLEANING
    # -----------------------------
    df = df.fillna(df.median(numeric_only=True))

    # Encode categorical
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Outlier removal
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    st.success(f"Cleaned shape: {df.shape}")

    # -----------------------------
    # TARGET
    # -----------------------------
    target = st.selectbox("Select Target Column", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # Validate target
    if y.nunique() < 2:
        st.error("Target must have at least 2 classes")
        st.stop()

    # -----------------------------
    # SPLIT (NO LEAKAGE)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # FEATURE SELECTION
    # -----------------------------
    selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)

    # -----------------------------
    # SAFE SMOTE
    # -----------------------------
    class_counts = Counter(y_train)
    min_class = min(class_counts.values())

    st.write("Class distribution:", class_counts)

    if type_of_target(y_train) not in ["binary", "multiclass"]:
        st.error("Invalid classification target")
        st.stop()

    if min_class < 6:
        st.warning("SMOTE skipped (not enough samples)")
        X_train_final, y_train_final = X_train_sel, y_train
    else:
        k = min(5, min_class - 1)
        smote = SMOTE(k_neighbors=k, random_state=42)
        X_train_final, y_train_final = smote.fit_resample(X_train_sel, y_train)

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "EDA", "Modeling", "Prediction", "Explainability"
    ])

    # =============================
    # EDA
    # =============================
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        if "Risk_Score" in df.columns:
            st.subheader("Risk Score Distribution")
            fig = plt.figure()
            plt.hist(df["Risk_Score"], bins=20)
            st.pyplot(fig)

        if "MP_Count_per_L" in df.columns and "Risk_Score" in df.columns:
            st.subheader("MP Count vs Risk Score")
            fig = plt.figure()
            plt.scatter(df["MP_Count_per_L"], df["Risk_Score"])
            st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig = plt.figure()
        plt.imshow(df.corr(), cmap="coolwarm")
        plt.colorbar()
        st.pyplot(fig)

    # =============================
    # MODELING
    # =============================
    with tab2:
        st.subheader("Train Models")

        if st.button("Train"):
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(random_state=42),
                "SVM": SVC(probability=True)
            }

            results = {}
            best_model = None
            best_score = 0

            for name, model in models.items():
                model.fit(X_train_final, y_train_final)
                preds = model.predict(X_test_sel)
                acc = accuracy_score(y_test, preds)
                results[name] = acc

                if acc > best_score:
                    best_model = model
                    best_score = acc

            joblib.dump({
                "model": best_model,
                "scaler": scaler,
                "selector": selector
            }, "pipeline.pkl")

            st.write(results)

            fig = plt.figure()
            plt.bar(results.keys(), results.values())
            st.pyplot(fig)

            st.success("Model saved successfully")

    # =============================
    # PREDICTION
    # =============================
    with tab3:
        st.subheader("Make Prediction")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, value=float(X[col].mean()))

        if st.button("Predict"):
            pipe = joblib.load("pipeline.pkl")

            model = pipe["model"]
            scaler = pipe["scaler"]
            selector = pipe["selector"]

            input_df = pd.DataFrame([input_data])

            input_scaled = scaler.transform(input_df)
            input_selected = selector.transform(input_scaled)

            pred = model.predict(input_selected)

            st.success(f"Prediction: {pred[0]}")

    # =============================
    # EXPLAINABILITY
    # =============================
    with tab4:
        st.subheader("Feature Importance")

        try:
            pipe = joblib.load("pipeline.pkl")
            model = pipe["model"]

            if isinstance(model, RandomForestClassifier):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sel)

                fig = plt.figure()
                shap.summary_plot(shap_values, X_test_sel, show=False)
                st.pyplot(fig)
            else:
                st.warning("SHAP only enabled for Random Forest in production mode")

        except Exception as e:
            st.warning(str(e))
