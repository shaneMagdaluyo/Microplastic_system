import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from ml_pipeline import train_models, preprocess_data

st.set_page_config(page_title="Microplastic Risk System", layout="wide")

st.title("🌊 Microplastic Risk Analysis Dashboard")

# =========================
# UPLOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("🚀 Run Analysis"):

        # =========================
        # TRAIN MODELS
        # =========================
        results, best_name, best_model, X_test, y_test, X_processed = train_models(df, target)

        st.success(f"Best Model: {best_name}")

        # =========================
        # MODEL COMPARISON
        # =========================
        st.subheader("📌 Model Comparison")

        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        fig, ax = plt.subplots()
        results_df["accuracy"].plot(kind="bar", ax=ax)
        ax.set_title("Model Accuracy Comparison")
        st.pyplot(fig)

        # =========================
        # RISK DISTRIBUTION (SAFE)
        # =========================
        st.subheader("📊 Risk Distribution")

        if df[target].dtype == "object":
            encoded = df[target].astype("category").cat.codes
        else:
            encoded = pd.to_numeric(df[target], errors="coerce")

        fig, ax = plt.subplots()
        ax.hist(encoded.dropna(), bins=20)
        ax.set_title("Risk Distribution")
        st.pyplot(fig)

        # =========================
        # FEATURE IMPORTANCE
        # =========================
        st.subheader("🔥 Feature Importance")

        if hasattr(best_model, "feature_importances_"):
            importances = best_model.feature_importances_
            feat_names = X_processed.columns

            imp_df = pd.DataFrame({
                "Feature": feat_names,
                "Importance": importances
            }).sort_values("Importance", ascending=False)

            st.dataframe(imp_df)

            fig, ax = plt.subplots()
            ax.barh(imp_df["Feature"], imp_df["Importance"])
            ax.set_title("Feature Importance")
            st.pyplot(fig)

        else:
            st.info("Model does not support feature importance")

        # =========================
        # CORRELATION MATRIX (FIXED)
        # =========================
        st.subheader("📈 Correlation Matrix")

        corr_df = X_processed.copy()
        corr_df["target"] = pd.factorize(df[target])[0]

        if corr_df.shape[1] < 2:
            st.warning("Not enough numeric features for correlation")
        else:
            corr = corr_df.corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            cax = ax.imshow(corr)
            plt.colorbar(cax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45)
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)

else:
    st.info("Upload a CSV file to begin analysis")
