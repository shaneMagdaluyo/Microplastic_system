import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import train_models

# =========================
# UI SETUP
# =========================
st.set_page_config(page_title="Microplastic Risk System", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")

# =========================
# UPLOAD DATA
# =========================
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    if st.button("🚀 Train Models"):

        try:
            results, best_name, best_model, X_processed = train_models(df, target)

            st.success(f"🏆 Best Model: {best_name}")

            # =========================
            # MODEL COMPARISON
            # =========================
            st.subheader("📈 Model Performance")

            results_df = pd.DataFrame(results).T
            st.dataframe(results_df)

            fig, ax = plt.subplots()
            results_df["accuracy"].plot(kind="bar", ax=ax)
            ax.set_title("Model Accuracy Comparison")
            st.pyplot(fig)

            # =========================
            # RISK DISTRIBUTION
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
            # CORRELATION MATRIX
            # =========================
            st.subheader("📌 Correlation Matrix")

            corr_df = X_processed.copy()
            corr_df["target"] = pd.factorize(df[target])[0]

            corr = corr_df.corr()

            fig, ax = plt.subplots(figsize=(8, 5))
            cax = ax.imshow(corr)
            plt.colorbar(cax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45)
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)

        except Exception as e:
            st.error(str(e))

else:
    st.info("⬅️ Upload dataset to start")
