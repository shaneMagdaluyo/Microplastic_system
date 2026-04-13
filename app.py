import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MP Risk Intelligence", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")


# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload CSV", type=["csv"])


if file:

    df = load_data(file)

    # =========================
    # OVERVIEW
    # =========================
    st.subheader("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Numeric Features", df.select_dtypes(include="number").shape[1])

    target = st.selectbox("🎯 Select Target Column", df.columns)

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Analysis", "🤖 ML Models"])


    # =========================
    # TAB 1 - DASHBOARD
    # =========================
    with tab1:

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Risk Distribution")

        fig, ax = plt.subplots()

        if df[target].dtype == "object":
            df[target].value_counts().plot(kind="bar", ax=ax)
        else:
            ax.hist(pd.to_numeric(df[target], errors="coerce").dropna(), bins=20)

        st.pyplot(fig)


    # =========================
    # TAB 2 - CORRELATION MATRIX (FIXED)
    # =========================
    with tab2:

        st.subheader("🔥 Correlation Matrix")

        numeric_df = df.select_dtypes(include="number").copy()

        numeric_df = numeric_df.dropna(axis=1, how="all")
        numeric_df = numeric_df.loc[:, numeric_df.nunique() > 1]

        if numeric_df.shape[1] < 2:
            st.warning("Not enough numeric features")
        else:
            corr = numeric_df.corr().fillna(0)

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

            plt.colorbar(im, ax=ax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))

            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)


    # =========================
    # TAB 3 - ML MODELS
    # =========================
    with tab3:

        if st.button("Train Models"):

            results, best_name, best_model, X_processed = train_models(df, target)

            st.success(f"Best Model: {best_name}")

            results_df = pd.DataFrame(results).T
            st.dataframe(results_df)

            fig, ax = plt.subplots()
            results_df["accuracy"].plot(kind="bar", ax=ax)
            ax.set_title("Model Comparison")
            st.pyplot(fig)

else:
    st.info("⬅️ Upload a CSV file to start")
