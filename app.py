import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Microplastic Dashboard", layout="wide")

st.title("🌊 Microplastic Research Dashboard")


# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload CSV", type=["csv"])

if file:

    df = load_data(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())


    target = st.selectbox("Select Target Column", df.columns)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔬 Analysis",
        "🤖 ML Models",
        "🧾 Article Comparison"
    ])


    # =========================
    # TAB 1 - OVERVIEW
    # =========================
    with tab1:

        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])

        st.subheader("Risk Distribution")

        fig, ax = plt.subplots()

        if df[target].dtype == "object":
            df[target].value_counts().plot(kind="bar", ax=ax)
        else:
            ax.hist(pd.to_numeric(df[target], errors="coerce"), bins=20)

        st.pyplot(fig)


    # =========================
    # TAB 2 - CORRELATION
    # =========================
    with tab2:

        st.subheader("🔥 Correlation Matrix")

        num = df.select_dtypes(include="number")

        num = num.dropna(axis=1, how="all")
        num = num.loc[:, num.nunique() > 1]

        if num.shape[1] < 2:
            st.warning("Not enough numeric data")
        else:
            corr = num.corr().fillna(0)

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

            res_df = pd.DataFrame(results).T
            st.dataframe(res_df)

            fig, ax = plt.subplots()
            res_df["accuracy"].plot(kind="bar", ax=ax)
            st.pyplot(fig)


    # =========================
    # TAB 4 - ARTICLE COMPARISON
    # =========================
    with tab4:

        st.subheader("🧾 Microplastic Article Comparison")

        # detect article column
        article_col = None
        for col in df.columns:
            if "article" in col.lower() or "source" in col.lower():
                article_col = col
                break

        if article_col is None:
            st.warning("No Article/Source column found")
        else:

            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            metric = st.selectbox("Select Metric", numeric_cols)

            comparison = df.groupby(article_col)[metric].agg([
                "mean", "min", "max", "count"
            ])

            st.dataframe(comparison)

            st.subheader("📊 Mean Comparison")

            fig, ax = plt.subplots(figsize=(10, 5))
            comparison["mean"].plot(kind="bar", ax=ax)
            ax.set_title(f"{metric} by Article")
            st.pyplot(fig)

            st.subheader("🏆 Top Polluted Articles")

            st.dataframe(comparison.sort_values("mean", ascending=False).head(5))

            st.subheader("📦 Distribution by Article")

            fig, ax = plt.subplots(figsize=(10, 5))
            df.boxplot(column=metric, by=article_col, ax=ax)
            plt.xticks(rotation=45)

            st.pyplot(fig)

else:
    st.info("⬅️ Upload CSV file to start")
    
