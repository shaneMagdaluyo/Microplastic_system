import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from ml_pipeline import load_data, train_models, save_model


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MP Risk Intelligence", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")
st.caption("Full Dashboard • Analysis • Machine Learning")


# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])


# =========================
# MAIN
# =========================
if file:

    df = load_data(file)

    # =========================
    # KPIs
    # =========================
    st.subheader("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", int(df.isnull().sum().sum()))
    c4.metric("Numeric Features", df.select_dtypes(include="number").shape[1])

    st.divider()

    # TARGET
    target = st.sidebar.selectbox("🎯 Select Risk Column", df.columns)

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Analysis", "🤖 ML"])

    # ==================================================
    # DASHBOARD
    # ==================================================
    with tab1:

        st.subheader("📌 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("🌊 Risk Distribution")

        target_data = df[target]
        col1, col2 = st.columns(2)

        if target_data.dtype == "object":

            counts = target_data.value_counts()
            col1.bar_chart(counts)

            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%")
            col2.pyplot(fig)

        else:
            clean = pd.to_numeric(target_data, errors="coerce").dropna()

            fig, ax = plt.subplots()
            ax.hist(clean, bins=20)
            col1.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.boxplot(clean)
            col2.pyplot(fig2)

    # ==================================================
    # ANALYSIS
    # ==================================================
    with tab2:

        # =========================
        # FEATURE COMPARISON
        # =========================
        st.subheader("🔬 Feature Comparison")

        cols = df.columns.tolist()
        cols.remove(target)

        feature = st.selectbox("Select Feature", cols)

        col1, col2 = st.columns(2)

        if pd.api.types.is_numeric_dtype(df[feature]):

            agg = st.selectbox("Aggregation", ["Mean", "Median"])

            try:
                grouped = (
                    df.groupby(target)[feature].mean()
                    if agg == "Mean"
                    else df.groupby(target)[feature].median()
                )

                col1.bar_chart(grouped)

                fig, ax = plt.subplots()
                df.boxplot(column=feature, by=target, ax=ax)
                col2.pyplot(fig)

            except Exception as e:
                st.warning(str(e))

        else:

            try:
                cross = pd.crosstab(df[target], df[feature])
                col1.bar_chart(cross)
            except Exception as e:
                st.warning(str(e))

        st.divider()

        # =========================
        # 🔥 CORRELATION MATRIX (AUTO FIX)
        # =========================
        st.subheader("🔥 Correlation Matrix (Auto-Working)")

        corr_df = df.copy()
        le = LabelEncoder()

        # Encode categorical columns
        for col in corr_df.columns:
            if corr_df[col].dtype == "object":
                try:
                    corr_df[col] = le.fit_transform(corr_df[col].astype(str))
                except:
                    pass

        # Fill missing values
        corr_df = corr_df.fillna(corr_df.mean(numeric_only=True))

        if corr_df.select_dtypes(include="number").shape[1] < 2:
            st.error("Still not enough numeric data")
        else:

            method = st.selectbox(
                "Method",
                ["pearson", "spearman", "kendall"]
            )

            corr = corr_df.corr(method=method)

            fig, ax = plt.subplots(figsize=(10, 6))
            cax = ax.imshow(corr)
            plt.colorbar(cax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)

            # Strong relationships
            st.subheader("📌 Strong Relationships")

            threshold = st.slider("Threshold", 0.3, 1.0, 0.6)

            found = False
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) >= threshold:
                        st.write(f"{corr.columns[i]} ↔ {corr.columns[j]} = {val:.2f}")
                        found = True

            if not found:
                st.info("No strong correlations")

            # Top correlations
            st.subheader("🏆 Top Correlations")

            top = (
                corr.abs()
                .unstack()
                .sort_values(ascending=False)
            )

            top = top[top < 1].drop_duplicates().head(5)
            st.write(top)

    # ==================================================
    # ML
    # ==================================================
    with tab3:

        st.subheader("🤖 Model Training")

        if st.button("Train Models"):

            with st.spinner("Training..."):

                try:
                    results, best_name, best_model = train_models(df, target)

                    st.success("Training Complete")

                    names = list(results.keys())
                    accs = [results[n]["accuracy"] for n in names]

                    fig, ax = plt.subplots()
                    ax.bar(names, accs)
                    st.pyplot(fig)

                    st.success(f"Best Model: {best_name}")
                    st.info(f"Accuracy: {results[best_name]['accuracy']:.4f}")

                    for name in results:
                        with st.expander(name):
                            st.text(results[name]["report"])

                    save_model(best_model)
                    st.success("Model Saved!")

                except Exception as e:
                    st.error(str(e))

else:
    st.info("⬅️ Upload a dataset to begin")
