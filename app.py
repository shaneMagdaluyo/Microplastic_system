import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MP Risk Intelligence", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")
st.caption("Advanced Dashboard with Analytics & Machine Learning")


# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])


# =========================
# MAIN APP
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
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Numeric Features", df.select_dtypes(include="number").shape[1])

    st.divider()

    # TARGET
    target = st.sidebar.selectbox("🎯 Select Risk Column", df.columns)

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Analysis", "🤖 ML Models"])

    # ==================================================
    # TAB 1: DASHBOARD
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
    # TAB 2: ANALYSIS (🔥 FULLY FUNCTIONAL)
    # ==================================================
    with tab2:

        st.subheader("🔬 Feature Comparison by Risk")

        cols = df.columns.tolist()
        cols.remove(target)

        feature = st.selectbox("Select Feature", cols)

        col1, col2 = st.columns(2)

        # NUMERIC
        if pd.api.types.is_numeric_dtype(df[feature]):

            agg = st.selectbox("Aggregation", ["Mean", "Median"])

            try:
                if agg == "Mean":
                    grouped = df.groupby(target)[feature].mean()
                else:
                    grouped = df.groupby(target)[feature].median()

                col1.bar_chart(grouped)

                fig, ax = plt.subplots()
                df.boxplot(column=feature, by=target, ax=ax)
                col2.pyplot(fig)

                st.markdown(f"""
                ### 📌 Insights
                - Highest: **{grouped.idxmax()}**
                - Lowest: **{grouped.idxmin()}**
                """)

            except Exception as e:
                st.warning(str(e))

        # CATEGORICAL
        else:

            try:
                cross = pd.crosstab(df[target], df[feature])
                col1.bar_chart(cross)

                st.markdown(f"""
                ### 📌 Insights
                - Most common: **{df[feature].value_counts().idxmax()}**
                - Categories: **{df[feature].nunique()}**
                """)

            except Exception as e:
                st.warning(str(e))

        st.divider()

        # ==================================================
        # 🔥 CORRELATION MATRIX (ENHANCED)
        # ==================================================
        st.subheader("🔥 Correlation Matrix")

        num_df = df.select_dtypes(include="number").copy()

        if num_df.shape[1] < 2:
            st.warning("Not enough numeric features")
        else:

            method = st.selectbox(
                "Correlation Method",
                ["pearson", "spearman", "kendall"]
            )

            num_df = num_df.fillna(num_df.mean())
            corr = num_df.corr(method=method)

            fig, ax = plt.subplots(figsize=(10, 6))
            cax = ax.imshow(corr)

            plt.colorbar(cax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))

            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)

            ax.set_title(f"{method.capitalize()} Correlation")

            st.pyplot(fig)

            # STRONG RELATIONSHIPS
            st.subheader("📌 Strong Relationships")

            threshold = st.slider("Threshold", 0.5, 1.0, 0.7)

            strong = []
            for i in range(len(corr.columns)):
                for j in range(i + 1, len(corr.columns)):
                    val = corr.iloc[i, j]
                    if abs(val) >= threshold:
                        strong.append((corr.columns[i], corr.columns[j], val))

            if strong:
                for f1, f2, val in strong:
                    st.write(f"{f1} ↔ {f2} = {val:.2f}")
            else:
                st.info("No strong correlations")

            # TOP CORRELATIONS
            st.subheader("🏆 Top Correlations")

            top_corr = (
                corr.abs()
                .unstack()
                .sort_values(ascending=False)
            )

            top_corr = top_corr[top_corr < 1].drop_duplicates().head(5)

            st.write(top_corr)

    # ==================================================
    # TAB 3: ML
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
    st.info("⬅️ Upload a CSV file to begin")
