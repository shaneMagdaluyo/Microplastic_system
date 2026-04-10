import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MP Risk Intelligence", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")
st.caption("Professional Dashboard with ML & Risk Analytics")


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
    # DASHBOARD KPIs
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

        # FIXED HISTOGRAM ERROR HERE
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
    # TAB 2: FEATURE ANALYSIS (🔥 FIXED)
    # ==================================================
    with tab2:

        st.subheader("🔬 Feature Comparison by Risk")

        cols = df.columns.tolist()
        cols.remove(target)

        feature = st.selectbox("Select Feature", cols)

        col1, col2 = st.columns(2)

        # NUMERIC FEATURE
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
                - Highest value group: **{grouped.idxmax()}**
                - Lowest value group: **{grouped.idxmin()}**
                """)

            except Exception as e:
                st.warning(str(e))

        # CATEGORICAL FEATURE
        else:

            try:
                cross = pd.crosstab(df[target], df[feature])

                col1.bar_chart(cross)

                st.markdown(f"""
                ### 📌 Insights
                - Most common: **{df[feature].value_counts().idxmax()}**
                - Total categories: **{df[feature].nunique()}**
                """)

            except Exception as e:
                st.warning(str(e))

        st.divider()

        # CORRELATION
        st.subheader("🔥 Correlation Matrix")

        corr = df.select_dtypes(include="number").corr()

        if not corr.empty:
            fig, ax = plt.subplots()
            cax = ax.imshow(corr)
            plt.colorbar(cax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)

    # ==================================================
    # TAB 3: ML
    # ==================================================
    with tab3:

        st.subheader("🤖 Model Training")

        if st.button("Train Models"):

            with st.spinner("Training..."):

                try:
                    results, best_name, best_model = train_models(df, target)

                    st.success("Training Done")

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
    st.info("⬅️ Upload a CSV file to start")
