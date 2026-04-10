import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MP Risk Intelligence System", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")
st.caption("Advanced Dashboard for Risk Analysis & Machine Learning")


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
    # KPI DASHBOARD
    # =========================
    st.subheader("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", int(df.isnull().sum().sum()))
    c4.metric("Numeric Features", df.select_dtypes(include="number").shape[1])

    st.divider()

    # =========================
    # TARGET
    # =========================
    target = st.sidebar.selectbox("🎯 Select MP Risk Column", df.columns)

    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Analysis", "🤖 ML Models"])

    # =========================
    # TAB 1: DASHBOARD
    # =========================
    with tab1:

        st.subheader("📌 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        # =========================
        # 🌊 ENHANCED RISK DISTRIBUTION
        # =========================
        st.subheader("🌊 Risk Distribution Analysis")

        target_data = df[target]

        col1, col2 = st.columns(2)

        if target_data.dtype == "object":

            counts = target_data.value_counts()
            percent = (counts / counts.sum()) * 100

            col1.write("### 📊 Risk Count")
            col1.bar_chart(counts)

            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct='%1.1f%%')
            ax.set_title("Risk Share")
            col2.pyplot(fig)

            top_risk = counts.idxmax()

            st.markdown(f"""
            ### 📌 Key Insights
            - Most common risk: **{top_risk}**
            - Highest percentage: **{percent.max():.2f}%**
            - Total categories: **{len(counts)}**
            """)

        else:

            clean = pd.to_numeric(target_data, errors="coerce").dropna()

            fig, ax = plt.subplots()
            ax.hist(clean, bins=20)
            ax.set_title("Risk Distribution")
            col1.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.boxplot(clean)
            ax2.set_title("Risk Spread")
            col2.pyplot(fig2)

            st.markdown(f"""
            ### 📌 Key Insights
            - Mean Risk: **{clean.mean():.2f}**
            - Max Risk: **{clean.max():.2f}**
            - Min Risk: **{clean.min():.2f}**
            - Std Dev: **{clean.std():.2f}**
            """)

    # =========================
    # TAB 2: ANALYSIS
    # =========================
    with tab2:

        st.subheader("🔬 Feature Comparison")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if target in numeric_cols:
            numeric_cols.remove(target)

        if numeric_cols:
            feature = st.selectbox("Select Feature", numeric_cols)

            try:
                group = df.groupby(target)[feature].mean()
                st.bar_chart(group)
            except:
                st.warning("Cannot analyze this feature")

        st.divider()

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

    # =========================
    # TAB 3: ML MODELS
    # =========================
    with tab3:

        st.subheader("🤖 Model Training")

        if st.button("🚀 Train Models"):

            with st.spinner("Training..."):

                try:
                    results, best_name, best_model = train_models(df, target)

                    st.success("Training Completed!")

                    names = list(results.keys())
                    accs = [results[n]["accuracy"] for n in names]

                    fig, ax = plt.subplots()
                    ax.bar(names, accs)
                    ax.set_title("Model Performance")

                    st.pyplot(fig)

                    st.subheader("🏆 Best Model")
                    st.success(best_name)
                    st.info(f"Accuracy: {results[best_name]['accuracy']:.4f}")

                    for name in results:
                        with st.expander(name):
                            st.text(results[name]["report"])

                    save_model(best_model)
                    st.success("💾 Model saved!")

                except Exception as e:
                    st.error(str(e))

else:
    st.info("⬅️ Upload a dataset to begin")
