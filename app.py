import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MP Risk Dashboard", layout="wide")

st.title("🌊 Microplastic Risk Analysis Dashboard")
st.write("Compare MP risk levels + train ML models + analyze patterns")


# =========================
# UPLOAD
# =========================
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])


if file:

    df = load_data(file)

    # =========================
    # BASIC INFO
    # =========================
    st.subheader("📌 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.dataframe(df.head())

    st.divider()


    # =========================
    # TARGET SELECTION
    # =========================
    target = st.sidebar.selectbox("🎯 Select MP Risk Column", df.columns)


    # =========================
    # 🌊 MP RISK ANALYSIS SECTION
    # =========================
    st.subheader("🌊 Microplastic Risk Analysis")

    if df[target].dtype == "object":
        risk_counts = df[target].value_counts()

        st.write("### 📊 Risk Distribution")
        st.bar_chart(risk_counts)

        st.write("### 📈 Risk Percentage")

        risk_percent = (risk_counts / risk_counts.sum()) * 100
        st.bar_chart(risk_percent)

    else:
        st.write("### 📊 Risk Value Distribution")
        fig, ax = plt.subplots()
        ax.hist(df[target], bins=20)
        st.pyplot(fig)


    # =========================
    # RISK vs FEATURES COMPARISON
    # =========================
    st.subheader("🔬 Feature Comparison by Risk Level")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if target in numeric_cols:
        numeric_cols.remove(target)

    if len(numeric_cols) > 0:

        selected_feature = st.selectbox("Select Feature to Compare", numeric_cols)

        group_avg = df.groupby(target)[selected_feature].mean()

        st.write("### 📊 Average Feature Value per Risk Level")

        st.bar_chart(group_avg)

    else:
        st.warning("No numeric features available for comparison")


    # =========================
    # CORRELATION HEATMAP (OPTIONAL INSIGHT)
    # =========================
    st.subheader("🔥 Correlation Insight")

    corr = df.select_dtypes(include="number").corr()

    if not corr.empty:
        fig, ax = plt.subplots()
        cax = ax.imshow(corr, cmap="coolwarm")
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        plt.colorbar(cax)
        st.pyplot(fig)


    st.divider()


    # =========================
    # ML TRAINING SECTION
    # =========================
    if st.sidebar.button("🚀 Train ML Models"):

        with st.spinner("Training models..."):

            try:
                results, best_name, best_model = train_models(df, target)

                st.success("Training Completed!")

                # =========================
                # MODEL COMPARISON
                # =========================
                st.subheader("🤖 Model Comparison")

                names = list(results.keys())
                accs = [results[n]["accuracy"] for n in names]

                fig, ax = plt.subplots()
                ax.bar(names, accs)
                ax.set_ylabel("Accuracy")
                ax.set_title("Model Performance")

                st.pyplot(fig)

                # =========================
                # BEST MODEL
                # =========================
                st.subheader("🏆 Best Model")

                st.success(best_name)
                st.info(f"Accuracy: {results[best_name]['accuracy']:.4f}")

                # =========================
                # FULL REPORTS
                # =========================
                for name in results:
                    with st.expander(f"📌 {name} Report"):
                        st.text(results[name]["report"])

                # save model
                save_model(best_model)
                st.success("💾 Model saved successfully!")

            except Exception as e:
                st.error(f"Error: {str(e)}")

else:
    st.info("⬆️ Upload a dataset to start analysis")
