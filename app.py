import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="MP Risk Dashboard", layout="wide")

st.title("🌊 Microplastic Risk Analysis Dashboard")


# =========================
# UPLOAD
# =========================
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])


if file:

    df = load_data(file)

    # =========================
    # OVERVIEW
    # =========================
    st.subheader("📌 Dataset Overview")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", df.isnull().sum().sum())

    st.dataframe(df.head())

    st.divider()


    # =========================
    # TARGET
    # =========================
    target = st.sidebar.selectbox("🎯 Select MP Risk Column", df.columns)


    # =========================
    # 🌊 SAFE MP RISK ANALYSIS
    # =========================
    st.subheader("🌊 Microplastic Risk Analysis")

    target_data = df[target]

    if target_data.dtype == "object":

        st.write("### 📊 Risk Categories")

        counts = target_data.value_counts()
        st.bar_chart(counts)

        st.write("### 📈 Risk Percentage")

        st.bar_chart((counts / counts.sum()) * 100)

    else:

        st.write("### 📊 Risk Distribution")

        clean = pd.to_numeric(target_data, errors="coerce").dropna()

        fig, ax = plt.subplots()
        ax.hist(clean, bins=20)
        ax.set_title("MP Risk Distribution")

        st.pyplot(fig)


    # =========================
    # FEATURE COMPARISON
    # =========================
    st.subheader("🔬 Feature vs MP Risk Comparison")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if target in numeric_cols:
        numeric_cols.remove(target)

    if len(numeric_cols) > 0:

        feature = st.selectbox("Select Feature", numeric_cols)

        group = df.groupby(target)[feature].mean()

        st.bar_chart(group)

    else:
        st.warning("No numeric features available")


    # =========================
    # CORRELATION
    # =========================
    st.subheader("🔥 Correlation Heatmap")

    corr = df.select_dtypes(include="number").corr()

    if not corr.empty:
        fig, ax = plt.subplots()
        ax.imshow(corr)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_yticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=90)
        ax.set_yticklabels(corr.columns)

        st.pyplot(fig)


    st.divider()


    # =========================
    # TRAIN MODELS
    # =========================
    if st.sidebar.button("🚀 Train Models"):

        try:
            results, best_name, best_model = train_models(df, target)

            st.success("Training Completed!")

            st.subheader("🤖 Model Comparison")

            names = list(results.keys())
            accs = [results[n]["accuracy"] for n in names]

            fig, ax = plt.subplots()
            ax.bar(names, accs)
            ax.set_title("Model Accuracy")

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
    st.info("Upload dataset to start")
    
