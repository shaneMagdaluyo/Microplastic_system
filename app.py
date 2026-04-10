import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# UI SETUP
# =========================
st.set_page_config(page_title="ML Pipeline App", layout="wide")

st.title("🚀 Advanced ML Pipeline (Error-Free Version)")
st.write("Upload CSV → Auto Clean → Train Models → Compare Results")

# =========================
# UPLOAD FILE
# =========================
file = st.file_uploader("Upload CSV File", type=["csv"])

if file:

    df = load_data(file)

    st.subheader("📊 Raw Data Preview")
    st.dataframe(df.head())

    st.subheader("📌 Missing Values Report")
    st.write(df.isnull().sum())

    target = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("🚀 Train Models"):

        with st.spinner("Cleaning data + training models..."):

            results, best_name, best_model = train_models(df, target)

        st.success("Training Completed!")

        # =========================
        # ACCURACY CHART
        # =========================
        st.subheader("📈 Model Accuracy Comparison")

        names = list(results.keys())
        accs = [results[n]["accuracy"] for n in names]

        fig, ax = plt.subplots()
        ax.bar(names, accs)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison")

        st.pyplot(fig)

        # =========================
        # RESULTS
        # =========================
        st.subheader("📋 Detailed Results")

        for name in results:
            st.markdown(f"### 🤖 {name}")
            st.write("Accuracy:", results[name]["accuracy"])
            st.text(results[name]["report"])

        # =========================
        # CONFUSION MATRIX
        # =========================
        st.subheader("🧮 Confusion Matrix (Best Model)")

        st.write(results[best_name]["confusion_matrix"])

        # =========================
        # BEST MODEL
        # =========================
        st.subheader("🏆 Best Model")

        st.write("Model:", best_name)
        st.write("Accuracy:", results[best_name]["accuracy"])

        # =========================
        # SAVE MODEL
        # =========================
        save_model(best_model)

        st.success("Best model saved as best_model.pkl")
