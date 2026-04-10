import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model

st.set_page_config(page_title="ML Pro Pipeline", layout="wide")

st.title("🚀 Advanced ML Pipeline Dashboard")

file = st.file_uploader("Upload CSV file", type=["csv"])

if file:
    df = load_data(file)

    st.subheader("📊 Data Preview")
    st.dataframe(df.head())

    target = st.selectbox("🎯 Select Target Column", df.columns)

    if st.button("Train Models"):
        with st.spinner("Training models..."):

            results, best_name, best_model = train_models(df, target)

        st.success("Training Completed!")

        # =========================
        # ACCURACY BAR CHART
        # =========================
        st.subheader("📈 Model Accuracy Comparison")

        model_names = list(results.keys())
        accuracies = [results[m]["accuracy"] for m in model_names]

        fig, ax = plt.subplots()
        ax.bar(model_names, accuracies)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Performance")

        st.pyplot(fig)

        # =========================
        # RESULTS DETAILS
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

        cm = results[best_name]["confusion_matrix"]
        st.write(cm)

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
