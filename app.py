import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="ML Dashboard",
    layout="wide"
)

st.title("📊 Machine Learning Dashboard")
st.write("Upload data → Explore → Train models → Compare performance")

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
    # DATA OVERVIEW DASHBOARD
    # =========================
    st.subheader("📌 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", int(df.isnull().sum().sum()))

    st.divider()

    # =========================
    # DATA PREVIEW
    # =========================
    st.subheader("👀 Data Preview")
    st.dataframe(df.head())

    # =========================
    # MISSING VALUES
    # =========================
    st.subheader("⚠️ Missing Values Per Column")
    st.bar_chart(df.isnull().sum())

    # =========================
    # TARGET SELECTION
    # =========================
    target = st.sidebar.selectbox("🎯 Select Target Column", df.columns)

    # =========================
    # TRAIN BUTTON
    # =========================
    if st.sidebar.button("🚀 Train Models"):

        with st.spinner("Training models... please wait"):

            results, best_name, best_model = train_models(df, target)

        st.success("Training Completed!")

        # =========================
        # MODEL PERFORMANCE CHART
        # =========================
        st.subheader("📈 Model Performance Comparison")

        names = list(results.keys())
        accs = [results[n]["accuracy"] for n in names]

        fig, ax = plt.subplots()
        ax.bar(names, accs)
        ax.set_ylabel("Accuracy")
        ax.set_title("Model Comparison")

        st.pyplot(fig)

        # =========================
        # BEST MODEL CARD
        # =========================
        st.subheader("🏆 Best Model")

        st.success(f"Best Model: {best_name}")
        st.info(f"Accuracy: {results[best_name]['accuracy']:.4f}")

        # =========================
        # DETAILED REPORT
        # =========================
        st.subheader("📋 Full Model Reports")

        for name in results:
            with st.expander(f"🤖 {name} Details"):
                st.write("Accuracy:", results[name]["accuracy"])
                st.text(results[name]["report"])

        # =========================
        # SAVE MODEL
        # =========================
        save_model(best_model)
        st.success("💾 Best model saved as best_model.pkl")

else:
    st.info("👆 Upload a CSV file to start the dashboard")
    
