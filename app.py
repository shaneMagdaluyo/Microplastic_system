import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import train_models

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Microplastic Dashboard", layout="wide")

st.title("🌊 Microplastic Risk Intelligence Dashboard")

# =========================
# UPLOAD DATA
# =========================
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target = st.selectbox("Select Target Column", df.columns)

    # =========================
    # TRAIN MODELS
    # =========================
    if st.button("🚀 Train Models"):

        results, best_name, best_model, X_test, y_test = train_models(df, target)

        st.success(f"🏆 Best Model: {best_name}")

        # =========================
        # MODEL PERFORMANCE
        # =========================
        st.subheader("📈 Model Comparison")

        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        fig, ax = plt.subplots()
        results_df["accuracy"].plot(kind="bar", ax=ax)
        ax.set_title("Model Accuracy")
        st.pyplot(fig)

        # =========================
        # PREDICTION TABLE (🔥 MAIN FEATURE)
        # =========================
        st.subheader("🎯 Prediction Table (Test Set Results)")

        predictions = best_model.predict(X_test)

        pred_df = X_test.copy()
        pred_df["Actual"] = y_test.values
        pred_df["Predicted"] = predictions

        st.dataframe(pred_df)

        # =========================
        # DOWNLOAD BUTTON
        # =========================
        csv = pred_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "📥 Download Predictions",
            csv,
            "predictions.csv",
            "text/csv"
        )

        # =========================
        # VISUALIZATION
        # =========================
        st.subheader("📊 Risk Distribution")

        fig, ax = plt.subplots()
        ax.hist(y_test, bins=20)
        ax.set_title("Risk Distribution")
        st.pyplot(fig)

else:
    st.info("⬅️ Upload dataset to start")
