import streamlit as st
from ml_pipeline import load_data, train_models

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="ML Pipeline App", layout="centered")

st.title("🚀 Machine Learning Pipeline App")
st.write("Upload a dataset, select target column, and train models automatically.")

# =========================
# UPLOAD FILE
# =========================
file = st.file_uploader("📂 Upload CSV File", type=["csv"])

if file:
    df = load_data(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    st.subheader("📌 Columns")
    st.write(list(df.columns))

    # select target column
    target = st.selectbox("🎯 Select Target Column", df.columns)

    # train button
    if st.button("🚀 Train Models"):
        with st.spinner("Training models... please wait"):

            results, best_model = train_models(df, target)

        st.success("Training Complete!")

        # =========================
        # SHOW RESULTS
        # =========================
        st.subheader("📊 Model Results")

        for name, metrics in results.items():
            st.markdown(f"### 🤖 {name}")
            st.write(f"Accuracy: {metrics['accuracy']:.4f}")
            st.text(metrics["report"])

        # =========================
        # BEST MODEL
        # =========================
        st.subheader("🏆 Best Model")

        best_name = best_model[0]
        st.write("Best Model:", best_name)
        st.write("Best Accuracy:", f"{results[best_name]['accuracy']:.4f}")
    
