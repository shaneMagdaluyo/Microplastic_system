import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MP Risk System", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")


# =========================
# UPLOAD
# =========================
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])


if file:

    df = load_data(file)

    # =========================
    # DATA OVERVIEW
    # =========================
    st.subheader("📌 Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing", int(df.isnull().sum().sum()))

    st.dataframe(df.head())


    # =========================
    # TARGET SELECTION
    # =========================
    target = st.sidebar.selectbox("🎯 Select MP Risk Column", df.columns)

    st.write("### 📊 Risk Distribution Preview")

    if df[target].dtype == "object":
        st.bar_chart(df[target].value_counts())
    else:
        clean = pd.to_numeric(df[target], errors="coerce").dropna()
        fig, ax = plt.subplots()
        ax.hist(clean, bins=20)
        st.pyplot(fig)


    # =========================
    # FEATURE COMPARISON
    # =========================
    st.subheader("🔬 MP Risk vs Features")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    if target in numeric_cols:
        numeric_cols.remove(target)

    if numeric_cols:

        feature = st.selectbox("Select Feature", numeric_cols)

        try:
            group = df.groupby(target)[feature].mean()
            st.bar_chart(group)
        except:
            st.warning("Cannot group this column")

    else:
        st.warning("No numeric features found")


    # =========================
    # TRAIN BUTTON
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
                # REPORTS
                # =========================
                for name in results:
                    with st.expander(name):
                        st.text(results[name]["report"])

                save_model(best_model)
                st.success("💾 Model saved successfully!")

            except Exception as e:
                st.error(f"Error: {e}")

else:
    st.info("⬆️ Upload CSV to start analysis")
