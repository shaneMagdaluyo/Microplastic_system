import streamlit as st
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="ML Dashboard", layout="wide")

st.title("📊 Machine Learning Dashboard")

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

    st.subheader("📌 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing", df.isnull().sum().sum())

    st.dataframe(df.head())

    st.subheader("⚠️ Missing Values")
    st.bar_chart(df.isnull().sum())

    target = st.sidebar.selectbox("🎯 Select Target Column", df.columns)

    st.write("Class distribution:")
    st.write(df[target].value_counts())

    if st.sidebar.button("🚀 Train Models"):

        with st.spinner("Training..."):

            try:
                results, best_name, best_model = train_models(df, target)

                st.success("Training Completed!")

                # =========================
                # ACCURACY CHART
                # =========================
                st.subheader("📈 Model Comparison")

                names = list(results.keys())
                accs = [results[n]["accuracy"] for n in names]

                fig, ax = plt.subplots()
                ax.bar(names, accs)
                ax.set_title("Model Accuracy")

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

                # save model
                save_model(best_model)
                st.success("💾 Model saved!")

            except Exception as e:
                st.error(f"Error: {str(e)}")

else:
    st.info("Upload a CSV to start")
