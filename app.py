import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MP Risk Intelligence", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")
st.caption("Advanced Dashboard with Analytics & Machine Learning")


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

    st.subheader("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Numeric Features", df.select_dtypes(include="number").shape[1])

    target = st.sidebar.selectbox("🎯 Select Risk Column", df.columns)

    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Analysis", "🤖 ML Models"])


    # =========================
    # TAB 1
    # =========================
    with tab1:

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Risk Distribution")

        fig, ax = plt.subplots()

        if df[target].dtype == "object":
            df[target].value_counts().plot(kind="bar", ax=ax)
        else:
            ax.hist(pd.to_numeric(df[target], errors="coerce").dropna(), bins=20)

        st.pyplot(fig)


    # =========================
    # TAB 2 (CORRELATION FIXED)
    # =========================
    with tab2:

        st.subheader("🔥 Correlation Matrix")

        num_df = df.select_dtypes(include="number").copy()
        num_df = num_df.dropna(axis=1, how="all")
        num_df = num_df.loc[:, num_df.nunique() > 1]

        if num_df.shape[1] < 2:
            st.warning("Not enough numeric features")
        else:

            corr = num_df.corr().fillna(0)

            fig, ax = plt.subplots(figsize=(8, 5))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

            plt.colorbar(im, ax=ax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))

            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)


    # =========================
    # TAB 3 (ML MODELS)
    # =========================
    with tab3:

        st.subheader("🤖 Model Training")

        if st.button("Train Models"):

            with st.spinner("Training models..."):

                try:
                    results, best_name, best_model = train_models(df, target)

                    st.success("Training Completed")

                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df)

                    fig, ax = plt.subplots()
                    results_df["accuracy"].plot(kind="bar", ax=ax)
                    st.pyplot(fig)

                    st.success(f"Best Model: {best_name}")
                    st.info(f"Accuracy: {results[best_name]['accuracy']:.4f}")

                    for name in results:
                        with st.expander(name):
                            st.text(results[name]["report"])

                    save_model(best_model)
                    st.success("Model Saved Successfully")

                except Exception as e:
                    st.error(str(e))

else:
    st.info("⬅️ Upload a CSV file to start")
