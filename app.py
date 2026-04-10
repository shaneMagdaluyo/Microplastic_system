import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="MP Risk Intelligence System",
    layout="wide",
    page_icon="🌊"
)

# =========================
# CUSTOM STYLE
# =========================
st.markdown("""
<style>
.metric-box {
    background-color: #f5f7fa;
    padding: 15px;
    border-radius: 10px;
    text-align: center;
}
</style>
""", unsafe_allow_html=True)


# =========================
# HEADER
# =========================
st.title("🌊 Microplastic Risk Intelligence System")
st.caption("Professional Dashboard for Risk Analysis & Machine Learning")


# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")
file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])


# =========================
# MAIN
# =========================
if file:

    df = load_data(file)

    # =========================
    # KPI DASHBOARD
    # =========================
    st.subheader("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.markdown(f"<div class='metric-box'><h3>{df.shape[0]}</h3><p>Rows</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-box'><h3>{df.shape[1]}</h3><p>Columns</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-box'><h3>{df.isnull().sum().sum()}</h3><p>Missing</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-box'><h3>{df.select_dtypes(include='number').shape[1]}</h3><p>Numeric Features</p></div>", unsafe_allow_html=True)

    st.divider()


    # =========================
    # TABS
    # =========================
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Analysis", "🤖 ML Models"])


    # =========================
    # TAB 1: DASHBOARD
    # =========================
    with tab1:

        st.subheader("📌 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        target = st.sidebar.selectbox("🎯 Select MP Risk Column", df.columns)

        st.subheader("🌊 Risk Distribution")

        if df[target].dtype == "object":
            st.bar_chart(df[target].value_counts())
        else:
            clean = pd.to_numeric(df[target], errors="coerce").dropna()
            fig, ax = plt.subplots()
            ax.hist(clean, bins=20)
            st.pyplot(fig)


    # =========================
    # TAB 2: ANALYSIS
    # =========================
    with tab2:

        st.subheader("🔬 Feature Analysis")

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if target in numeric_cols:
            numeric_cols.remove(target)

        if numeric_cols:

            col1, col2 = st.columns(2)

            feature = col1.selectbox("Select Feature", numeric_cols)

            try:
                group = df.groupby(target)[feature].mean()
                col2.bar_chart(group)
            except:
                st.warning("Cannot analyze this feature")

        else:
            st.warning("No numeric features found")

        st.divider()

        st.subheader("🔥 Correlation Matrix")

        corr = df.select_dtypes(include="number").corr()

        if not corr.empty:
            fig, ax = plt.subplots()
            cax = ax.imshow(corr)
            plt.colorbar(cax)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=90)
            ax.set_yticklabels(corr.columns)
            st.pyplot(fig)


    # =========================
    # TAB 3: ML MODELS
    # =========================
    with tab3:

        st.subheader("🤖 Model Training")

        if st.button("🚀 Train Models"):

            with st.spinner("Training..."):

                try:
                    results, best_name, best_model = train_models(df, target)

                    st.success("Training Completed!")

                    # comparison chart
                    names = list(results.keys())
                    accs = [results[n]["accuracy"] for n in names]

                    fig, ax = plt.subplots()
                    ax.bar(names, accs)
                    ax.set_title("Model Performance")

                    st.pyplot(fig)

                    # best model
                    st.subheader("🏆 Best Model")
                    st.success(best_name)
                    st.info(f"Accuracy: {results[best_name]['accuracy']:.4f}")

                    # reports
                    for name in results:
                        with st.expander(name):
                            st.text(results[name]["report"])

                    save_model(best_model)
                    st.success("💾 Model saved!")

                except Exception as e:
                    st.error(str(e))


else:
    st.info("⬅️ Upload a dataset to begin")
