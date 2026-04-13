import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import load_data, train_models, save_model


# =========================
# RISK MATRIX ENGINE (REAL NAME)
# =========================
def create_risk_matrix(series, name_series):

    numeric = pd.to_numeric(series, errors="coerce")

    df = pd.DataFrame({
        "Name": name_series,
        "Value": numeric
    })

    df = df.dropna()

    if len(df) < 3:
        return None

    min_val = df["Value"].min()
    max_val = df["Value"].max()

    # Normalize 0–100
    if max_val == min_val:
        df["Risk Score (0–100)"] = 50
    else:
        df["Risk Score (0–100)"] = (
            (df["Value"] - min_val) / (max_val - min_val)
        ) * 100

    # Risk classification
    def classify(x):
        if x < 25:
            return "Low"
        elif x < 50:
            return "Medium"
        elif x < 75:
            return "High"
        else:
            return "Critical"

    df["Risk Level"] = df["Risk Score (0–100)"].apply(classify)

    return df


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

    # =========================
    # OVERVIEW
    # =========================
    st.subheader("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", int(df.isnull().sum().sum()))
    c4.metric("Numeric Features", df.select_dtypes(include="number").shape[1])

    st.divider()

    # TARGET
    target = st.sidebar.selectbox("🎯 Select Risk Column", df.columns)

    # NAME COLUMN (REAL IDENTIFIER)
    name_col = st.sidebar.selectbox("🏷️ Select Name Column", df.columns)

    # TABS
    tab1, tab2, tab3 = st.tabs(["📊 Dashboard", "🔬 Analysis", "🤖 ML Models"])


    # ==================================================
    # TAB 1: DASHBOARD
    # ==================================================
    with tab1:

        st.subheader("📌 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("🌊 Risk Distribution")

        target_data = df[target]

        col1, col2 = st.columns(2)

        if target_data.dtype == "object":

            counts = target_data.value_counts()

            col1.bar_chart(counts)

            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%")
            col2.pyplot(fig)

        else:

            clean = pd.to_numeric(target_data, errors="coerce").dropna()

            fig, ax = plt.subplots()
            ax.hist(clean, bins=20)
            col1.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.boxplot(clean)
            col2.pyplot(fig2)

        # =========================
        # ⚠️ RISK MATRIX SECTION
        # =========================
        st.subheader("⚠️ Risk Matrix Analysis")

        risk_df = create_risk_matrix(df[target], df[name_col])

        if risk_df is None:
            st.warning("Not enough valid numeric data for risk matrix")
        else:

            col3, col4 = st.columns(2)

            level_counts = risk_df["Risk Level"].value_counts()

            order = ["Low", "Medium", "High", "Critical"]
            level_counts = level_counts.reindex(order).fillna(0)

            col3.bar_chart(level_counts)

            fig, ax = plt.subplots()
            ax.pie(level_counts, labels=level_counts.index, autopct="%1.1f%%")
            col4.pyplot(fig)

            # =========================
            # TABLE (REAL NAME USED)
            # =========================
            st.subheader("📊 Risk Matrix Table")

            st.dataframe(risk_df, use_container_width=True)

            # =========================
            # TOP RISKS
            # =========================
            st.subheader("🏆 Highest Risk Entries")

            st.dataframe(
                risk_df.sort_values("Risk Score (0–100)", ascending=False).head(10)
            )

            # =========================
            # INTERPRETATION
            # =========================
            st.markdown("""
            ### 📌 Risk Interpretation Matrix

            | Level | Score Range | Meaning |
            |------|------------|---------|
            | 🟢 Low | 0–24 | Safe level |
            | 🟡 Medium | 25–49 | Moderate contamination |
            | 🟠 High | 50–74 | Elevated risk |
            | 🔴 Critical | 75–100 | Dangerous pollution |
            """)


    # ==================================================
    # TAB 2: ANALYSIS
    # ==================================================
    with tab2:

        st.subheader("🔬 Feature Comparison by Risk")

        cols = df.columns.tolist()
        cols.remove(target)

        feature = st.selectbox("Select Feature", cols)

        col1, col2 = st.columns(2)

        if pd.api.types.is_numeric_dtype(df[feature]):

            agg = st.selectbox("Aggregation", ["Mean", "Median"])

            try:
                if agg == "Mean":
                    grouped = df.groupby(target)[feature].mean()
                else:
                    grouped = df.groupby(target)[feature].median()

                col1.bar_chart(grouped)

                fig, ax = plt.subplots()
                df.boxplot(column=feature, by=target, ax=ax)
                col2.pyplot(fig)

            except Exception as e:
                st.warning(str(e))

        else:

            try:
                cross = pd.crosstab(df[target], df[feature])
                col1.bar_chart(cross)

            except Exception as e:
                st.warning(str(e))


        st.divider()

        # CORRELATION MATRIX
        st.subheader("🔥 Correlation Matrix")

        num_df = df.select_dtypes(include="number").copy()

        if num_df.shape[1] < 2:
            st.warning("Not enough numeric features")
        else:

            method = st.selectbox("Correlation Method", ["pearson", "spearman", "kendall"])

            num_df = num_df.fillna(num_df.mean())
            corr = num_df.corr(method=method)

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(corr)

            plt.colorbar(im)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))

            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)


    # ==================================================
    # TAB 3: ML MODELS
    # ==================================================
    with tab3:

        st.subheader("🤖 Model Training")

        if st.button("Train Models"):

            with st.spinner("Training..."):

                try:
                    results, best_name, best_model = train_models(df, target)

                    st.success("Training Complete")

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
                    st.success("Model Saved!")

                except Exception as e:
                    st.error(str(e))

else:
    st.info("⬅️ Upload a CSV file to begin")
