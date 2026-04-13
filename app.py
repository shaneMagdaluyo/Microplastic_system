import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import train_models

st.set_page_config(page_title="MP Risk Intelligence", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")
st.caption("Dashboard • Analysis • ML • Risk Scoring Engine")

# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    # =========================
    # OVERVIEW
    # =========================
    st.subheader("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", int(df.isnull().sum().sum()))
    c4.metric("Numeric", df.select_dtypes(include="number").shape[1])

    st.dataframe(df.head())

    # =========================
    # TARGET
    # =========================
    target = st.selectbox("Select Target Column (Risk)", df.columns)

    if st.button("🚀 Run Full Analysis"):

        # =========================
        # TRAIN MODELS
        # =========================
        results, best_name, best_model, X_processed = train_models(df, target)

        st.success(f"🏆 Best Model: {best_name}")

        # =========================
        # MODEL COMPARISON
        # =========================
        st.subheader("📊 Model Comparison")

        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        fig, ax = plt.subplots()
        results_df["accuracy"].plot(kind="bar", ax=ax)
        st.pyplot(fig)

        # =========================
        # RISK DISTRIBUTION
        # =========================
        st.subheader("🌊 Risk Distribution")

        if df[target].dtype == "object":
            encoded = df[target].astype("category").cat.codes
        else:
            encoded = pd.to_numeric(df[target], errors="coerce")

        fig, ax = plt.subplots()
        ax.hist(encoded.dropna(), bins=20)
        st.pyplot(fig)

        # =========================
        # CORRELATION MATRIX (AUTO SAFE)
        # =========================
        st.subheader("🔥 Correlation Matrix")

        corr_df = X_processed.copy()
        corr_df["target"] = pd.factorize(df[target])[0]

        if corr_df.shape[1] < 2:
            st.warning("Not enough numeric features")
        else:
            corr = corr_df.corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            cax = ax.imshow(corr)
            plt.colorbar(cax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45)
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)

        # =========================
        # 🎯 RISK SCORE ENGINE
        # =========================
        st.subheader("🎯 Microplastic Risk Score Engine")

        input_data = {}

        st.write("Enter environmental parameters:")

        for col in X_processed.columns:
            input_data[col] = st.number_input(f"{col}", value=0.0)

        if st.button("Predict Risk Score"):

            input_df = pd.DataFrame([input_data])

            # Probability → Score
            if hasattr(best_model, "predict_proba"):
                prob = best_model.predict_proba(input_df)[0][1]
            else:
                prob = best_model.predict(input_df)[0]

            risk_score = prob * 100

            # Risk Level
            if risk_score < 33:
                level = "🟢 Low"
            elif risk_score < 66:
                level = "🟡 Medium"
            else:
                level = "🔴 High"

            st.metric("Risk Score", f"{risk_score:.2f}/100")
            st.progress(int(risk_score))
            st.success(f"Risk Level: {level}")

else:
    st.info("⬅️ Upload a dataset to begin")import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ml_pipeline import train_models

st.set_page_config(page_title="MP Risk Intelligence", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")
st.caption("Dashboard • Analysis • ML • Risk Scoring Engine")

# =========================
# UPLOAD
# =========================
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:

    df = pd.read_csv(file)

    # =========================
    # OVERVIEW
    # =========================
    st.subheader("📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", int(df.isnull().sum().sum()))
    c4.metric("Numeric", df.select_dtypes(include="number").shape[1])

    st.dataframe(df.head())

    # =========================
    # TARGET
    # =========================
    target = st.selectbox("Select Target Column (Risk)", df.columns)

    if st.button("🚀 Run Full Analysis"):

        # =========================
        # TRAIN MODELS
        # =========================
        results, best_name, best_model, X_processed = train_models(df, target)

        st.success(f"🏆 Best Model: {best_name}")

        # =========================
        # MODEL COMPARISON
        # =========================
        st.subheader("📊 Model Comparison")

        results_df = pd.DataFrame(results).T
        st.dataframe(results_df)

        fig, ax = plt.subplots()
        results_df["accuracy"].plot(kind="bar", ax=ax)
        st.pyplot(fig)

        # =========================
        # RISK DISTRIBUTION
        # =========================
        st.subheader("🌊 Risk Distribution")

        if df[target].dtype == "object":
            encoded = df[target].astype("category").cat.codes
        else:
            encoded = pd.to_numeric(df[target], errors="coerce")

        fig, ax = plt.subplots()
        ax.hist(encoded.dropna(), bins=20)
        st.pyplot(fig)

        # =========================
        # CORRELATION MATRIX (AUTO SAFE)
        # =========================
        st.subheader("🔥 Correlation Matrix")

        corr_df = X_processed.copy()
        corr_df["target"] = pd.factorize(df[target])[0]

        if corr_df.shape[1] < 2:
            st.warning("Not enough numeric features")
        else:
            corr = corr_df.corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            cax = ax.imshow(corr)
            plt.colorbar(cax)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45)
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)

        # =========================
        # 🎯 RISK SCORE ENGINE
        # =========================
        st.subheader("🎯 Microplastic Risk Score Engine")

        input_data = {}

        st.write("Enter environmental parameters:")

        for col in X_processed.columns:
            input_data[col] = st.number_input(f"{col}", value=0.0)

        if st.button("Predict Risk Score"):

            input_df = pd.DataFrame([input_data])

            # Probability → Score
            if hasattr(best_model, "predict_proba"):
                prob = best_model.predict_proba(input_df)[0][1]
            else:
                prob = best_model.predict(input_df)[0]

            risk_score = prob * 100

            # Risk Level
            if risk_score < 33:
                level = "🟢 Low"
            elif risk_score < 66:
                level = "🟡 Medium"
            else:
                level = "🔴 High"

            st.metric("Risk Score", f"{risk_score:.2f}/100")
            st.progress(int(risk_score))
            st.success(f"Risk Level: {level}")

else:
    st.info("⬅️ Upload a dataset to begin")
