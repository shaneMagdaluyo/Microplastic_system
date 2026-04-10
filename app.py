import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Polymer Risk ML System", layout="wide")
st.title("🌊 Enterprise Polymer Risk ML System")

# =========================
# SESSION STATE
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

# =========================
# LOAD DATA
# =========================
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.session_state.df = df
    st.sidebar.success("Data Loaded")

df = st.session_state.df

# =========================
# SIDEBAR MENU
# =========================
menu = st.sidebar.radio(
    "ML Pipeline Steps",
    [
        "1. Data Overview",
        "2. EDA Analysis",
        "3. Preprocessing",
        "4. Feature Engineering",
        "5. Feature Selection",
        "6. Risk Type Modeling",
        "7. Model Evaluation",
        "8. Feature Importance",
        "9. Summary"
    ]
)

# =========================
# 1. DATA OVERVIEW
# =========================
if menu == "1. Data Overview":

    if df is not None:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Data Types")
        st.write(df.dtypes)

# =========================
# 2. EDA ANALYSIS
# =========================
elif menu == "2. EDA Analysis":

    if df is not None:

        st.subheader("📊 Risk Score Distribution")

        if "risk_score" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df["risk_score"], kde=True, ax=ax)
            st.pyplot(fig)

        st.subheader("📈 Risk Score vs MP Count per L")

        if "risk_score" in df.columns and "mp_count_per_l" in df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df["mp_count_per_l"], y=df["risk_score"], ax=ax)
            st.pyplot(fig)

        st.subheader("📊 Risk Level Comparison")

        if "risk_level" in df.columns and "risk_score" in df.columns:
            fig, ax = plt.subplots()
            sns.boxplot(x=df["risk_level"], y=df["risk_score"], ax=ax)
            st.pyplot(fig)

# =========================
# 3. PREPROCESSING
# =========================
elif menu == "3. Preprocessing":

    if df is not None:

        data = df.copy()

        st.subheader("Encoding Categorical Variables")
        cat_cols = data.select_dtypes(include="object").columns

        for col in cat_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

        st.success("Categorical Encoding Done")

        st.subheader("Outlier Handling (Clipping)")
        num_cols = data.select_dtypes(include=np.number).columns

        for col in num_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            data[col] = np.clip(data[col], lower, upper)

        st.success("Outliers Handled")

        st.subheader("Skew Transformation")
        pt = PowerTransformer()
        data[num_cols] = pt.fit_transform(data[num_cols])

        st.success("Skewness Reduced")

        st.session_state.df = data
        st.dataframe(data.head())

# =========================
# 4. FEATURE ENGINEERING
# =========================
elif menu == "4. Feature Engineering":

    if df is not None:

        st.subheader("Feature Scaling")

        scaler = StandardScaler()
        num_cols = df.select_dtypes(include=np.number).columns

        df[num_cols] = scaler.fit_transform(df[num_cols])

        st.session_state.df = df

        st.success("Scaling Completed")
        st.dataframe(df.head())

# =========================
# 5. FEATURE SELECTION
# =========================
elif menu == "5. Feature Selection":

    if df is not None:

        target = st.selectbox("Select Target Column", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        selector = SelectKBest(score_func=f_classif, k=min(5, X.shape[1]))
        X_new = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()]

        st.subheader("Selected Features")
        st.write(list(selected_features))

        st.session_state.selected_features = selected_features

# =========================
# 6. MODEL TRAINING (RISK TYPE)
# =========================
elif menu == "6. Risk Type Modeling":

    if df is not None:

        target = st.selectbox("Target (Risk_Type)", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        # SMOTE
        smote = SMOTE()
        X_res, y_res = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            results[name] = {
                "model": model,
                "acc": accuracy_score(y_test, pred)
            }

        st.session_state.results = results
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.success("Models Trained")

# =========================
# 7. MODEL EVALUATION
# =========================
elif menu == "7. Model Evaluation":

    if "results" in st.session_state:

        results = st.session_state.results

        best = max(results.items(), key=lambda x: x[1]["acc"])

        st.subheader(f"Best Model: {best[0]}")

        y_pred = best[1]["model"].predict(st.session_state.X_test)

        st.text(classification_report(st.session_state.y_test, y_pred))

# =========================
# 8. FEATURE IMPORTANCE
# =========================
elif menu == "8. Feature Importance":

    if "results" in st.session_state:

        best_model = max(st.session_state.results.items(), key=lambda x: x[1]["acc"])[1]["model"]

        if hasattr(best_model, "feature_importances_"):
            importance = best_model.feature_importances_

            feat_df = pd.DataFrame({
                "Feature": df.columns[:-1],
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(feat_df)

            fig, ax = plt.subplots()
            sns.barplot(x="Importance", y="Feature", data=feat_df, ax=ax)
            st.pyplot(fig)

# =========================
# 9. SUMMARY
# =========================
elif menu == "9. Summary":

    st.markdown("""
    # 🧾 FINAL SUMMARY

    ## ✔ Data Processing
    - Encoded categorical variables
    - Handled outliers
    - Transformed skewed features
    - Applied scaling

    ## ✔ EDA
    - Risk score distribution analyzed
    - MP count relationship studied
    - Risk level differences compared

    ## ✔ Feature Engineering
    - Scaling applied
    - Feature selection performed

    ## ✔ Modeling
    - SMOTE applied for imbalance
    - Multiple ML models trained
    - Best model selected

    ## ✔ Interpretation
    - Feature importance extracted
    - Model performance evaluated

    ## 🚀 SYSTEM COMPLETE
    """)
