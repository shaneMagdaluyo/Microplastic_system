import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Polymer Risk ML System", layout="wide")
st.title("🌊 Enterprise Polymer Risk ML System (FIXED)")

# =========================
# SESSION STATE
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

if "processed_df" not in st.session_state:
    st.session_state.processed_df = None

if "X" not in st.session_state:
    st.session_state.X = None

if "y" not in st.session_state:
    st.session_state.y = None


# =========================
# UPLOAD DATA
# =========================
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    st.session_state.df = pd.read_csv(uploaded_file)
    st.sidebar.success("Data Loaded Successfully")

df = st.session_state.df


# =========================
# MENU
# =========================
menu = st.sidebar.radio(
    "Pipeline Steps",
    [
        "1. Data Overview",
        "2. EDA",
        "4. Feature Engineering",
        "5. Feature Selection",
        "6. Risk Type Modeling",
        "7. Model Evaluation",
        "8. Summary"
    ]
)


# =========================
# FIXED CLEAN FUNCTION (IMPORTANT FIX)
# =========================
def clean_data(df):
    df = df.copy()

    for col in df.columns:

        # try numeric conversion first
        numeric = pd.to_numeric(df[col], errors="coerce")

        # if mostly numeric → treat as numeric column
        if numeric.notna().sum() > len(df) * 0.5:
            df[col] = numeric
            df[col] = df[col].fillna(df[col].median())

        # otherwise treat as categorical/text
        else:
            df[col] = df[col].astype(str).fillna("missing")

    return df


# =========================
# LABEL ENCODING
# =========================
def encode(df):
    df = df.copy()

    for col in df.select_dtypes(include="object").columns:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    return df


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
# 2. EDA
# =========================
elif menu == "2. EDA":
    if df is not None:

        st.subheader("Risk Score Distribution")
        if "risk_score" in df.columns:
            fig, ax = plt.subplots()
            sns.histplot(df["risk_score"], kde=True, ax=ax)
            st.pyplot(fig)

        st.subheader("MP Count vs Risk Score")
        if "mp_count_per_l" in df.columns and "risk_score" in df.columns:
            fig, ax = plt.subplots()
            sns.scatterplot(x=df["mp_count_per_l"], y=df["risk_score"], ax=ax)
            st.pyplot(fig)


# =========================
# 4. FEATURE ENGINEERING (FIXED)
# =========================
elif menu == "4. Feature Engineering":

    if df is not None:

        st.subheader("Feature Engineering Pipeline")

        data = clean_data(df)

        # numeric columns only
        num_cols = data.select_dtypes(include=np.number).columns

        # outlier clipping
        for col in num_cols:
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            iqr = q3 - q1

            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr

            data[col] = np.clip(data[col], lower, upper)

        # skew fix
        if len(num_cols) > 0:
            pt = PowerTransformer()
            data[num_cols] = pt.fit_transform(data[num_cols])

        st.session_state.processed_df = data

        st.success("Feature Engineering Completed Successfully")
        st.dataframe(data.head())


# =========================
# 5. FEATURE SELECTION
# =========================
elif menu == "5. Feature Selection":

    if st.session_state.processed_df is None:
        st.warning("Run Feature Engineering first")
    else:

        df2 = encode(st.session_state.processed_df)

        target = st.selectbox("Select Target Column", df2.columns)

        X = df2.drop(columns=[target])
        y = df2[target]

        selector = SelectKBest(f_classif, k=min(8, X.shape[1]))
        X_new = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()]

        st.subheader("Selected Features")
        st.write(list(selected_features))

        st.session_state.X = X[selected_features]
        st.session_state.y = y


# =========================
# 6. MODEL TRAINING (SMOTE FIXED)
# =========================
elif menu == "6. Risk Type Modeling":

    if st.session_state.X is None:
        st.warning("Run Feature Selection first")
    else:

        X = st.session_state.X
        y = st.session_state.y

        st.subheader("Applying SMOTE")

        smote = SMOTE(random_state=42)

        X_res, y_res = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res,
            test_size=0.2,
            random_state=42,
            stratify=y_res
        )

        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000),
            "Random Forest": RandomForestClassifier(n_estimators=200),
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

            st.write(f"{name}: {results[name]['acc']:.4f}")

        st.session_state.results = results
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test

        st.success("Models Trained Successfully")


# =========================
# 7. MODEL EVALUATION
# =========================
elif menu == "7. Model Evaluation":

    if "results" not in st.session_state:
        st.warning("Train models first")
    else:

        results = st.session_state.results

        best_name = max(results, key=lambda x: results[x]["acc"])
        best_model = results[best_name]["model"]

        st.subheader(f"Best Model: {best_name}")

        y_pred = best_model.predict(st.session_state.X_test)

        st.text(classification_report(st.session_state.y_test, y_pred))

        acc_df = pd.DataFrame({
            "Model": list(results.keys()),
            "Accuracy": [results[m]["acc"] for m in results]
        })

        st.subheader("Model Comparison")
        st.dataframe(acc_df)

        fig, ax = plt.subplots()
        sns.barplot(data=acc_df, x="Model", y="Accuracy", ax=ax)
        plt.xticks(rotation=20)
        st.pyplot(fig)


# =========================
# 8. SUMMARY
# =========================
elif menu == "8. Summary":

    st.markdown("""
# 🧾 SYSTEM SUMMARY (FIXED VERSION)

✔ Safe data cleaning (no median string crash)  
✔ Robust numeric detection  
✔ Outlier handling  
✔ Skew transformation  
✔ Feature selection  
✔ SMOTE balancing  
✔ Multi-model training  
✔ Model comparison dashboard  

🚀 SYSTEM NOW FULLY STABLE
""")
