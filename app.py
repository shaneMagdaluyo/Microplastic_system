import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Microplastic Dashboard", layout="wide")

st.title("🌊 Microplastic Risk Intelligence Dashboard")

# =========================
# SIDEBAR NAVIGATION
# =========================
menu = st.sidebar.radio(
    "📊 Navigation",
    [
        "📁 Upload Data",
        "📊 Dataset Overview",
        "📈 EDA",
        "⚙️ Preprocessing",
        "🤖 Model Training",
        "📉 Evaluation",
        "🔍 Feature Importance"
    ]
)

# =========================
# SESSION STATE
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

if "data" not in st.session_state:
    st.session_state.data = None

# =========================
# UPLOAD DATA
# =========================
if menu == "📁 Upload Data":
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])

    if uploaded_file is not None:
        st.session_state.df = pd.read_csv(uploaded_file)
        st.success("Dataset Loaded Successfully!")

        st.dataframe(st.session_state.df.head())

# =========================
# DATASET OVERVIEW
# =========================
elif menu == "📊 Dataset Overview":

    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("Dataset Shape")
        st.write(df.shape)

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Summary Statistics")
        st.write(df.describe())

    else:
        st.warning("Upload dataset first")

# =========================
# EDA
# =========================
elif menu == "📈 EDA":

    if st.session_state.df is not None:
        df = st.session_state.df

        st.subheader("Correlation Heatmap")

        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        st.subheader("Distribution Plot")

        col = st.selectbox("Select Column", df.columns)

        fig, ax = plt.subplots()

        if df[col].dtype != "object":
            sns.histplot(df[col], kde=True, ax=ax)
        else:
            sns.countplot(x=df[col], ax=ax)

        st.pyplot(fig)

    else:
        st.warning("Upload dataset first")

# =========================
# PREPROCESSING
# =========================
elif menu == "⚙️ Preprocessing":

    if st.session_state.df is not None:
        df = st.session_state.df.copy()

        st.subheader("Running Preprocessing Pipeline...")

        num_cols = df.select_dtypes(include=np.number).columns
        cat_cols = df.select_dtypes(include="object").columns

        # numeric
        for col in num_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col].fillna(df[col].median(), inplace=True)

        # categorical
        for col in cat_cols:
            df[col].fillna(df[col].mode()[0], inplace=True)

        # encoding
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

        st.session_state.data = df

        st.success("Preprocessing Completed")
        st.dataframe(df.head())

    else:
        st.warning("Upload dataset first")

# =========================
# MODEL TRAINING
# =========================
elif menu == "🤖 Model Training":

    if st.session_state.data is not None:
        df = st.session_state.data

        target = st.selectbox("Select Target", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model_name = st.selectbox(
            "Choose Model",
            ["Logistic Regression", "Random Forest", "Decision Tree"]
        )

        if model_name == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
        elif model_name == "Random Forest":
            model = RandomForestClassifier()
        else:
            model = DecisionTreeClassifier()

        model.fit(X_train, y_train)

        st.session_state.model = model
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.X_columns = X.columns

        st.success("Model Trained Successfully 🚀")

    else:
        st.warning("Run preprocessing first")

# =========================
# EVALUATION
# =========================
elif menu == "📉 Evaluation":

    if "model" in st.session_state:

        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test

        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.write("Accuracy:", acc)
        st.write("F1 Score:", f1)

        st.text(classification_report(y_test, y_pred))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Train model first")

# =========================
# FEATURE IMPORTANCE
# =========================
elif menu == "🔍 Feature Importance":

    if "model" in st.session_state:

        model = st.session_state.model
        cols = st.session_state.X_columns

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = model.coef_[0]
        else:
            importance = None

        if importance is not None:
            feat_df = pd.DataFrame({
                "Feature": cols,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.subheader("Feature Importance Ranking")
            st.dataframe(feat_df)

            fig, ax = plt.subplots()
            sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
            st.pyplot(fig)

    else:
        st.warning("Train model first")
