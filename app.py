import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Microplastic Risk System", layout="wide")

st.title("🌊 Microplastic Risk Analysis & Prediction System")

# =========================
# 1. LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("File loaded successfully!")

    st.dataframe(df.head())

    # NOW SAFE TO COPY
    data = df.copy()

else:
    st.warning("Please upload a CSV file to continue.")
    st.stop()

    # =========================
    # 2. TARGET SELECTION
    # =========================
    target = st.selectbox("Select Target Column", df.columns)

    if target:

        # =========================
        # 3. BASIC EDA
        # =========================
        st.subheader("📈 EDA")

        if df[target].dtype != "object":
            fig, ax = plt.subplots()
            sns.histplot(df[target], kde=True, ax=ax)
            st.pyplot(fig)

        # Correlation heatmap
        st.write("### Correlation Heatmap")
        numeric_df = df.select_dtypes(include=np.number)
        if numeric_df.shape[1] > 1:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # =========================
        # 4. PREPROCESSING
        # =========================
        # =========================
# CLEAN PREPROCESSING (FIXED)
# =========================

# =========================
# CLEAN & SAFE PREPROCESSING (FINAL FIX)
# =========================

data = df.copy()

# STEP 1: Convert everything safely to numeric where possible
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# STEP 2: Handle missing values
for col in data.columns:

    if pd.api.types.is_numeric_dtype(data[col]):
        data[col].fillna(data[col].median(), inplace=True)
    else:
        data[col].fillna(data[col].mode()[0], inplace=True)

# STEP 3: Encode categorical columns (if any remain)
from sklearn.preprocessing import LabelEncoder

encoders = {}

for col in data.columns:
    if data[col].dtype == "object":
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

        # =========================
        # 5. SPLIT DATA
        # =========================
        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # =========================
        # 6. SMOTE
        # =========================
        st.write("Applying SMOTE (if classification)...")

        try:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
        except:
            st.warning("SMOTE not applied (check target type)")

        # =========================
        # 7. SCALING
        # =========================
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # =========================
        # 8. MODEL SELECTION
        # =========================
        st.subheader("🤖 Model Training")

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
        y_pred = model.predict(X_test)

        # =========================
        # 9. EVALUATION
        # =========================
        st.subheader("📊 Model Evaluation")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.write("Accuracy:", acc)
        st.write("F1 Score:", f1)

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        # Confusion Matrix
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        # =========================
        # 10. FEATURE IMPORTANCE
        # =========================
        st.subheader("🔍 Feature Importance")

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        elif hasattr(model, "coef_"):
            importance = model.coef_[0]
        else:
            importance = None

        if importance is not None:
            feat_df = pd.DataFrame({
                "Feature": X.columns,
                "Importance": importance
            }).sort_values(by="Importance", ascending=False)

            st.dataframe(feat_df)

            fig, ax = plt.subplots()
            sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
            st.pyplot(fig)

        st.success("Model pipeline completed successfully 🚀")

else:
    st.info("Upload a CSV file to begin analysis")
