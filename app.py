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
# STREAMLIT CONFIG
# =========================
st.set_page_config(page_title="Microplastic Risk System", layout="wide")
st.title("🌊 Microplastic Risk Analysis System")

# =========================
# SAFE SMOTE FUNCTION
# =========================
def handle_class_imbalance(X_train, y_train, method="smote"):
    """
    Safe SMOTE implementation (prevents crashes)
    """

    if method != "smote":
        return X_train, y_train

    from imblearn.over_sampling import SMOTE

    class_counts = pd.Series(y_train).value_counts()
    min_class_size = class_counts.min()

    st.write("### 📊 Class Distribution (Before SMOTE)")
    st.write(class_counts)

    # If dataset too small → skip SMOTE
    if min_class_size < 2:
        st.warning("SMOTE skipped: not enough samples in smallest class")
        return X_train, y_train

    # Safe k_neighbors
    k = min(5, min_class_size - 1)

    try:
        smote = SMOTE(random_state=42, k_neighbors=k)
        X_res, y_res = smote.fit_resample(X_train, y_train)

        st.success("SMOTE Applied Successfully")
        return X_res, y_res

    except Exception as e:
        st.warning(f"SMOTE failed safely: {e}")
        return X_train, y_train


# =========================
# UPLOAD DATA
# =========================
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.success("Dataset Loaded Successfully")
    st.dataframe(df.head())

    # =========================
    # TARGET SELECTION
    # =========================
    target = st.selectbox("🎯 Select Target Column", df.columns)

    if target:

        # =========================
        # EDA
        # =========================
        st.subheader("📊 Exploratory Data Analysis")

        if df[target].dtype != "object":
            fig, ax = plt.subplots()
            sns.histplot(df[target], kde=True, ax=ax)
            st.pyplot(fig)

        num_df = df.select_dtypes(include=np.number)
        if num_df.shape[1] > 1:
            st.write("### Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # =========================
        # PREPROCESSING
        # =========================
        st.subheader("⚙️ Preprocessing")

        data = df.copy()

        num_cols = data.select_dtypes(include=np.number).columns
        cat_cols = data.select_dtypes(include="object").columns

        # numeric cleaning
        for col in num_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
            data[col].fillna(data[col].median(), inplace=True)

        # categorical cleaning
        for col in cat_cols:
            data[col].fillna(data[col].mode()[0], inplace=True)

        # encoding
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

        st.success("Preprocessing Completed")

        # =========================
        # SPLIT DATA
        # =========================
        X = data.drop(columns=[target])
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # =========================
        # CLASS IMBALANCE
        # =========================
        st.subheader("⚖️ Class Imbalance Handling")

        use_smote = st.checkbox("Apply SMOTE")

        if use_smote:
            X_train, y_train = handle_class_imbalance(X_train, y_train, method="smote")

        # =========================
        # SCALING
        # =========================
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # =========================
        # MODEL SELECTION
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
        # EVALUATION
        # =========================
        st.subheader("📈 Model Evaluation")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        st.write("Accuracy:", acc)
        st.write("F1 Score:", f1)

        st.text("Classification Report")
        st.text(classification_report(y_test, y_pred))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

        # =========================
        # FEATURE IMPORTANCE
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

        st.success("Model Completed Successfully 🚀")

else:
    st.warning("📂 Please upload a dataset to begin")
