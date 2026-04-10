import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Enterprise ML System", layout="wide")

st.title("🚀 Enterprise ML Dashboard System")

# =========================
# SESSION STATE
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

if "model_results" not in st.session_state:
    st.session_state.model_results = {}

# =========================
# SIDEBAR
# =========================
menu = st.sidebar.radio(
    "Navigation",
    [
        "📁 Upload",
        "📊 Overview",
        "⚙️ Preprocessing",
        "⚖️ Imbalance Handling",
        "🤖 Training",
        "📉 Evaluation",
        "🔍 Feature Importance",
        "🏆 Model Leaderboard"
    ]
)

# =========================
# SAFE SMOTE SYSTEM
# =========================
def handle_class_imbalance(X, y, method="smote"):

    X = pd.DataFrame(X).copy()

    # Convert everything to numeric safely
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.fillna(X.median())

    class_counts = y.value_counts()
    min_class = class_counts.min()

    # If too small → fallback
    if min_class < 2:
        st.warning("⚠️ Too few samples → using RandomOverSampler")
        return RandomOverSampler().fit_resample(X, y)

    k_neighbors = min(5, min_class - 1)

    try:
        if method == "smote":
            sampler = SMOTE(k_neighbors=k_neighbors, random_state=42)

        elif method == "tomek":
            sampler = SMOTETomek(random_state=42)

        elif method == "under":
            sampler = RandomUnderSampler(random_state=42)

        else:
            return X, y

        return sampler.fit_resample(X, y)

    except Exception as e:
        st.error(f"SMOTE failed → fallback used: {e}")
        return RandomUnderSampler().fit_resample(X, y)

# =========================
# UPLOAD
# =========================
if menu == "📁 Upload":
    file = st.file_uploader("Upload CSV", type=["csv"])

    if file:
        st.session_state.df = pd.read_csv(file)
        st.success("Dataset Loaded")
        st.dataframe(st.session_state.df.head())

# =========================
# OVERVIEW
# =========================
elif menu == "📊 Overview":

    df = st.session_state.df

    if df is not None:
        st.subheader("Shape")
        st.write(df.shape)

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Summary")
        st.write(df.describe(include="all"))

    else:
        st.warning("Upload dataset first")

# =========================
# PREPROCESSING
# =========================
elif menu == "⚙️ Preprocessing":

    df = st.session_state.df

    if df is not None:
        data = df.copy()

        cat_cols = data.select_dtypes(include="object").columns
        num_cols = data.select_dtypes(include=np.number).columns

        # numeric cleanup
        for col in num_cols:
            data[col] = pd.to_numeric(data[col], errors="coerce")
            data[col].fillna(data[col].median(), inplace=True)

        # categorical encoding
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

        st.session_state.df = data

        st.success("Preprocessing Done")
        st.dataframe(data.head())

    else:
        st.warning("Upload dataset first")

# =========================
# IMBALANCE HANDLING
# =========================
elif menu == "⚖️ Imbalance Handling":

    df = st.session_state.df

    if df is not None:

        target = st.selectbox("Select Target", df.columns)

        method = st.selectbox(
            "Method",
            ["none", "smote", "tomek", "under"]
        )

        if st.button("Apply"):
            X = df.drop(columns=[target])
            y = df[target]

            X_res, y_res = handle_class_imbalance(X, y, method)

            st.session_state.df = pd.concat([pd.DataFrame(X_res), pd.DataFrame(y_res, columns=[target])], axis=1)

            st.success("Imbalance Handling Applied")
            st.write("New distribution:")
            st.write(pd.Series(y_res).value_counts())

    else:
        st.warning("Upload dataset first")

# =========================
# TRAINING
# =========================
elif menu == "🤖 Training":

    df = st.session_state.df

    if df is not None:

        target = st.selectbox("Target Column", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Decision Tree": DecisionTreeClassifier(),
            "Random Forest": RandomForestClassifier(),
            "Gradient Boosting": GradientBoostingClassifier()
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            pred = model.predict(X_test)

            results[name] = {
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred, average="weighted"),
                "model": model
            }

        st.session_state.model_results = results
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.features = X.columns

        st.success("Training Completed")

# =========================
# EVALUATION
# =========================
elif menu == "📉 Evaluation":

    results = st.session_state.model_results

    if results:

        best_model = max(results.items(), key=lambda x: x[1]["accuracy"])

        st.subheader("Best Model")
        st.write(best_model[0])

        model = best_model[1]["model"]

        y_pred = model.predict(st.session_state.X_test)

        st.write("Accuracy:", accuracy_score(st.session_state.y_test, y_pred))
        st.write("F1 Score:", f1_score(st.session_state.y_test, y_pred, average="weighted"))

        st.text(classification_report(st.session_state.y_test, y_pred))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(st.session_state.y_test, y_pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Train models first")

# =========================
# FEATURE IMPORTANCE
# =========================
elif menu == "🔍 Feature Importance":

    results = st.session_state.model_results

    if results:

        best_model = max(results.items(), key=lambda x: x[1]["accuracy"])[1]["model"]

        features = st.session_state.features

        if hasattr(best_model, "feature_importances_"):
            importance = best_model.feature_importances_
        else:
            importance = best_model.coef_[0]

        feat_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(feat_df)

        fig, ax = plt.subplots()
        sns.barplot(data=feat_df, x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)

    else:
        st.warning("Train model first")

# =========================
# LEADERBOARD
# =========================
elif menu == "🏆 Model Leaderboard":

    results = st.session_state.model_results

    if results:

        leaderboard = pd.DataFrame([
            {
                "Model": name,
                "Accuracy": res["accuracy"],
                "F1 Score": res["f1"]
            }
            for name, res in results.items()
        ]).sort_values(by="Accuracy", ascending=False)

        st.dataframe(leaderboard)

        st.bar_chart(leaderboard.set_index("Model"))

    else:
        st.warning("Train models first")
