import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Microplastic Dashboard", layout="wide")

st.title("🌊 Microplastic Risk Dashboard")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)

    # =========================
    # SIDEBAR OPTIONS
    # =========================
    target = st.sidebar.selectbox("Select Target", df.columns)
    top_n = st.sidebar.slider("Top Categories", 5, 20, 10)
    use_smote = st.sidebar.checkbox("Apply SMOTE")

    # =========================
    # MAIN DASHBOARD
    # =========================
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Unique Target", df[target].nunique())

    st.dataframe(df.head())

    # =========================
    # TARGET VISUALIZATION
    # =========================
    st.subheader("📊 Target Distribution")

    top_data = df[target].value_counts().nlargest(top_n).reset_index()
    top_data.columns = ["Category", "Count"]

    fig = px.bar(
        top_data,
        x="Count",
        y="Category",
        orientation="h",
        title="Top Categories"
    )
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # PREPROCESSING
    # =========================
    data = df.copy()

    for col in data.columns:
        if data[col].dtype == "object":
            data[col].fillna(data[col].mode()[0], inplace=True)
        else:
            data[col].fillna(data[col].median(), inplace=True)

    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))

    # =========================
    # SPLIT
    # =========================
    X = data.drop(columns=[target])
    y = data[target]

    if use_smote:
        smote = SMOTE(random_state=42)
        X, y = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # MODEL TRAINING
    # =========================
    st.subheader("🤖 Model Performance")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier()
    }

    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1 Score": f1
        })

    results_df = pd.DataFrame(results)

    # KPI DISPLAY
    col1, col2 = st.columns(2)
    col1.metric("Best Accuracy", round(results_df["Accuracy"].max(), 3))
    col2.metric("Best F1 Score", round(results_df["F1 Score"].max(), 3))

    # =========================
    # MODEL COMPARISON CHART
    # =========================
    st.subheader("📊 Model Comparison")

    fig2 = px.bar(
        results_df,
        x="Model",
        y=["Accuracy", "F1 Score"],
        barmode="group"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    st.subheader("🔍 Feature Importance")

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    importance = rf.feature_importances_

    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False)

    fig3 = px.bar(
        feat_df.head(10),
        x="Importance",
        y="Feature",
        orientation="h"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # =========================
    # HYPERPARAMETER TUNING
    # =========================
    st.subheader("⚙️ Hyperparameter Tuning")

    if st.button("Run Logistic Regression Tuning"):

        param_grid = {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }

        grid = GridSearchCV(LogisticRegression(max_iter=1000), param_grid, cv=3)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        st.success(f"Best Params: {grid.best_params_}")
        st.write("Accuracy:", acc)
        st.write("F1 Score:", f1)

    st.success("🚀 Dashboard Ready!")

else:
    st.info("Upload a dataset from the sidebar to begin")
