import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Microplastic Dashboard", layout="wide")
st.title("🌊 Microplastic Risk Dashboard")

# =========================
# SIDEBAR
# =========================
st.sidebar.header("⚙️ Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# =========================
# SAFE CLEANING FUNCTION
# =========================
def clean_data(df):
    data = df.copy()

    for col in data.columns:

        # Convert everything to string first
        data[col] = data[col].astype(str)

        # Extract numbers if present (e.g., "12 mg" → 12)
        extracted = data[col].str.extract('(\d+\.?\d*)')[0]

        # Try convert extracted values to numeric
        numeric_col = pd.to_numeric(extracted, errors='coerce')

        # If most values are numeric → treat as numeric column
        if numeric_col.notna().sum() > len(data) * 0.5:
            data[col] = numeric_col
            data[col].fillna(data[col].median(), inplace=True)

        else:
            # Treat as categorical
            data[col] = data[col].fillna("Unknown")
            data[col] = data[col].astype(str)

    return data

# =========================
# MAIN
# =========================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Dataset")
    st.dataframe(df.head())

    target = st.sidebar.selectbox("🎯 Select Target Column", df.columns)
    top_n = st.sidebar.slider("Top Categories", 5, 20, 10)
    use_smote = st.sidebar.checkbox("Apply SMOTE")

    # =========================
    # KPI CARDS
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Unique Classes", df[target].nunique())

    # =========================
    # TARGET VISUALIZATION (FIXED)
    # =========================
    st.subheader("📊 Target Distribution")

    top_data = df[target].value_counts().nlargest(top_n).reset_index()
    top_data.columns = ["Category", "Count"]

    fig = px.bar(
        top_data,
        x="Count",
        y="Category",
        orientation='h',
        title="Top Categories (Readable)"
    )
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # CLEAN DATA
    # =========================
    st.subheader("⚙️ Data Cleaning")

    data = clean_data(df)
    st.success("Data cleaned successfully")

    # Encode categorical AFTER cleaning
    encoders = {}
    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    # =========================
    # SPLIT
    # =========================
    X = data.drop(columns=[target])
    y = data[target]

    # =========================
    # SMOTE
    # =========================
    if use_smote:
        try:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            st.success("SMOTE applied successfully")
        except:
            st.warning("SMOTE failed (dataset too small or imbalanced)")

    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # =========================
    # SCALING
    # =========================
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

    # KPI RESULTS
    col1, col2 = st.columns(2)
    col1.metric("Best Accuracy", round(results_df["Accuracy"].max(), 3))
    col2.metric("Best F1 Score", round(results_df["F1 Score"].max(), 3))

    # =========================
    # MODEL COMPARISON
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

    feat_df = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    fig3 = px.bar(
        feat_df.head(10),
        x="Importance",
        y="Feature",
        orientation='h'
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

        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
        st.plotly_chart(fig_cm)

    st.success("🚀 Dashboard Ready!")

else:
    st.info("Upload a dataset from the sidebar to begin")
