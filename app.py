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
        data[col] = data[col].astype(str)
        extracted = data[col].str.extract(r'(\d+\.?\d*)')[0]
        numeric_col = pd.to_numeric(extracted, errors='coerce')

        if numeric_col.notna().sum() > len(data) * 0.5:
            data[col] = numeric_col
            data[col] = data[col].fillna(data[col].median())
        else:
            data[col] = data[col].fillna("Unknown").astype(str)

    return data

# =========================
# MAIN APP
# =========================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Raw Dataset")
    st.dataframe(df.head())

    target = st.sidebar.selectbox("🎯 Select Target Column", df.columns)
    top_n = st.sidebar.slider("Top Categories", 5, 20, 10)
    use_smote = st.sidebar.checkbox("Apply SMOTE")

    # =========================
    # CLEAN DATA
    # =========================
    data = clean_data(df)

    # FIX: Replace inf and NaN early
    data = data.replace([np.inf, -np.inf], np.nan)

    # =========================
    # ENCODING
    # =========================
    encoders = {}

    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col].astype(str))
        encoders[col] = le

    # =========================
    # TARGET SPLIT
    # =========================
    X = data.drop(columns=[target])
    y = data[target]

    # Force numeric safety
    X = X.apply(pd.to_numeric, errors='coerce')

    # Fill NaNs
    X = X.fillna(X.median())
    y = y.fillna(y.mode()[0])

    # =========================
    # KPI
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Unique Classes", len(np.unique(y)))

    # =========================
    # TARGET VISUALIZATION
    # =========================
    st.subheader("📊 Target Distribution")

    top_data = pd.Series(y).value_counts().nlargest(top_n).reset_index()
    top_data.columns = ["Category", "Count"]

    fig = px.bar(
        top_data,
        x="Count",
        y="Category",
        orientation='h'
    )
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # FINAL SAFETY CLEAN
    X_train = np.nan_to_num(X_train)
    X_test = np.nan_to_num(X_test)

    # =========================
    # SMOTE
    # =========================
    if use_smote and len(np.unique(y_train)) > 1:
        try:
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            st.success("SMOTE applied successfully")
        except Exception as e:
            st.warning(f"SMOTE failed: {e}")

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # MODELS
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

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred, average='weighted')
        })

    results_df = pd.DataFrame(results)

    col1, col2 = st.columns(2)
    col1.metric("Best Accuracy", round(results_df["Accuracy"].max(), 3))
    col2.metric("Best F1 Score", round(results_df["F1 Score"].max(), 3))

    fig2 = px.bar(results_df, x="Model", y=["Accuracy", "F1 Score"], barmode="group")
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

    fig3 = px.bar(feat_df.head(10), x="Importance", y="Feature", orientation='h')
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

        st.success(f"Best Params: {grid.best_params_}")
        st.write("Accuracy:", accuracy_score(y_test, y_pred))
        st.write("F1 Score:", f1_score(y_test, y_pred, average='weighted'))
        st.text(classification_report(y_test, y_pred))

        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True)
        st.plotly_chart(fig_cm)

    st.success("🚀 Dashboard Ready!")

else:
    st.info("Upload a dataset from the sidebar to begin")
