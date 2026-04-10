import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
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
st.set_page_config(page_title="Microplastic Risk System", layout="wide")
st.title("🌊 Microplastic Risk Analysis System")

# =========================
# UPLOAD
# =========================
uploaded_file = st.file_uploader("📂 Upload CSV Dataset", type=["csv"])

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    st.success("Dataset Loaded")
    st.dataframe(df.head())

    # =========================
    # TARGET
    # =========================
    target = st.selectbox("🎯 Select Target Column", df.columns)

    if target:

        # =========================
        # PREPROCESSING
        # =========================
        st.subheader("⚙️ Data Preprocessing")

        data = df.copy()

        for col in data.columns:
            if data[col].dtype == "object":
                data[col].fillna(data[col].mode()[0], inplace=True)
            else:
                data[col].fillna(data[col].median(), inplace=True)

        encoders = {}
        for col in data.select_dtypes(include="object").columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            encoders[col] = le

        st.success("Preprocessing Done")

        # =========================
        # EDA (FIXED READABLE)
        # =========================
        st.subheader("📊 Data Visualization")

        top_n = st.slider("Select number of categories", 5, 20, 10)

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
        # CLASS BALANCE
        # =========================
        st.subheader("⚖️ Class Distribution")

        st.write(df[target].value_counts())

        fig2 = px.histogram(df, x=target, title="Class Distribution")
        st.plotly_chart(fig2, use_container_width=True)

        # =========================
        # SPLIT
        # =========================
        X = data.drop(columns=[target])
        y = data[target]

        # =========================
        # SMOTE
        # =========================
        use_smote = st.checkbox("Apply SMOTE (Fix Imbalance)")

        if use_smote:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)
            st.success("SMOTE Applied")

        # =========================
        # SPLIT DATA
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
        # MODELS
        # =========================
        st.subheader("🤖 Model Training")

        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier()
        }

        results = {}

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')

            results[name] = (acc, f1)

        st.write("### Model Results")
        st.write(results)

        # =========================
        # HYPERPARAMETER TUNING
        # =========================
        st.subheader("⚙️ Hyperparameter Tuning (Logistic Regression)")

        if st.button("Run Tuning"):

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

            results["Tuned Logistic"] = (acc, f1)

            st.success(f"Best Params: {grid.best_params_}")
            st.write("Accuracy:", acc)
            st.write("F1 Score:", f1)

            st.text(classification_report(y_test, y_pred))

            # Confusion Matrix
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, title="Confusion Matrix")
            st.plotly_chart(fig_cm)

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
            orientation='h',
            title="Top Features"
        )
        st.plotly_chart(fig3, use_container_width=True)

        # =========================
        # MODEL COMPARISON
        # =========================
        st.subheader("📊 Model Comparison")

        model_names = list(results.keys())
        acc_vals = [v[0] for v in results.values()]
        f1_vals = [v[1] for v in results.values()]

        comp_df = pd.DataFrame({
            "Model": model_names,
            "Accuracy": acc_vals,
            "F1 Score": f1_vals
        })

        fig4 = px.bar(comp_df, x="Model", y=["Accuracy", "F1 Score"], barmode="group")
        st.plotly_chart(fig4, use_container_width=True)

        st.success("🚀 Analysis Complete!")

else:
    st.warning("Please upload a dataset")
