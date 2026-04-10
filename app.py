import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Polymer Risk ML System", layout="wide")
st.title("🌊 Polymer Risk Classification System (Full ML Workflow)")

# =========================
# SESSION STATE
# =========================
if "df" not in st.session_state:
    st.session_state.df = None

if "models" not in st.session_state:
    st.session_state.models = {}

# =========================
# SIDEBAR WORKFLOW
# =========================
step = st.sidebar.radio(
    "📌 ML Workflow Steps",
    [
        "1️⃣ Load Data",
        "2️⃣ Polymer Type Distribution",
        "3️⃣ Preprocess Data",
        "4️⃣ Class Distribution (Risk_Type)",
        "5️⃣ Handle Imbalance (SMOTE)",
        "6️⃣ Train Models",
        "7️⃣ Hyperparameter Tuning",
        "8️⃣ Evaluate Models",
        "9️⃣ Compare Models",
        "🔟 Feature Importance",
        "📊 Visualization Dashboard",
        "🧾 Summary"
    ]
)

# =========================
# 1. LOAD DATA
# =========================
if step == "1️⃣ Load Data":

    file = st.file_uploader("Upload Dataset (CSV)", type=["csv"])

    if file:
        df = pd.read_csv(file)
        st.session_state.df = df

        st.success("Data Loaded Successfully")
        st.dataframe(df.head())

# =========================
# 2. POLYMER DISTRIBUTION
# =========================
elif step == "2️⃣ Polymer Type Distribution":

    df = st.session_state.df

    if df is not None:

        col = st.selectbox("Select Polymer Column", df.columns)

        fig, ax = plt.subplots()
        sns.countplot(x=df[col], ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    else:
        st.warning("Load data first")

# =========================
# 3. PREPROCESSING
# =========================
elif step == "3️⃣ Preprocess Data":

    df = st.session_state.df

    if df is not None:

        data = df.copy()

        cat_cols = data.select_dtypes(include="object").columns

        for col in cat_cols:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))

        data = data.apply(pd.to_numeric, errors="coerce")
        data = data.fillna(data.median())

        st.session_state.df = data

        st.success("Data Preprocessed")
        st.dataframe(data.head())

    else:
        st.warning("Load data first")

# =========================
# 4. CLASS DISTRIBUTION
# =========================
elif step == "4️⃣ Class Distribution (Risk_Type)":

    df = st.session_state.df

    if df is not None:

        target = st.selectbox("Select Risk_Type Column", df.columns)

        fig, ax = plt.subplots()
        sns.countplot(x=df[target], ax=ax)
        st.pyplot(fig)

        st.write(df[target].value_counts())

    else:
        st.warning("Load data first")

# =========================
# 5. SMOTE BALANCING
# =========================
elif step == "5️⃣ Handle Imbalance (SMOTE)":

    df = st.session_state.df

    if df is not None:

        target = st.selectbox("Target Column", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X, y)

        new_df = pd.concat([
            pd.DataFrame(X_res),
            pd.DataFrame(y_res, columns=[target])
        ], axis=1)

        st.session_state.df = new_df

        st.success("SMOTE Applied")
        st.write(pd.Series(y_res).value_counts())

    else:
        st.warning("Load data first")

# =========================
# 6. TRAIN MODELS
# =========================
elif step == "6️⃣ Train Models":

    df = st.session_state.df

    if df is not None:

        target = st.selectbox("Target Column", df.columns)

        X = df.drop(columns=[target])
        y = df[target]

        X = X.apply(pd.to_numeric, errors="coerce").fillna(0)

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
                "model": model,
                "accuracy": accuracy_score(y_test, pred),
                "f1": f1_score(y_test, pred, average="weighted")
            }

        st.session_state.models = results
        st.session_state.X_test = X_test
        st.session_state.y_test = y_test
        st.session_state.features = X.columns

        st.success("Models Trained")

# =========================
# 7. HYPERPARAMETER TUNING
# =========================
elif step == "7️⃣ Hyperparameter Tuning":

    df = st.session_state.df

    if df is not None:

        target = st.selectbox("Target Column", df.columns)

        X = df.drop(columns=[target]).apply(pd.to_numeric, errors="coerce").fillna(0)
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        param_grid = {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, None]
        }

        model = RandomForestClassifier()

        grid = GridSearchCV(model, param_grid, cv=3)
        grid.fit(X_train, y_train)

        st.success("Best Params Found")
        st.write(grid.best_params_)

        st.session_state.best_model = grid.best_estimator_

# =========================
# 8. EVALUATION
# =========================
elif step == "8️⃣ Evaluate Models":

    models = st.session_state.models

    if models:

        best = max(models.items(), key=lambda x: x[1]["accuracy"])
        model = best[1]["model"]

        y_pred = model.predict(st.session_state.X_test)

        st.subheader(f"Best Model: {best[0]}")

        st.write(classification_report(st.session_state.y_test, y_pred))

        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(st.session_state.y_test, y_pred), annot=True, fmt="d", ax=ax)
        st.pyplot(fig)

# =========================
# 9. COMPARE MODELS
# =========================
elif step == "9️⃣ Compare Models":

    models = st.session_state.models

    if models:

        df = pd.DataFrame([
            {"Model": k, "Accuracy": v["accuracy"], "F1": v["f1"]}
            for k, v in models.items()
        ])

        st.dataframe(df)

        st.bar_chart(df.set_index("Model"))

# =========================
# 10. FEATURE IMPORTANCE
# =========================
elif step == "🔟 Feature Importance":

    models = st.session_state.models

    if models:

        best = max(models.items(), key=lambda x: x[1]["accuracy"])[1]["model"]

        features = st.session_state.features

        importance = best.feature_importances_ if hasattr(best, "feature_importances_") else best.coef_[0]

        feat_df = pd.DataFrame({
            "Feature": features,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.dataframe(feat_df)

# =========================
# 11. VISUALIZATION DASHBOARD
# =========================
elif step == "📊 Visualization Dashboard":

    df = st.session_state.df

    if df is not None:

        st.subheader("Correlation Heatmap")

        fig, ax = plt.subplots()
        sns.heatmap(df.corr(), ax=ax)
        st.pyplot(fig)

# =========================
# 12. SUMMARY
# =========================
elif step == "🧾 Summary":

    st.markdown("""
    ## Summary:

    - Data Loaded and Preprocessed
    - Polymer Type Distribution analyzed
    - Risk_Type class imbalance checked
    - SMOTE applied for balancing
    - Multiple ML models trained
    - Hyperparameter tuning performed
    - Best model evaluated and compared
    - Feature importance analyzed
    - Visual dashboards generated

    ## Workflow Completed Successfully 🚀
    """)
