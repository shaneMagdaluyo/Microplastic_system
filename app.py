import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from collections import Counter

st.set_page_config(page_title="Microplastic Risk System", layout="wide")

st.title("🌊 Microplastic Risk Prediction System (Final Version)")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data & EDA",
    "⚖️ Modeling",
    "🔮 Prediction",
    "📌 Explainability"
])

file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    # -----------------------------
    # CLEANING
    # -----------------------------
    df = df.fillna(df.median(numeric_only=True))

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Outliers
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    target = st.selectbox("Select Target (Risk_Type)", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE SAFE
    class_counts = Counter(y)
    min_samples = min(class_counts.values())

    if min_samples > 5:
        smote = SMOTE(k_neighbors=min(5, min_samples - 1))
        X_res, y_res = smote.fit_resample(X_scaled, y)
    else:
        X_res, y_res = X_scaled, y

# -----------------------------
# TAB 1: EDA
# -----------------------------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Statistics")
        st.write(df.describe())

        if "Risk_Score" in df.columns:
            fig = plt.figure()
            plt.hist(df["Risk_Score"], bins=20)
            plt.title("Risk Score Distribution")
            st.pyplot(fig)

        if "MP_Count_per_L" in df.columns and "Risk_Score" in df.columns:
            fig = plt.figure()
            plt.scatter(df["MP_Count_per_L"], df["Risk_Score"])
            plt.xlabel("MP Count")
            plt.ylabel("Risk Score")
            st.pyplot(fig)

        # Correlation
        st.subheader("Correlation Heatmap")
        corr = df.corr()
        fig = plt.figure()
        plt.imshow(corr)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        st.pyplot(fig)

# -----------------------------
# TAB 2: MODELING
# -----------------------------
    with tab2:
        st.subheader("Model Training")

        st.write("Class Distribution:", class_counts)

        if st.button("Train Models"):

            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42
            )

            models = {
                "Logistic": LogisticRegression(max_iter=1000),
                "RandomForest": RandomForestClassifier(),
                "SVM": SVC(probability=True)
            }

            results = {}
            best_model = None
            best_score = 0

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                results[name] = acc

                if acc > best_score:
                    best_score = acc
                    best_model = model

            joblib.dump(best_model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")

            st.write("Model Accuracy:", results)

            fig = plt.figure()
            plt.bar(results.keys(), results.values())
            st.pyplot(fig)

            st.success("Best model saved!")

# -----------------------------
# TAB 3: PREDICTION
# -----------------------------
    with tab3:
        st.subheader("Make Prediction")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

        if st.button("Predict"):
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            pred = model.predict(input_scaled)
            st.success(f"Prediction: {pred[0]}")

# -----------------------------
# TAB 4: EXPLAINABILITY (SHAP)
# -----------------------------
    with tab4:
        st.subheader("Model Explainability (SHAP)")

        try:
            model = joblib.load("model.pkl")

            explainer = shap.Explainer(model, X_res)
            shap_values = explainer(X_res)

            st.write("Feature Impact Summary")
            fig = plt.figure()
            shap.plots.bar(shap_values, show=False)
            st.pyplot(fig)

        except Exception as e:
            st.warning("Train model first or SHAP not supported")import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from collections import Counter

st.set_page_config(page_title="Microplastic Risk System", layout="wide")

st.title("🌊 Microplastic Risk Prediction System (Final Version)")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Data & EDA",
    "⚖️ Modeling",
    "🔮 Prediction",
    "📌 Explainability"
])

file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    # -----------------------------
    # CLEANING
    # -----------------------------
    df = df.fillna(df.median(numeric_only=True))

    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Outliers
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    target = st.selectbox("Select Target (Risk_Type)", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE SAFE
    class_counts = Counter(y)
    min_samples = min(class_counts.values())

    if min_samples > 5:
        smote = SMOTE(k_neighbors=min(5, min_samples - 1))
        X_res, y_res = smote.fit_resample(X_scaled, y)
    else:
        X_res, y_res = X_scaled, y

# -----------------------------
# TAB 1: EDA
# -----------------------------
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Statistics")
        st.write(df.describe())

        if "Risk_Score" in df.columns:
            fig = plt.figure()
            plt.hist(df["Risk_Score"], bins=20)
            plt.title("Risk Score Distribution")
            st.pyplot(fig)

        if "MP_Count_per_L" in df.columns and "Risk_Score" in df.columns:
            fig = plt.figure()
            plt.scatter(df["MP_Count_per_L"], df["Risk_Score"])
            plt.xlabel("MP Count")
            plt.ylabel("Risk Score")
            st.pyplot(fig)

        # Correlation
        st.subheader("Correlation Heatmap")
        corr = df.corr()
        fig = plt.figure()
        plt.imshow(corr)
        plt.colorbar()
        plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
        plt.yticks(range(len(corr.columns)), corr.columns)
        st.pyplot(fig)

# -----------------------------
# TAB 2: MODELING
# -----------------------------
    with tab2:
        st.subheader("Model Training")

        st.write("Class Distribution:", class_counts)

        if st.button("Train Models"):

            X_train, X_test, y_train, y_test = train_test_split(
                X_res, y_res, test_size=0.2, random_state=42
            )

            models = {
                "Logistic": LogisticRegression(max_iter=1000),
                "RandomForest": RandomForestClassifier(),
                "SVM": SVC(probability=True)
            }

            results = {}
            best_model = None
            best_score = 0

            for name, model in models.items():
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                acc = accuracy_score(y_test, preds)
                results[name] = acc

                if acc > best_score:
                    best_score = acc
                    best_model = model

            joblib.dump(best_model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")

            st.write("Model Accuracy:", results)

            fig = plt.figure()
            plt.bar(results.keys(), results.values())
            st.pyplot(fig)

            st.success("Best model saved!")

# -----------------------------
# TAB 3: PREDICTION
# -----------------------------
    with tab3:
        st.subheader("Make Prediction")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, value=float(df[col].mean()))

        if st.button("Predict"):
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            pred = model.predict(input_scaled)
            st.success(f"Prediction: {pred[0]}")

# -----------------------------
# TAB 4: EXPLAINABILITY (SHAP)
# -----------------------------
    with tab4:
        st.subheader("Model Explainability (SHAP)")

        try:
            model = joblib.load("model.pkl")

            explainer = shap.Explainer(model, X_res)
            shap_values = explainer(X_res)

            st.write("Feature Impact Summary")
            fig = plt.figure()
            shap.plots.bar(shap_values, show=False)
            st.pyplot(fig)

        except Exception as e:
            st.warning("Train model first or SHAP not supported")
