import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif

from imblearn.over_sampling import SMOTE
from collections import Counter

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Microplastic Risk System", layout="wide")
st.title("🌊 Microplastic Risk Prediction System (Final Polished)")

# -----------------------------
# UPLOAD DATA
# -----------------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    # -----------------------------
    # MISSING VALUES
    # -----------------------------
    df = df.fillna(df.median(numeric_only=True))

    # -----------------------------
    # ENCODE CATEGORICAL
    # -----------------------------
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # -----------------------------
    # SKEW TRANSFORMATION
    # -----------------------------
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if abs(df[col].skew()) > 1:
            df[col] = np.log1p(df[col])

    # -----------------------------
    # OUTLIER REMOVAL (IQR)
    # -----------------------------
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    st.success(f"Cleaned dataset shape: {df.shape}")

    # -----------------------------
    # TARGET
    # -----------------------------
    target = st.selectbox("Select Target Column (Risk_Type)", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # -----------------------------
    # TRAIN TEST SPLIT (avoid leakage)
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -----------------------------
    # FEATURE SCALING
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # FEATURE SELECTION
    # -----------------------------
    selector = SelectKBest(score_func=f_classif, k=min(10, X.shape[1]))
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_test_sel = selector.transform(X_test_scaled)

    # -----------------------------
    # SMOTE (ONLY TRAIN DATA)
    # -----------------------------
    class_counts = Counter(y_train)
    st.write("Class distribution:", class_counts)

    min_samples = min(class_counts.values())

    if min_samples > 5:
        smote = SMOTE(k_neighbors=min(5, min_samples - 1))
        X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)
        st.success("SMOTE applied")
    else:
        X_train_res, y_train_res = X_train_sel, y_train
        st.warning("SMOTE skipped")

    # -----------------------------
    # TABS
    # -----------------------------
    tab1, tab2, tab3, tab4 = st.tabs([
        "EDA",
        "Modeling",
        "Prediction",
        "Explainability"
    ])

    # =============================
    # TAB 1 - EDA
    # =============================
    with tab1:
        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        st.subheader("Risk Score Distribution (if exists)")
        if "Risk_Score" in df.columns:
            fig = plt.figure()
            plt.hist(df["Risk_Score"], bins=20)
            st.pyplot(fig)

        st.subheader("MP Count vs Risk Score")
        if "MP_Count_per_L" in df.columns and "Risk_Score" in df.columns:
            fig = plt.figure()
            plt.scatter(df["MP_Count_per_L"], df["Risk_Score"])
            plt.xlabel("MP Count per L")
            plt.ylabel("Risk Score")
            st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        fig = plt.figure()
        plt.imshow(df.corr(), cmap="coolwarm")
        plt.colorbar()
        plt.title("Correlation Matrix")
        st.pyplot(fig)

    # =============================
    # TAB 2 - MODELING
    # =============================
    with tab2:
        st.subheader("Train Models")

        if st.button("Train Models"):
            models = {
                "Logistic Regression": LogisticRegression(max_iter=1000),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(probability=True)
            }

            results = {}
            best_model = None
            best_score = 0

            for name, model in models.items():
                model.fit(X_train_res, y_train_res)
                preds = model.predict(X_test_sel)
                acc = accuracy_score(y_test, preds)
                results[name] = acc

                if acc > best_score:
                    best_score = acc
                    best_model = model

            joblib.dump(best_model, "model.pkl")
            joblib.dump(scaler, "scaler.pkl")
            joblib.dump(selector, "selector.pkl")

            st.write(results)

            fig = plt.figure()
            plt.bar(results.keys(), results.values())
            st.pyplot(fig)

            st.success("Best model saved!")

    # =============================
    # TAB 3 - PREDICTION
    # =============================
    with tab3:
        st.subheader("Make Prediction")

        input_data = {}
        for col in X.columns:
            input_data[col] = st.number_input(col, value=float(X[col].mean()))

        if st.button("Predict"):
            try:
                model = joblib.load("model.pkl")
                scaler = joblib.load("scaler.pkl")
                selector = joblib.load("selector.pkl")

                input_df = pd.DataFrame([input_data])

                input_scaled = scaler.transform(input_df)
                input_selected = selector.transform(input_scaled)

                pred = model.predict(input_selected)
                st.success(f"Prediction: {pred[0]}")

            except Exception as e:
                st.error(str(e))

    # =============================
    # TAB 4 - SHAP
    # =============================
    with tab4:
        st.subheader("Explainability")

        try:
            model = joblib.load("model.pkl")

            if hasattr(model, "feature_importances_"):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X_test_sel)

                fig = plt.figure()
                shap.summary_plot(shap_values, X_test_sel, show=False)
                st.pyplot(fig)

            else:
                st.warning("SHAP not supported for this model type")

        except Exception as e:
            st.warning(str(e))
