import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

# -----------------------------
# APP TITLE
# -----------------------------
st.title("🌊 Advanced Microplastic Risk Prediction System")

# -----------------------------
# FILE UPLOAD
# -----------------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file:
    df = pd.read_csv(file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    # -----------------------------
    # CLEANING
    # -----------------------------
    df = df.fillna(df.median(numeric_only=True))

    # Encode categorical
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # -----------------------------
    # OUTLIER REMOVAL
    # -----------------------------
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    st.write("✅ Cleaned Data Shape:", df.shape)

    # -----------------------------
    # SELECT TARGET
    # -----------------------------
    target = st.selectbox("Select Target (Risk_Type)", df.columns)

    X = df.drop(columns=[target])
    y = df[target]

    # -----------------------------
    # SCALING
    # -----------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # -----------------------------
    # HANDLE IMBALANCE (SMOTE)
    # -----------------------------
    smote = SMOTE()
    X_res, y_res = smote.fit_resample(X_scaled, y)

    # -----------------------------
    # EDA SECTION
    # -----------------------------
    st.header("📊 Exploratory Data Analysis")

    # Histogram
    if "Risk_Score" in df.columns:
        fig = plt.figure()
        plt.hist(df["Risk_Score"], bins=20)
        plt.title("Risk Score Distribution")
        st.pyplot(fig)

    # Scatter
    if "Risk_Score" in df.columns and "MP_Count_per_L" in df.columns:
        fig = plt.figure()
        plt.scatter(df["MP_Count_per_L"], df["Risk_Score"])
        plt.xlabel("MP Count per L")
        plt.ylabel("Risk Score")
        plt.title("Risk Score vs MP Count")
        st.pyplot(fig)

    # Correlation Heatmap
    st.subheader("🔥 Correlation Heatmap")
    corr = df.corr()

    fig = plt.figure()
    plt.imshow(corr)
    plt.colorbar()
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    st.pyplot(fig)

    # -----------------------------
    # TRAIN MODELS
    # -----------------------------
    if st.button("🚀 Train & Tune Models"):

        X_train, X_test, y_train, y_test = train_test_split(
            X_res, y_res, test_size=0.2, random_state=42
        )

        models = {
            "Logistic Regression": (
                LogisticRegression(max_iter=1000),
                {"C": [0.1, 1, 10]}
            ),
            "Random Forest": (
                RandomForestClassifier(),
                {"n_estimators": [50, 100]}
            ),
            "SVM": (
                SVC(probability=True),
                {"C": [0.1, 1], "kernel": ["linear", "rbf"]}
            )
        }

        results = {}
        best_model = None
        best_score = 0

        for name, (model, params) in models.items():
            grid = GridSearchCV(model, params, cv=3)
            grid.fit(X_train, y_train)

            preds = grid.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc

            if acc > best_score:
                best_score = acc
                best_model = grid.best_estimator_

        # Save model
        joblib.dump(best_model, "model.pkl")
        joblib.dump(scaler, "scaler.pkl")

        # Results
        st.subheader("📊 Model Comparison")
        st.write(results)

        # Bar chart
        fig = plt.figure()
        plt.bar(results.keys(), results.values())
        plt.xticks(rotation=30)
        plt.title("Model Accuracy Comparison")
        st.pyplot(fig)

        st.success("✅ Best model trained and saved!")

        # -----------------------------
        # ROC CURVE
        # -----------------------------
        try:
            probs = best_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, probs)
            roc_auc = auc(fpr, tpr)

            fig = plt.figure()
            plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
            plt.plot([0, 1], [0, 1], linestyle='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC Curve")
            plt.legend()
            st.pyplot(fig)
        except:
            st.info("ROC not available for this model")

    # -----------------------------
    # PREDICTION
    # -----------------------------
    st.header("🔮 Prediction")

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(f"{col}", value=float(df[col].mean()))

    if st.button("Predict"):
        try:
            model = joblib.load("model.pkl")
            scaler = joblib.load("scaler.pkl")

            input_df = pd.DataFrame([input_data])
            input_scaled = scaler.transform(input_df)

            pred = model.predict(input_scaled)
            st.success(f"Prediction: {pred[0]}")
        except Exception as e:
            st.error(e)

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    st.header("📌 Feature Importance")

    try:
        model = joblib.load("model.pkl")

        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_

            fig = plt.figure()
            plt.barh(X.columns, importance)
            plt.title("Feature Importance")
            st.pyplot(fig)
        else:
            st.info("Feature importance not available for this model")

    except:
        st.info("Train model first")
