# app.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    roc_curve,
)
from imblearn.over_sampling import SMOTE


st.set_page_config(page_title="Microplastic Risk Analysis", layout="wide")


@st.cache_data
def generate_data(n=1000):
    np.random.seed(42)
    return pd.DataFrame({
        "risk score": np.random.gamma(2, 2, n) * 10,
        "mp count per l": np.random.exponential(100, n),
        "risk level": np.random.choice(["Low", "Medium", "High"], n, p=[0.5, 0.3, 0.2]),
        "Risk_Type": np.random.choice(["Type A", "Type B"], n, p=[0.8, 0.2]),
        "Polymer Type": np.random.choice(["PET", "PE", "PP", "PVC", "PS"], n),
    })


@st.cache_data
def preprocess(df):
    skewed = ["risk score", "mp count per l"]

    pt = PowerTransformer(method="yeo-johnson")
    df[skewed] = pt.fit_transform(df[skewed])

    scaler = RobustScaler()
    df[skewed] = scaler.fit_transform(df[skewed])

    df = pd.get_dummies(df, columns=["risk level", "Polymer Type"], drop_first=True)

    enc = LabelEncoder()
    df["Risk_Type"] = enc.fit_transform(df["Risk_Type"])

    return df, enc


def plot(fig):
    st.pyplot(fig)
    plt.close(fig)


def main():
    st.title("Microplastic Risk Analysis System")

    df = generate_data()
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ================= EDA =================
    st.subheader("EDA")

    fig1, ax1 = plt.subplots()
    sns.countplot(data=df, x="Polymer Type", ax=ax1)
    ax1.set_title("Polymer Distribution")
    plot(fig1)

    fig2, ax2 = plt.subplots()
    sns.histplot(df["risk score"], kde=True, ax=ax2)
    ax2.set_title("Risk Score Distribution")
    plot(fig2)

    fig3, ax3 = plt.subplots()
    sns.scatterplot(data=df, x="mp count per l", y="risk score", ax=ax3)
    ax3.set_title("MP vs Risk Score")
    plot(fig3)

    # ================= PREPROCESS =================
    df, encoder = preprocess(df)

    X = df.drop("Risk_Type", axis=1)
    y = df["Risk_Type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    # ================= MODELS =================
    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(),
        "GB": GradientBoostingClassifier(),
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_prob),
        }

    st.subheader("Model Comparison")
    st.dataframe(pd.DataFrame(results).T)

    # ================= TUNING =================
    st.subheader("Tuned Logistic Regression")

    grid = {"C": [0.01, 0.1, 1, 10]}
    gs = GridSearchCV(LogisticRegression(max_iter=1000), grid, cv=3, scoring="roc_auc")
    gs.fit(X_train, y_train)

    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]

    st.write("Best Params:", gs.best_params_)
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    fig4, ax4 = plt.subplots()
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax4,
                xticklabels=encoder.classes_,
                yticklabels=encoder.classes_)
    ax4.set_title("Confusion Matrix")
    plot(fig4)

    # ROC Curve
    fig5, ax5 = plt.subplots()
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    ax5.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_prob):.2f}")
    ax5.plot([0, 1], [0, 1], linestyle="--")
    ax5.legend()
    ax5.set_title("ROC Curve")
    plot(fig5)


if __name__ == "__main__":
    main()
