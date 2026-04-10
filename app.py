# app.py

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

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


def main():
    st.title("Microplastic Risk Analysis System")

    df = generate_data()

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # ================= EDA =================
    st.subheader("EDA")

    fig1 = px.histogram(df, x="Polymer Type", title="Polymer Distribution")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.histogram(df, x="risk score", nbins=30, title="Risk Score Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.scatter(df, x="mp count per l", y="risk score", title="MP vs Risk Score")
    st.plotly_chart(fig3, use_container_width=True)

    fig4 = px.box(df, x="risk level", y="risk score", title="Risk Score by Level")
    st.plotly_chart(fig4, use_container_width=True)

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

    # Confusion Matrix (Plotly)
    cm = confusion_matrix(y_test, y_pred)
    fig_cm = px.imshow(
        cm,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual"),
        x=encoder.classes_,
        y=encoder.classes_,
        title="Confusion Matrix",
    )
    st.plotly_chart(fig_cm, use_container_width=True)

    # ROC Curve (Plotly)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig_roc = px.line(
        x=fpr,
        y=tpr,
        title=f"ROC Curve (AUC={roc_auc_score(y_test, y_prob):.2f})",
        labels={"x": "False Positive Rate", "y": "True Positive Rate"},
    )
    fig_roc.add_shape(type="line", line=dict(dash="dash"), x0=0, x1=1, y0=0, y1=1)
    st.plotly_chart(fig_roc, use_container_width=True)


if __name__ == "__main__":
    main()
