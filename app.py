import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.multiclass import type_of_target

from imblearn.over_sampling import SMOTE
from collections import Counter

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="Microplastic Risk System", layout="wide")
st.title("🌊 Microplastic Risk Prediction System (Stable Version)")

# -----------------------------
# UPLOAD FILE (FIXED)
# -----------------------------
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is None:
    st.info("Upload dataset to continue")
    st.stop()

df = pd.read_csv(file)

# -----------------------------
# CLEAN DATA
# -----------------------------
df = df.fillna(df.median(numeric_only=True))

# Encode categorical
for col in df.select_dtypes(include=['object']).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# -----------------------------
# OUTLIER REMOVAL
# -----------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

st.success(f"Dataset shape: {df.shape}")

# -----------------------------
# TARGET
# -----------------------------
target = st.selectbox("Select Target Column", df.columns)

X = df.drop(columns=[target])
y = df[target]

# -----------------------------
# VALIDATION
# -----------------------------
if y.nunique() < 2:
    st.error("Target must have at least 2 classes")
    st.stop()

class_counts = Counter(y)
st.write("Class distribution:", class_counts)

# -----------------------------
# SAFE TRAIN SPLIT (NO STRATIFY CRASH)
# -----------------------------
use_stratify = min(class_counts.values()) > 1

if use_stratify:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
else:
    st.warning("Stratify disabled due to small class size")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

# -----------------------------
# SCALING
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# FEATURE SELECTION
# -----------------------------
selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# -----------------------------
# SAFE SMOTE
# -----------------------------
min_class = min(Counter(y_train).values())

if type_of_target(y_train) in ["binary", "multiclass"] and min_class >= 6:
    smote = SMOTE(k_neighbors=min(5, min_class - 1), random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    st.success("SMOTE applied")
else:
    st.warning("SMOTE skipped (not enough samples)")

# -----------------------------
# MODELS
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}
best_model = None
best_score = 0

# -----------------------------
# TRAIN MODELS
# -----------------------------
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    results[name] = acc

    if acc > best_score:
        best_score = acc
        best_model = model

# -----------------------------
# SAVE PIPELINE
# -----------------------------
joblib.dump({
    "model": best_model,
    "scaler": scaler,
    "selector": selector
}, "pipeline.pkl")

# -----------------------------
# TABS
# -----------------------------
tab1, tab2 = st.tabs(["EDA", "Prediction"])

# -----------------------------
# EDA
# -----------------------------
with tab1:
    st.subheader("EDA")

    if "Risk_Score" in df.columns:
        fig = plt.figure()
        plt.hist(df["Risk_Score"], bins=20)
        st.pyplot(fig)

    if "MP_Count_per_L" in df.columns and "Risk_Score" in df.columns:
        fig = plt.figure()
        plt.scatter(df["MP_Count_per_L"], df["Risk_Score"])
        st.pyplot(fig)

    st.subheader("Model Results")
    st.write(results)

    fig = plt.figure()
    plt.bar(results.keys(), results.values())
    st.pyplot(fig)

# -----------------------------
# PREDICTION
# -----------------------------
with tab2:
    st.subheader("Make Prediction")

    input_data = {}
    for col in X.columns:
        input_data[col] = st.number_input(col, value=float(X[col].mean()))

    if st.button("Predict"):
        pipe = joblib.load("pipeline.pkl")

        model = pipe["model"]
        scaler = pipe["scaler"]
        selector = pipe["selector"]

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        input_selected = selector.transform(input_scaled)

        pred = model.predict(input_selected)

        st.success(f"Prediction: {pred[0]}")
