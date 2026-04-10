import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from collections import Counter

# =============================
# APP CONFIG
# =============================
st.set_page_config(page_title="Microplastic ML System", layout="wide")
st.title("🌊 Microplastic Risk Prediction System")

# =============================
# UPLOAD DATA
# =============================
file = st.file_uploader("Upload CSV Dataset", type=["csv"])

if file is None:
    st.info("Please upload a dataset to continue")
    st.stop()

df = pd.read_csv(file)

# =============================
# CLEAN DATA
# =============================
df = df.fillna(df.median(numeric_only=True))

# Encode categorical variables
for col in df.select_dtypes(include=["object"]).columns:
    df[col] = LabelEncoder().fit_transform(df[col])

# Outlier removal
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

st.success(f"Cleaned dataset shape: {df.shape}")

# =============================
# TARGET SELECTION
# =============================
target = st.selectbox("Select Target Column", df.columns)

X = df.drop(columns=[target])
y = df[target]

# =============================
# EDA SECTION
# =============================
st.subheader("📊 Exploratory Data Analysis")

# Risk Score Distribution
if "Risk_Score" in df.columns:
    fig = plt.figure()
    plt.hist(df["Risk_Score"], bins=20)
    plt.title("Risk Score Distribution")
    st.pyplot(fig)

# MP vs Risk Score
if "MP_Count_per_L" in df.columns and "Risk_Score" in df.columns:
    fig = plt.figure()
    plt.scatter(df["MP_Count_per_L"], df["Risk_Score"])
    plt.xlabel("MP Count per L")
    plt.ylabel("Risk Score")
    plt.title("MP vs Risk Score")
    st.pyplot(fig)

# Risk Level comparison
if "Risk_Level" in df.columns:
    fig = plt.figure()
    df.boxplot(column="Risk_Score", by="Risk_Level")
    plt.title("Risk Score by Risk Level")
    st.pyplot(fig)

# =============================
# POLYMER TYPE (FIXED NUMBERING)
# =============================
if "Polymer_Type" in df.columns:
    st.subheader("📦 Polymer Type Distribution (Numbered)")

    poly_counts = df["Polymer_Type"].value_counts().sort_values(ascending=False)

    numbered_labels = [str(i + 1) for i in range(len(poly_counts))]

    fig = plt.figure()
    plt.bar(numbered_labels, poly_counts.values)

    plt.xlabel("Polymer Type (Numbered)")
    plt.ylabel("Count")
    plt.title("Polymer Type Distribution")

    st.pyplot(fig)

    # Mapping table
    st.write("📌 Polymer Mapping (Original → Number)")
    mapping_df = pd.DataFrame({
        "Polymer_Type": poly_counts.index,
        "Number": range(1, len(poly_counts) + 1)
    })

    st.dataframe(mapping_df)

# =============================
# FEATURE SELECTION
# =============================
selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
X_selected = selector.fit_transform(X, y)

# =============================
# TRAIN TEST SPLIT
# =============================
X_train, X_test, y_train, y_test = train_test_split(
    X_selected, y, test_size=0.2, random_state=42
)

# =============================
# SCALING
# =============================
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# =============================
# SMOTE HANDLING
# =============================
class_counts = Counter(y_train)

if min(class_counts.values()) > 5:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

st.write("Class Distribution:", class_counts)

# =============================
# MODELS
# =============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
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
        best_model = model
        best_score = acc

# =============================
# RESULTS
# =============================
st.subheader("📈 Model Performance")

st.write(results)

fig = plt.figure()
plt.bar(results.keys(), results.values())
plt.title("Model Comparison")
st.pyplot(fig)

# =============================
# PREDICTION
# =============================
st.subheader("🔮 Prediction System")

input_data = {}

for col in X.columns:
    input_data[col] = st.number_input(col, value=float(X[col].mean()))

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])

    input_scaled = scaler.transform(input_df)
    input_selected = selector.transform(input_scaled)

    pred = best_model.predict(input_selected)

    st.success(f"Prediction: {pred[0]}")
