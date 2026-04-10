import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from collections import Counter

# -----------------------------
# APP CONFIG
# -----------------------------
st.set_page_config(page_title="Microplastic ML System", layout="wide")
st.title("🌊 Microplastic Risk Analysis & Prediction System")

# -----------------------------
# UPLOAD DATA
# -----------------------------
file = st.file_uploader("Upload CSV", type=["csv"])

if file is None:
    st.stop()

df = pd.read_csv(file)

# =============================
# 1. DATA PREPROCESSING
# =============================
df = df.fillna(df.median(numeric_only=True))

# Encode categorical variables
encoders = {}
for col in df.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# Transform skewed data
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    if abs(df[col].skew()) > 1:
        df[col] = np.log1p(df[col])

# Remove outliers
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

st.success(f"Cleaned dataset: {df.shape}")

# =============================
# TARGET
# =============================
target = st.selectbox("Select Target Column", df.columns)

X = df.drop(columns=[target])
y = df[target]

# =============================
# 2. EDA FEATURES
# =============================
st.subheader("📊 Exploratory Data Analysis")

if "Risk_Score" in df.columns:
    st.write("Risk Score Distribution")
    fig = plt.figure()
    plt.hist(df["Risk_Score"], bins=20)
    st.pyplot(fig)

if "MP_Count_per_L" in df.columns and "Risk_Score" in df.columns:
    st.write("MP vs Risk Score")
    fig = plt.figure()
    plt.scatter(df["MP_Count_per_L"], df["Risk_Score"])
    st.pyplot(fig)

if "Risk_Level" in df.columns:
    st.write("Risk Score by Risk Level")
    fig = plt.figure()
    df.boxplot(column="Risk_Score", by="Risk_Level")
    st.pyplot(fig)

if "Polymer_Type" in df.columns:
    st.write("Polymer Type Distribution")
    fig = plt.figure()
    df["Polymer_Type"].value_counts().plot(kind="bar")
    st.pyplot(fig)

# =============================
# 3. FEATURE SELECTION
# =============================
selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
X_selected = selector.fit_transform(X, y)

st.write("Top Features Selected")

# =============================
# SPLIT DATA
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
# 7. CLASS IMBALANCE (SMOTE)
# =============================
st.write("Class Distribution:", Counter(y_train))

if min(Counter(y_train).values()) > 5:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

# =============================
# 4. MODEL BUILDING
# =============================
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    results[name] = accuracy_score(y_test, preds)

# =============================
# 6. HYPERPARAMETER TUNING
# =============================
st.subheader("⚙ Hyperparameter Tuning")

param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None]
}

rf = RandomForestClassifier()
grid = GridSearchCV(rf, param_grid, cv=3)
grid.fit(X_train, y_train)

best_model = grid.best_estimator_

# =============================
# 5. MODEL EVALUATION
# =============================
st.subheader("📈 Model Comparison")

results["Tuned RF"] = accuracy_score(y_test, best_model.predict(X_test))

st.write(results)

fig = plt.figure()
plt.bar(results.keys(), results.values())
st.pyplot(fig)

# =============================
# FEATURE IMPORTANCE
# =============================
st.subheader("📌 Feature Importance")

if hasattr(best_model, "feature_importances_"):
    plt.figure()
    plt.bar(range(len(best_model.feature_importances_)), best_model.feature_importances_)
    st.pyplot(plt)

# =============================
# 8. SUMMARY
# =============================
st.subheader("📄 Summary Report")

best_name = max(results, key=results.get)

st.write(f"""
- Dataset cleaned and preprocessed
- Categorical encoding applied
- Outliers removed
- Skewed data transformed
- SMOTE applied for imbalance
- Models trained: Logistic Regression, Random Forest, SVM
- Best model: {best_name}
- Best accuracy: {results[best_name]:.4f}
""")

# =============================
# 9. PREDICTION
# =============================
st.subheader("🔮 Prediction")

input_data = {}

for col in X.shape[1:]:
    input_data[col] = st.number_input(f"Feature {col}", value=0.0)

if st.button("Predict"):
    model = best_model
    input_df = np.array(list(input_data.values())).reshape(1, -1)
    input_scaled = scaler.transform(input_df)

    pred = model.predict(input_scaled)
    st.success(f"Prediction: {pred[0]}")
