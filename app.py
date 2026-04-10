import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE
from collections import Counter

# -----------------------------
# LOAD DATA
# -----------------------------
df = pd.read_csv("data.csv")

# -----------------------------
# MISSING VALUES
# -----------------------------
df = df.fillna(df.median(numeric_only=True))

# -----------------------------
# ENCODE CATEGORICAL VARIABLES
# -----------------------------
label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# -----------------------------
# TRANSFORM SKEWED DATA
# -----------------------------
num_cols = df.select_dtypes(include=np.number).columns
for col in num_cols:
    if abs(df[col].skew()) > 1:
        df[col] = np.log1p(df[col])

# -----------------------------
# OUTLIER HANDLING (IQR)
# -----------------------------
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1
df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# -----------------------------
# TARGET SELECTION
# -----------------------------
target = "Risk_Type"  # change if needed
X = df.drop(columns=[target])
y = df[target]

# -----------------------------
# EDA ANALYSIS (REQUIRED TASKS)
# -----------------------------

# Risk Score distribution
if "Risk_Score" in df.columns:
    plt.figure()
    sns.histplot(df["Risk_Score"], kde=True)
    plt.title("Risk Score Distribution")
    plt.show()

# MP vs Risk Score relationship
if "MP_Count_per_L" in df.columns and "Risk_Score" in df.columns:
    plt.figure()
    sns.scatterplot(x=df["MP_Count_per_L"], y=df["Risk_Score"])
    plt.title("MP Count vs Risk Score")
    plt.show()

# Risk level differences
if "Risk_Level" in df.columns:
    plt.figure()
    sns.boxplot(x=df["Risk_Level"], y=df["Risk_Score"])
    plt.title("Risk Score by Risk Level")
    plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# FEATURE SCALING
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
# CLASS IMBALANCE HANDLING
# -----------------------------
print("Before SMOTE:", Counter(y_train))

if min(Counter(y_train).values()) > 5:
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

print("After SMOTE:", Counter(y_train))

# -----------------------------
# MODEL TRAINING
# -----------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC()
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc}")

# -----------------------------
# BEST MODEL SELECTION
# -----------------------------
best_model = max(models.items(), key=lambda x: accuracy_score(y_test, x[1].predict(X_test)))[1]

# -----------------------------
# FEATURE IMPORTANCE (RF ONLY)
# -----------------------------
if hasattr(best_model, "feature_importances_"):
    importance = best_model.feature_importances_

    plt.figure()
    plt.bar(range(len(importance)), importance)
    plt.title("Feature Importance")
    plt.show()

# -----------------------------
# FINAL REPORT OUTPUT
# -----------------------------
print("\nMODEL COMPARISON:")
for k, v in results.items():
    print(k, ":", v)

print("\nBest Model:", type(best_model).__name__)
