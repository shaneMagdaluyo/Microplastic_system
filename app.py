import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import type_of_target

from imblearn.over_sampling import SMOTE
from collections import Counter

# =============================
# LOAD & PREPROCESS FUNCTION
# =============================
def preprocess_data(df, target_column):
    """
    Full preprocessing pipeline:
    - encoding
    - skew handling
    - outlier removal
    - scaling
    - SMOTE balancing
    """

    # -----------------------------
    # 1. Handle missing values
    # -----------------------------
    df = df.fillna(df.median(numeric_only=True))

    # -----------------------------
    # 2. Encode categorical variables
    # -----------------------------
    encoders = {}
    for col in df.select_dtypes(include=["object"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    # -----------------------------
    # 3. Transform skewed numerical columns
    # -----------------------------
    num_cols = df.select_dtypes(include=[np.number]).columns

    for col in num_cols:
        if abs(df[col].skew()) > 1:
            df[col] = np.log1p(df[col])

    # -----------------------------
    # 4. Outlier handling (IQR method)
    # -----------------------------
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1

    df = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # -----------------------------
    # 5. Split features/target
    # -----------------------------
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # -----------------------------
    # 6. Validate target
    # -----------------------------
    if y.nunique() < 2:
        raise ValueError("Target must have at least 2 classes")

    # -----------------------------
    # 7. Train-test split
    # -----------------------------
    use_stratify = min(Counter(y).values()) > 1

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y if use_stratify else None
    )

    # -----------------------------
    # 8. Feature scaling
    # -----------------------------
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # -----------------------------
    # 9. Handle class imbalance (SMOTE)
    # -----------------------------
    class_counts = Counter(y_train)

    if type_of_target(y_train) in ["binary", "multiclass"] and min(class_counts.values()) >= 6:
        smote = SMOTE(k_neighbors=5, random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)

    # -----------------------------
    # OUTPUT
    # -----------------------------
    return X_train, X_test, y_train, y_test, scaler, encoders
