import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE
from collections import Counter

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Polymer ML Objective 2", layout="wide")
st.title("🌊 OBJECTIVE #2 – Polymer Risk ML System")

# =========================
# LOAD DATA
# =========================
uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("Data Loaded")
else:
    df = None


# =========================
# CLEAN DATA
# =========================
def clean_data(df):
    df = df.copy()

    for col in df.columns:
        num = pd.to_numeric(df[col], errors="coerce")

        if num.notna().sum() > len(df) * 0.5:
            df[col] = num.fillna(num.median())
        else:
            df[col] = df[col].astype(str).fillna("missing")

    return df


# =========================
# ENCODE DATA
# =========================
def encode(df):
    df = df.copy()
    encoders = {}

    for col in df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    return df


# =========================
# SAFE SMOTE
# =========================
def safe_smote(X, y):
    X = pd.DataFrame(X)

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median())

    counts = Counter(y)
    min_class = min(counts.values())

    if min_class < 2:
        st.warning("SMOTE skipped (not enough samples)")
        return X, y

    k = min(5, min_class - 1)
    smote = SMOTE(k_neighbors=k, random_state=42)

    return smote.fit_resample(X, y)


# =========================
# PREPROCESS PIPELINE
# =========================
if df is not None:

    df = clean_data(df)

    st.subheader("1. Dataset Preview")
    st.dataframe(df.head())


    # =========================
    # POLYMER TYPE DISTRIBUTION
    # =========================
    if "polymer_type" in df.columns:
        st.subheader("2. Polymer Type Distribution")
        fig, ax = plt.subplots()
        df["polymer_type"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)


    # =========================
    # RISK TYPE DISTRIBUTION
    # =========================
    if "Risk_Type" in df.columns:
        st.subheader("3. Risk_Type Distribution")

        st.write(Counter(df["Risk_Type"]))

        fig, ax = plt.subplots()
        df["Risk_Type"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)


    # =========================
    # ENCODING
    # =========================
    df = encode(df)


    # =========================
    # FEATURE ENGINEERING
    # =========================
    num_cols = df.select_dtypes(include=np.number).columns

    pt = PowerTransformer()
    df[num_cols] = pt.fit_transform(df[num_cols])


    # =========================
    # TARGET SELECTION
    # =========================
    target = st.selectbox("Select Target", df.columns)

    X = df.drop(columns=[target])
    y = df[target]


    # =========================
    # FEATURE SELECTION
    # =========================
    selector = SelectKBest(f_classif, k=min(10, X.shape[1]))
    X_new = selector.fit_transform(X, y)

    selected_features = X.columns[selector.get_support()]
    X = X[selected_features]


    st.subheader("Selected Features")
    st.write(list(selected_features))


    # =========================
    # SMOTE
    # =========================
    st.subheader("4. SMOTE Class Balancing")

    X_res, y_res = safe_smote(X, y)


    st.write("After SMOTE:", Counter(y_res))


    # =========================
    # TRAIN TEST SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res,
        test_size=0.2,
        random_state=42,
        stratify=y_res
    )


    # =========================
    # MODELS
    # =========================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = {}

    st.subheader("5. Model Training")

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)

        results[name] = {
            "model": model,
            "acc": acc
        }

        st.write(f"{name}: {acc:.4f}")


    # =========================
    # MODEL COMPARISON
    # =========================
    st.subheader("6. Model Comparison")

    comp_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": [results[m]["acc"] for m in results]
    })

    st.dataframe(comp_df)

    fig, ax = plt.subplots()
    sns.barplot(data=comp_df, x="Model", y="Accuracy", ax=ax)
    plt.xticks(rotation=20)
    st.pyplot(fig)


    # =========================
    # HYPERPARAMETER TUNING
    # =========================
    st.subheader("7. Hyperparameter Tuning (Logistic Regression)")

    param_grid = {
        "C": [0.1, 1, 10],
        "solver": ["lbfgs"]
    }

    grid = GridSearchCV(
        LogisticRegression(max_iter=2000),
        param_grid,
        cv=3
    )

    grid.fit(X_train, y_train)

    best_lr = grid.best_estimator_
    pred = best_lr.predict(X_test)

    st.write("Best Params:", grid.best_params_)
    st.write("Tuned LR Accuracy:", accuracy_score(y_test, pred))


    # =========================
    # FEATURE IMPORTANCE
    # =========================
    st.subheader("8. Feature Importance (Random Forest)")

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    imp = pd.DataFrame({
        "Feature": X.columns,
        "Importance": rf.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    st.dataframe(imp)

    fig, ax = plt.subplots()
    sns.barplot(data=imp, x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)


    # =========================
    # BEST MODEL
    # =========================
    best_model = max(results, key=lambda x: results[x]["acc"])

    st.subheader("9. Best Model")
    st.write(best_model)


    # =========================
    # SUMMARY
    # =========================
    st.subheader("10. Summary")

    st.markdown("""
### ✔ OBJECTIVE #2 COMPLETED

- Data loaded & cleaned
- Polymer distribution analyzed
- Risk_Type distribution checked
- Encoding applied
- Feature selection performed
- SMOTE balancing applied
- Multiple models trained
- Model comparison visualized
- Hyperparameter tuning done
- Feature importance analyzed
- Best model selected

🚀 SYSTEM READY FOR ENTERPRISE ML PIPELINE
""")
