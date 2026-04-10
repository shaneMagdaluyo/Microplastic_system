import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import tempfile

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

from imblearn.over_sampling import SMOTE

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Microplastic AI Pro", layout="wide")
st.title("🌊 Microplastic AI Pro Dashboard")

# =========================
# UPLOAD
# =========================
uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

# =========================
# CLEAN FUNCTION (ROBUST)
# =========================
def clean_data(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan)

    for col in df.columns:
        df[col] = df[col].astype(str)
        extracted = df[col].str.extract(r'(\d+\.?\d*)')[0]
        numeric = pd.to_numeric(extracted, errors='coerce')

        if numeric.notna().mean() > 0.5:
            df[col] = numeric.fillna(numeric.median())
        else:
            df[col] = df[col].fillna("Unknown")

    return df


# =========================
# MAIN APP
# =========================
if uploaded_file:

    df = pd.read_csv(uploaded_file)

    st.subheader("📊 Dataset Preview")
    st.dataframe(df.head())

    target = st.sidebar.selectbox("🎯 Target Column", df.columns)
    apply_smote = st.sidebar.checkbox("⚖️ Apply SMOTE")
    test_size = st.sidebar.slider("Test Size", 0.1, 0.4, 0.2)

    # =========================
    # CLEAN
    # =========================
    data = clean_data(df)

    # =========================
    # ENCODE
    # =========================
    encoders = {}
    for col in data.select_dtypes(include="object").columns:
        le = LabelEncoder()
        data[col] = le.fit_transform(data[col])
        encoders[col] = le

    X = data.drop(columns=[target])
    y = data[target]

    X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

    # =========================
    # SPLIT
    # =========================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # =========================
    # SMOTE
    # =========================
    if apply_smote:
        try:
            smote = SMOTE()
            X_train, y_train = smote.fit_resample(X_train, y_train)
            st.success("SMOTE applied")
        except:
            st.warning("SMOTE skipped (data issue)")

    # =========================
    # SCALING
    # =========================
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # =========================
    # MODELS (AUTO COMPARE)
    # =========================
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = []
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="weighted")

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1 Score": f1
        })

        trained_models[name] = model

    results_df = pd.DataFrame(results)

    # =========================
    # BEST MODEL
    # =========================
    best_model_name = results_df.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
    best_model = trained_models[best_model_name]

    st.subheader("🏆 Best Model")
    st.success(f"{best_model_name}")

    col1, col2 = st.columns(2)
    col1.metric("Best Accuracy", round(results_df["Accuracy"].max(), 3))
    col2.metric("Best F1 Score", round(results_df["F1 Score"].max(), 3))

    fig = px.bar(results_df, x="Model", y=["Accuracy", "F1 Score"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    # =========================
    # SAVE MODEL
    # =========================
    model_data = {
        "model": best_model,
        "scaler": scaler,
        "encoders": encoders,
        "columns": X.columns.tolist()
    }

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        joblib.dump(model_data, tmp.name)
        st.download_button(
            "⬇️ Download Best Model",
            open(tmp.name, "rb"),
            file_name="microplastic_model.pkl"
        )

    # =========================
    # FEATURE IMPORTANCE
    # =========================
    st.subheader("🔍 Feature Importance")

    if hasattr(best_model, "feature_importances_"):
        feat_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        fig2 = px.bar(feat_df.head(10), x="Importance", y="Feature", orientation="h")
        st.plotly_chart(fig2, use_container_width=True)

    # =========================
    # REAL-TIME PREDICTION
    # =========================
    st.subheader("🔮 Real-Time Prediction")

    input_data = []

    for col in X.columns:
        val = st.number_input(f"{col}", value=0.0)
        input_data.append(val)

    if st.button("Predict"):
        input_array = np.array(input_data).reshape(1, -1)
        input_scaled = scaler.transform(input_array)

        prediction = best_model.predict(input_scaled)[0]

        st.success(f"Prediction: {prediction}")

    st.success("🚀 PRO Dashboard Ready!")

else:
    st.info("Upload a CSV file to start")
