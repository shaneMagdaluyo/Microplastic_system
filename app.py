import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from ml_pipeline import load_data

# =========================
# SAFE ARIMA IMPORT
# =========================
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False


# =========================
# RISK MATRIX
# =========================
def create_risk_matrix(series, name_series):

    numeric = pd.to_numeric(series, errors="coerce")

    df = pd.DataFrame({
        "Name": name_series,
        "Value": numeric
    }).dropna()

    if len(df) < 3:
        return None

    min_val = df["Value"].min()
    max_val = df["Value"].max()

    df["Risk Score"] = 50 if max_val == min_val else (
        (df["Value"] - min_val) / (max_val - min_val) * 100
    )

    def classify(x):
        if x < 25:
            return "Low"
        elif x < 50:
            return "Medium"
        elif x < 75:
            return "High"
        return "Critical"

    df["Risk Level"] = df["Risk Score"].apply(classify)

    return df


# =========================
# HIGH RISK ENGINE
# =========================
def high_risk_engine(df, target):

    values = pd.to_numeric(df[target], errors="coerce")
    threshold = values.quantile(0.75)

    df = df.copy()
    df["Risk Category"] = values.apply(
        lambda x: "HIGH RISK" if x >= threshold else "NORMAL"
    )

    return df, threshold


# =========================
# KMEANS
# =========================
def run_kmeans(df, k=3):

    data = df.copy()

    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = pd.get_dummies(data, drop_first=True)
    data = data.select_dtypes(include="number").fillna(0)

    if data.shape[1] < 2:
        data["extra"] = range(len(data))

    X = StandardScaler().fit_transform(data)

    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(X)

    result = df.copy()
    result["Cluster"] = clusters

    pca = PCA(n_components=2)
    comp = pca.fit_transform(X)

    result["PCA1"] = comp[:, 0]
    result["PCA2"] = comp[:, 1]

    return result


# =========================
# CLASSIFICATION TABLE
# =========================
def build_report_table(y_true, y_pred, model_name):

    report = classification_report(y_true, y_pred, output_dict=True)

    rows = []

    for label, metrics in report.items():

        if label in ["accuracy", "macro avg", "weighted avg"]:
            continue

        rows.append({
            "Model": model_name,
            "Class": label,
            "Precision": round(metrics["precision"], 3),
            "Recall": round(metrics["recall"], 3),
            "F1-score": round(metrics["f1-score"], 3),
            "Support": int(metrics["support"])
        })

    return pd.DataFrame(rows)


# =========================
# TIME SERIES FIX (IMPORTANT)
# =========================
def prepare_time_series(df, date_col, value_col):

    temp = df[[date_col, value_col]].copy()

    # 🔥 FIX: force string first to avoid "duplicate keys" bug
    temp[date_col] = pd.to_datetime(temp[date_col].astype(str), errors="coerce")
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")

    temp = temp.dropna()

    # 🔥 FIX: remove duplicate dates
    temp = temp.groupby(date_col)[value_col].mean().reset_index()

    temp = temp.sort_values(date_col).set_index(date_col)

    return temp if not temp.empty else None


# =========================
# ARIMA
# =========================
def run_arima(series, steps=10):

    if not ARIMA_AVAILABLE:
        return None

    try:
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()
        return model_fit.forecast(steps=steps)
    except:
        return None


# =========================
# APP
# =========================
st.set_page_config(page_title="MP Risk Intelligence", layout="wide")
st.title("🌊 Microplastic Risk Intelligence System")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    df = load_data(file)

    target = st.sidebar.selectbox("🎯 Risk Column", df.columns)
    name_col = st.sidebar.selectbox("🏷️ Name Column", df.columns)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard",
        "Risk Analysis",
        "ML Models",
        "Clustering",
        "Forecasting"
    ])

    # ================= DASHBOARD =================
    with tab1:

        clean = pd.to_numeric(df[target], errors="coerce").dropna()

        fig, ax = plt.subplots()
        ax.hist(clean, bins=20)
        st.pyplot(fig)

        df_risk, threshold = high_risk_engine(df, target)

        st.info(f"High Risk Threshold: {threshold:.2f}")
        st.bar_chart(df_risk["Risk Category"].value_counts())

    # ================= ML MODELS =================
    with tab3:

        st.subheader("ML + Cross Validation")

        if st.button("Run Models"):

            df_ml = df.copy().dropna(subset=[target])

            if df_ml[target].dtype == "object":
                df_ml[target] = LabelEncoder().fit_transform(df_ml[target].astype(str))

            y = df_ml[target]
            X = pd.get_dummies(df_ml.drop(columns=[target]), drop_first=True).fillna(0)

            rf = RandomForestClassifier(n_estimators=200)
            svm = SVC()

            # 🔥 CROSS VALIDATION
            rf_cv = cross_val_score(rf, X, y, cv=5).mean()
            svm_cv = cross_val_score(svm, X, y, cv=5).mean()

            st.subheader("Cross Validation Accuracy")

            cv_df = pd.DataFrame({
                "Model": ["Random Forest", "SVM"],
                "CV Accuracy": [rf_cv, svm_cv]
            })

            st.dataframe(cv_df)
            st.bar_chart(cv_df.set_index("Model"))

            # Train-test
            X_train, X_test, y_train, y_test = train_test_split(X, y)

            rf.fit(X_train, y_train)
            svm.fit(X_train, y_train)

            rf_pred = rf.predict(X_test)
            svm_pred = svm.predict(X_test)

            st.subheader("Classification Table")

            st.dataframe(pd.concat([
                build_report_table(y_test, rf_pred, "Random Forest"),
                build_report_table(y_test, svm_pred, "SVM")
            ]))

    # ================= FORECAST =================
    with tab5:

        st.subheader("Time-Series Forecast")

        date_col = st.selectbox("Date Column", df.columns)
        value_col = st.selectbox("Value Column", df.columns)

        if st.button("Run Forecast"):

            ts = prepare_time_series(df, date_col, value_col)

            if ts is None or len(ts) < 10:
                st.warning("Not enough time-series data")
            else:

                st.line_chart(ts[value_col])

                forecast = run_arima(ts[value_col])

                if forecast is not None:

                    future_index = pd.date_range(
                        ts.index[-1],
                        periods=len(forecast)+1
                    )[1:]

                    forecast_series = pd.Series(forecast, index=future_index)

                    st.line_chart(pd.concat([ts[value_col], forecast_series]))

                else:
                    st.warning("ARIMA not available or failed")

else:
    st.info("Upload CSV to start")
