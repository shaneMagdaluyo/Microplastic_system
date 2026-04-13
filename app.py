import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# ✅ SAFE ARIMA IMPORT
try:
    from statsmodels.tsa.arima.model import ARIMA
    ARIMA_AVAILABLE = True
except:
    ARIMA_AVAILABLE = False

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


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
# TIME SERIES
# =========================
def prepare_time_series(df, date_col, value_col):

    temp = df[[date_col, value_col]].copy()
    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")

    temp = temp.dropna().sort_values(date_col)
    temp = temp.set_index(date_col)

    return temp


def run_arima(series, steps=10):

    if not ARIMA_AVAILABLE:
        return np.zeros(steps)

    try:
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()
        return model_fit.forecast(steps=steps)
    except:
        return np.zeros(steps)


def run_lstm(series, steps=10):

    if len(series) < 15:
        return np.zeros(steps)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.values.reshape(-1, 1))

    X, y = [], []

    for i in range(10, len(scaled)):
        X.append(scaled[i-10:i])
        y.append(scaled[i])

    X, y = np.array(X), np.array(y)

    model = Sequential([
        LSTM(50, activation="relu", input_shape=(X.shape[1], 1)),
        Dense(1)
    ])

    model.compile(optimizer="adam", loss="mse")
    model.fit(X, y, epochs=5, batch_size=8, verbose=0)

    last_seq = scaled[-10:]
    preds = []

    for _ in range(steps):
        pred = model.predict(last_seq.reshape(1, 10, 1), verbose=0)[0][0]
        preds.append(pred)
        last_seq = np.append(last_seq[1:], [[pred]], axis=0)

    preds = scaler.inverse_transform(np.array(preds).reshape(-1, 1))

    return preds.flatten()


# =========================
# APP
# =========================
st.set_page_config(page_title="MP Risk Intelligence", layout="wide")
st.title("🌊 Microplastic Risk Intelligence System")

file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    df = pd.read_csv(file)

    target = st.sidebar.selectbox("🎯 Risk Column", df.columns)
    name_col = st.sidebar.selectbox("🏷️ Name Column", df.columns)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard",
        "Risk Analysis",
        "ML Models",
        "Clustering",
        "Forecasting"
    ])

    # DASHBOARD
    with tab1:
        st.dataframe(df.head())

    # RISK ANALYSIS
    with tab2:
        st.write("Risk Analysis Ready")

    # ML MODELS
    with tab3:

        if st.button("Run Models"):

            df_ml = df.dropna(subset=[target]).copy()

            if df_ml[target].dtype == "object":
                df_ml[target] = LabelEncoder().fit_transform(df_ml[target].astype(str))

            y = df_ml[target]
            X = pd.get_dummies(df_ml.drop(columns=[target]), drop_first=True).fillna(0)

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            rf = RandomForestClassifier(n_estimators=100)
            rf.fit(X_train, y_train)
            rf_pred = rf.predict(X_test)

            svm = SVC()
            svm.fit(X_train, y_train)
            svm_pred = svm.predict(X_test)

            rf_df = build_report_table(y_test, rf_pred, "Random Forest")
            svm_df = build_report_table(y_test, svm_pred, "SVM")

            st.dataframe(pd.concat([rf_df, svm_df]))

    # CLUSTERING
    with tab4:
        if st.button("Run Clustering"):
            result = run_kmeans(df)
            st.dataframe(result)

    # FORECASTING
    with tab5:

        date_col = st.selectbox("Date Column", df.columns)
        value_col = st.selectbox("Value Column", df.columns)

        if st.button("Run Forecast"):

            ts = prepare_time_series(df, date_col, value_col)

            st.line_chart(ts[value_col])

            arima_pred = run_arima(ts[value_col])
            lstm_pred = run_lstm(ts[value_col])

            st.write("ARIMA Forecast:", arima_pred)
            st.write("LSTM Forecast:", lstm_pred)

else:
    st.info("Upload CSV to start")
