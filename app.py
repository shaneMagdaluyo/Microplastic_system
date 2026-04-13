import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from ml_pipeline import load_data, train_models, save_model


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

    if max_val == min_val:
        df["Risk Score"] = 50
    else:
        df["Risk Score"] = (df["Value"] - min_val) / (max_val - min_val) * 100

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

    scaler = StandardScaler()
    X = scaler.fit_transform(data)

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
# TIME SERIES FUNCTIONS
# =========================
def prepare_time_series(df, date_col, value_col):

    temp = df[[date_col, value_col]].copy()

    temp[date_col] = pd.to_datetime(temp[date_col], errors="coerce")
    temp[value_col] = pd.to_numeric(temp[value_col], errors="coerce")

    temp = temp.dropna().sort_values(date_col)

    if temp.empty:
        return None

    temp = temp.set_index(date_col)

    return temp


def run_arima(series, steps=10):

    if not ARIMA_AVAILABLE:
        return None

    try:
        model = ARIMA(series, order=(2, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=steps)
        return forecast
    except:
        return None


# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="MP Risk Intelligence", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")


# =========================
# UPLOAD
# =========================
file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

if file:

    df = load_data(file)

    st.subheader("📊 Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing", int(df.isnull().sum().sum()))
    c4.metric("Numeric", df.select_dtypes(include="number").shape[1])

    target = st.sidebar.selectbox("🎯 Risk Column", df.columns)
    name_col = st.sidebar.selectbox("🏷️ Name Column", df.columns)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Dashboard",
        "Risk Analysis",
        "ML Models",
        "Clustering + Classification",
        "Forecasting"
    ])


    # ================= DASHBOARD =================
    with tab1:

        clean = pd.to_numeric(df[target], errors="coerce").dropna()

        col1, col2 = st.columns(2)

        if len(clean) > 0:
            fig, ax = plt.subplots()
            ax.hist(clean, bins=20)
            col1.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.boxplot(clean)
            col2.pyplot(fig2)

        st.subheader("High Risk Detection")

        df_risk, threshold = high_risk_engine(df, target)

        st.info(f"High Risk Threshold: {threshold:.2f}")
        st.bar_chart(df_risk["Risk Category"].value_counts())
        st.dataframe(df_risk[df_risk["Risk Category"] == "HIGH RISK"])

        st.subheader("Risk Level Matrix")

        risk_df = create_risk_matrix(df[target], df[name_col])

        if risk_df is not None:
            st.bar_chart(risk_df["Risk Level"].value_counts())
            st.dataframe(risk_df)


    # ================= RISK ANALYSIS =================
    with tab2:

        st.subheader("Risk Comparison")

        feature = st.selectbox("Select Feature", df.columns)

        df_clean = df.dropna(subset=[feature, target]).copy()
        df_clean[target] = pd.to_numeric(df_clean[target], errors="coerce")

        if pd.api.types.is_numeric_dtype(df_clean[feature]):

            plot_df = pd.DataFrame({
                "Feature": pd.to_numeric(df_clean[feature], errors="coerce"),
                "Risk": df_clean[target]
            }).dropna()

            if len(plot_df) > 0:
                fig, ax = plt.subplots()
                ax.scatter(plot_df["Feature"], plot_df["Risk"])
                st.pyplot(fig)

        else:

            grouped = (
                df_clean.groupby(feature)[target]
                .mean()
                .dropna()
                .reset_index()
            )

            grouped.columns = [feature, "Risk"]

            if len(grouped) > 0:

                st.bar_chart(grouped.set_index(feature))

                st.write("🏆 Highest Risk:",
                         grouped.loc[grouped["Risk"].idxmax(), feature])

                st.write("⬇️ Lowest Risk:",
                         grouped.loc[grouped["Risk"].idxmin(), feature])


    # ================= ML MODELS =================
    with tab3:

        st.subheader("Random Forest & SVM Table")

        if st.button("Run Models"):

            df_ml = df.copy().dropna(subset=[target])

            le = LabelEncoder()
            if df_ml[target].dtype == "object":
                df_ml[target] = le.fit_transform(df_ml[target].astype(str))

            y = df_ml[target]
            X = pd.get_dummies(df_ml.drop(columns=[target]), drop_first=True).fillna(0)

            if X.shape[1] > 0:

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                rf = RandomForestClassifier(n_estimators=200)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)

                svm = SVC()
                svm.fit(X_train, y_train)
                svm_pred = svm.predict(X_test)

                rf_df = build_report_table(y_test, rf_pred, "Random Forest")
                svm_df = build_report_table(y_test, svm_pred, "SVM")

                st.dataframe(pd.concat([rf_df, svm_df]))

                acc_df = pd.DataFrame({
                    "Model": ["Random Forest", "SVM"],
                    "Accuracy": [
                        accuracy_score(y_test, rf_pred),
                        accuracy_score(y_test, svm_pred)
                    ]
                })

                st.bar_chart(acc_df.set_index("Model"))


    # ================= CLUSTERING =================
    with tab4:

        k = st.slider("Clusters", 2, 10, 3)

        if st.button("Run Clustering"):

            result = run_kmeans(df, k)

            st.bar_chart(result["Cluster"].value_counts())

            fig, ax = plt.subplots()
            ax.scatter(result["PCA1"], result["PCA2"], c=result["Cluster"])
            st.pyplot(fig)

            st.dataframe(result)


    # ================= FORECASTING =================
    with tab5:

        st.subheader("📈 Time-Series Forecasting")

        date_col = st.selectbox("Date Column", df.columns)
        value_col = st.selectbox("Value Column", df.columns)

        steps = st.slider("Forecast Steps", 5, 30, 10)

        if st.button("Run Forecast"):

            ts = prepare_time_series(df, date_col, value_col)

            if ts is None or len(ts) < 10:
                st.warning("Not enough valid time-series data")
            else:

                st.line_chart(ts[value_col])

                forecast = run_arima(ts[value_col], steps)

                if forecast is None:
                    st.warning("ARIMA not available. Install statsmodels.")
                else:

                    future_index = pd.date_range(
                        ts.index[-1],
                        periods=steps + 1
                    )[1:]

                    forecast_series = pd.Series(forecast, index=future_index)

                    combined = pd.concat([ts[value_col], forecast_series])

                    st.line_chart(combined)

else:
    st.info("Upload a CSV to begin")
