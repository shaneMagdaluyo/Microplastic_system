import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ml_pipeline import load_data, train_models, save_model


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
# K-MEANS (ROBUST FIX)
# =========================
def run_kmeans(df, k=3):

    data = df.copy()

    # convert everything possible to numeric
    for col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="ignore")

    # encode categorical columns
    data = pd.get_dummies(data, drop_first=True)

    # keep numeric only
    data = data.select_dtypes(include="number").fillna(0)

    # FORCE SAFE FEATURE COUNT
    if data.shape[1] < 2:
        data["extra_feature"] = range(len(data))

    # scale
    scaler = StandardScaler()
    X = scaler.fit_transform(data)

    # kmeans
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = model.fit_predict(X)

    result = df.copy()
    result["Cluster"] = clusters

    # PCA
    pca = PCA(n_components=2)
    comp = pca.fit_transform(X)

    result["PCA1"] = comp[:, 0]
    result["PCA2"] = comp[:, 1]

    return result


# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="MP Risk Intelligence", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")
st.caption("Full AI Dashboard (Risk + ML + Clustering)")


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

    st.divider()

    target = st.sidebar.selectbox("🎯 Risk Column", df.columns)
    name_col = st.sidebar.selectbox("🏷️ Name Column", df.columns)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔬 Analysis",
        "🤖 ML Models",
        "🧩 Clustering"
    ])

    # =========================
    # TAB 1 - DASHBOARD
    # =========================
    with tab1:

        st.subheader("📌 Data Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("🌊 Risk Distribution")

        clean = pd.to_numeric(df[target], errors="coerce").dropna()

        if len(clean) > 0:

            col1, col2 = st.columns(2)

            fig, ax = plt.subplots()
            ax.hist(clean, bins=20)
            col1.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.boxplot(clean)
            col2.pyplot(fig2)
        else:
            st.warning("Target column is not numeric enough for plots")

        # =========================
        # RISK MATRIX
        # =========================
        st.subheader("⚠️ Risk Matrix")

        risk_df = create_risk_matrix(df[target], df[name_col])

        if risk_df is not None:

            st.bar_chart(risk_df["Risk Level"].value_counts())

            st.dataframe(risk_df, use_container_width=True)

        else:
            st.warning("Not enough valid data for risk matrix")


    # =========================
    # TAB 2 - ANALYSIS
    # =========================
    with tab2:

        st.subheader("🔥 Correlation Matrix")

        num_df = df.select_dtypes(include="number").fillna(0)

        if num_df.shape[1] >= 2:

            corr = num_df.corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(corr, cmap="coolwarm")

            plt.colorbar(im)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))

            ax.set_xticklabels(corr.columns, rotation=45, ha="right")
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)

        else:
            st.warning("Not enough numeric features for correlation")

        st.divider()

        st.subheader("⚖️ Risk Comparison")

        feature = st.selectbox("Select Feature", df.columns)

        if pd.api.types.is_numeric_dtype(df[feature]):

            fig, ax = plt.subplots()

            ax.scatter(
                df[feature],
                pd.to_numeric(df[target], errors="coerce")
            )

            ax.set_xlabel(feature)
            ax.set_ylabel("Risk")

            st.pyplot(fig)

        else:
            st.info("Select numeric feature only")


    # =========================
    # TAB 3 - ML MODELS
    # =========================
    with tab3:

        st.subheader("🤖 Model Training")

        if st.button("Train Models"):

            try:
                results, best_name, best_model = train_models(df, target)

                st.success("Training Done")

                results_df = pd.DataFrame(results).T
                st.dataframe(results_df)

                fig, ax = plt.subplots()
                results_df["accuracy"].plot(kind="bar", ax=ax)
                st.pyplot(fig)

                save_model(best_model)

                st.success(f"Best Model: {best_name}")

            except Exception as e:
                st.error(str(e))


    # =========================
    # TAB 4 - CLUSTERING
    # =========================
    with tab4:

        st.subheader("🧩 K-Means Clustering")

        k = st.slider("Number of Clusters", 2, 10, 3)

        if st.button("Run Clustering"):

            result = run_kmeans(df, k)

            st.success("Clustering Completed")

            st.subheader("📊 Cluster Distribution")
            st.bar_chart(result["Cluster"].value_counts().sort_index())

            st.subheader("📍 PCA Visualization")

            fig, ax = plt.subplots()

            scatter = ax.scatter(
                result["PCA1"],
                result["PCA2"],
                c=result["Cluster"],
                cmap="viridis"
            )

            ax.set_xlabel("PCA 1")
            ax.set_ylabel("PCA 2")

            plt.colorbar(scatter)

            st.pyplot(fig)

            st.subheader("📌 Cluster Data")
            st.dataframe(result, use_container_width=True)

            st.subheader("📊 Cluster Insights")

            st.write(result.groupby("Cluster").mean(numeric_only=True))

else:
    st.info("⬅️ Upload a CSV to begin")
