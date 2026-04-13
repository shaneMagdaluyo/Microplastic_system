import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from ml_pipeline import load_data, train_models, save_model


# =========================
# RISK MATRIX ENGINE
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
        df["Risk Score (0–100)"] = 50
    else:
        df["Risk Score (0–100)"] = (
            (df["Value"] - min_val) / (max_val - min_val)
        ) * 100

    def classify(x):
        if x < 25:
            return "Low"
        elif x < 50:
            return "Medium"
        elif x < 75:
            return "High"
        else:
            return "Critical"

    df["Risk Level"] = df["Risk Score (0–100)"].apply(classify)

    return df


# =========================
# K-MEANS CLUSTERING
# =========================
def run_kmeans(df, n_clusters):

    data = df.select_dtypes(include="number").copy()
    data = data.fillna(0)

    if data.shape[1] < 2:
        return None

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)

    result = df.copy()
    result["Cluster"] = clusters

    # PCA for visualization
    pca = PCA(n_components=2)
    comp = pca.fit_transform(X_scaled)

    result["PCA1"] = comp[:, 0]
    result["PCA2"] = comp[:, 1]

    return result


# =========================
# CONFIG
# =========================
st.set_page_config(page_title="MP Risk Intelligence", layout="wide")

st.title("🌊 Microplastic Risk Intelligence System")
st.caption("AI Dashboard: Risk + Correlation + ML + Clustering")


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


    # ==================================================
    # TAB 1 - DASHBOARD
    # ==================================================
    with tab1:

        st.subheader("📌 Dataset Preview")
        st.dataframe(df.head(), use_container_width=True)

        st.subheader("🌊 Risk Distribution")

        col1, col2 = st.columns(2)

        clean = pd.to_numeric(df[target], errors="coerce").dropna()

        fig, ax = plt.subplots()
        ax.hist(clean, bins=20)
        col1.pyplot(fig)

        fig2, ax2 = plt.subplots()
        ax2.boxplot(clean)
        col2.pyplot(fig2)


        # =========================
        # RISK MATRIX
        # =========================
        st.subheader("⚠️ Risk Matrix")

        risk_df = create_risk_matrix(df[target], df[name_col])

        if risk_df is not None:

            counts = risk_df["Risk Level"].value_counts()

            order = ["Low", "Medium", "High", "Critical"]
            counts = counts.reindex(order).fillna(0)

            col3, col4 = st.columns(2)

            col3.bar_chart(counts)

            fig, ax = plt.subplots()
            ax.pie(counts, labels=counts.index, autopct="%1.1f%%")
            col4.pyplot(fig)

            st.dataframe(risk_df, use_container_width=True)


    # ==================================================
    # TAB 2 - ANALYSIS
    # ==================================================
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
            st.warning("Not enough numeric features")

        st.divider()

        st.subheader("⚖️ Risk Comparison")

        feature = st.selectbox("Compare Feature", df.columns)

        if pd.api.types.is_numeric_dtype(df[feature]):

            fig, ax = plt.subplots()

            ax.scatter(
                df[feature],
                pd.to_numeric(df[target], errors="coerce")
            )

            ax.set_xlabel(feature)
            ax.set_ylabel("Risk Value")

            st.pyplot(fig)


    # ==================================================
    # TAB 3 - ML MODELS
    # ==================================================
    with tab3:

        st.subheader("🤖 Model Training")

        if st.button("Train Models"):

            try:
                results, best_name, best_model = train_models(df, target)

                st.success("Training Complete")

                results_df = pd.DataFrame(results).T
                st.dataframe(results_df)

                fig, ax = plt.subplots()
                results_df["accuracy"].plot(kind="bar", ax=ax)
                st.pyplot(fig)

                save_model(best_model)

                st.success(f"Best Model: {best_name}")

            except Exception as e:
                st.error(str(e))


    # ==================================================
    # TAB 4 - CLUSTERING (K-MEANS)
    # ==================================================
    with tab4:

        st.subheader("🧩 K-Means Clustering")

        k = st.slider("Select Number of Clusters", 2, 10, 3)

        if st.button("Run Clustering"):

            result = run_kmeans(df, k)

            if result is None:
                st.warning("Not enough numeric features for clustering")
            else:

                st.success("Clustering Completed")

                st.subheader("📊 Cluster Distribution")

                st.bar_chart(result["Cluster"].value_counts().sort_index())

                st.subheader("📍 Cluster Visualization (PCA)")

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

                st.subheader("📌 Clustered Data")

                st.dataframe(result, use_container_width=True)

                st.subheader("📊 Cluster Insights")

                st.write(result.groupby("Cluster").mean(numeric_only=True))

else:
    st.info("⬅️ Upload a CSV to begin")
