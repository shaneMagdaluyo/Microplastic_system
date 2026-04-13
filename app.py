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
# K-MEANS
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
    st.dataframe(df.head())

    target = st.sidebar.selectbox("Risk Column", df.columns)
    name_col = st.sidebar.selectbox("Name Column", df.columns)

    tab1, tab2, tab3, tab4 = st.tabs([
        "Dashboard",
        "Analysis",
        "ML Models",
        "Clustering"
    ])


    # =========================
    # DASHBOARD
    # =========================
    with tab1:

        st.subheader("Risk Distribution")

        clean = pd.to_numeric(df[target], errors="coerce").dropna()

        col1, col2 = st.columns(2)

        if len(clean) > 0:
            fig, ax = plt.subplots()
            ax.hist(clean, bins=20)
            col1.pyplot(fig)

            fig2, ax2 = plt.subplots()
            ax2.boxplot(clean)
            col2.pyplot(fig2)

        st.subheader("Risk Matrix")

        risk_df = create_risk_matrix(df[target], df[name_col])

        if risk_df is not None:
            st.bar_chart(risk_df["Risk Level"].value_counts())
            st.dataframe(risk_df)
        else:
            st.warning("Not enough data")


    # =========================
    # ANALYSIS
    # =========================
    with tab2:

        st.subheader("Correlation Matrix")

        num_df = df.select_dtypes(include="number").dropna(axis=1, how="all")
        num_df = num_df.loc[:, num_df.nunique() > 1]

        if num_df.shape[1] >= 2:

            corr = num_df.corr()

            fig, ax = plt.subplots(figsize=(10, 6))
            im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)

            plt.colorbar(im)

            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))

            ax.set_xticklabels(corr.columns, rotation=45)
            ax.set_yticklabels(corr.columns)

            st.pyplot(fig)

        else:
            st.warning("Not enough numeric features")


        st.subheader("Risk Comparison (FIXED)")

        feature = st.selectbox("Select Feature", df.columns)

        target_numeric = pd.to_numeric(df[target], errors="coerce")


        # =========================
        # NUMERIC FEATURE
        # =========================
        if pd.api.types.is_numeric_dtype(df[feature]):

            x = pd.to_numeric(df[feature], errors="coerce")

            plot_df = pd.DataFrame({
                "Feature": x,
                "Risk": target_numeric
            }).dropna()

            if len(plot_df) > 0:

                fig, ax = plt.subplots()
                ax.scatter(plot_df["Feature"], plot_df["Risk"])

                ax.set_xlabel(feature)
                ax.set_ylabel("Risk")

                st.pyplot(fig)

            else:
                st.warning("No valid numeric data")

        # =========================
        # CATEGORICAL FEATURE
        # =========================
        else:

            grouped = df.copy()
            grouped[target] = pd.to_numeric(grouped[target], errors="coerce")

            grouped = grouped.groupby(feature)[target].mean().dropna()

            if len(grouped) == 0:
                st.warning("No valid grouped data")
            else:

                grouped = grouped.reset_index()
                grouped.columns = [feature, "Risk"]

                st.bar_chart(grouped.set_index(feature))

                st.write("Highest Risk:", grouped.loc[grouped["Risk"].idxmax(), feature])
                st.write("Lowest Risk:", grouped.loc[grouped["Risk"].idxmin(), feature])


    # =========================
    # ML MODELS
    # =========================
    with tab3:

        st.subheader("Model Training")

        if st.button("Train Models"):

            try:
                results, best_name, best_model = train_models(df, target)

                st.success("Training Done")

                st.dataframe(pd.DataFrame(results).T)

                save_model(best_model)

                st.success(f"Best Model: {best_name}")

            except Exception as e:
                st.error(str(e))


    # =========================
    # CLUSTERING
    # =========================
    with tab4:

        st.subheader("K-Means Clustering")

        k = st.slider("Clusters", 2, 10, 3)

        if st.button("Run Clustering"):

            result = run_kmeans(df, k)

            st.bar_chart(result["Cluster"].value_counts())

            fig, ax = plt.subplots()
            ax.scatter(result["PCA1"], result["PCA2"], c=result["Cluster"])

            st.pyplot(fig)

            st.dataframe(result)

else:
    st.info("Upload a CSV to start")
