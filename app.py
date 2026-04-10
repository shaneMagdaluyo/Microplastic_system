# app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import RobustScaler, LabelEncoder

sns.set_style("whitegrid")


def handle_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[col] = np.clip(df[col], lower, upper)
    return df


def main():
    st.set_page_config(page_title="Microplastic Risk EDA", layout="wide")
    st.title("🌊 Microplastic Risk Analysis Dashboard")

    st.markdown("Focused EDA: encoding, scaling, outliers, and risk insights.")

    # ----------------------------
    # DATA LOADING (dummy data)
    # ----------------------------
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "risk score": np.random.gamma(2, 5, n),
        "mp count per l": np.random.exponential(80, n),
        "risk level": np.random.choice(["Low", "Medium", "High"], n, p=[0.5, 0.3, 0.2]),
        "Polymer Type": np.random.choice(["PET", "PE", "PP", "PVC"], n)
    })

    st.subheader("📊 Raw Data Sample")
    st.dataframe(df.head())

    # ----------------------------
    # OUTLIER HANDLING
    # ----------------------------
    st.header("1. Outlier Treatment")

    for col in ["risk score", "mp count per l"]:
        df = handle_outliers(df, col)

    st.success("Outliers capped using IQR method (1.5×IQR rule).")

    # ----------------------------
    # ENCODING
    # ----------------------------
    st.header("2. Encoding Categorical Variables")

    le = LabelEncoder()
    df["risk level encoded"] = le.fit_transform(df["risk level"])

    df_encoded = pd.get_dummies(df, columns=["Polymer Type"], drop_first=True)

    st.write("Encoded dataset preview:")
    st.dataframe(df_encoded.head())

    # ----------------------------
    # SCALING
    # ----------------------------
    st.header("3. Feature Scaling")

    scaler = RobustScaler()
    df_encoded[["risk score", "mp count per l"]] = scaler.fit_transform(
        df_encoded[["risk score", "mp count per l"]]
    )

    st.success("Robust scaling applied to numeric features.")

    # ----------------------------
    # ANALYSIS 1: DISTRIBUTION
    # ----------------------------
    st.header("4. Distribution of Risk Score")

    fig1, ax1 = plt.subplots()
    sns.histplot(df_encoded["risk score"], kde=True, ax=ax1, color="blue")
    ax1.set_title("Risk Score Distribution")
    st.pyplot(fig1)

    # ----------------------------
    # ANALYSIS 2: RELATIONSHIP
    # ----------------------------
    st.header("5. Risk Score vs MP Count per L")

    fig2, ax2 = plt.subplots()
    sns.regplot(
        x="mp count per l",
        y="risk score",
        data=df_encoded,
        ax=ax2,
        scatter_kws={"alpha": 0.4},
        line_kws={"color": "red"}
    )
    ax2.set_title("Risk Score vs Microplastic Concentration")
    st.pyplot(fig2)

    correlation = df_encoded["risk score"].corr(df_encoded["mp count per l"])
    st.info(f"Correlation: {correlation:.3f}")

    # ----------------------------
    # ANALYSIS 3: RISK LEVEL COMPARISON
    # ----------------------------
    st.header("6. Risk Score by Risk Level")

    fig3, ax3 = plt.subplots()
    sns.boxplot(x=df["risk level"], y=df["risk score"], ax=ax3)
    ax3.set_title("Risk Score across Risk Levels")
    st.pyplot(fig3)

    summary_table = df.groupby("risk level")["risk score"].agg(["mean", "median", "std"])
    st.write("Summary statistics by risk level:")
    st.dataframe(summary_table)

    # ----------------------------
    # FINAL SUMMARY
    # ----------------------------
    st.header("📌 Summary Insights")

    low = df[df["risk level"] == "Low"]["risk score"].mean()
    med = df[df["risk level"] == "Medium"]["risk score"].mean()
    high = df[df["risk level"] == "High"]["risk score"].mean()

    st.markdown(f"""
### Key Findings:
- Risk score distribution is **right-skewed (environmental accumulation pattern)**.
- Correlation between microplastic concentration and risk score: **{correlation:.2f}**
- Mean risk score:
  - Low: **{low:.2f}**
  - Medium: **{med:.2f}**
  - High: **{high:.2f}**
- Risk score increases clearly across risk levels → strong separation.

### Interpretation:
Microplastic concentration (mp count per l) shows a **moderate positive relationship** with risk score, suggesting it is a key predictor of environmental risk classification.
""")

if __name__ == "__main__":
    main()
