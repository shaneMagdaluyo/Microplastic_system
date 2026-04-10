import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler, PowerTransformer, LabelEncoder
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, roc_curve

from imblearn.over_sampling import SMOTE

sns.set_style("whitegrid")


# -----------------------------
# OUTLIER HANDLING
# -----------------------------
def cap_outliers(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1

    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    df[col] = np.clip(df[col], lower, upper)
    return df


# -----------------------------
# MAIN APP
# -----------------------------
def main():
    st.set_page_config(page_title="Microplastic ML Pipeline", layout="wide")
    st.title("🌊 Microplastic Risk ML Pipeline")

    # =========================
    # LOAD DATA (dummy)
    # =========================
    np.random.seed(42)
    n = 1000

    df = pd.DataFrame({
        "risk score": np.random.gamma(2, 5, n),
        "mp count per l": np.random.exponential(80, n),
        "risk level": np.random.choice(["Low", "Medium", "High"], n),
        "Polymer Type": np.random.choice(["PET", "PE", "PP", "PVC"], n),
        "Risk_Type": np.random.choice(["Type A", "Type B"], n, p=[0.8, 0.2])
    })

    st.subheader("Raw Data")
    st.dataframe(df.head())

    # =========================
    # OUTLIERS
    # =========================
    st.header("1. Outlier Handling")
    for col in ["risk score", "mp count per l"]:
        df = cap_outliers(df, col)

    st.success("Outliers capped using IQR method")

    # =========================
    # SKEW TRANSFORM
    # =========================
    st.header("2. Skew Transformation")

    pt = PowerTransformer(method="yeo-johnson")
    df[["risk score", "mp count per l"]] = pt.fit_transform(
        df[["risk score", "mp count per l"]]
    )

    st.success("Skewness reduced using PowerTransformer")

    # =========================
    # ENCODING
    # =========================
    st.header("3. Encoding")

    le = LabelEncoder()
    df["risk level"] = le.fit_transform(df["risk level"])
    df["Risk_Type"] = le.fit_transform(df["Risk_Type"])

    df = pd.get_dummies(df, columns=["Polymer Type"], drop_first=True)

    # =========================
    # SCALING
    # =========================
    st.header("4. Scaling")

    scaler = RobustScaler()
    scale_cols = ["risk score", "mp count per l"]
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # =========================
    # EDA
    # =========================
    st.header("5. EDA")

    fig1, ax1 = plt.subplots()
    sns.histplot(df["risk score"], kde=True, ax=ax1)
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df["mp count per l"], y=df["risk score"], ax=ax2)
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    sns.boxplot(x=df["risk level"], y=df["risk score"], ax=ax3)
    st.pyplot(fig3)

    # =========================
    # FEATURES
    # =========================
    st.header("6. Feature Selection")

    X = df.drop("Risk_Type", axis=1)
    y = df["Risk_Type"]

    selector = SelectKBest(mutual_info_classif, k="all")
    selector.fit(X, y)

    feat = pd.DataFrame({
        "Feature": X.columns,
        "Score": selector.scores_
    }).sort_values("Score", ascending=False)

    fig4, ax4 = plt.subplots()
    sns.barplot(data=feat, y="Feature", x="Score", ax=ax4)
    st.pyplot(fig4)

    # =========================
    # MODELING
    # =========================
    st.header("7. Modeling")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    smote = SMOTE()
    X_train, y_train = smote.fit_resample(X_train, y_train)

    models = {
        "LogReg": LogisticRegression(max_iter=1000),
        "RF": RandomForestClassifier(),
        "GB": GradientBoostingClassifier()
    }

    results = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)

        results[name] = {
            "Accuracy": accuracy_score(y_test, pred),
            "ROC_AUC": roc_auc_score(y_test, pred)
        }

    st.dataframe(pd.DataFrame(results).T)

    # =========================
    # TUNING
    # =========================
    st.header("8. Hyperparameter Tuning")

    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        {"C": [0.01, 0.1, 1, 10]},
        cv=5,
        scoring="roc_auc"
    )

    grid.fit(X_train, y_train)

    st.write("Best Params:", grid.best_params_)

    best = grid.best_estimator_
    pred = best.predict(X_test)

    cm = confusion_matrix(y_test, pred)

    fig5, ax5 = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax5)
    st.pyplot(fig5)

    # =========================
    # SUMMARY
    # =========================
    st.header("📌 Summary")

    st.markdown("""
- Data was cleaned using IQR outlier capping
- Skewness reduced using PowerTransformer
- Categorical variables encoded successfully
- Features scaled using RobustScaler
- SMOTE applied to handle imbalance
- Multiple models trained and compared
- Logistic Regression tuned using GridSearchCV
- Feature importance visualized using mutual information
""")


if __name__ == "__main__":
    main()
