import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Check Imports for Hyperparameter Tuning
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler, PowerTransformer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

def main():
    st.set_page_config(page_title="Microplastic Risk Modeling", layout="wide")
    st.title("Microplastic System Risk Modeling 🌊")
    
    st.markdown("This application runs a full machine learning pipeline on microplastic data, including EDA, preprocessing, feature selection, modeling, and evaluation.")
    
    st.header("1. Data Loading and Preprocessing")
    with st.spinner("Loading data..."):
        # Creating a dummy dataframe for demonstration (Replace with real data later)
        np.random.seed(42)
        n_samples = 1000
        df = pd.DataFrame({
            'risk score': np.random.gamma(shape=2, scale=2, size=n_samples) * 10,
            'mp count per l': np.random.exponential(scale=100, size=n_samples),
            'risk level': np.random.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.5, 0.3, 0.2]),
            'Risk_Type': np.random.choice(['Type A', 'Type B'], size=n_samples, p=[0.8, 0.2]), 
            'Polymer Type': np.random.choice(['PET', 'PE', 'PP', 'PVC', 'PS'], size=n_samples)
        })
        st.write("Sample Data Overview:", df.head())

    st.header("2. Exploratory Data Analysis (EDA)")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Polymer Type")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.countplot(data=df, x='Polymer Type', palette='viridis', ax=ax1)
        st.pyplot(fig1)

        st.subheader("Distribution of Risk Score")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(df['risk score'], kde=True, bins=30, color='blue', ax=ax2)
        st.pyplot(fig2)

    with col2:
        st.subheader("Risk Score vs MP Count")
        fig3, ax3 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(data=df, x='mp count per l', y='risk score', alpha=0.6, ax=ax3)
        st.pyplot(fig3)

        st.subheader("Risk Score by Risk Level")
        fig4, ax4 = plt.subplots(figsize=(6, 4))
        sns.boxplot(data=df, x='risk level', y='risk score', palette='Set2', ax=ax4)
        st.pyplot(fig4)

    st.write("Class distribution for 'Risk_Type':")
    st.dataframe(df['Risk_Type'].value_counts(normalize=True) * 100)

    st.header("3. Data Preprocessing & Feature Selection")
    
    skewed_cols = ['risk score', 'mp count per l']
    pt = PowerTransformer(method='yeo-johnson')
    df[skewed_cols] = pt.fit_transform(df[skewed_cols])
    
    robust_scaler = RobustScaler()
    df[skewed_cols] = robust_scaler.fit_transform(df[skewed_cols])

    categorical_cols = ['risk level', 'Polymer Type']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    target_encoder = LabelEncoder()
    df_encoded['Risk_Type'] = target_encoder.fit_transform(df_encoded['Risk_Type'])
    
    X = df_encoded.drop('Risk_Type', axis=1)
    y = df_encoded['Risk_Type']

    selector = SelectKBest(score_func=mutual_info_classif, k='all')
    selector.fit(X, y)
    
    feature_relevance = pd.DataFrame({'Feature': X.columns, 'Score': selector.scores_})
    feature_relevance = feature_relevance.sort_values(by='Score', ascending=False)
    
    fig5, ax5 = plt.subplots(figsize=(8, 4))
    sns.barplot(data=feature_relevance, x='Score', y='Feature', palette='magma', ax=ax5)
    ax5.set_title("Feature Relevance for Risk_Type Prediction")
    st.pyplot(fig5)

    st.header("4. Modeling & Classification Evaluation")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42)
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_resampled, y_train_resampled)
        y_pred = model.predict(X_test_scaled)
        acc = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, model.predict_proba(X_test_scaled)[:, 1])
        results[name] = {'Accuracy': acc, 'ROC-AUC': roc_auc}
    
    results_df = pd.DataFrame(results).T
    st.write("Initial Model Performance (Trained with SMOTE-resampled data):")
    st.dataframe(results_df)

    st.header("5. Hyperparameter Tuning (Logistic Regression)")
    with st.spinner("Tuning hyperparameters..."):
        param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2']}
        grid_search = GridSearchCV(LogisticRegression(random_state=42, max_iter=1000), param_grid, cv=5, scoring='roc_auc')
        grid_search.fit(X_train_resampled, y_train_resampled)
        
        best_log_reg = grid_search.best_estimator_
        y_pred_tuned = best_log_reg.predict(X_test_scaled)
        y_prob_tuned = best_log_reg.predict_proba(X_test_scaled)[:, 1]

    st.write(f"**Best Parameters:** {grid_search.best_params_}")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred_tuned):.4f}")
    st.write(f"**ROC-AUC:** {roc_auc_score(y_test, y_prob_tuned):.4f}")

    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Confusion Matrix")
        fig6, ax6 = plt.subplots(figsize=(5, 4))
        cm = confusion_matrix(y_test, y_pred_tuned)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_encoder.classes_, yticklabels=target_encoder.classes_, ax=ax6)
        st.pyplot(fig6)

    with col4:
        st.subheader("ROC Curve")
        fig7, ax7 = plt.subplots(figsize=(5, 4))
        fpr, tpr, _ = roc_curve(y_test, y_prob_tuned)
        ax7.plot(fpr, tpr, color='orange', label=f'AUC = {roc_auc_score(y_test, y_prob_tuned):.2f}')
        ax7.plot([0, 1], [0, 1], color='navy', linestyle='--')
        ax7.legend(loc='lower right')
        st.pyplot(fig7)

if __name__ == "__main__":
    main()
