"""
Microplastic Risk Analysis Dashboard
A comprehensive Streamlit application for analyzing microplastic risk data,
featuring data preprocessing, EDA, model training, cross validation, and model comparison.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report)
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
import time

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Microplastic Risk Analysis Dashboard",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-top: 1rem; margin-bottom: 1rem; }
    .subsection-header { font-size: 1.4rem; font-weight: 500; color: #34495e; margin-top: 0.8rem; }
    .stButton > button { width: 100%; background-color: #1f77b4; color: white; font-weight: 600; border-radius: 8px; padding: 0.5rem 1rem; }
    .stButton > button:hover { background-color: #2980b9; border-color: #2980b9; }
    .explanation-box { background-color: #f0f8ff; border-left: 4px solid #1f77b4; padding: 15px; margin: 15px 0; border-radius: 5px; }
    .result-box { background-color: #f9f9f9; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 5px; }
    .metric-explain { font-size: 0.9rem; color: #666; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if 'data' not in st.session_state: st.session_state.data = None
    if 'processed_data' not in st.session_state: st.session_state.processed_data = None
    if 'models' not in st.session_state: st.session_state.models = {}
    if 'comparison_results' not in st.session_state: st.session_state.comparison_results = None
    if 'feature_importance' not in st.session_state: st.session_state.feature_importance = None
    if 'mutual_info' not in st.session_state: st.session_state.mutual_info = None
    if 'chi2_scores' not in st.session_state: st.session_state.chi2_scores = None
    if 'X_selected' not in st.session_state: st.session_state.X_selected = None
    if 'best_model' not in st.session_state: st.session_state.best_model = None
    if 'preprocessing_log' not in st.session_state: st.session_state.preprocessing_log = []
    if 'trained' not in st.session_state: st.session_state.trained = False
    if 'X_test' not in st.session_state: st.session_state.X_test = None
    if 'y_test' not in st.session_state: st.session_state.y_test = None
    if 'X_train' not in st.session_state: st.session_state.X_train = None
    if 'y_train' not in st.session_state: st.session_state.y_train = None
    if 'encoders' not in st.session_state: st.session_state.encoders = {}
    if 'scaler' not in st.session_state: st.session_state.scaler = None
    if 'target_encoder' not in st.session_state: st.session_state.target_encoder = None
    if 'selected_features' not in st.session_state: st.session_state.selected_features = None
    if 'scaled_columns' not in st.session_state: st.session_state.scaled_columns = None
    if 'scaled_data' not in st.session_state: st.session_state.scaled_data = None
    if 'encoded_data' not in st.session_state: st.session_state.encoded_data = None
    if 'encoded_shape' not in st.session_state: st.session_state.encoded_shape = None

init_session_state()

# [KEEP ALL YOUR EXISTING FUNCTIONS: load_dataset, generate_sample_data, handle_missing_values, 
#  cap_outliers_iqr, encode_categorical, one_hot_encode, scale_features, detect_outliers,
#  analyze_skewness, apply_log_transform, calculate_mutual_info, calculate_chi2, 
#  calculate_rf_importance, train_and_evaluate_for_target, plot_distribution, plot_correlation_heatmap]

def train_and_evaluate_detailed(df, target_col):
    """Train models for a specific target and return detailed metrics with precision and recall."""
    feature_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
    if target_col in feature_cols: feature_cols.remove(target_col)
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    mask = y.notna()
    X = X[mask]; y = y[mask]
    if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
    X = X.fillna(X.median())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    data_split_info = {
        'X_train_shape': X_train.shape,
        'X_test_shape': X_test.shape,
        'y_train_shape': y_train.shape,
        'y_test_shape': y_test.shape,
        'target_col': target_col,
        'total_samples': len(df),
        'train_pct': round(len(X_train)/len(X)*100, 1),
        'test_pct': round(len(X_test)/len(X)*100, 1)
    }
    
    models = {}
    
    try:
        lr = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', n_jobs=-1)
        lr.fit(X_train, y_train)
        models['Logistic Regression'] = lr
    except: pass
    
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
        rf.fit(X_train, y_train)
        models['RandomForestClassifier'] = rf
    except: pass
    
    try:
        gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
        gb.fit(X_train, y_train)
        models['GradientBoostingClassifier'] = gb
    except: pass
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
    
    return results, data_split_info

def main():
    """Main application function."""
    
    st.markdown('<p class="main-header">🔬 Microplastic Risk Analysis Dashboard</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio("Select Section", [
        "🏠 Home", "🔧 Preprocessing", "🛠️ Feature Selection & Relevance", 
        "🤖 Modeling", "📊 Cross Validation & Evaluation"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard analyzes microplastic risk data to predict risk types and identify key factors.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Status")
    if st.session_state.data is not None: st.sidebar.success("✅ Data Loaded")
    else: st.sidebar.warning("⚠️ No Data")
    if st.session_state.trained: st.sidebar.success(f"✅ Models Trained ({len(st.session_state.models)})")
    else: st.sidebar.warning("⚠️ Models Not Trained")
    
    # ==================== HOME ====================
    if section == "🏠 Home":
        st.markdown('<p class="section-header">🏠 Home - Upload Dataset</p>', unsafe_allow_html=True)
        
        # Explanation of this section
        with st.expander("ℹ️ About this section", expanded=False):
            st.markdown("""
            **Purpose of the Home Page:**
            This is the starting point of your analysis. Here you can:
            - **Upload your dataset** (CSV or Excel format) containing microplastic risk data
            - **Generate sample data** to explore the dashboard's capabilities
            - **Preview your data** to understand its structure
            - **Apply initial scaling** to see how StandardScaler transforms numerical columns
            
            **Why this matters:** Before any analysis, you need to load and understand your data.
            The preview helps you identify column types, missing values, and basic statistics.
            """)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            f = st.file_uploader("Upload dataset (CSV/Excel)", type=['csv','xlsx','xls'])
            if f: load_dataset(f)
        with c2:
            st.markdown("#### Quick Start")
            if st.button("Generate Sample Dataset", type="primary"):
                st.session_state.data = generate_sample_data()
                st.success("✅ Sample dataset generated!")
                st.rerun()
        
        if st.session_state.data is not None:
            df = st.session_state.data
            st.markdown("---")
            st.markdown('<p class="subsection-header">📋 Dataset Preview</p>', unsafe_allow_html=True)
            
            st.markdown("""
            <div class="explanation-box">
            <b>📖 Understanding the Dataset Preview:</b><br>
            - <b>Samples:</b> The number of rows (observations) in your dataset<br>
            - <b>Features:</b> The number of columns (variables) available for analysis<br>
            - <b>Missing Values:</b> Gaps in data that need to be addressed during preprocessing
            </div>
            """, unsafe_allow_html=True)
            
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Samples", df.shape[0])
            with c2: st.metric("Features", df.shape[1])
            with c3: st.metric("Missing", df.isnull().sum().sum())
            st.dataframe(df.head(10), use_container_width=True)
            
            # Feature Scaling
            st.markdown("---")
            st.markdown("### 📏 Feature Scaling Preview")
            
            st.markdown("""
            <div class="explanation-box">
            <b>📖 Why Feature Scaling?</b><br>
            Feature scaling (StandardScaler) transforms numerical columns to have <b>mean=0</b> and <b>standard deviation=1</b>.
            This is essential because:<br>
            • Machine learning algorithms perform better when features are on the same scale<br>
            • Features with larger values don't dominate the model's learning process<br>
            • It helps gradient-based algorithms converge faster
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🔧 Apply StandardScaler", type="primary", key="scale_home"):
                with st.spinner('Scaling...'):
                    nums = df.select_dtypes(include=['float64','int64']).columns.tolist()
                    cols = [c for c in nums if 'ID' not in c and 'Sample' not in c]
                    if len(cols) > 0:
                        scaler = StandardScaler()
                        sd = scaler.fit_transform(df[cols].fillna(df[cols].median()))
                        sdf = pd.DataFrame(sd, columns=cols)
                        st.session_state.scaler = scaler
                        st.success(f"✅ {len(cols)} columns scaled! Mean=0, Std=1")
                        st.dataframe(sdf.head(), column_config={c: st.column_config.NumberColumn(c,format="%.6f") for c in cols}, use_container_width=True)
            
            # Risk Score vs MP Count
            if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
                st.markdown("---")
                st.markdown("### 🔬 Risk Score vs MP Count per L")
                
                st.markdown("""
                <div class="explanation-box">
                <b>📖 Why analyze this relationship?</b><br>
                This scatter plot explores the relationship between <b>Microplastic Count per Liter</b> and <b>Risk Score</b>.
                Understanding this relationship helps determine:<br>
                • Whether higher microplastic concentrations correlate with higher risk scores<br>
                • If MP Count is a strong predictor of Risk Score<br>
                • The nature of the relationship (linear, non-linear, or no correlation)
                </div>
                """, unsafe_allow_html=True)
                
                df['MP_Count_per_L'] = pd.to_numeric(df['MP_Count_per_L'], errors='coerce')
                df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                clean = df.dropna(subset=['MP_Count_per_L','Risk_Score'])
                if len(clean) > 0:
                    tab1, tab2 = st.tabs(["📊 Scatter", "📈 Trendline"])
                    with tab1:
                        fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                        color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                        title='MP Count vs Risk Score', opacity=0.7)
                        st.plotly_chart(fig, use_container_width=True)
                    with tab2:
                        try:
                            fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                            color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                            trendline='ols', title='MP Count vs Risk Score with Trendline', opacity=0.7)
                            st.plotly_chart(fig, use_container_width=True)
                        except: st.warning("⚠️ Trendline not available")
            
            # Risk Score by Risk Level
            if 'Risk_Score' in df.columns and 'Risk_Level' in df.columns:
                st.markdown("---")
                st.markdown("### 📊 Risk Score by Risk Level")
                
                st.markdown("""
                <div class="explanation-box">
                <b>📖 Why investigate Risk Score by Risk Level?</b><br>
                Box plots show the distribution of Risk Scores across different Risk Level categories.
                This analysis reveals:<br>
                • Whether different risk levels have distinct risk score ranges<br>
                • The spread and central tendency of scores within each level<br>
                • Potential outliers that may need attention during preprocessing
                </div>
                """, unsafe_allow_html=True)
                
                clean = df.dropna(subset=['Risk_Score'])
                clean['Risk_Level'] = clean['Risk_Level'].astype(str)
                if len(clean) > 0:
                    tab1, tab2 = st.tabs(["📦 Box Plot", "📊 Stats"])
                    with tab1:
                        fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                                    title='Risk Score by Risk Level', points='outliers')
                        st.plotly_chart(fig, use_container_width=True)
                    with tab2:
                        stats = clean.groupby('Risk_Level')['Risk_Score'].agg(['count','mean','median','std','min','max']).round(2)
                        stats.columns = ['Count','Mean','Median','Std Dev','Min','Max']
                        st.dataframe(stats, use_container_width=True)
            
            # Quality Check
            st.markdown("---")
            st.markdown("### 🔍 Data Quality Check")
            st.markdown("""
            <div class="explanation-box">
            <b>📖 Why check data quality?</b><br>
            Before analysis, it's crucial to assess data quality:<br>
            • <b>Missing Data:</b> High percentages may require imputation or removal<br>
            • <b>Duplicates:</b> Redundant rows can bias analysis<br>
            • <b>Column Types:</b> Understanding numeric vs categorical helps plan preprocessing
            </div>
            """, unsafe_allow_html=True)
            
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Missing %", f"{(df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%")
            with c2: st.metric("Duplicates", df.duplicated().sum())
            with c3: st.metric("Numeric Cols", len(df.select_dtypes(include=['float64','int64']).columns))
            with c4: st.metric("Categorical Cols", len(df.select_dtypes(include=['object']).columns))
    
    # ==================== PREPROCESSING ====================
    elif section == "🔧 Preprocessing":
        st.markdown('<p class="section-header">🔧 Data Preprocessing</p>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ About Preprocessing", expanded=False):
            st.markdown("""
            **Purpose of Preprocessing:**
            Data preprocessing transforms raw data into a clean, structured format suitable for machine learning.
            
            **Key Steps:**
            1. **Feature Scaling**: Standardizes numerical values to mean=0, std=1
            2. **Categorical Encoding**: Converts text categories to numerical format
            3. **Outlier Capping**: Limits extreme values to reduce their impact
            4. **Skewness Transformation**: Normalizes skewed distributions
            
            **Why this matters:** Machine learning models require numerical, well-scaled data.
            Without preprocessing, models may perform poorly or fail entirely.
            """)
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            return
        
        df = st.session_state.data.copy()
        
        prep_tab1, prep_tab2, prep_tab3, prep_tab4, prep_tab5 = st.tabs([
            "📏 Feature Scaling", "🔄 Categorical Encoding", "🎯 Outlier Capping", 
            "📊 Skewness & Transform", "📋 Summary & Next Steps"
        ])
        
        with prep_tab1:
            st.markdown("### 📏 Perform Feature Scaling")
            st.markdown("""
            <div class="explanation-box">
            <b>📖 What is Feature Scaling?</b><br>
            <b>StandardScaler</b> transforms each numerical feature to have:<br>
            • <b>Mean = 0</b>: Centers the data around zero<br>
            • <b>Standard Deviation = 1</b>: Ensures all features have the same spread<br><br>
            <b>Why is this necessary?</b><br>
            • Prevents features with larger ranges from dominating the model<br>
            • Essential for distance-based algorithms (SVM, KNN)<br>
            • Helps gradient descent converge faster in neural networks and logistic regression
            </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_to_scale = [col for col in numeric_cols if 'ID' not in col and 'Sample' not in col]
            
            if st.button("🔧 Apply Feature Scaling (StandardScaler)", type="primary", key="scale_tab"):
                with st.spinner('Applying StandardScaler...'):
                    if len(cols_to_scale) > 0:
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[cols_to_scale].fillna(df[cols_to_scale].median()))
                        scaled_df = pd.DataFrame(scaled_data, columns=cols_to_scale)
                        st.session_state.scaler = scaler
                        st.session_state.scaled_data = scaled_df
                        st.success(f"✅ Numerical columns scaled successfully! Mean=0, Std=1")
                        st.markdown("**First 5 rows of scaled numerical data (Mean≈0, Std≈1):**")
                        st.dataframe(scaled_df.head(), column_config={col: st.column_config.NumberColumn(col, format="%.6f") for col in cols_to_scale}, use_container_width=True)
        
        # [KEEP YOUR EXISTING prep_tab2, prep_tab3, prep_tab4, prep_tab5 CODE with similar explanations added]
        
        with prep_tab2:
            st.markdown("### 🔄 Encode Categorical Variables")
            st.markdown("""
            <div class="explanation-box">
            <b>📖 What is Categorical Encoding?</b><br>
            <b>One-Hot Encoding</b> creates binary (0/1) columns for each category in a categorical variable.<br><br>
            <b>Why is this necessary?</b><br>
            • Machine learning models require numerical input<br>
            • One-hot encoding prevents the model from assuming ordinal relationships<br>
            • Each category becomes an independent feature for the model to learn from<br>
            <b>Note:</b> This significantly increases the number of columns in your dataset.
            </div>
            """, unsafe_allow_html=True)
            
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            cols_to_encode = [col for col in categorical_cols if 'ID' not in col and 'Sample' not in col]
            
            if len(cols_to_encode) > 0:
                st.markdown(f"**Categorical columns identified ({len(cols_to_encode)}):** {', '.join(cols_to_encode)}")
            
            if st.button("🔄 Apply One-Hot Encoding", type="primary", key="encode_tab"):
                with st.spinner('Applying One-Hot Encoding...'):
                    if len(cols_to_encode) > 0:
                        encoded_df, new_cols, original_cols, encoded_shape = one_hot_encode(df)
                        st.session_state.encoded_data = encoded_df
                        st.session_state.encoded_shape = encoded_shape
                        st.success(f"✅ One-Hot Encoding applied! Created {len(new_cols)} new columns.")
                        st.markdown(f"**Original shape:** {df.shape} → **Encoded shape:** {encoded_shape}")
                        st.markdown("**First 5 rows of the DataFrame after one-hot encoding:**")
                        st.dataframe(encoded_df.head(), use_container_width=True)
        
        # [Keep prep_tab3, prep_tab4, prep_tab5 with similar explanation patterns]
        with prep_tab3:
            st.markdown("### 🎯 Address Outliers")
            st.markdown("""
            <div class="explanation-box">
            <b>📖 What is Outlier Capping?</b><br>
            Outliers are extreme values that can significantly skew statistical analysis and model training.<br>
            <b>IQR Method:</b> Values beyond Q1-1.5×IQR and Q3+1.5×IQR are capped to these bounds.<br><br>
            <b>Why cap outliers?</b><br>
            • Prevents extreme values from distorting model training<br>
            • Reduces the influence of measurement errors or rare events<br>
            • Improves model stability and generalization
            </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_for_outliers = [col for col in numeric_cols if 'ID' not in col and 'Sample' not in col]
            if len(cols_for_outliers) > 0:
                outlier_info = detect_outliers(df, cols_for_outliers)
                outlier_summary = [{'Column':col,'Outliers':info['count'],'Percentage':f"{info['percentage']:.1f}%"} for col,info in outlier_info.items()]
                st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True, hide_index=True)
            
            if st.button("🎯 Cap Outliers (IQR Method)", type="primary", key="outlier_tab"):
                with st.spinner('Capping outliers...'):
                    if len(cols_for_outliers) > 0:
                        df_capped, cap_logs = cap_outliers_iqr(df, cols_for_outliers)
                        st.session_state.processed_data = df_capped
                        st.success(f"✅ Outliers capped!")
                        for log in cap_logs: st.write(f"- {log}")
        
        with prep_tab4:
            st.markdown("### 📊 Skewness Analysis & Log Transformation")
            st.markdown("""
            <div class="explanation-box">
            <b>📖 What is Skewness?</b><br>
            Skewness measures the asymmetry of a distribution. A skewed distribution has a long tail on one side.<br>
            • <b>|Skewness| > 0.5</b>: Significantly skewed, may benefit from transformation<br>
            • <b>Log Transformation</b>: Reduces right-skewness by compressing large values<br><br>
            <b>Why transform skewed data?</b><br>
            • Many ML models assume normally distributed features<br>
            • Reduces the impact of extreme values<br>
            • Can improve model performance and interpretability
            </div>
            """, unsafe_allow_html=True)
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_for_skew = [col for col in numeric_cols if 'ID' not in col and 'Sample' not in col]
            if len(cols_for_skew) > 0:
                skew_df = analyze_skewness(df, cols_for_skew)
                st.dataframe(skew_df, use_container_width=True, hide_index=True)
            
            if st.button("📊 Apply Log Transformation", type="primary", key="skew_tab"):
                with st.spinner('Applying log transformation...'):
                    if len(cols_for_skew) > 0:
                        df_transformed, transform_logs = apply_log_transform(df, cols_for_skew)
                        st.session_state.processed_data = df_transformed
                        st.success(f"✅ Log transformation applied!")
                        for log in transform_logs: st.write(f"- {log}")
        
        with prep_tab5:
            st.markdown("### 📋 Summary & Next Steps")
            actions = []
            if st.session_state.get('scaled_data') is not None:
                actions.append("✅ **Feature Scaling**: Numerical columns scaled using StandardScaler.")
            if st.session_state.get('encoded_data') is not None:
                shape = st.session_state.get('encoded_shape', 'N/A')
                actions.append(f"✅ **Categorical Encoding**: One-hot encoding applied. Shape: {shape}")
            if st.session_state.get('processed_data') is not None:
                actions.append("✅ **Outlier Capping**: Outliers addressed using IQR method.")
            
            if len(actions) == 0:
                st.info("Run preprocessing steps to see the summary.")
            else:
                for action in actions: st.markdown(action)
                st.markdown("---")
                st.markdown("""
                ### 🚀 Next Steps
                The preprocessing steps have prepared the dataset for subsequent modeling by:
                - Handling outliers
                - Scaling numerical features  
                - Encoding categorical variables
                - Transforming skewed distributions
                
                **The dataset is now ready for model training and evaluation.**
                Proceed to **🛠️ Feature Selection & Relevance** for EDA and feature importance analysis.
                """)
    
    # ==================== FEATURE SELECTION & RELEVANCE ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ About Feature Selection", expanded=False):
            st.markdown("""
            **Purpose of Feature Selection:**
            Feature selection identifies the most important variables for predicting the target.
            
            **Three Methods Used:**
            1. **Mutual Information**: Measures the dependency between features and target
            2. **Chi-squared Test**: Tests the independence between categorical features and target
            3. **Random Forest Importance**: Uses tree-based model to rank feature importance
            
            **Why this matters:** Selecting the right features reduces noise, speeds up training,
            and improves model interpretability.
            """)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data.copy()
        
        st.markdown("### 📈 Exploratory Data Analysis")
        
        # [KEEP YOUR EXISTING EDA CODE with similar explanation patterns]
        
        st.markdown("---")
        st.markdown("### 🎯 Feature Selection Methods")
        
        st.markdown("""
        <div class="explanation-box">
        <b>📖 Understanding Feature Selection Methods:</b><br>
        • <b>Mutual Information:</b> Measures how much knowing a feature reduces uncertainty about the target<br>
        • <b>Chi-squared Test:</b> Tests if there's a statistically significant relationship between feature and target<br>
        • <b>Random Forest Importance:</b> Shows how much each feature contributes to accurate predictions<br><br>
        <b>Higher scores = More important features</b>
        </div>
        """, unsafe_allow_html=True)
        
        target_col = st.selectbox("Select Target Variable", df.columns.tolist(),
                                  index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        if target_col in numeric_cols: numeric_cols.remove(target_col)
        
        if st.button("Calculate All Feature Importance Metrics", type="primary", use_container_width=True):
            with st.spinner('Calculating...'):
                X = df[numeric_cols].copy()
                y = df[target_col].copy()
                X = X.fillna(X.median())
                if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
                X = X.dropna(axis=1, how='any')
                
                mi_df = calculate_mutual_info(X, y)
                chi2_df = calculate_chi2(X, y)
                rf_df = calculate_rf_importance(X, y)
                
                st.session_state.feature_importance = rf_df
                st.session_state.mutual_info = mi_df
                st.session_state.chi2_scores = chi2_df
                
                X_selected = X[mi_df.head(20)['Feature'].tolist()]
                st.session_state.X_selected = X_selected
                st.session_state.selected_features = rf_df.head(10)['Feature'].tolist()
                
                ft1, ft2, ft3 = st.tabs(["🌲 Random Forest", "📊 Mutual Information", "🔢 Chi-squared"])
                
                with ft1:
                    st.markdown("**Top 20 features - RandomForest Feature Importances:**")
                    st.markdown("*Higher importance means the feature contributes more to accurate predictions*")
                    top20_rf = rf_df.head(20)
                    fig_rf = px.bar(top20_rf, x='Importance', y='Feature', orientation='h',
                                   title='Top 20 Features - Random Forest', height=500)
                    st.plotly_chart(fig_rf, use_container_width=True)
                
                with ft2:
                    st.markdown("**Top 20 features - Mutual Information:**")
                    st.markdown("*Higher Mutual Information means the feature has stronger dependency with the target*")
                    top20_mi = mi_df.head(20)
                    fig_mi = px.bar(top20_mi, x='Mutual_Info', y='Feature', orientation='h',
                                   title='Top 20 Features - Mutual Information', height=500)
                    st.plotly_chart(fig_mi, use_container_width=True)
                
                with ft3:
                    st.markdown("**Top 20 features - Chi-squared Test:**")
                    st.markdown("*Higher Chi2 score indicates stronger statistical relationship with the target*")
                    top20_chi2 = chi2_df.head(20)
                    fig_chi2 = px.bar(top20_chi2, x='Chi2_Score', y='Feature', orientation='h',
                                     title='Top 20 Features - Chi-squared Test', height=500)
                    st.plotly_chart(fig_chi2, use_container_width=True)
                
                st.success(f"✅ Feature selection completed!")
    
    # ==================== MODELING ====================
    elif section == "🤖 Modeling":
        st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)
        
        # [KEEP YOUR EXISTING MODELING CODE]
    
    # ==================== CROSS VALIDATION & EVALUATION ====================
    elif section == "📊 Cross Validation & Evaluation":
        st.markdown('<p class="section-header">📊 Cross Validation & Model Evaluation</p>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ About Model Evaluation", expanded=False):
            st.markdown("""
            **Purpose of Model Evaluation:**
            After training models, we need to evaluate how well they perform on unseen data.
            
            **Key Metrics Explained:**
            - **Accuracy**: Overall percentage of correct predictions
            - **Precision**: How many predicted positives are actually positive (minimizes false alarms)
            - **Recall**: How many actual positives were correctly identified (minimizes missed cases)
            - **F1-Score**: Harmonic mean of Precision and Recall (balanced measure)
            
            **Cross Validation**: Tests model stability by training on different data splits
            
            **Why this matters:** These metrics help you choose the best model for your specific needs.
            """)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data.copy()
        
        eval_tab1, eval_tab2, eval_tab3 = st.tabs([
            "📊 Evaluate Models", 
            "📊 Compare Both Targets",
            "🔄 Cross Validation"
        ])
        
        # ===== TAB 1: EVALUATE MODELS =====
        with eval_tab1:
            st.markdown("### 📊 Evaluate the Models")
            st.markdown("""
            <div class="explanation-box">
            <b>📖 What this section does:</b><br>
            • <b>Trains models</b> on the training data (80% of dataset)<br>
            • <b>Evaluates performance</b> on the testing data (20% held out)<br>
            • <b>Calculates metrics</b> using 'weighted' averaging for multi-class problems<br><br>
            <b>Weighted averaging</b> accounts for class imbalance by weighting each class's metric by its support (number of samples).
            </div>
            """, unsafe_allow_html=True)
            
            target_col = 'Risk_Type'
            if target_col not in df.columns:
                st.error(f"❌ '{target_col}' column not found!")
            else:
                if st.button("🚀 Evaluate Models", type="primary", key="eval_detail"):
                    with st.spinner('Training and evaluating models...'):
                        results, split_info = train_and_evaluate_detailed(df, target_col)
                    
                    if results:
                        # Data split info
                        st.markdown("---")
                        st.markdown("### 📊 Data Split Information")
                        st.markdown("""
                        <div class="explanation-box">
                        <b>📖 Understanding the Data Split:</b><br>
                        • <b>Training Set (80%)</b>: Used to teach the model patterns in the data<br>
                        • <b>Testing Set (20%)</b>: Used to evaluate how well the model generalizes to unseen data<br>
                        • This split prevents <b>overfitting</b> - where a model memorizes training data but fails on new data
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div style="background: #e8f4fd; border: 2px solid #1f77b4; border-radius: 10px; padding: 20px; margin: 15px 0;">
                            <p style="margin: 5px 0;"><b>Target Variable:</b> {split_info['target_col']}</p>
                            <p style="margin: 5px 0;"><b>X_train shape:</b> {split_info['X_train_shape']} (Training features)</p>
                            <p style="margin: 5px 0;"><b>X_test shape:</b> {split_info['X_test_shape']} (Testing features)</p>
                            <p style="margin: 5px 0;"><b>y_train shape:</b> {split_info['y_train_shape']} (Training labels)</p>
                            <p style="margin: 5px 0;"><b>y_test shape:</b> {split_info['y_test_shape']} (Testing labels)</p>
                            <p style="margin: 5px 0;"><b>Total samples:</b> {split_info['total_samples']} | <b>Train:</b> {split_info['train_pct']}% | <b>Test:</b> {split_info['test_pct']}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Evaluate each model individually
                        st.markdown("---")
                        
                        # Evaluate Logistic Regression
                        if 'Logistic Regression' in results:
                            res = results['Logistic Regression']
                            st.markdown("### # Evaluate Logistic Regression Model")
                            st.markdown("*Logistic Regression is a linear model that estimates probabilities using a logistic function. It works well when features have a linear relationship with the target.*")
                            st.markdown(f"**Accuracy:** {res['accuracy']:.4f} <span class='metric-explain'>- Overall correct predictions</span>")
                            st.markdown(f"**Precision:** {res['precision']:.4f} <span class='metric-explain'>- How many predicted positives were correct</span>")
                            st.markdown(f"**Recall:** {res['recall']:.4f} <span class='metric-explain'>- How many actual positives were found</span>")
                            st.markdown(f"**F1-Score:** {res['f1_score']:.4f} <span class='metric-explain'>- Balance between precision and recall</span>")
                            st.markdown("---")
                            st.markdown("")
                        
                        # Evaluate RandomForestClassifier
                        if 'RandomForestClassifier' in results:
                            res = results['RandomForestClassifier']
                            st.markdown("### # Evaluate RandomForestClassifier Model")
                            st.markdown("*Random Forest is an ensemble of decision trees that reduces overfitting by averaging multiple trees trained on different data subsets.*")
                            st.markdown(f"**Accuracy:** {res['accuracy']:.4f} <span class='metric-explain'>- Overall correct predictions</span>")
                            st.markdown(f"**Precision:** {res['precision']:.4f} <span class='metric-explain'>- How many predicted positives were correct</span>")
                            st.markdown(f"**Recall:** {res['recall']:.4f} <span class='metric-explain'>- How many actual positives were found</span>")
                            st.markdown(f"**F1-Score:** {res['f1_score']:.4f} <span class='metric-explain'>- Balance between precision and recall</span>")
                            st.markdown("---")
                            st.markdown("")
                        
                        # Evaluate GradientBoostingClassifier
                        if 'GradientBoostingClassifier' in results:
                            res = results['GradientBoostingClassifier']
                            st.markdown("### # Evaluate GradientBoostingClassifier Model")
                            st.markdown("*Gradient Boosting builds trees sequentially, where each new tree corrects errors made by previous trees. It often achieves high accuracy but can be slower to train.*")
                            st.markdown(f"**Accuracy:** {res['accuracy']:.4f} <span class='metric-explain'>- Overall correct predictions</span>")
                            st.markdown(f"**Precision:** {res['precision']:.4f} <span class='metric-explain'>- How many predicted positives were correct</span>")
                            st.markdown(f"**Recall:** {res['recall']:.4f} <span class='metric-explain'>- How many actual positives were found</span>")
                            st.markdown(f"**F1-Score:** {res['f1_score']:.4f} <span class='metric-explain'>- Balance between precision and recall</span>")
                            st.markdown("---")
                            st.markdown("")
                        
                        # Comparison Table
                        st.markdown("### 📊 Model Performance Comparison ---")
                        st.markdown("""
                        <div class="explanation-box">
                        <b>📖 How to interpret this comparison table:</b><br>
                        • <b>Higher values are better</b> for all metrics (closer to 1.0 = better)<br>
                        • <b>Accuracy</b>: Best when classes are balanced<br>
                        • <b>F1-Score</b>: Better metric when classes are imbalanced<br>
                        • Choose the model that best balances all metrics for your specific needs
                        </div>
                        """, unsafe_allow_html=True)
                        
                        metrics_data = []
                        for name, res in results.items():
                            metrics_data.append({
                                'Model': name,
                                'Accuracy': res['accuracy'],
                                'Precision': res['precision'],
                                'Recall': res['recall'],
                                'F1-Score': res['f1_score']
                            })
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        st.dataframe(metrics_df, column_config={
                            "Model": st.column_config.TextColumn("Model"),
                            "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.6f"),
                            "Precision": st.column_config.NumberColumn("Precision", format="%.6f"),
                            "Recall": st.column_config.NumberColumn("Recall", format="%.6f"),
                            "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.6f"),
                        }, use_container_width=True)
                        
                        # Bar chart
                        fig = px.bar(metrics_df, x='Model', y=['Accuracy','Precision','Recall','F1-Score'],
                                    barmode='group', title='Model Performance Metrics Comparison',
                                    color_discrete_sequence=['#3498db','#e74c3c','#2ecc71','#f39c12'], height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        best_acc = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
                        best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
                        st.markdown(f"""
                        <div style="background: #d4edda; border: 2px solid #27ae60; border-radius: 10px; padding: 20px; margin: 15px 0;">
                            <p style="margin: 5px 0; color: #155724;">
                                Based on <b>Accuracy</b>, the best performing model is: <b>{best_acc['Model']}</b> with Accuracy: <b>{best_acc['Accuracy']:.4f}</b>
                            </p>
                            <p style="margin: 5px 0; color: #155724;">
                                Based on <b>F1-Score</b>, the best performing model is: <b>{best_f1['Model']}</b> with F1-Score: <b>{best_f1['F1-Score']:.4f}</b>
                            </p>
                            <p style="margin: 10px 0 0 0; font-size: 0.9rem; color: #666;">
                                <b>Recommendation:</b> If your data has balanced classes, use Accuracy to choose. 
                                If classes are imbalanced, F1-Score is a better indicator of model quality.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # ===== TAB 2: COMPARE BOTH TARGETS =====
        with eval_tab2:
            st.markdown("### 📊 Compare Model Performance (Both Targets)")
            st.markdown("""
            <div class="explanation-box">
            <b>📖 Why compare both Risk_Type and Risk_Level?</b><br>
            • Different targets may have different predictability<br>
            • Some models may perform better on one target than another<br>
            • This comparison helps you understand which prediction task is more feasible<br>
            • <b>Risk_Level</b> (Low/Medium/High/Critical) may be easier to predict than <b>Risk_Type</b> (Type_A/B/C)
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("🚀 Train & Compare for Both Targets", type="primary", key="compare_both"):
                all_comparisons = {}
                
                for target_col in ['Risk_Type', 'Risk_Level']:
                    if target_col not in df.columns: continue
                    
                    with st.spinner(f'Training models for {target_col}...'):
                        results, _ = train_and_evaluate_detailed(df, target_col)
                        all_comparisons[target_col] = results
                
                for target_col, results in all_comparisons.items():
                    st.markdown("---")
                    st.markdown(f"## 📊 Analysis for **'{target_col}'**")
                    
                    if results:
                        metrics_data = []
                        for name, res in results.items():
                            metrics_data.append({'Model': name, 'Accuracy': res['accuracy'], 'F1-Score': res['f1_score']})
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        best_acc = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
                        best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
                        
                        st.markdown(f"""
                        <div style="background: #d4edda; border: 2px solid #27ae60; border-radius: 10px; padding: 20px; margin: 15px 0;">
                            <p style="margin: 5px 0; color: #155724;">
                                Based on <b>Accuracy</b>, best: <b>{best_acc['Model']}</b> ({best_acc['Accuracy']:.4f})
                            </p>
                            <p style="margin: 5px 0; color: #155724;">
                                Based on <b>F1-Score</b>, best: <b>{best_f1['Model']}</b> ({best_f1['F1-Score']:.4f})
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.dataframe(metrics_df, column_config={
                            "Model": "Model",
                            "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                            "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.4f"),
                        }, use_container_width=True, hide_index=True)
                        
                        fig = px.bar(metrics_df, x='Model', y=['Accuracy','F1-Score'], barmode='group',
                                    title=f'Model Performance - {target_col}',
                                    color_discrete_sequence=['#3498db','#e74c3c'], height=400)
                        st.plotly_chart(fig, use_container_width=True)
                
                if len(all_comparisons) > 1:
                    st.markdown("---")
                    st.markdown("## 📊 Overall Summary")
                    st.markdown("""
                    <div class="explanation-box">
                    <b>📖 How to interpret the Overall Summary:</b><br>
                    • Compare which target variable achieves higher accuracy/F1 scores<br>
                    • The target with higher scores is easier to predict<br>
                    • Use this to decide which prediction task to prioritize
                    </div>
                    """, unsafe_allow_html=True)
                    
                    summary_data = []
                    for target_col, results in all_comparisons.items():
                        if results:
                            best_f1 = max(results.items(), key=lambda x: x[1]['f1_score'])
                            best_acc = max(results.items(), key=lambda x: x[1]['accuracy'])
                            summary_data.append({
                                'Target Variable': target_col,
                                'Best (Accuracy)': f"{best_acc[0]} ({best_acc[1]['accuracy']:.4f})",
                                'Best (F1-Score)': f"{best_f1[0]} ({best_f1[1]['f1_score']:.4f})"
                            })
                    if summary_data:
                        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)
        
        # ===== TAB 3: CROSS VALIDATION =====
        with eval_tab3:
            st.markdown("### 🔄 Cross Validation Analysis")
            st.markdown("""
            <div class="explanation-box">
            <b>📖 What is Cross Validation?</b><br>
            Cross Validation (CV) evaluates model stability by training on different subsets of data.<br>
            • <b>K-Fold CV</b>: Splits data into K parts, trains K times on K-1 parts, tests on the remaining part<br>
            • <b>Stratified CV</b>: Preserves class distribution in each fold<br><br>
            <b>Why use Cross Validation?</b><br>
            • More reliable than a single train/test split<br>
            • Shows how consistent model performance is across different data subsets<br>
            • <b>Mean ± Std</b> tells you the expected performance range
            </div>
            """, unsafe_allow_html=True)
            
            target = st.selectbox("Target Variable for CV", df.columns.tolist(),
                                 index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
            nums = df.select_dtypes(include=['float64','int64','int32']).columns.tolist()
            if target in nums: nums.remove(target)
            folds = st.slider("CV Folds", 3, 10, 5)
            
            if st.button("🔄 Run Cross Validation", type="primary", key="cv_run"):
                X = df[nums].copy(); y = df[target].copy()
                mask = y.notna(); X = X[mask]; y = y[mask]
                if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
                X = X.fillna(X.median())
                
                cv_models = {
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', n_jobs=-1),
                    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1),
                    'GradientBoosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
                }
                cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
                
                cv_results = []; all_scores = {}
                for name, model in cv_models.items():
                    try:
                        acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                        f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
                        all_scores[name] = f1
                        cv_results.append({
                            'Model':name,
                            'Mean Accuracy':round(acc.mean(),4),
                            'Std Accuracy':round(acc.std(),4),
                            'Mean F1':round(f1.mean(),4),
                            'Std F1':round(f1.std(),4)
                        })
                    except: pass
                
                if cv_results:
                    cv_df = pd.DataFrame(cv_results)
                    st.markdown("#### 📊 Cross Validation Results")
                    st.markdown("*Lower Std means more consistent performance across different data splits*")
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
                    
                    best_cv = cv_df.loc[cv_df['Mean F1'].idxmax()]
                    st.success(f"🏆 Best CV Model: **{best_cv['Model']}** (Mean F1: {best_cv['Mean F1']:.4f} ±{best_cv['Std F1']:.4f})")
                    
                    fig_cv = go.Figure()
                    for name, scores in all_scores.items():
                        fig_cv.add_trace(go.Box(y=scores, name=name, boxmean='sd'))
                    fig_cv.update_layout(
                        title=f'Cross Validation F1 Scores ({folds}-Fold Stratified)<br><sub>Box shows spread of scores across folds</sub>',
                        yaxis_title='F1 Score (Weighted)', height=400)
                    st.plotly_chart(fig_cv, use_container_width=True)


if __name__ == "__main__":
    main()
