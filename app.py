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
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, r2_score)
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from imblearn.over_sampling import SMOTE
from scipy import stats
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Microplastic Risk Analysis Dashboard", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-top: 1rem; margin-bottom: 1rem; }
    .stButton > button { width: 100%; background-color: #1f77b4; color: white; font-weight: 600; border-radius: 8px; padding: 0.5rem 1rem; }
    .stButton > button:hover { background-color: #2980b9; border-color: #2980b9; }
    .stMarkdown, .stMarkdown p, .stMarkdown li { color: #2c3e50 !important; }
    .outlier-box { padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .outlier-before { background-color: #ffeaa7; }
    .outlier-after { background-color: #55efc4; }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize all session state variables"""
    if 'data' not in st.session_state: st.session_state.data = None
    if 'processed_data' not in st.session_state: st.session_state.processed_data = None
    if 'models' not in st.session_state: st.session_state.models = {}
    if 'feature_importance' not in st.session_state: st.session_state.feature_importance = None
    if 'mutual_info' not in st.session_state: st.session_state.mutual_info = None
    if 'chi2_scores' not in st.session_state: st.session_state.chi2_scores = None
    if 'trained' not in st.session_state: st.session_state.trained = False
    if 'selected_features' not in st.session_state: st.session_state.selected_features = None
    if 'scaler' not in st.session_state: st.session_state.scaler = None
    if 'scaled_data' not in st.session_state: st.session_state.scaled_data = None
    if 'scaled_columns' not in st.session_state: st.session_state.scaled_columns = None
    if 'encoded_data' not in st.session_state: st.session_state.encoded_data = None
    if 'encoded_shape' not in st.session_state: st.session_state.encoded_shape = None
    if 'evaluation_ran' not in st.session_state: st.session_state.evaluation_ran = False
    if 'comparison_ran' not in st.session_state: st.session_state.comparison_ran = False
    if 'cv_ran' not in st.session_state: st.session_state.cv_ran = False
    if 'outlier_stats_before' not in st.session_state: st.session_state.outlier_stats_before = None
    if 'outlier_stats_after' not in st.session_state: st.session_state.outlier_stats_after = None
    if 'outlier_columns_processed' not in st.session_state: st.session_state.outlier_columns_processed = []

init_session_state()

def load_dataset(uploaded_file):
    """Load dataset from uploaded file with encoding detection"""
    try:
        if uploaded_file.name.endswith('.csv'):
            for enc in ['utf-8', 'latin1', 'cp1252']:
                try: 
                    uploaded_file.seek(0)
                    data = pd.read_csv(uploaded_file, encoding=enc)
                    break
                except: continue
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(uploaded_file)
        else: 
            st.error("Unsupported file format. Please upload CSV or Excel.")
            return None
        st.session_state.data = data
        st.success(f"✅ Dataset loaded successfully! Shape: {data.shape[0]} rows × {data.shape[1]} columns")
        return data
    except Exception as e: 
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def generate_sample_data():
    """Generate sample microplastic dataset"""
    np.random.seed(42)
    n = 1000
    data = {
        'Sample_ID': [f'MP_{i:04d}' for i in range(n)],
        'MP_Count_per_L': np.random.poisson(lam=50, size=n),
        'Particle_Size_um': np.random.normal(100, 30, n),
        'Microplastic_Size_mm_midpoint': np.random.normal(2.5, 1.5, n),
        'Density_midpoint': np.random.normal(1.0, 0.1, n),
        'Polymer_Type': np.random.choice(['PE','PP','PS','PET','PVC','Nylon'], n),
        'pH': np.random.normal(7, 0.5, n), 
        'Temperature_C': np.random.normal(20, 5, n),
        'Risk_Score': np.random.uniform(0, 100, n),
        'Risk_Level': np.random.choice(['Low','Medium','High','Critical'], n, p=[0.3,0.35,0.25,0.1]),
        'Risk_Type': np.random.choice(['Type_A','Type_B','Type_C'], n, p=[0.5,0.3,0.2]),
        'Location': np.random.choice(['Urban','Rural','Industrial','Coastal'], n),
        'Author': np.random.choice(['Author_A','Author_B','Author_C'], n),
        'Source': np.random.choice(['Source_1','Source_2','Source_3'], n)
    }
    df = pd.DataFrame(data)
    for col in ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']:
        if col in df.columns:
            outlier_indices = np.random.choice(n, size=int(n*0.05), replace=False)
            if col == 'Risk_Score':
                df.loc[outlier_indices, col] = np.random.uniform(150, 200, len(outlier_indices))
            elif col == 'MP_Count_per_L':
                df.loc[outlier_indices, col] = np.random.poisson(lam=200, size=len(outlier_indices))
            elif col == 'Microplastic_Size_mm_midpoint':
                df.loc[outlier_indices, col] = np.random.uniform(10, 20, len(outlier_indices))
            elif col == 'Density_midpoint':
                df.loc[outlier_indices, col] = np.random.uniform(1.5, 2.0, len(outlier_indices))
    for col in df.columns:
        if col != 'Sample_ID' and df[col].dtype in ['float64','int64']:
            df.loc[np.random.random(n) < 0.05, col] = np.nan
    return df

def detect_outliers_detailed(df, columns):
    outlier_info = {}
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            clean_data = df[col].dropna()
            if len(clean_data) == 0: continue
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            outlier_info[col] = {
                'Q1': Q1, 'Q3': Q3, 'IQR': IQR,
                'lower_bound': lower_bound, 'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(clean_data)) * 100,
                'min_value': clean_data.min(), 'max_value': clean_data.max(),
                'mean': clean_data.mean(), 'std': clean_data.std(),
                'outliers_below': len(clean_data[clean_data < lower_bound]),
                'outliers_above': len(clean_data[clean_data > upper_bound])
            }
    return outlier_info

def cap_outliers_iqr_detailed(df, columns):
    df_capped = df.copy()
    stats_before = {}
    for col in columns:
        if df_capped[col].dtype in ['float64', 'int64']:
            clean_data = df_capped[col].dropna()
            stats_before[col] = clean_data.describe()
    outlier_counts = {}
    for col in columns:
        if df_capped[col].dtype in ['float64', 'int64']:
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df_capped[col][(df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
            df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    stats_after = {}
    for col in columns:
        if df_capped[col].dtype in ['float64', 'int64']:
            clean_data = df_capped[col].dropna()
            stats_after[col] = clean_data.describe()
    return df_capped, stats_before, stats_after, outlier_counts

def analyze_skewness(df, columns):
    info = []
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            s = df[col].skew()
            info.append({'Column': col, 'Skewness': round(s, 4), 'Skewed': 'Yes' if abs(s) > 0.5 else 'No'})
    return pd.DataFrame(info)

def calculate_mutual_info(X, y):
    scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    return mi_series

def calculate_chi2(X, y):
    X_s = X - X.min() + 1
    scores, pvals = chi2(X_s, y)
    chi2_series = pd.Series(scores, index=X.columns).sort_values(ascending=False)
    return chi2_series

def calculate_rf_importance(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_series = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    return rf_series

def train_and_evaluate_detailed(df, target_col):
    feature_cols = df.select_dtypes(include=['float64','int64','int32']).columns.tolist()
    if target_col in feature_cols: feature_cols.remove(target_col)
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
    X = X.fillna(X.median())
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    info = {'X_train': X_train.shape, 'X_test': X_test.shape, 'target': target_col}
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
        }
    return results, info

def plot_distribution(data, column, title):
    try:
        clean = data[column].dropna()
        if clean.empty: return go.Figure()
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
        fig.add_trace(go.Histogram(x=clean, nbinsx=30, marker_color='#3498db', name='Histogram'), row=1, col=1)
        fig.add_trace(go.Box(y=clean, marker_color='#e74c3c', name='Box Plot'), row=1, col=2)
        fig.update_layout(title_text=title, showlegend=False, height=500)
        return fig
    except: return go.Figure()

def main():
    st.markdown('<p class="main-header">🔬 Microplastic Risk Analysis Dashboard</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio("Select Section", [
        "🏠 Home", "🔧 Preprocessing", "🛠️ Feature Selection & Relevance", 
        "🤖 Modeling", "📊 Cross Validation & Evaluation"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard analyzes microplastic risk data using advanced ML techniques.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Status")
    if st.session_state.data is not None: st.sidebar.success("✅ Data Loaded")
    else: st.sidebar.warning("⚠️ No Data")
    if st.session_state.trained: st.sidebar.success("✅ Models Trained")
    else: st.sidebar.warning("⚠️ Models Not Trained")
    if st.session_state.processed_data is not None: st.sidebar.success("✅ Preprocessing Done")
    
    # ==================== HOME ====================
    if section == "🏠 Home":
        st.markdown('<p class="section-header">🏠 Home - Dataset Overview</p>', unsafe_allow_html=True)
        home_tab1, home_tab2, home_tab3, home_tab4, home_tab5 = st.tabs([
            "📤 Upload & Preview", "📊 Risk Score Distribution", 
            "🔬 MP Count vs Risk Score", "📊 Risk Score by Risk Level", "🔍 Data Quality Check"
        ])
        
        with home_tab1:
            st.markdown("### 📤 Upload Dataset")
            c1, c2 = st.columns([2, 1])
            with c1:
                f = st.file_uploader("Upload dataset (CSV/Excel)", type=['csv','xlsx','xls'])
                if f: load_dataset(f)
            with c2:
                if st.button("Generate Sample Dataset", type="primary"):
                    st.session_state.data = generate_sample_data()
                    st.success("✅ Sample dataset generated!")
                    st.rerun()
            if st.session_state.data is not None:
                df = st.session_state.data
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Samples", df.shape[0])
                with c2: st.metric("Features", df.shape[1])
                with c3: st.metric("Missing Values", df.isnull().sum().sum())
                st.markdown("#### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                st.markdown("#### Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns, 'Type': df.dtypes.values,
                    'Missing': df.isnull().sum().values,
                    'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
                })
                st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        with home_tab2:
            st.markdown("### 📊 Risk Score Distribution")
            if st.session_state.data is not None and 'Risk_Score' in st.session_state.data.columns:
                df = st.session_state.data
                df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                clean = df['Risk_Score'].dropna()
                if len(clean) > 0:
                    st.plotly_chart(plot_distribution(df, 'Risk_Score', 'Risk Score Distribution'), use_container_width=True)
        
        with home_tab3:
            st.markdown("### 🔬 MP Count vs Risk Score")
            if st.session_state.data is not None:
                df = st.session_state.data
                if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
                    clean = df.dropna(subset=['MP_Count_per_L', 'Risk_Score'])
                    if len(clean) > 0:
                        fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score', 
                                       color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                       trendline='ols', title='MP Count vs Risk Score', opacity=0.7)
                        st.plotly_chart(fig, use_container_width=True)
        
        with home_tab4:
            st.markdown("### 📊 Risk Score by Risk Level")
            if st.session_state.data is not None:
                df = st.session_state.data
                if 'Risk_Score' in df.columns and 'Risk_Level' in df.columns:
                    clean = df.dropna(subset=['Risk_Score'])
                    if len(clean) > 0:
                        fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                                    title='Risk Score by Risk Level')
                        st.plotly_chart(fig, use_container_width=True)
        
        with home_tab5:
            st.markdown("### 🔍 Data Quality Check")
            if st.session_state.data is not None:
                df = st.session_state.data
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Missing %", f"{(df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%")
                with c2: st.metric("Duplicates", df.duplicated().sum())
                with c3: st.metric("Numeric Cols", len(df.select_dtypes(include=['float64', 'int64']).columns))
                with c4: st.metric("Categorical Cols", len(df.select_dtypes(include=['object']).columns))
                st.dataframe(df.describe(), use_container_width=True)
    
    # ==================== PREPROCESSING ====================
    elif section == "🔧 Preprocessing":
        st.markdown('<p class="section-header">🔧 Data Preprocessing</p>', unsafe_allow_html=True)
        if st.session_state.data is None: 
            st.warning("⚠️ Please upload data first!")
            return
        
        df = st.session_state.data.copy()
        
        p1, p2, p3, p4, p5 = st.tabs([
            "📏 Feature Scaling", "🔄 Categorical Encoding", 
            "🎯 Outlier Handling", "📊 Skewness & Transform", "📋 Summary"
        ])
        
        with p1:
            st.markdown("### 📏 Perform Feature Scaling")
            numerical_cols = ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']
            available_cols = [col for col in numerical_cols if col in df.columns]
            if len(available_cols) > 0:
                scaler = StandardScaler()
                # Fill NaN with median before scaling
                df[available_cols] = df[available_cols].fillna(df[available_cols].median())
                df[available_cols] = scaler.fit_transform(df[available_cols])
        
        with p2:
            st.markdown("### 🔄 Encode Categorical Variables")
            categorical_cols = ['Location', 'Shape', 'Polymer_Type', 'pH', 'Salinity', 
                               'Industrial_Activity', 'Population_Density', 'Risk_Type', 
                               'Risk_Level', 'Author', 'Source']
            available_cats = [col for col in categorical_cols if col in df.columns]
            if len(available_cats) > 0:
                df_encoded = pd.get_dummies(df, columns=available_cats, drop_first=False)
                st.session_state.processed_data = df_encoded
                st.session_state.encoded_data = df_encoded
                st.session_state.encoded_shape = df_encoded.shape
                st.success(f"✅ One-hot encoding applied! Shape: {df_encoded.shape}")
                st.markdown("First 5 rows after one-hot encoding:")
                st.dataframe(df_encoded.head())
        
        with p3:
            st.markdown("### 🎯 Address Outliers")
            nums = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_out = [c for c in nums if 'ID' not in c and 'Sample' not in c]
            if len(cols_out) > 0:
                default_cols = ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']
                selected_cols = [c for c in default_cols if c in cols_out] or cols_out[:4]
                df_capped, stats_before, stats_after, outlier_counts = cap_outliers_iqr_detailed(df, selected_cols)
                st.session_state.processed_data = df_capped
                st.session_state.outlier_columns_processed = selected_cols
                st.success(f"✅ Capped outliers in {len(selected_cols)} columns!")
        
        with p4:
            st.markdown("### 📊 Skewness & Log Transform")
            numerical_cols = ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']
            available_cols = [col for col in numerical_cols if col in df.columns]
            if len(available_cols) > 0:
                skew_before = df[available_cols].skew()
                skew_before_df = pd.DataFrame({'Column': skew_before.index, 'Skewness': skew_before.values})
                st.markdown("**Skewness before transformation:**")
                st.dataframe(skew_before_df, use_container_width=True, hide_index=True)
                
                df_transformed = df.copy()
                for col in available_cols:
                    shift = abs(df_transformed[col].min()) + 1 if df_transformed[col].min() <= 0 else 0
                    df_transformed[col] = np.log1p(df_transformed[col] + shift)
                st.session_state.processed_data = df_transformed
                
                skew_after = df_transformed[available_cols].skew()
                skew_after_df = pd.DataFrame({'Column': skew_after.index, 'Skewness': skew_after.values})
                st.success("✅ Log transform applied!")
                st.markdown("**Skewness after transformation:**")
                st.dataframe(skew_after_df, use_container_width=True, hide_index=True)
        
        with p5:
            st.markdown("### 📋 Summary")
            current_data = st.session_state.processed_data if st.session_state.processed_data is not None else df
            st.markdown(f"**Current Data Shape:** {current_data.shape}")
            st.success("✅ Preprocessing complete! Proceed to Feature Selection.")
    
    # ==================== FEATURE SELECTION ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        
        st.markdown("### 🎯 Understand the Goal")
        col1, col2 = st.columns(2)
        with col1:
            target = st.selectbox("Select Target Variable:", df.columns.tolist(),
                                 index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        with col2:
            model_type = st.selectbox("Select Model Type:", ["Classification", "Regression"],
                                     index=0 if df[target].dtype == 'object' or df[target].nunique() < 10 else 1)
        
        st.markdown(f"**Target:** `{target}` | **Model Type:** {model_type}")
        st.markdown("---")
        
        st.markdown("### 📚 Explore Feature Selection Methods")
        method_tabs = st.tabs(["📋 Overview", "🔍 Filter", "🔄 Wrapper", "🌲 Embedded"])
        with method_tabs[0]:
            st.markdown("**Filter:** Statistical scores | **Wrapper:** Model-based subsets | **Embedded:** Built-in during training")
        with method_tabs[1]:
            st.markdown("**Selected:** Mutual Information & Chi-Squared")
        with method_tabs[2]:
            st.markdown("**Not selected** - Computationally expensive")
        with method_tabs[3]:
            st.markdown("**Selected:** Random Forest Importance")
        
        st.markdown("---")
        st.markdown("### 🎯 Implement Selected Methods")
        st.markdown("Apply Mutual Information, Chi-squared tests, and Random Forest feature importances.")
        
        nums = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        if target in nums: nums.remove(target)
        
        if st.button("🚀 Calculate Feature Importance", type="primary", use_container_width=True):
            with st.spinner('Calculating...'):
                X = df[nums].copy()
                y = df[target].copy()
                mask = y.notna()
                X = X[mask]
                y = y[mask]
                X = X.fillna(X.median())
                if y.dtype == 'object': 
                    y = LabelEncoder().fit_transform(y)
                else:
                    y = pd.qcut(y, q=4, labels=False, duplicates='drop')
                X = X.dropna(axis=1, how='any')
                
                # Calculate scores
                mi_series = calculate_mutual_info(X, y)
                chi2_series = calculate_chi2(X, y)
                rf_series = calculate_rf_importance(X, y)
                
                st.session_state.feature_importance = rf_series
                st.session_state.mutual_info = mi_series
                st.session_state.chi2_scores = chi2_series
                st.session_state.selected_features = rf_series.head(10).index.tolist()
                
                # Display results
                ft1, ft2, ft3 = st.tabs(["📊 Mutual Information", "🔢 Chi-squared", "🌲 Random Forest"])
                
                with ft1:
                    st.markdown("**Top 20 features based on Mutual Information:**")
                    mi_df = mi_series.head(20).reset_index()
                    mi_df.columns = ['Feature', 'Mutual Information Scores']
                    st.dataframe(mi_df, use_container_width=True, hide_index=True)
                    
                    fig = px.bar(mi_df, x='Mutual Information Scores', y='Feature', orientation='h',
                                title='Mutual Information Scores (Top 20)', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with ft2:
                    st.markdown("**Top 20 features based on Chi-squared Test:**")
                    chi2_df = chi2_series.head(20).reset_index()
                    chi2_df.columns = ['Feature', 'Chi-squared Scores']
                    st.dataframe(chi2_df, use_container_width=True, hide_index=True)
                    
                    fig = px.bar(chi2_df, x='Chi-squared Scores', y='Feature', orientation='h',
                                title='Chi-squared Scores (Top 20)', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with ft3:
                    st.markdown("**Top 20 features based on RandomForest Feature Importances:**")
                    rf_df = rf_series.head(20).reset_index()
                    rf_df.columns = ['Feature', 'Feature Importances']
                    st.dataframe(rf_df, use_container_width=True, hide_index=True)
                    
                    fig = px.bar(rf_df, x='Feature Importances', y='Feature', orientation='h',
                                title='Random Forest Feature Importances (Top 20)', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success("✅ Feature selection completed!")
    
    # ==================== MODELING ====================
    elif section == "🤖 Modeling":
        st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data
        target = st.selectbox("Target Variable", df.columns.tolist(), key='train_target')
        all_f = [c for c in df.columns if c != target]
        default = st.session_state.get('selected_features', df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:5])
        default = [f for f in default if f in all_f]
        features = st.multiselect("Features", all_f, default=default)
        
        c1, c2 = st.columns(2)
        with c1: ts = st.slider("Test Size", 0.1, 0.5, 0.2)
        with c2: use_smote = st.checkbox("Use SMOTE", value=True)
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if len(features) == 0: 
                st.error("Select at least one feature!")
                return
            
            X = df[features].select_dtypes(include=['float64', 'int64', 'int32'])
            y = df[target]
            mask = y.notna()
            X = X[mask]; y = y[mask]
            if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
            X = X.fillna(X.median())
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
            
            if use_smote:
                tc = pd.Series(y_train).value_counts()
                if tc.min() >= 2:
                    try: X_train, y_train = SMOTE(random_state=42, k_neighbors=min(5, tc.min()-1)).fit_resample(X_train, y_train)
                    except: pass
            
            models = {}
            try:
                lr = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', n_jobs=-1)
                lr.fit(X_train, y_train); models['Logistic Regression'] = lr
            except: pass
            try:
                rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
                rf.fit(X_train, y_train); models['RandomForestClassifier'] = rf
            except: pass
            try:
                gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
                gb.fit(X_train, y_train); models['GradientBoostingClassifier'] = gb
            except: pass
            
            if models:
                st.session_state.models = models
                st.session_state.trained = True
                st.success(f"✅ {len(models)} models trained!")
                for name, model in models.items():
                    y_pred = model.predict(X_test)
                    st.markdown(f"**{name}:** Acc={accuracy_score(y_test, y_pred):.4f} | F1={f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    # ==================== CROSS VALIDATION & EVALUATION ====================
    elif section == "📊 Cross Validation & Evaluation":
        st.markdown('<p class="section-header">📊 Cross Validation & Model Evaluation</p>', unsafe_allow_html=True)
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        
        et1, et2, et3, et4 = st.tabs([
            "📊 Evaluate", "📊 Compare Targets", "🔄 Cross Validation", "📋 Summary"
        ])
        
        with et1:
            st.markdown("### 📊 Evaluate on Risk_Type")
            if 'Risk_Type' in df.columns and st.button("🚀 Evaluate", type="primary", key="eval"):
                results, _ = train_and_evaluate_detailed(df, 'Risk_Type')
                st.session_state.evaluation_ran = True
                if results:
                    md = [{'Model': n, 'Accuracy': r['accuracy'], 'F1': r['f1_score']} for n, r in results.items()]
                    st.dataframe(pd.DataFrame(md), use_container_width=True)
        
        with et2:
            st.markdown("### 📊 Compare Both Targets")
            if st.button("🚀 Compare", type="primary", key="cmp"):
                for tc in ['Risk_Type', 'Risk_Level']:
                    if tc in df.columns:
                        results, _ = train_and_evaluate_detailed(df, tc)
                        st.markdown(f"**{tc}**")
                        if results:
                            md = [{'Model': n, 'Accuracy': r['accuracy'], 'F1': r['f1_score']} for n, r in results.items()]
                            st.dataframe(pd.DataFrame(md), use_container_width=True, hide_index=True)
                st.session_state.comparison_ran = True
        
        with et3:
            st.markdown("### 🔄 Cross Validation")
            target = st.selectbox("Target", df.columns.tolist(), key="cv_target")
            nums = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            if target in nums: nums.remove(target)
            folds = st.slider("Folds", 3, 10, 5)
            
            if st.button("🔄 Run CV", type="primary", key="cv"):
                X = df[nums].fillna(df[nums].median())
                y = df[target].dropna()
                X = X.loc[y.index]
                if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
                
                models = {
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', n_jobs=-1),
                    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
                }
                cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
                cv_res = []
                for nm, md in models.items():
                    try:
                        acc = cross_val_score(md, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                        f1 = cross_val_score(md, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
                        cv_res.append({'Model': nm, 'Mean Acc': acc.mean(), 'Mean F1': f1.mean()})
                    except: pass
                if cv_res:
                    st.dataframe(pd.DataFrame(cv_res), use_container_width=True)
                st.session_state.cv_ran = True
        
        with et4:
            st.markdown("### 📋 Pipeline Summary")
            if st.button("🔄 Generate Summary", type="primary", key="pipe"):
                summary = [
                    {'Stage': 'Data Loading', 'Status': '✅' if st.session_state.data is not None else '❌'},
                    {'Stage': 'Preprocessing', 'Status': '✅' if st.session_state.processed_data is not None else '⬜'},
                    {'Stage': 'Feature Selection', 'Status': '✅' if st.session_state.feature_importance is not None else '⬜'},
                    {'Stage': 'Modeling', 'Status': '✅' if st.session_state.trained else '⬜'},
                    {'Stage': 'Evaluation', 'Status': '✅' if st.session_state.evaluation_ran else '⬜'},
                ]
                st.dataframe(pd.DataFrame(summary), use_container_width=True)

if __name__ == "__main__":
    main()
