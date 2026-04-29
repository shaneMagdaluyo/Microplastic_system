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
        'Water_Source': np.random.choice(['River','Lake','Ocean','Groundwater','Tap'], n),
        'pH': np.random.normal(7, 0.5, n), 
        'Temperature_C': np.random.normal(20, 5, n),
        'Risk_Score': np.random.uniform(0, 100, n),
        'Risk_Level': np.random.choice(['Low','Medium','High','Critical'], n, p=[0.3,0.35,0.25,0.1]),
        'Risk_Type': np.random.choice(['Type_A','Type_B','Type_C'], n, p=[0.5,0.3,0.2]),
        'Location': np.random.choice(['Urban','Rural','Industrial','Coastal'], n),
        'Season': np.random.choice(['Winter','Spring','Summer','Fall'], n),
        'Author': np.random.choice(['Author_A','Author_B','Author_C'], n),
        'Source': np.random.choice(['Source_1','Source_2','Source_3'], n)
    }
    df = pd.DataFrame(data)
    # Add some outliers
    for col in ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']:
        if col in df.columns:
            # Add extreme values
            outlier_indices = np.random.choice(n, size=int(n*0.05), replace=False)
            if col == 'Risk_Score':
                df.loc[outlier_indices, col] = np.random.uniform(150, 200, len(outlier_indices))
            elif col == 'MP_Count_per_L':
                df.loc[outlier_indices, col] = np.random.poisson(lam=200, size=len(outlier_indices))
            elif col == 'Microplastic_Size_mm_midpoint':
                df.loc[outlier_indices, col] = np.random.uniform(10, 20, len(outlier_indices))
            elif col == 'Density_midpoint':
                df.loc[outlier_indices, col] = np.random.uniform(1.5, 2.0, len(outlier_indices))
    
    # Add some missing values
    for col in df.columns:
        if col != 'Sample_ID' and df[col].dtype in ['float64','int64']:
            df.loc[np.random.random(n) < 0.05, col] = np.nan
    
    return df

def detect_outliers_detailed(df, columns):
    """
    Detailed outlier detection using IQR method
    Returns comprehensive outlier information
    """
    outlier_info = {}
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            clean_data = df[col].dropna()
            if len(clean_data) == 0:
                continue
                
            Q1 = clean_data.quantile(0.25)
            Q3 = clean_data.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = clean_data[(clean_data < lower_bound) | (clean_data > upper_bound)]
            
            outlier_info[col] = {
                'Q1': Q1,
                'Q3': Q3,
                'IQR': IQR,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outlier_count': len(outliers),
                'outlier_percentage': (len(outliers) / len(clean_data)) * 100,
                'min_value': clean_data.min(),
                'max_value': clean_data.max(),
                'mean': clean_data.mean(),
                'std': clean_data.std(),
                'outliers_below': len(clean_data[clean_data < lower_bound]),
                'outliers_above': len(clean_data[clean_data > upper_bound])
            }
    return outlier_info

def cap_outliers_iqr_detailed(df, columns):
    """
    Cap outliers using IQR method with detailed statistics
    Returns capped dataframe and statistics before/after
    """
    df_capped = df.copy()
    
    # Store statistics before capping
    stats_before = {}
    for col in columns:
        if df_capped[col].dtype in ['float64', 'int64']:
            clean_data = df_capped[col].dropna()
            stats_before[col] = clean_data.describe()
    
    # Cap outliers
    outlier_counts = {}
    for col in columns:
        if df_capped[col].dtype in ['float64', 'int64']:
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Count outliers before capping
            outliers = df_capped[col][(df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
            
            # Cap the outliers
            df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
    
    # Store statistics after capping
    stats_after = {}
    for col in columns:
        if df_capped[col].dtype in ['float64', 'int64']:
            clean_data = df_capped[col].dropna()
            stats_after[col] = clean_data.describe()
    
    return df_capped, stats_before, stats_after, outlier_counts

def create_outlier_summary_table(stats_before, stats_after, columns):
    """Create a formatted summary table comparing before/after outlier handling"""
    summary_data = []
    for col in columns:
        if col in stats_before and col in stats_after:
            summary_data.append({
                'Column': col,
                'Before Mean': f"{stats_before[col]['mean']:.4f}",
                'After Mean': f"{stats_after[col]['mean']:.4f}",
                'Before Std': f"{stats_before[col]['std']:.4f}",
                'After Std': f"{stats_after[col]['std']:.4f}",
                'Before Min': f"{stats_before[col]['min']:.4f}",
                'After Min': f"{stats_after[col]['min']:.4f}",
                'Before Max': f"{stats_before[col]['max']:.4f}",
                'After Max': f"{stats_after[col]['max']:.4f}"
            })
    return pd.DataFrame(summary_data)

def one_hot_encode(df):
    """Perform one-hot encoding on categorical columns"""
    try:
        cats = df.select_dtypes(include=['object']).columns.tolist()
        cols = [c for c in cats if 'ID' not in c and 'Sample' not in c]
        if len(cols) == 0: 
            return df, [], [], df.shape
        df_enc = pd.get_dummies(df, columns=cols, drop_first=False)
        new = [c for c in df_enc.columns if c not in df.columns]
        return df_enc, new, cols, df_enc.shape
    except Exception as e:
        st.error(f"Encoding error: {e}")
        return df, [], [], df.shape

def analyze_skewness(df, columns):
    """Analyze skewness of numerical columns"""
    info = []
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            s = df[col].skew()
            info.append({
                'Column': col, 
                'Skewness': round(s, 4), 
                'Skewed': 'Yes' if abs(s) > 0.5 else 'No'
            })
    return pd.DataFrame(info)

def apply_log_transform(df, columns):
    """Apply log transformation to skewed columns"""
    df_t = df.copy()
    for col in columns:
        if df_t[col].dtype in ['float64', 'int64'] and abs(df_t[col].skew()) > 0.5:
            shift = abs(df_t[col].min()) + 1 if df_t[col].min() <= 0 else 0
            df_t[col] = np.log1p(df_t[col] + shift)
    return df_t

def calculate_mutual_info(X, y):
    """Calculate mutual information scores"""
    scores = mutual_info_classif(X, y, random_state=42)
    return pd.DataFrame({'Feature': X.columns, 'Mutual_Info': scores}).sort_values('Mutual_Info', ascending=False)

def calculate_chi2(X, y):
    """Calculate chi-squared scores"""
    X_s = X - X.min() + 1
    scores, pvals = chi2(X_s, y)
    return pd.DataFrame({'Feature': X.columns, 'Chi2_Score': scores, 'P_Value': pvals}).sort_values('Chi2_Score', ascending=False)

def calculate_rf_importance(X, y):
    """Calculate Random Forest feature importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)

def train_and_evaluate_detailed(df, target_col):
    """Train and evaluate models with detailed metrics"""
    feature_cols = df.select_dtypes(include=['float64','int64','int32']).columns.tolist()
    if target_col in feature_cols: 
        feature_cols.remove(target_col)
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    mask = y.notna()
    X = X[mask]
    y = y[mask]
    
    if y.dtype == 'object': 
        y = LabelEncoder().fit_transform(y)
    
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
    """Create distribution plot with histogram and box plot"""
    try:
        clean = data[column].dropna()
        if clean.empty: 
            return go.Figure()
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
        fig.add_trace(go.Histogram(x=clean, nbinsx=30, marker_color='#3498db', name='Histogram'), row=1, col=1)
        fig.add_trace(go.Box(y=clean, marker_color='#e74c3c', name='Box Plot'), row=1, col=2)
        fig.update_layout(title_text=title, showlegend=False, height=500)
        return fig
    except Exception as e:
        return go.Figure()

def main():
    st.markdown('<p class="main-header">🔬 Microplastic Risk Analysis Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio("Select Section", [
        "🏠 Home", 
        "🔧 Preprocessing", 
        "🛠️ Feature Selection & Relevance", 
        "🤖 Modeling", 
        "📊 Cross Validation & Evaluation"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard analyzes microplastic risk data using advanced ML techniques.")
    st.sidebar.markdown("---")
    
    # Status indicators
    st.sidebar.markdown("### 📌 Status")
    if st.session_state.data is not None: 
        st.sidebar.success("✅ Data Loaded")
    else: 
        st.sidebar.warning("⚠️ No Data")
    
    if st.session_state.trained: 
        st.sidebar.success("✅ Models Trained")
    else: 
        st.sidebar.warning("⚠️ Models Not Trained")
    
    if st.session_state.processed_data is not None:
        st.sidebar.success("✅ Preprocessing Done")
    
    # ==================== HOME ====================
    if section == "🏠 Home":
        st.markdown('<p class="section-header">🏠 Home - Dataset Overview</p>', unsafe_allow_html=True)
        
        home_tab1, home_tab2, home_tab3, home_tab4, home_tab5 = st.tabs([
            "📤 Upload & Preview", 
            "📊 Risk Score Distribution", 
            "🔬 MP Count vs Risk Score", 
            "📊 Risk Score by Risk Level", 
            "🔍 Data Quality Check"
        ])
        
        with home_tab1:
            st.markdown("### 📤 Upload Dataset")
            c1, c2 = st.columns([2, 1])
            with c1:
                f = st.file_uploader("Upload dataset (CSV/Excel)", type=['csv','xlsx','xls'])
                if f: 
                    load_dataset(f)
            with c2:
                if st.button("Generate Sample Dataset", type="primary"):
                    st.session_state.data = generate_sample_data()
                    st.success("✅ Sample dataset generated with outliers!")
                    st.rerun()
            
            if st.session_state.data is not None:
                df = st.session_state.data
                c1, c2, c3 = st.columns(3)
                with c1: 
                    st.metric("Samples", df.shape[0])
                with c2: 
                    st.metric("Features", df.shape[1])
                with c3: 
                    st.metric("Missing Values", df.isnull().sum().sum())
                st.markdown("#### Data Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # Show column info
                st.markdown("#### Column Information")
                col_info = pd.DataFrame({
                    'Column': df.columns,
                    'Type': df.dtypes.values,
                    'Missing': df.isnull().sum().values,
                    'Missing %': (df.isnull().sum() / len(df) * 100).round(2).values
                })
                st.dataframe(col_info, use_container_width=True, hide_index=True)
        
        with home_tab2:
            st.markdown("### 📊 Analyze the Distribution of Risk Score")
            if st.session_state.data is None: 
                st.warning("⚠️ Upload data first!")
            else:
                df = st.session_state.data
                if 'Risk_Score' in df.columns:
                    df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                    clean = df['Risk_Score'].dropna()
                    if len(clean) > 0:
                        st.plotly_chart(plot_distribution(df, 'Risk_Score', 'Risk Score Distribution'), use_container_width=True)
                        q1, q3 = clean.quantile(0.25), clean.quantile(0.75)
                        c1, c2 = st.columns(2)
                        with c1:
                            stats_data = [
                                ('Count', f'{len(clean):,}'),
                                ('Mean', f'{clean.mean():.4f}'),
                                ('Median', f'{clean.median():.4f}'),
                                ('Std', f'{clean.std():.4f}'),
                                ('Min', f'{clean.min():.4f}'),
                                ('Q1', f'{q1:.4f}'),
                                ('Q3', f'{q3:.4f}'),
                                ('Max', f'{clean.max():.4f}'),
                                ('Skewness', f'{clean.skew():.4f}')
                            ]
                            st.dataframe(pd.DataFrame(stats_data, columns=['Statistic', 'Value']), use_container_width=True, hide_index=True)
                        with c2:
                            cats = [
                                ('🟢 Low (0-25)', (clean < 25).sum()),
                                ('🟡 Med (25-50)', ((clean >= 25) & (clean < 50)).sum()),
                                ('🟠 High (50-75)', ((clean >= 50) & (clean < 75)).sum()),
                                ('🔴 Critical (75-100)', (clean >= 75).sum())
                            ]
                            for cat, cnt in cats:
                                st.markdown(f"**{cat}**: {cnt:,} ({(cnt/len(clean))*100:.1f}%)")
                                st.progress(int((cnt/len(clean))*100))
        
        with home_tab3:
            st.markdown("### 🔬 Explore the Relationship Between Risk Score and MP Count per L")
            if st.session_state.data is None: 
                st.warning("⚠️ Upload data first!")
            else:
                df = st.session_state.data
                if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
                    df['MP_Count_per_L'] = pd.to_numeric(df['MP_Count_per_L'], errors='coerce')
                    df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                    clean = df.dropna(subset=['MP_Count_per_L', 'Risk_Score'])
                    if len(clean) > 0:
                        st1, st2, st3 = st.tabs(["📊 Scatter", "📈 Trendline", "📋 Correlation"])
                        with st1:
                            fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                            color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                            title='MP Count per Liter vs Risk Score',
                                            labels={'MP_Count_per_L': 'MP Count per Liter', 'Risk_Score': 'Risk Score'}, 
                                            opacity=0.7)
                            st.plotly_chart(fig, use_container_width=True)
                        with st2:
                            try:
                                fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                                color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                                trendline='ols', 
                                                title='MP Count vs Risk Score with Trendline',
                                                labels={'MP_Count_per_L': 'MP Count per Liter', 'Risk_Score': 'Risk Score'}, 
                                                opacity=0.7)
                                st.plotly_chart(fig, use_container_width=True)
                            except: 
                                st.warning("Trendline not available for this data")
                        with st3:
                            corr = clean['MP_Count_per_L'].corr(clean['Risk_Score'])
                            c1, c2 = st.columns(2)
                            with c1:
                                st.dataframe(pd.DataFrame([
                                    ('Correlation (r)', f'{corr:.4f}'),
                                    ('Samples', f'{len(clean):,}')
                                ], columns=['Metric', 'Value']), use_container_width=True, hide_index=True)
                            with c2:
                                st.metric("Correlation", f"{corr:.4f}")
                                st.progress(min(abs(corr), 1.0))
        
        with home_tab4:
            st.markdown("### 📊 Investigate Difference: Risk Score by Risk Level")
            if st.session_state.data is None: 
                st.warning("⚠️ Upload data first!")
            else:
                df = st.session_state.data
                if 'Risk_Score' in df.columns and 'Risk_Level' in df.columns:
                    df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                    clean = df.dropna(subset=['Risk_Score'])
                    clean['Risk_Level'] = clean['Risk_Level'].astype(str)
                    if len(clean) > 0:
                        fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                                    title='Risk Score by Risk Level', points='outliers')
                        st.plotly_chart(fig, use_container_width=True)
                        stats = clean.groupby('Risk_Level')['Risk_Score'].agg(['count', 'mean', 'median', 'std', 'min', 'max']).round(2)
                        stats.columns = ['Count', 'Mean', 'Median', 'Std', 'Min', 'Max']
                        st.dataframe(stats, use_container_width=True)
        
        with home_tab5:
            st.markdown("### 🔍 Data Quality Check")
            if st.session_state.data is None: 
                st.warning("⚠️ Upload data first!")
            else:
                df = st.session_state.data
                c1, c2, c3, c4 = st.columns(4)
                with c1: 
                    st.metric("Missing %", f"{(df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%")
                with c2: 
                    st.metric("Duplicates", df.duplicated().sum())
                with c3: 
                    st.metric("Numeric Cols", len(df.select_dtypes(include=['float64', 'int64']).columns))
                with c4: 
                    st.metric("Categorical Cols", len(df.select_dtypes(include=['object']).columns))
                
                st.markdown("---")
                st.write("**Data Types:**")
                st.dataframe(pd.DataFrame(df.dtypes, columns=['Data Type']).T, use_container_width=True)
                
                st.markdown("---")
                st.write("**Descriptive Statistics:**")
                st.dataframe(df.describe(), use_container_width=True)
    
    # ==================== PREPROCESSING ====================
    elif section == "🔧 Preprocessing":
        st.markdown('<p class="section-header">🔧 Data Preprocessing</p>', unsafe_allow_html=True)
        if st.session_state.data is None: 
            st.warning("⚠️ Please upload data first!")
            return
        
        df = st.session_state.data.copy()
        
        p1, p2, p3, p4, p5 = st.tabs([
            "📏 Feature Scaling", 
            "🔄 Categorical Encoding", 
            "🎯 Outlier Handling", 
            "📊 Skewness & Transform", 
            "📋 Summary"
        ])
        
        with p1:
            st.markdown("### 📏 Perform Feature Scaling")
            
            # Select the numerical columns
            numerical_cols = ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']
            
            # Check which columns exist in the dataframe
            available_cols = [col for col in numerical_cols if col in df.columns]
            
            if len(available_cols) == 0:
                st.error(f"None of the required columns found. Available numerical columns: {df.select_dtypes(include=['float64', 'int64']).columns.tolist()}")
            else:
                st.markdown(f"**Columns to be scaled:** {', '.join(available_cols)}")
                
                if st.button("🔧 Apply StandardScaler", type="primary"):
                    # Instantiate the scaler
                    scaler = StandardScaler()
                    
                    # Fit and transform the numerical data
                    df[available_cols] = scaler.fit_transform(df[available_cols])
                    
                    # Store in session state
                    st.session_state.processed_data = df
                    st.session_state.scaler = scaler
                    st.session_state.scaled_columns = available_cols
                    
                    # Display the first few rows of the scaled numerical data
                    st.markdown("First 5 rows of scaled numerical data:")
                    st.dataframe(df[available_cols].head())
        
         with p2:
            st.markdown("### 🔄 Encode Categorical Variables")
            
            # Define the categorical columns to encode
            categorical_cols = ['Location', 'Shape', 'Polymer_Type', 'pH', 'Salinity', 
                               'Industrial_Activity', 'Population_Density', 'Risk_Type', 
                               'Risk_Level', 'Author', 'Source']
            
            # Check which columns exist in the dataframe
            available_cats = [col for col in categorical_cols if col in df.columns]
            
            if len(available_cats) == 0:
                st.warning("None of the specified categorical columns found in the dataset.")
            else:
                st.markdown(f"**Categorical columns to encode ({len(available_cats)}):** {', '.join(available_cats)}")
                
                if st.button("🔄 Apply One-Hot Encoding", type="primary", key="encode_tab"):
                    # Perform one-hot encoding
                    df_encoded = pd.get_dummies(df, columns=available_cats, drop_first=False)
                    
                    # Store in session state
                    st.session_state.processed_data = df_encoded
                    st.session_state.encoded_data = df_encoded
                    st.session_state.encoded_shape = df_encoded.shape
                    
                    st.success(f"✅ One-hot encoding applied! New shape: {df_encoded.shape}")
                    
                    # Display the first 5 rows of the encoded data
                    st.markdown("First 5 rows of the DataFrame after one-hot encoding:")
                    st.dataframe(df_encoded.head())
        with p3:
            st.markdown("### 🎯 Address Outliers in Numerical Columns")
            st.markdown("""
            <div class='outlier-box outlier-before'>
            <strong>📌 Method:</strong> IQR (Interquartile Range) method will be used to identify and cap outliers.<br>
            <strong>Formula:</strong> Lower bound = Q1 - 1.5×IQR, Upper bound = Q3 + 1.5×IQR<br>
            <strong>Action:</strong> Values outside bounds will be capped to the nearest bound.
            </div>
            """, unsafe_allow_html=True)
            
            # Select numerical columns for outlier handling
            nums = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_out = [c for c in nums if 'ID' not in c and 'Sample' not in c]
            
            if len(cols_out) > 0:
                # Let user select which columns to process
                default_cols = ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']
                available_defaults = [c for c in default_cols if c in cols_out]
                
                selected_cols = st.multiselect(
                    "Select columns for outlier handling:",
                    cols_out,
                    default=available_defaults if available_defaults else cols_out[:4]
                )
                
                if len(selected_cols) > 0:
                    # Show current outlier statistics
                    st.markdown("#### Current Outlier Detection")
                    outlier_info = detect_outliers_detailed(df, selected_cols)
                    
                    outlier_summary = []
                    for col, info in outlier_info.items():
                        outlier_summary.append({
                            'Column': col,
                            'Outliers': info['outlier_count'],
                            '%': f"{info['outlier_percentage']:.1f}%",
                            'Lower Bound': f"{info['lower_bound']:.4f}",
                            'Upper Bound': f"{info['upper_bound']:.4f}",
                            'Below': info['outliers_below'],
                            'Above': info['outliers_above']
                        })
                    
                    outlier_df = pd.DataFrame(outlier_summary)
                    st.dataframe(outlier_df, use_container_width=True, hide_index=True)
                    
                    # Visualize outliers
                    st.markdown("#### Outlier Visualization")
                    for col in selected_cols:
                        fig = go.Figure()
                        clean = df[col].dropna()
                        fig.add_trace(go.Box(y=clean, name=col, boxmean='sd'))
                        fig.update_layout(title=f'Box Plot: {col}', height=300)
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Apply outlier capping
                    if st.button("🎯 Cap Outliers (IQR Method)", type="primary", key="outlier_tab"):
                        with st.spinner('Capping outliers...'):
                            try:
                                # Apply outlier capping
                                df_capped, stats_before, stats_after, outlier_counts = cap_outliers_iqr_detailed(df, selected_cols)
                                
                                # Store results
                                st.session_state.processed_data = df_capped
                                st.session_state.outlier_stats_before = stats_before
                                st.session_state.outlier_stats_after = stats_after
                                st.session_state.outlier_columns_processed = selected_cols
                                
                                st.success(f"✅ Successfully capped outliers in {len(selected_cols)} columns!")
                                
                                # Show comparison
                                st.markdown("""
                                <div class='outlier-box outlier-after'>
                                <strong>✅ Outliers handled successfully!</strong><br>
                                Values outside IQR bounds have been capped to the nearest bound.
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Display before/after statistics
                                st.markdown("### 📊 Descriptive Statistics After Handling Outliers")
                                
                                # Create the exact table format you requested
                                stats_table_data = {}
                                for col in selected_cols:
                                    if col in stats_after:
                                        stats_table_data[col] = stats_after[col]
                                
                                stats_df = pd.DataFrame(stats_table_data).round(6)
                                st.dataframe(stats_df, use_container_width=True)
                                
                                # Show outlier count reduction
                                st.markdown("### 📊 Outlier Count Comparison")
                                comp_data = []
                                for col in selected_cols:
                                    if col in outlier_counts:
                                        comp_data.append({
                                            'Column': col,
                                            'Outliers Before': outlier_counts[col],
                                            'Outliers After': 0,
                                            'Reduction': '100%'
                                        })
                                
                                comp_df = pd.DataFrame(comp_data)
                                st.dataframe(comp_df, use_container_width=True, hide_index=True)
                                
                                # Show distribution after capping
                                st.markdown("### 📈 Distribution After Outlier Handling")
                                for col in selected_cols:
                                    st.plotly_chart(plot_distribution(df_capped, col, f'{col} Distribution (After Capping)'), use_container_width=True)
                                    
                            except Exception as e:
                                st.error(f"Outlier handling failed: {e}")
        
        with p4:
            st.markdown("### 📊 Skewness Analysis & Log Transform")
            nums = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_sk = [c for c in nums if 'ID' not in c and 'Sample' not in c]
            
            if len(cols_sk) > 0:
                skew_df = analyze_skewness(df, cols_sk)
                st.dataframe(skew_df, use_container_width=True, hide_index=True)
                
                skewed_cols = skew_df[skew_df['Skewed'] == 'Yes']['Column'].tolist()
                if len(skewed_cols) > 0:
                    st.markdown(f"**Skewed columns ({len(skewed_cols)}):** {', '.join(skewed_cols)}")
                    
                    if st.button("📊 Apply Log Transform to Skewed Columns", type="primary", key="skew_tab"):
                        with st.spinner('Applying log transformation...'):
                            current_data = st.session_state.processed_data if st.session_state.processed_data is not None else df
                            transformed_df = apply_log_transform(current_data, skewed_cols)
                            st.session_state.processed_data = transformed_df
                            st.success("✅ Log transform applied to skewed columns!")
                            
                            st.markdown("#### Post-Transform Skewness")
                            new_skew = analyze_skewness(transformed_df, skewed_cols)
                            st.dataframe(new_skew, use_container_width=True, hide_index=True)
        
        with p5:
            st.markdown("### 📋 Preprocessing Summary")
            actions = []
            
            if st.session_state.get('scaled_columns') is not None: 
                actions.append("✅ Feature Scaling applied")
            if st.session_state.get('encoded_data') is not None: 
                actions.append("✅ Categorical Encoding applied")
            if st.session_state.get('processed_data') is not None and len(st.session_state.get('outlier_columns_processed', [])) > 0: 
                actions.append(f"✅ Outliers handled in {len(st.session_state.outlier_columns_processed)} columns")
            
            if actions:
                for a in actions: 
                    st.markdown(a)
                
                # Show final data shape
                current_data = st.session_state.processed_data if st.session_state.processed_data is not None else df
                st.markdown(f"**Current Data Shape:** {current_data.shape}")
                
                st.markdown("---")
                st.markdown("### 🚀 Next Steps")
                st.markdown("Proceed to **🛠️ Feature Selection & Relevance** to analyze feature importance.")
            else:
                st.info("No preprocessing steps have been applied yet.")
    
    # ==================== FEATURE SELECTION ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        st.markdown("### 📈 Exploratory Data Analysis")
        
        if 'Risk_Score' in df.columns:
            clean = df['Risk_Score'].dropna()
            if len(clean) > 0: 
                st.plotly_chart(plot_distribution(df, 'Risk_Score', 'Risk Score Distribution'), use_container_width=True)
        
        if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
            clean = df.dropna(subset=['MP_Count_per_L', 'Risk_Score'])
            if not clean.empty:
                try: 
                    fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score', 
                                   color='Risk_Level' if 'Risk_Level' in clean.columns else None, 
                                   trendline='ols', title='MP Count vs Risk Score')
                except: 
                    fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score', title='MP Count vs Risk Score')
                st.plotly_chart(fig, use_container_width=True)
        
        if 'Risk_Level' in df.columns and 'Risk_Score' in df.columns:
            clean = df.dropna(subset=['Risk_Score'])
            clean['Risk_Level'] = clean['Risk_Level'].astype(str)
            if len(clean) > 0: 
                st.plotly_chart(px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level', 
                                      title='Risk Score by Risk Level'), use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Feature Selection")
        target = st.selectbox("Target Variable", df.columns.tolist(), 
                             index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        nums = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        if target in nums: 
            nums.remove(target)
        
        if st.button("Calculate Feature Importance", type="primary", use_container_width=True):
            with st.spinner('Calculating feature importance...'):
                X = df[nums].copy()
                y = df[target].copy()
                X = X.fillna(X.median())
                if y.dtype == 'object': 
                    y = LabelEncoder().fit_transform(y)
                X = X.dropna(axis=1, how='any')
                
                mi_df = calculate_mutual_info(X, y)
                chi2_df = calculate_chi2(X, y)
                rf_df = calculate_rf_importance(X, y)
                
                st.session_state.feature_importance = rf_df
                st.session_state.mutual_info = mi_df
                st.session_state.chi2_scores = chi2_df
                st.session_state.selected_features = rf_df.head(10)['Feature'].tolist()
                
                ft1, ft2, ft3 = st.tabs(["🌲 Random Forest", "📊 Mutual Information", "🔢 Chi-squared"])
                with ft1: 
                    st.plotly_chart(px.bar(rf_df.head(20), x='Importance', y='Feature', orientation='h', 
                                          title='RF Feature Importance', height=500), use_container_width=True)
                with ft2: 
                    st.plotly_chart(px.bar(mi_df.head(20), x='Mutual_Info', y='Feature', orientation='h', 
                                          title='Mutual Information Scores', height=500), use_container_width=True)
                with ft3: 
                    st.plotly_chart(px.bar(chi2_df.head(20), x='Chi2_Score', y='Feature', orientation='h', 
                                          title='Chi-squared Scores', height=500), use_container_width=True)
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
        with c1: 
            ts = st.slider("Test Size", 0.1, 0.5, 0.2)
        with c2: 
            use_smote = st.checkbox("Use SMOTE", value=True)
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if len(features) == 0: 
                st.error("Select at least one feature!")
                return
            
            X = df[features].select_dtypes(include=['float64', 'int64', 'int32'])
            y = df[target]
            mask = y.notna()
            X = X[mask]
            y = y[mask]
            
            if y.dtype == 'object': 
                y = LabelEncoder().fit_transform(y)
            X = X.fillna(X.median())
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
            
            if use_smote:
                tc = pd.Series(y_train).value_counts()
                if tc.min() >= 2:
                    try: 
                        X_train, y_train = SMOTE(random_state=42, k_neighbors=min(5, tc.min()-1)).fit_resample(X_train, y_train)
                    except: 
                        pass
            
            models = {}
            try:
                lr = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', n_jobs=-1)
                lr.fit(X_train, y_train)
                models['Logistic Regression'] = lr
            except: 
                pass
            
            try:
                rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
                rf.fit(X_train, y_train)
                models['RandomForestClassifier'] = rf
            except: 
                pass
            
            try:
                gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
                gb.fit(X_train, y_train)
                models['GradientBoostingClassifier'] = gb
            except: 
                pass
            
            if models:
                st.session_state.models = models
                st.session_state.trained = True
                st.success(f"✅ {len(models)} models trained successfully!")
                for name, model in models.items():
                    y_pred = model.predict(X_test)
                    st.markdown(f"**{name}:** Accuracy={accuracy_score(y_test, y_pred):.4f} | F1={f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    # ==================== CROSS VALIDATION & EVALUATION ====================
    elif section == "📊 Cross Validation & Evaluation":
        st.markdown('<p class="section-header">📊 Cross Validation & Model Evaluation</p>', unsafe_allow_html=True)
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        
        et1, et2, et3, et4 = st.tabs([
            "📊 Evaluate Models", 
            "📊 Compare Both Targets", 
            "🔄 Cross Validation", 
            "📋 Pipeline Summary"
        ])
        
        with et1:
            st.markdown("### 📊 Evaluate Models on Risk_Type")
            tc = 'Risk_Type'
            if tc not in df.columns: 
                st.error(f"'{tc}' column not found!")
            elif st.button("🚀 Evaluate Models", type="primary", key="eval"):
                with st.spinner('Evaluating models...'):
                    results, info = train_and_evaluate_detailed(df, tc)
                    st.session_state.evaluation_ran = True
                
                if results:
                    for mn in ['Logistic Regression', 'RandomForestClassifier', 'GradientBoostingClassifier']:
                        if mn in results:
                            r = results[mn]
                            st.markdown(f"### {mn}")
                            st.markdown(f"Accuracy={r['accuracy']:.4f} | Precision={r['precision']:.4f} | Recall={r['recall']:.4f} | F1={r['f1_score']:.4f}")
                    
                    md = [{'Model': n, 'Accuracy': r['accuracy'], 'Precision': r['precision'], 
                          'Recall': r['recall'], 'F1': r['f1_score']} for n, r in results.items()]
                    mdf = pd.DataFrame(md)
                    st.dataframe(mdf, column_config={
                        "Accuracy": st.column_config.NumberColumn(format="%.6f"),
                        "Precision": st.column_config.NumberColumn(format="%.6f"),
                        "Recall": st.column_config.NumberColumn(format="%.6f"),
                        "F1": st.column_config.NumberColumn(format="%.6f")
                    }, use_container_width=True)
                    st.plotly_chart(px.bar(mdf, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1'], 
                                          barmode='group', height=400), use_container_width=True)
        
        with et2:
            st.markdown("### 📊 Compare Models on Both Targets")
            if st.button("🚀 Compare Targets", type="primary", key="cmp"):
                all_c = {}
                for tc in ['Risk_Type', 'Risk_Level']:
                    if tc not in df.columns: 
                        continue
                    with st.spinner(f'Processing {tc}...'):
                        results, _ = train_and_evaluate_detailed(df, tc)
                        all_c[tc] = results
                st.session_state.comparison_ran = True
                
                for tc, results in all_c.items():
                    st.markdown(f"## {tc}")
                    if results:
                        md = [{'Model': n, 'Accuracy': r['accuracy'], 'F1': r['f1_score']} 
                             for n, r in results.items()]
                        mdf = pd.DataFrame(md)
                        st.dataframe(mdf, column_config={
                            "Accuracy": st.column_config.NumberColumn(format="%.4f"),
                            "F1": st.column_config.NumberColumn(format="%.4f")
                        }, use_container_width=True, hide_index=True)
                        st.markdown("---")
        
        with et3:
            st.markdown("### 🔄 Cross Validation")
            target = st.selectbox("Target for CV", df.columns.tolist(), 
                                 index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
            nums = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
            if target in nums: 
                nums.remove(target)
            folds = st.slider("CV Folds", 3, 10, 5)
            
            if st.button("🔄 Run Cross Validation", type="primary", key="cv"):
                X = df[nums].copy()
                y = df[target].copy()
                mask = y.notna()
                X = X[mask]
                y = y[mask]
                
                if y.dtype == 'object': 
                    y = LabelEncoder().fit_transform(y)
                X = X.fillna(X.median())
                
                cv_models = {
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', n_jobs=-1),
                    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1),
                    'Gradient Boosting': GradientBoostingClassifier(n_estimators=50, random_state=42)
                }
                
                cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
                cv_res = []
                all_s = {}
                
                for nm, md in cv_models.items():
                    try:
                        acc = cross_val_score(md, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                        f1 = cross_val_score(md, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
                        all_s[nm] = f1
                        cv_res.append({
                            'Model': nm,
                            'Mean Accuracy': round(acc.mean(), 4),
                            'Std Accuracy': round(acc.std(), 4),
                            'Mean F1': round(f1.mean(), 4),
                            'Std F1': round(f1.std(), 4)
                        })
                    except: 
                        pass
                
                st.session_state.cv_ran = True
                
                if cv_res:
                    cv_df = pd.DataFrame(cv_res)
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
                    best = cv_df.loc[cv_df['Mean F1'].idxmax()]
                    st.success(f"🏆 Best Model: **{best['Model']}** (F1: {best['Mean F1']:.4f})")
                    
                    fig_cv = go.Figure()
                    for nm, sc in all_s.items(): 
                        fig_cv.add_trace(go.Box(y=sc, name=nm, boxmean='sd'))
                    fig_cv.update_layout(title=f'Cross Validation F1 Scores ({folds}-Fold)', yaxis_title='F1 Score', height=400)
                    st.plotly_chart(fig_cv, use_container_width=True)
        
        with et4:
            st.markdown("### 📋 Complete Pipeline Summary")
            if st.button("🔄 Generate Pipeline Summary", type="primary", key="pipe"):
                pd_data = []
                
                # Data loading
                if st.session_state.data is not None:
                    df = st.session_state.data
                    pd_data.append({
                        'Stage': '1. Data Loading',
                        'Step': 'Dataset Loaded',
                        'Status': '✅',
                        'Details': f'{df.shape[0]} rows × {df.shape[1]} columns'
                    })
                else:
                    pd_data.append({
                        'Stage': '1. Data Loading',
                        'Step': 'Dataset',
                        'Status': '❌',
                        'Details': 'No data loaded'
                    })
                
                # Preprocessing
                pd_data.append({
                    'Stage': '2. Preprocessing',
                    'Step': 'Feature Scaling',
                    'Status': '✅' if st.session_state.get('scaled_columns') is not None else '⬜',
                    'Details': f'{len(st.session_state.get("scaled_columns", []))} columns scaled' 
                    if st.session_state.get('scaled_columns') is not None else 'Not applied'
                })
                
                pd_data.append({
                    'Stage': '2. Preprocessing',
                    'Step': 'Categorical Encoding',
                    'Status': '✅' if st.session_state.get('encoded_data') is not None else '⬜',
                    'Details': f'Shape: {st.session_state.get("encoded_shape", "N/A")}'
                    if st.session_state.get('encoded_data') is not None else 'Not applied'
                })
                
                pd_data.append({
                    'Stage': '2. Preprocessing',
                    'Step': 'Outlier Handling',
                    'Status': '✅' if len(st.session_state.get('outlier_columns_processed', [])) > 0 else '⬜',
                    'Details': f'IQR method on {len(st.session_state.get("outlier_columns_processed", []))} columns'
                    if len(st.session_state.get('outlier_columns_processed', [])) > 0 else 'Not applied'
                })
                
                # Feature selection
                pd_data.append({
                    'Stage': '3. Feature Selection',
                    'Step': 'Feature Importance',
                    'Status': '✅' if st.session_state.get('feature_importance') is not None else '⬜',
                    'Details': 'MI, Chi², RF methods' if st.session_state.get('feature_importance') is not None else 'Not computed'
                })
                
                # Modeling
                pd_data.append({
                    'Stage': '4. Modeling',
                    'Step': 'Models Trained',
                    'Status': '✅' if st.session_state.get('trained') else '⬜',
                    'Details': f'{len(st.session_state.get("models", {}))} models'
                    if st.session_state.get('trained') else 'Not trained'
                })
                
                # Evaluation
                pd_data.append({
                    'Stage': '5. Evaluation',
                    'Step': 'Model Evaluation',
                    'Status': '✅' if st.session_state.get('evaluation_ran') else '⬜',
                    'Details': 'Accuracy, Precision, Recall, F1' if st.session_state.get('evaluation_ran') else 'Not evaluated'
                })
                
                pd_data.append({
                    'Stage': '5. Evaluation',
                    'Step': 'Target Comparison',
                    'Status': '✅' if st.session_state.get('comparison_ran') else '⬜',
                    'Details': 'Risk_Type & Risk_Level' if st.session_state.get('comparison_ran') else 'Not compared'
                })
                
                pd_data.append({
                    'Stage': '5. Evaluation',
                    'Step': 'Cross Validation',
                    'Status': '✅' if st.session_state.get('cv_ran') else '⬜',
                    'Details': 'K-Fold validation' if st.session_state.get('cv_ran') else 'Not performed'
                })
                
                pdf = pd.DataFrame(pd_data)
                st.dataframe(pdf, column_config={
                    "Stage": "Pipeline Stage",
                    "Step": "Step",
                    "Status": "Status",
                    "Details": "Details"
                }, use_container_width=True, height=400)
                
                # Progress bar
                completed = sum(1 for d in pd_data if d['Status'] == '✅')
                total = len(pd_data)
                progress = int((completed / total) * 100) if total > 0 else 0
                st.progress(progress, text=f"Pipeline Progress: {progress}% ({completed}/{total} steps completed)")
                
                if progress == 100:
                    st.success("🎉 Complete pipeline executed successfully!")
                elif progress >= 60:
                    st.info("📊 Pipeline partially complete. Continue with remaining steps.")
                else:
                    st.warning("🚧 Pipeline in early stages. Complete preprocessing and modeling steps.")


if __name__ == "__main__":
    main()
