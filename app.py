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

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-top: 1rem; margin-bottom: 1rem; }
    .stButton > button { width: 100%; background-color: #1f77b4; color: white; font-weight: 600; border-radius: 8px; padding: 0.5rem 1rem; }
    .stButton > button:hover { background-color: #2980b9; border-color: #2980b9; }
    .stMarkdown, .stMarkdown p, .stMarkdown li { color: #2c3e50 !important; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    if 'data' not in st.session_state: st.session_state.data = None
    if 'processed_data' not in st.session_state: st.session_state.processed_data = None
    if 'models' not in st.session_state: st.session_state.models = {}
    if 'feature_importance' not in st.session_state: st.session_state.feature_importance = None
    if 'mutual_info' not in st.session_state: st.session_state.mutual_info = None
    if 'chi2_scores' not in st.session_state: st.session_state.chi2_scores = None
    if 'X_selected' not in st.session_state: st.session_state.X_selected = None
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

init_session_state()

# ==================== FUNCTIONS ====================

def load_dataset(uploaded_file):
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
            st.error("Unsupported file format.")
            return None
        st.session_state.data = data
        st.success(f"✅ Dataset loaded! Shape: {data.shape}")
        return data
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def generate_sample_data():
    np.random.seed(42)
    n = 1000
    data = {
        'Sample_ID': [f'MP_{i:04d}' for i in range(n)],
        'Latitude': np.random.uniform(12.8, 13.0, n),
        'Longitude': np.random.uniform(123.9, 124.1, n),
        'MP_Count_per_L': np.random.poisson(lam=50, size=n),
        'Microplastic_Size_mm': np.random.choice(['0.1-5.0', '5.0-10.0', '0.1-1.0'], n),
        'Density': np.random.choice(['1.3-1.4', '1.2-1.3', '1.0-1.2'], n),
        'Particle_Size_um': np.random.normal(100, 30, n),
        'Polymer_Type': np.random.choice(['PE', 'PP', 'PS', 'PET', 'PVC', 'Nylon'], n),
        'Water_Source': np.random.choice(['River', 'Lake', 'Ocean', 'Groundwater', 'Tap'], n),
        'pH': np.random.normal(7, 0.5, n),
        'Temperature_C': np.random.normal(20, 5, n),
        'Risk_Score': np.random.uniform(0, 100, n),
        'Risk_Level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n, p=[0.3,0.35,0.25,0.1]),
        'Risk_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n, p=[0.5,0.3,0.2]),
        'Location': np.random.choice(['Urban', 'Rural', 'Industrial', 'Coastal'], n),
        'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n),
        'Author': np.random.choice(['Author_A', 'Author_B', 'Author_C'], n),
        'Source': np.random.choice(['Source_1', 'Source_2', 'Source_3'], n)
    }
    df = pd.DataFrame(data)
    for col in df.columns:
        if col != 'Sample_ID' and df[col].dtype in ['float64', 'int64']:
            df.loc[np.random.random(n) < 0.05, col] = np.nan
    return df

def one_hot_encode(df):
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        cols_to_encode = [c for c in categorical_cols if 'ID' not in c and 'Sample' not in c]
        if len(cols_to_encode) == 0: return df, [], [], df.shape
        df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
        new_cols = [c for c in df_encoded.columns if c not in df.columns]
        return df_encoded, new_cols, cols_to_encode, df_encoded.shape
    except: return df, [], [], df.shape

def detect_outliers(df, columns):
    info = {}
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            out = df[(df[col] < lo) | (df[col] > hi)]
            info[col] = {'count': len(out), 'percentage': (len(out)/len(df))*100 if len(df)>0 else 0}
    return info

def cap_outliers_iqr(df, columns):
    df_capped = df.copy()
    for col in columns:
        if df_capped[col].dtype in ['float64', 'int64']:
            Q1, Q3 = df_capped[col].quantile(0.25), df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            df_capped[col] = df_capped[col].clip(lower=Q1 - 1.5*IQR, upper=Q3 + 1.5*IQR)
    return df_capped

def analyze_skewness(df, columns):
    info = []
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            s = df[col].skew()
            info.append({'Column': col, 'Skewness': round(s,4), 'Skewed (>0.5)': 'Yes' if abs(s)>0.5 else 'No'})
    return pd.DataFrame(info)

def apply_log_transform(df, columns):
    df_t = df.copy()
    for col in columns:
        if df_t[col].dtype in ['float64', 'int64']:
            if abs(df_t[col].skew()) > 0.5:
                shift = abs(df_t[col].min()) + 1 if df_t[col].min() <= 0 else 0
                df_t[col] = np.log1p(df_t[col] + shift)
    return df_t

def calculate_mutual_info(X, y):
    scores = mutual_info_classif(X, y, random_state=42)
    return pd.DataFrame({'Feature': X.columns, 'Mutual_Info': scores}).sort_values('Mutual_Info', ascending=False)

def calculate_chi2(X, y):
    X_s = X - X.min() + 1
    scores, pvals = chi2(X_s, y)
    return pd.DataFrame({'Feature': X.columns, 'Chi2_Score': scores, 'P_Value': pvals}).sort_values('Chi2_Score', ascending=False)

def calculate_rf_importance(X, y):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)

def train_and_evaluate_detailed(df, target_col):
    feature_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
    if target_col in feature_cols: feature_cols.remove(target_col)
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    mask = y.notna(); X = X[mask]; y = y[mask]
    if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
    X = X.fillna(X.median())
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    info = {'X_train': X_train.shape, 'X_test': X_test.shape, 'y_train': y_train.shape, 'y_test': y_test.shape, 'target': target_col}
    
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
    
    results = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'cm': confusion_matrix(y_test, y_pred),
            'report': classification_report(y_test, y_pred, zero_division=0)
        }
    return results, info

def plot_distribution(data, column, title):
    try:
        clean = data[column].dropna()
        if clean.empty: return go.Figure()
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
        fig.add_trace(go.Histogram(x=clean, nbinsx=30, marker_color='#3498db'), row=1, col=1)
        fig.add_trace(go.Box(y=clean, marker_color='#e74c3c'), row=1, col=2)
        fig.update_layout(title_text=title, showlegend=False, height=500)
        return fig
    except: return go.Figure()

# ==================== MAIN APP ====================

def main():
    st.markdown('<p class="main-header">🔬 Microplastic Risk Analysis Dashboard</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio("Select Section", [
        "🏠 Home", "🔧 Preprocessing", "🛠️ Feature Selection & Relevance", 
        "🤖 Modeling", "📊 Cross Validation & Evaluation"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard analyzes microplastic risk data.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Status")
    if st.session_state.data is not None: st.sidebar.success("✅ Data Loaded")
    else: st.sidebar.warning("⚠️ No Data")
    if st.session_state.trained: st.sidebar.success("✅ Models Trained")
    else: st.sidebar.warning("⚠️ Models Not Trained")
    
    # ==================== HOME ====================
    if section == "🏠 Home":
        st.markdown('<p class="section-header">🏠 Home - Upload Dataset</p>', unsafe_allow_html=True)
        
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
            st.markdown("---")
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Samples", df.shape[0])
            with c2: st.metric("Features", df.shape[1])
            with c3: st.metric("Missing", df.isnull().sum().sum())
            st.dataframe(df.head(10), use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📏 Feature Scaling Preview")
            if st.button("🔧 Apply StandardScaler", type="primary", key="scale_home"):
                with st.spinner('Scaling...'):
                    nums = df.select_dtypes(include=['float64','int64']).columns.tolist()
                    cols = [c for c in nums if 'ID' not in c and 'Sample' not in c]
                    if len(cols) > 0:
                        scaler = StandardScaler()
                        sd = scaler.fit_transform(df[cols].fillna(df[cols].median()))
                        sdf = pd.DataFrame(sd, columns=cols)
                        st.session_state.scaler = scaler
                        st.session_state.scaled_columns = cols
                        st.session_state.scaled_data = sdf
                        st.success(f"✅ {len(cols)} columns scaled!")
                        st.dataframe(sdf.head(), column_config={c: st.column_config.NumberColumn(c,format="%.6f") for c in cols}, use_container_width=True)
            
            if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
                st.markdown("---")
                st.markdown("### 🔬 MP Count vs Risk Score")
                clean = df.dropna(subset=['MP_Count_per_L','Risk_Score'])
                if len(clean) > 0:
                    try:
                        fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                        color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                        trendline='ols', title='MP Count vs Risk Score', opacity=0.7)
                    except:
                        fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score', title='MP Count vs Risk Score', opacity=0.7)
                    st.plotly_chart(fig, use_container_width=True)
            
            if 'Risk_Score' in df.columns and 'Risk_Level' in df.columns:
                st.markdown("---")
                st.markdown("### 📊 Risk Score by Risk Level")
                clean = df.dropna(subset=['Risk_Score'])
                clean['Risk_Level'] = clean['Risk_Level'].astype(str)
                if len(clean) > 0:
                    fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                                title='Risk Score by Risk Level', points='outliers')
                    st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 🔍 Data Quality Check")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Missing %", f"{(df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%")
            with c2: st.metric("Duplicates", df.duplicated().sum())
            with c3: st.metric("Numeric Cols", len(df.select_dtypes(include=['float64','int64']).columns))
            with c4: st.metric("Categorical Cols", len(df.select_dtypes(include=['object']).columns))
    
    # ==================== PREPROCESSING ====================
    elif section == "🔧 Preprocessing":
        st.markdown('<p class="section-header">🔧 Data Preprocessing</p>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            return
        
        df = st.session_state.data.copy()
        
        p1, p2, p3, p4, p5 = st.tabs(["📏 Feature Scaling", "🔄 Categorical Encoding", "🎯 Outlier Capping", "📊 Skewness & Transform", "📋 Summary"])
        
        with p1:
            st.markdown("### 📏 Perform Feature Scaling")
            numeric_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
            cols = [c for c in numeric_cols if 'ID' not in c and 'Sample' not in c]
            if st.button("🔧 Apply StandardScaler", type="primary", key="scale_tab"):
                with st.spinner('Applying...'):
                    if len(cols) > 0:
                        scaler = StandardScaler()
                        sd = scaler.fit_transform(df[cols].fillna(df[cols].median()))
                        sdf = pd.DataFrame(sd, columns=cols)
                        st.session_state.scaler = scaler
                        st.session_state.scaled_columns = cols
                        st.session_state.scaled_data = sdf
                        st.success(f"✅ {len(cols)} columns scaled!")
                        st.dataframe(sdf.head(), column_config={c: st.column_config.NumberColumn(c,format="%.6f") for c in cols}, use_container_width=True)
        
        with p2:
            st.markdown("### 🔄 Encode Categorical Variables")
            cat_cols = df.select_dtypes(include=['object']).columns.tolist()
            cols_enc = [c for c in cat_cols if 'ID' not in c and 'Sample' not in c]
            if len(cols_enc) > 0: st.markdown(f"**Categorical ({len(cols_enc)}):** {', '.join(cols_enc)}")
            if st.button("🔄 Apply One-Hot Encoding", type="primary", key="encode_tab"):
                with st.spinner('Applying...'):
                    if len(cols_enc) > 0:
                        enc_df, new_cols, _, enc_shape = one_hot_encode(df)
                        st.session_state.encoded_data = enc_df
                        st.session_state.encoded_shape = enc_shape
                        st.success(f"✅ Created {len(new_cols)} new columns! Shape: {enc_shape}")
                        st.dataframe(enc_df.head(), use_container_width=True)
        
        with p3:
            st.markdown("### 🎯 Address Outliers")
            num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
            cols_out = [c for c in num_cols if 'ID' not in c and 'Sample' not in c]
            if len(cols_out) > 0:
                oi = detect_outliers(df, cols_out)
                st.dataframe(pd.DataFrame([{'Column':c,'Outliers':i['count'],'%':f"{i['percentage']:.1f}%"} for c,i in oi.items()]), use_container_width=True, hide_index=True)
            if st.button("🎯 Cap Outliers (IQR)", type="primary", key="outlier_tab"):
                if len(cols_out) > 0:
                    df_capped = cap_outliers_iqr(df, cols_out)
                    st.session_state.processed_data = df_capped
                    st.success("✅ Outliers capped!")
        
        with p4:
            st.markdown("### 📊 Skewness & Log Transform")
            num_cols = df.select_dtypes(include=['float64','int64']).columns.tolist()
            cols_sk = [c for c in num_cols if 'ID' not in c and 'Sample' not in c]
            if len(cols_sk) > 0:
                sk_df = analyze_skewness(df, cols_sk)
                st.dataframe(sk_df, use_container_width=True, hide_index=True)
            if st.button("📊 Apply Log Transform", type="primary", key="skew_tab"):
                if len(cols_sk) > 0:
                    df_t = apply_log_transform(df, cols_sk)
                    st.session_state.processed_data = df_t
                    st.success("✅ Log transform applied!")
        
        with p5:
            st.markdown("### 📋 Summary")
            actions = []
            if st.session_state.get('scaled_data'): actions.append("✅ Feature Scaling applied")
            if st.session_state.get('encoded_data'): actions.append("✅ Categorical Encoding applied")
            if st.session_state.get('processed_data'): actions.append("✅ Outliers handled")
            if actions:
                for a in actions: st.markdown(a)
                st.markdown("---\n### 🚀 Next Steps\nProceed to **🛠️ Feature Selection & Relevance**.")
    
    # ==================== FEATURE SELECTION & RELEVANCE ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data.copy()
        
        st.markdown("### 📈 Exploratory Data Analysis")
        
        if 'Risk_Score' in df.columns:
            clean = df['Risk_Score'].dropna()
            if len(clean) > 0:
                st.plotly_chart(plot_distribution(df, 'Risk_Score', 'Risk Score Distribution'), use_container_width=True)
        
        if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
            st.markdown("---")
            st.markdown("#### 🔬 MP Count vs Risk Score")
            clean = df.dropna(subset=['MP_Count_per_L','Risk_Score'])
            if not clean.empty:
                try:
                    fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                    color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                    trendline='ols', title='MP Count vs Risk Score')
                except:
                    fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score', title='MP Count vs Risk Score')
                st.plotly_chart(fig, use_container_width=True)
        
        if 'Risk_Level' in df.columns and 'Risk_Score' in df.columns:
            st.markdown("---")
            st.markdown("#### 📊 Risk Score by Risk Level")
            clean = df.dropna(subset=['Risk_Score'])
            clean['Risk_Level'] = clean['Risk_Level'].astype(str)
            if len(clean) > 0:
                fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level', title='Risk Score by Risk Level')
                st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Feature Selection Methods")
        st.info("**📖 Higher scores = More important features**")
        
        target_col = st.selectbox("Select Target Variable", df.columns.tolist(),
                                  index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        if target_col in numeric_cols: numeric_cols.remove(target_col)
        
        if st.button("Calculate Feature Importance", type="primary", use_container_width=True):
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
                st.session_state.selected_features = rf_df.head(10)['Feature'].tolist()
                
                ft1, ft2, ft3 = st.tabs(["🌲 Random Forest", "📊 Mutual Information", "🔢 Chi-squared"])
                
                with ft1:
                    st.markdown("**Top 20 features - RandomForest Importance:**")
                    fig = px.bar(rf_df.head(20), x='Importance', y='Feature', orientation='h',
                               title='Random Forest Feature Importance', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(rf_df.head(20), use_container_width=True, hide_index=True)
                
                with ft2:
                    st.markdown("**Top 20 features - Mutual Information:**")
                    fig = px.bar(mi_df.head(20), x='Mutual_Info', y='Feature', orientation='h',
                               title='Mutual Information Scores', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(mi_df.head(20), use_container_width=True, hide_index=True)
                
                with ft3:
                    st.markdown("**Top 20 features - Chi-squared Test:**")
                    fig = px.bar(chi2_df.head(20), x='Chi2_Score', y='Feature', orientation='h',
                               title='Chi-squared Test Scores', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(chi2_df.head(20), use_container_width=True, hide_index=True)
                
                st.success(f"✅ Top {len(st.session_state.selected_features)} features selected!")
                st.markdown(f"**Selected Features:** {', '.join(st.session_state.selected_features[:5])}...")
    
    # ==================== MODELING ====================
    elif section == "🤖 Modeling":
        st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data
        
        target = st.selectbox("Target", df.columns.tolist(), key='train_target')
        all_f = [c for c in df.columns if c != target]
        default = st.session_state.get('selected_features', df.select_dtypes(include=['float64','int64']).columns.tolist()[:5])
        default = [f for f in default if f in all_f]
        features = st.multiselect("Features", all_f, default=default)
        c1,c2 = st.columns(2)
        with c1: ts = st.slider("Test Size", 0.1, 0.5, 0.2)
        with c2: use_smote = st.checkbox("Use SMOTE", value=True)
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if len(features) == 0: st.error("Select features!"); return
            X = df[features].select_dtypes(include=['float64','int64','int32'])
            y = df[target]
            mask = y.notna(); X = X[mask]; y = y[mask]
            if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
            X = X.fillna(X.median())
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
            if use_smote:
                tc = pd.Series(y_train).value_counts()
                if tc.min() >= 2:
                    try: X_train, y_train = SMOTE(random_state=42, k_neighbors=min(5,tc.min()-1)).fit_resample(X_train, y_train)
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
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data.copy()
        
        eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs([
            "📊 Evaluate Models", 
            "📊 Compare Both Targets",
            "🔄 Cross Validation",
            "📋 Pipeline Summary"
        ])
        
        # ===== TAB 1: EVALUATE MODELS =====
        with eval_tab1:
            st.markdown("### 📊 Evaluate the Models")
            
            target_col = 'Risk_Type'
            if target_col not in df.columns:
                st.error(f"❌ '{target_col}' column not found!")
            else:
                if st.button("🚀 Evaluate Models", type="primary", key="eval_detail"):
                    with st.spinner('Training and evaluating...'):
                        results, info = train_and_evaluate_detailed(df, target_col)
                        st.session_state.evaluation_ran = True
                    
                    if results:
                        st.markdown(f"**Target:** {info['target']} | X_train: {info['X_train']} | X_test: {info['X_test']}")
                        st.markdown("---")
                        
                        for model_name in ['Logistic Regression', 'RandomForestClassifier', 'GradientBoostingClassifier']:
                            if model_name in results:
                                res = results[model_name]
                                st.markdown(f"### # Evaluate {model_name} Model")
                                st.markdown(f"**Accuracy:** {res['accuracy']:.4f} | **Precision:** {res['precision']:.4f} | **Recall:** {res['recall']:.4f} | **F1-Score:** {res['f1_score']:.4f}")
                                st.markdown("---")
                        
                        metrics_data = []
                        for name, res in results.items():
                            metrics_data.append({'Model': name, 'Accuracy': res['accuracy'], 'Precision': res['precision'], 'Recall': res['recall'], 'F1-Score': res['f1_score']})
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        st.markdown("### Model Performance Comparison")
                        st.dataframe(metrics_df, column_config={
                            "Model": "Model", "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.6f"),
                            "Precision": st.column_config.NumberColumn("Precision", format="%.6f"),
                            "Recall": st.column_config.NumberColumn("Recall", format="%.6f"),
                            "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.6f"),
                        }, use_container_width=True)
                        
                        fig = px.bar(metrics_df, x='Model', y=['Accuracy','Precision','Recall','F1-Score'],
                                    barmode='group', title='Model Performance', height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        best_acc = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
                        best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
                        st.success(f"Best Accuracy: **{best_acc['Model']}** ({best_acc['Accuracy']:.4f}) | Best F1: **{best_f1['Model']}** ({best_f1['F1-Score']:.4f})")
        
        # ===== TAB 2: COMPARE BOTH TARGETS =====
        with eval_tab2:
            st.markdown("### 📊 Compare Model Performance (Both Targets)")
            
            if st.button("🚀 Train & Compare Both Targets", type="primary", key="compare_both"):
                all_comp = {}
                for tc in ['Risk_Type', 'Risk_Level']:
                    if tc not in df.columns: continue
                    with st.spinner(f'Training {tc}...'):
                        results, _ = train_and_evaluate_detailed(df, tc)
                        all_comp[tc] = results
                
                st.session_state.comparison_ran = True
                
                for tc, results in all_comp.items():
                    st.markdown(f"## 📊 Analysis for **'{tc}'**")
                    if results:
                        md = [{'Model':n,'Accuracy':r['accuracy'],'F1-Score':r['f1_score']} for n,r in results.items()]
                        mdf = pd.DataFrame(md)
                        best_a = mdf.loc[mdf['Accuracy'].idxmax()]
                        best_f = mdf.loc[mdf['F1-Score'].idxmax()]
                        st.success(f"Best Acc: **{best_a['Model']}** ({best_a['Accuracy']:.4f}) | Best F1: **{best_f['Model']}** ({best_f['F1-Score']:.4f})")
                        st.dataframe(mdf, column_config={"Model":"Model","Accuracy":st.column_config.NumberColumn("Accuracy",format="%.4f"),"F1-Score":st.column_config.NumberColumn("F1-Score",format="%.4f")}, use_container_width=True, hide_index=True)
                        st.markdown("---")
        
        # ===== TAB 3: CROSS VALIDATION =====
        with eval_tab3:
            st.markdown("### 🔄 Cross Validation Analysis")
            
            target = st.selectbox("Target for CV", df.columns.tolist(),
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
                        cv_results.append({'Model':name,'Mean Acc':round(acc.mean(),4),'Std Acc':round(acc.std(),4),
                                          'Mean F1':round(f1.mean(),4),'Std F1':round(f1.std(),4)})
                    except: pass
                
                st.session_state.cv_ran = True
                
                if cv_results:
                    cv_df = pd.DataFrame(cv_results)
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
                    best_cv = cv_df.loc[cv_df['Mean F1'].idxmax()]
                    st.success(f"🏆 Best CV Model: **{best_cv['Model']}** (F1: {best_cv['Mean F1']:.4f})")
                    
                    fig_cv = go.Figure()
                    for name, scores in all_scores.items():
                        fig_cv.add_trace(go.Box(y=scores, name=name, boxmean='sd'))
                    fig_cv.update_layout(title=f'CV F1 Scores ({folds}-Fold)', yaxis_title='F1 Score', height=400)
                    st.plotly_chart(fig_cv, use_container_width=True)
        
        # ===== TAB 4: PIPELINE SUMMARY =====
        with eval_tab4:
            st.markdown("### 📋 Overall Pipeline Summary")
            
            if st.button("🔄 Generate Pipeline Summary", type="primary", key="pipeline_summary", use_container_width=True):
                
                pipeline_data = []
                
                if st.session_state.data is not None:
                    df = st.session_state.data
                    pipeline_data.append({'Stage': '1. Data Loading', 'Step': 'Dataset Loaded', 'Status': '✅', 'Details': f'Shape: {df.shape[0]} × {df.shape[1]}'})
                    pipeline_data.append({'Stage': '1. Data Loading', 'Step': 'Missing Values', 'Status': '✅', 'Details': f'{df.isnull().sum().sum()} missing'})
                else:
                    pipeline_data.append({'Stage': '1. Data Loading', 'Step': 'Dataset', 'Status': '❌', 'Details': 'No data loaded'})
                
                pipeline_data.append({'Stage': '2. Preprocessing', 'Step': 'Feature Scaling', 'Status': '✅' if st.session_state.get('scaled_data') else '⬜', 'Details': f'{len(st.session_state.get("scaled_columns",[]))} cols' if st.session_state.get('scaled_data') else 'Not applied'})
                pipeline_data.append({'Stage': '2. Preprocessing', 'Step': 'One-Hot Encoding', 'Status': '✅' if st.session_state.get('encoded_data') else '⬜', 'Details': f'Shape: {st.session_state.get("encoded_shape","N/A")}' if st.session_state.get('encoded_data') else 'Not applied'})
                pipeline_data.append({'Stage': '2. Preprocessing', 'Step': 'Outlier Capping', 'Status': '✅' if st.session_state.get('processed_data') else '⬜', 'Details': 'IQR method' if st.session_state.get('processed_data') else 'Not applied'})
                
                pipeline_data.append({'Stage': '3. Feature Selection', 'Step': 'Feature Importance', 'Status': '✅' if st.session_state.get('feature_importance') else '⬜', 'Details': 'MI, Chi2, RF' if st.session_state.get('feature_importance') else 'Not run'})
                
                pipeline_data.append({'Stage': '4. Modeling', 'Step': 'Models Trained', 'Status': '✅' if st.session_state.get('trained') else '⬜', 'Details': f'{len(st.session_state.get("models",{}))} models' if st.session_state.get('trained') else 'Not trained'})
                
                pipeline_data.append({'Stage': '5. Evaluation', 'Step': 'Model Evaluation', 'Status': '✅' if st.session_state.get('evaluation_ran') else '⬜', 'Details': 'Metrics computed' if st.session_state.get('evaluation_ran') else 'Not run'})
                pipeline_data.append({'Stage': '5. Evaluation', 'Step': 'Target Comparison', 'Status': '✅' if st.session_state.get('comparison_ran') else '⬜', 'Details': 'Both targets' if st.session_state.get('comparison_ran') else 'Not run'})
                pipeline_data.append({'Stage': '5. Evaluation', 'Step': 'Cross Validation', 'Status': '✅' if st.session_state.get('cv_ran') else '⬜', 'Details': 'K-Fold CV' if st.session_state.get('cv_ran') else 'Not run'})
                
                pipeline_df = pd.DataFrame(pipeline_data)
                
                st.markdown("### 📊 Complete Analysis Pipeline")
                st.dataframe(pipeline_df, column_config={
                    "Stage": st.column_config.TextColumn("Stage", width="small"),
                    "Step": st.column_config.TextColumn("Step", width="medium"),
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Details": st.column_config.TextColumn("Details", width="large"),
                }, use_container_width=True, height=450)
                
                completed = sum(1 for d in pipeline_data if d['Status'] == '✅')
                not_run = sum(1 for d in pipeline_data if d['Status'] == '⬜')
                failed = sum(1 for d in pipeline_data if d['Status'] == '❌')
                total = len(pipeline_data)
                
                st.markdown("---")
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("✅ Completed", completed)
                with c2: st.metric("⬜ Pending", not_run)
                with c3: st.metric("❌ Failed", failed)
                st.progress(int((completed/total)*100) if total>0 else 0, text=f"Progress: {int((completed/total)*100)}% ({completed}/{total})")


if __name__ == "__main__":
    main()
