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
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.feature_selection import mutual_info_classif, chi2
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(page_title="Microplastic Risk Analysis Dashboard", page_icon="🔬", layout="wide", initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-top: 1rem; margin-bottom: 1rem; }
    .stButton > button { width: 100%; background-color: #1f77b4; color: white; font-weight: 600; border-radius: 8px; padding: 0.5rem 1rem; }
    .stButton > button:hover { background-color: #2980b9; border-color: #2980b9; }
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

init_session_state()

def load_dataset(uploaded_file):
    """Load dataset from uploaded file"""
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
    """Generate sample microplastic dataset with outliers"""
    np.random.seed(42)
    n = 1000
    
    data = {
        'Sample_ID': [f'MP_{i:04d}' for i in range(n)],
        'MP_Count_per_L': np.random.normal(2, 1.5, n),
        'Particle_Size_um': np.random.normal(100, 30, n),
        'Microplastic_Size_mm_midpoint': np.random.normal(2.5, 2.0, n),
        'Density_midpoint': np.random.normal(1.0, 0.1, n),
        'Polymer_Type': np.random.choice(['PE','PP','PS','PET','PVC','Nylon'], n),
        'Water_Source': np.random.choice(['River','Lake','Ocean','Groundwater','Tap'], n),
        'pH': np.random.normal(7, 0.5, n), 
        'Temperature_C': np.random.normal(20, 5, n),
        'Risk_Score': np.random.uniform(0, 1.5, n),
        'Risk_Level': np.random.choice(['Low','Medium','High','Critical'], n, p=[0.3,0.35,0.25,0.1]),
        'Risk_Type': np.random.choice(['Type_A','Type_B','Type_C'], n, p=[0.5,0.3,0.2]),
        'Location': np.random.choice(['Urban','Rural','Industrial','Coastal'], n),
        'Season': np.random.choice(['Winter','Spring','Summer','Fall'], n),
        'Author': np.random.choice(['Author_A','Author_B','Author_C'], n),
        'Source': np.random.choice(['Source_1','Source_2','Source_3'], n)
    }
    
    df = pd.DataFrame(data)
    
    # Add outliers to specific columns
    for col in ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']:
        outlier_indices = np.random.choice(n, size=int(n*0.05), replace=False)
        if col == 'MP_Count_per_L':
            df.loc[outlier_indices, col] = np.random.uniform(8, 15, len(outlier_indices))
        elif col == 'Risk_Score':
            df.loc[outlier_indices, col] = np.random.uniform(2, 3, len(outlier_indices))
        elif col == 'Microplastic_Size_mm_midpoint':
            df.loc[outlier_indices, col] = np.random.uniform(10, 20, len(outlier_indices))
        elif col == 'Density_midpoint':
            df.loc[outlier_indices, col] = np.random.uniform(1.5, 2.0, len(outlier_indices))
    
    # Add some missing values (to get count less than 1000)
    for col in ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']:
        df.loc[np.random.random(n) < 0.077, col] = np.nan
    
    return df

def one_hot_encode(df):
    """Perform one-hot encoding"""
    cats = df.select_dtypes(include=['object']).columns.tolist()
    cols = [c for c in cats if 'ID' not in c and 'Sample' not in c]
    if len(cols) == 0: 
        return df, [], [], df.shape
    df_enc = pd.get_dummies(df, columns=cols, drop_first=False)
    new = [c for c in df_enc.columns if c not in df.columns]
    return df_enc, new, cols, df_enc.shape

def analyze_skewness(df, columns):
    """Analyze skewness"""
    info = []
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            s = df[col].skew()
            info.append({'Column': col, 'Skewness': round(s, 4), 'Skewed': 'Yes' if abs(s) > 0.5 else 'No'})
    return pd.DataFrame(info)

def apply_log_transform(df, columns):
    """Apply log transformation"""
    df_t = df.copy()
    for col in columns:
        if df_t[col].dtype in ['float64', 'int64'] and abs(df_t[col].skew()) > 0.5:
            shift = abs(df_t[col].min()) + 1 if df_t[col].min() <= 0 else 0
            df_t[col] = np.log1p(df_t[col] + shift)
    return df_t

def calculate_mutual_info(X, y):
    """Calculate mutual information"""
    scores = mutual_info_classif(X, y, random_state=42)
    return pd.DataFrame({'Feature': X.columns, 'Mutual_Info': scores}).sort_values('Mutual_Info', ascending=False)

def calculate_chi2(X, y):
    """Calculate chi-squared scores"""
    X_s = X - X.min() + 1
    scores, pvals = chi2(X_s, y)
    return pd.DataFrame({'Feature': X.columns, 'Chi2_Score': scores, 'P_Value': pvals}).sort_values('Chi2_Score', ascending=False)

def calculate_rf_importance(X, y):
    """Calculate Random Forest importance"""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)

def train_and_evaluate(df, target_col):
    """Train and evaluate models"""
    feature_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
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
    
    return results

def plot_distribution(data, column, title):
    """Create distribution plot"""
    try:
        clean = data[column].dropna()
        if clean.empty: 
            return go.Figure()
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
        fig.add_trace(go.Histogram(x=clean, nbinsx=30, marker_color='#3498db'), row=1, col=1)
        fig.add_trace(go.Box(y=clean, marker_color='#e74c3c'), row=1, col=2)
        fig.update_layout(title_text=title, showlegend=False, height=500)
        return fig
    except:
        return go.Figure()

def main():
    st.markdown('<p class="main-header">🔬 Microplastic Risk Analysis Dashboard</p>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio("Select Section", [
        "🏠 Home", 
        "🔧 Preprocessing", 
        "🛠️ Feature Selection & Relevance", 
        "🤖 Modeling", 
        "📊 Cross Validation & Evaluation"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard analyzes microplastic risk data.")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Status")
    if st.session_state.data is not None: 
        st.sidebar.success("✅ Data Loaded")
    else: 
        st.sidebar.warning("⚠️ No Data")
    if st.session_state.trained: 
        st.sidebar.success("✅ Models Trained")
    else: 
        st.sidebar.warning("⚠️ Models Not Trained")
    
    # ==================== HOME ====================
    if section == "🏠 Home":
        st.markdown('<p class="section-header">🏠 Home - Dataset Overview</p>', unsafe_allow_html=True)
        
        home_tab1, home_tab2, home_tab3 = st.tabs(["📤 Upload & Preview", "📊 Risk Score Distribution", "🔍 Data Quality"])
        
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
                    st.success("✅ Sample dataset generated!")
                    st.rerun()
            
            if st.session_state.data is not None:
                df = st.session_state.data
                c1, c2, c3 = st.columns(3)
                with c1: st.metric("Samples", df.shape[0])
                with c2: st.metric("Features", df.shape[1])
                with c3: st.metric("Missing", df.isnull().sum().sum())
                st.dataframe(df.head(10), use_container_width=True)
        
        with home_tab2:
            st.markdown("### 📊 Risk Score Distribution")
            if st.session_state.data is not None:
                df = st.session_state.data
                if 'Risk_Score' in df.columns:
                    st.plotly_chart(plot_distribution(df, 'Risk_Score', 'Risk Score Distribution'), use_container_width=True)
        
        with home_tab3:
            st.markdown("### 🔍 Data Quality Check")
            if st.session_state.data is not None:
                df = st.session_state.data
                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Missing %", f"{(df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%")
                with c2: st.metric("Duplicates", df.duplicated().sum())
                with c3: st.metric("Numeric Cols", len(df.select_dtypes(include=['float64','int64']).columns))
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
            "📏 Feature Scaling", 
            "🔄 Categorical Encoding", 
            "🎯 Outlier Handling", 
            "📊 Skewness & Transform", 
            "📋 Summary"
        ])
        
        with p1:
            st.markdown("### 📏 Feature Scaling")
            nums = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols = [c for c in nums if 'ID' not in c and 'Sample' not in c]
            
            if len(cols) > 0:
                st.markdown(f"**Available columns ({len(cols)}):** {', '.join(cols[:5])}...")
                
                if st.button("🔧 Apply StandardScaler", type="primary"):
                    with st.spinner('Scaling...'):
                        scaler = StandardScaler()
                        sd = scaler.fit_transform(df[cols].fillna(df[cols].median()))
                        sdf = pd.DataFrame(sd, columns=cols)
                        st.session_state.scaler = scaler
                        st.session_state.scaled_columns = cols
                        st.session_state.scaled_data = sdf
                        st.success(f"✅ {len(cols)} columns scaled!")
                        st.dataframe(sdf.head(), use_container_width=True)
        
        with p2:
            st.markdown("### 🔄 Categorical Encoding")
            cats = df.select_dtypes(include=['object']).columns.tolist()
            cols_enc = [c for c in cats if 'ID' not in c and 'Sample' not in c]
            
            if len(cols_enc) > 0:
                st.markdown(f"**Categorical columns ({len(cols_enc)}):** {', '.join(cols_enc)}")
                
                if st.button("🔄 Apply One-Hot Encoding", type="primary"):
                    with st.spinner('Encoding...'):
                        enc_df, new_cols, _, enc_shape = one_hot_encode(df)
                        st.session_state.encoded_data = enc_df
                        st.session_state.encoded_shape = enc_shape
                        st.success(f"✅ Created {len(new_cols)} columns! Shape: {enc_shape}")
                        st.dataframe(enc_df.head(), use_container_width=True)
        
        with p3:
            st.markdown("### 🎯 Address Outliers in Numerical Columns")
            st.markdown("""
            **📌 IQR Method:** Outliers are capped using Q1 - 1.5×IQR and Q3 + 1.5×IQR bounds.
            """)
            
            # Define the numerical columns to process
            numerical_cols = ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']
            
            # Check which columns exist in the dataframe
            available_cols = [col for col in numerical_cols if col in df.columns]
            
            if len(available_cols) > 0:
                st.markdown(f"**Processing columns:** {', '.join(available_cols)}")
                
                # Show before statistics
                st.markdown("#### Before Outlier Handling")
                st.dataframe(df[available_cols].describe(), use_container_width=True)
                
                if st.button("🎯 Cap Outliers (IQR Method)", type="primary"):
                    with st.spinner('Handling outliers...'):
                        # EXACT IMPLEMENTATION AS REQUESTED
                        # Define the numerical columns
                        numerical_cols = ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']
                        
                        # Handle outliers using the IQR method (capping)
                        for col in numerical_cols:
                            if col in df.columns:
                                Q1 = df[col].quantile(0.25)
                                Q3 = df[col].quantile(0.75)
                                IQR = Q3 - Q1
                                
                                lower_bound = Q1 - 1.5 * IQR
                                upper_bound = Q3 + 1.5 * IQR
                                
                                # Cap the outliers
                                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
                        
                        # Store processed data
                        st.session_state.processed_data = df
                        
                        st.success("✅ Outliers handled successfully!")
                        
                        # Display descriptive statistics after outlier handling
                        st.markdown("### Descriptive statistics after handling outliers:")
                        st.dataframe(df[numerical_cols].describe(), use_container_width=True)
                        
                        # Show distributions after capping
                        st.markdown("### 📈 Distributions After Outlier Handling")
                        for col in available_cols:
                            st.plotly_chart(plot_distribution(df, col, f'{col} (After Capping)'), use_container_width=True)
            else:
                st.warning("None of the specified numerical columns found in the dataset.")
        
        with p4:
            st.markdown("### 📊 Skewness Analysis & Log Transform")
            nums = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_sk = [c for c in nums if 'ID' not in c and 'Sample' not in c]
            
            if len(cols_sk) > 0:
                skew_df = analyze_skewness(df, cols_sk)
                st.dataframe(skew_df, use_container_width=True, hide_index=True)
                
                skewed_cols = skew_df[skew_df['Skewed'] == 'Yes']['Column'].tolist()
                if len(skewed_cols) > 0:
                    st.markdown(f"**Skewed columns:** {', '.join(skewed_cols)}")
                    
                    if st.button("📊 Apply Log Transform", type="primary"):
                        with st.spinner('Applying log transform...'):
                            current_data = st.session_state.processed_data if st.session_state.processed_data is not None else df
                            transformed_df = apply_log_transform(current_data, skewed_cols)
                            st.session_state.processed_data = transformed_df
                            st.success("✅ Log transform applied!")
                            
                            new_skew = analyze_skewness(transformed_df, skewed_cols)
                            st.dataframe(new_skew, use_container_width=True, hide_index=True)
        
        with p5:
            st.markdown("### 📋 Preprocessing Summary")
            actions = []
            
            if st.session_state.get('scaled_data') is not None: 
                actions.append("✅ Feature Scaling applied")
            if st.session_state.get('encoded_data') is not None: 
                actions.append("✅ Categorical Encoding applied")
            if st.session_state.get('processed_data') is not None: 
                actions.append("✅ Outliers handled / Log transform applied")
            
            if actions:
                for a in actions: 
                    st.markdown(a)
                
                # Show the outlier stats table if available from processed data
                if st.session_state.get('processed_data') is not None:
                    numerical_cols = ['MP_Count_per_L', 'Risk_Score', 'Microplastic_Size_mm_midpoint', 'Density_midpoint']
                    available_cols = [col for col in numerical_cols if col in st.session_state.processed_data.columns]
                    if available_cols:
                        st.markdown("---")
                        st.markdown("### Descriptive statistics after handling outliers:")
                        st.dataframe(st.session_state.processed_data[available_cols].describe(), use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 🚀 Next Steps")
                st.markdown("Proceed to **🛠️ Feature Selection & Relevance**.")
            else:
                st.info("No preprocessing steps applied yet.")
    
    # ==================== FEATURE SELECTION ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: 
            st.warning("⚠️ Load data first!")
            return
        
        df = data.copy()
        
        st.markdown("### 📈 EDA")
        if 'Risk_Score' in df.columns:
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
        
        st.markdown("---")
        st.markdown("### 🎯 Feature Selection")
        target = st.selectbox("Target Variable", df.columns.tolist(), 
                             index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        nums = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        if target in nums: 
            nums.remove(target)
        
        if st.button("Calculate Feature Importance", type="primary", use_container_width=True):
            with st.spinner('Calculating...'):
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
                                          title='RF Importance', height=500), use_container_width=True)
                with ft2: 
                    st.plotly_chart(px.bar(mi_df.head(20), x='Mutual_Info', y='Feature', orientation='h', 
                                          title='Mutual Information', height=500), use_container_width=True)
                with ft3: 
                    st.plotly_chart(px.bar(chi2_df.head(20), x='Chi2_Score', y='Feature', orientation='h', 
                                          title='Chi-squared', height=500), use_container_width=True)
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
                st.error("Select features!")
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
                    except: pass
            
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
            
            if models:
                st.session_state.models = models
                st.session_state.trained = True
                st.success(f"✅ {len(models)} models trained!")
                for name, model in models.items():
                    y_pred = model.predict(X_test)
                    st.markdown(f"**{name}:** Acc={accuracy_score(y_test, y_pred):.4f} | F1={f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    # ==================== CROSS VALIDATION ====================
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
                st.error(f"'{tc}' not found!")
            elif st.button("🚀 Evaluate Models", type="primary", key="eval"):
                with st.spinner('Evaluating...'):
                    results = train_and_evaluate(df, tc)
                    st.session_state.evaluation_ran = True
                
                if results:
                    for mn in ['Logistic Regression', 'RandomForestClassifier', 'GradientBoostingClassifier']:
                        if mn in results:
                            r = results[mn]
                            st.markdown(f"### {mn}: Acc={r['accuracy']:.4f} | Prec={r['precision']:.4f} | Rec={r['recall']:.4f} | F1={r['f1_score']:.4f}")
                    
                    md = [{'Model': n, 'Accuracy': r['accuracy'], 'Precision': r['precision'], 
                          'Recall': r['recall'], 'F1': r['f1_score']} for n, r in results.items()]
                    mdf = pd.DataFrame(md)
                    st.dataframe(mdf, use_container_width=True)
                    st.plotly_chart(px.bar(mdf, x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1'], 
                                          barmode='group', height=400), use_container_width=True)
        
        with et2:
            st.markdown("### 📊 Compare Both Targets")
            if st.button("🚀 Compare", type="primary", key="cmp"):
                all_c = {}
                for tc in ['Risk_Type', 'Risk_Level']:
                    if tc not in df.columns: continue
                    with st.spinner(f'{tc}...'):
                        results = train_and_evaluate(df, tc)
                        all_c[tc] = results
                st.session_state.comparison_ran = True
                
                for tc, results in all_c.items():
                    st.markdown(f"## {tc}")
                    if results:
                        md = [{'Model': n, 'Accuracy': r['accuracy'], 'F1': r['f1_score']} for n, r in results.items()]
                        st.dataframe(pd.DataFrame(md), use_container_width=True, hide_index=True)
                        st.markdown("---")
        
        with et3:
            st.markdown("### 🔄 Cross Validation")
            target = st.selectbox("Target for CV", df.columns.tolist(), 
                                 index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
            nums = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
            if target in nums: nums.remove(target)
            folds = st.slider("CV Folds", 3, 10, 5)
            
            if st.button("🔄 Run CV", type="primary", key="cv"):
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
                            'Mean Acc': round(acc.mean(), 4),
                            'Std Acc': round(acc.std(), 4),
                            'Mean F1': round(f1.mean(), 4),
                            'Std F1': round(f1.std(), 4)
                        })
                    except: pass
                
                st.session_state.cv_ran = True
                
                if cv_res:
                    cv_df = pd.DataFrame(cv_res)
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
                    best = cv_df.loc[cv_df['Mean F1'].idxmax()]
                    st.success(f"🏆 Best: **{best['Model']}** (F1: {best['Mean F1']:.4f})")
                    
                    fig_cv = go.Figure()
                    for nm, sc in all_s.items(): 
                        fig_cv.add_trace(go.Box(y=sc, name=nm, boxmean='sd'))
                    fig_cv.update_layout(title=f'CV F1 ({folds}-Fold)', yaxis_title='F1', height=400)
                    st.plotly_chart(fig_cv, use_container_width=True)
        
        with et4:
            st.markdown("### 📋 Pipeline Summary")
            if st.button("🔄 Generate Summary", type="primary", key="pipe"):
                pd_data = [
                    {'Stage': '1. Data', 'Step': 'Loaded', 'Status': '✅' if st.session_state.data is not None else '❌', 
                     'Details': f'{st.session_state.data.shape[0]}×{st.session_state.data.shape[1]}' if st.session_state.data is not None else 'No data'},
                    {'Stage': '2. Preproc', 'Step': 'Scaling', 'Status': '✅' if st.session_state.get('scaled_data') is not None else '⬜', 
                     'Details': f'{len(st.session_state.get("scaled_columns", []))} cols'},
                    {'Stage': '2. Preproc', 'Step': 'Encoding', 'Status': '✅' if st.session_state.get('encoded_data') is not None else '⬜', 
                     'Details': f'{st.session_state.get("encoded_shape", "N/A")}'},
                    {'Stage': '2. Preproc', 'Step': 'Outliers', 'Status': '✅' if st.session_state.get('processed_data') is not None else '⬜', 
                     'Details': 'IQR capping'},
                    {'Stage': '3. Features', 'Step': 'Importance', 'Status': '✅' if st.session_state.get('feature_importance') is not None else '⬜', 
                     'Details': 'MI, Chi2, RF'},
                    {'Stage': '4. Models', 'Step': 'Trained', 'Status': '✅' if st.session_state.get('trained') else '⬜', 
                     'Details': f'{len(st.session_state.get("models", {}))} models'},
                    {'Stage': '5. Eval', 'Step': 'Evaluation', 'Status': '✅' if st.session_state.get('evaluation_ran') else '⬜', 
                     'Details': 'Metrics'},
                    {'Stage': '5. Eval', 'Step': 'Comparison', 'Status': '✅' if st.session_state.get('comparison_ran') else '⬜', 
                     'Details': 'Both targets'},
                    {'Stage': '5. Eval', 'Step': 'CV', 'Status': '✅' if st.session_state.get('cv_ran') else '⬜', 
                     'Details': 'K-Fold'}
                ]
                
                pdf = pd.DataFrame(pd_data)
                st.dataframe(pdf, use_container_width=True, height=400)
                
                completed = sum(1 for d in pd_data if d['Status'] == '✅')
                total = len(pd_data)
                st.progress(int((completed/total)*100), text=f"Progress: {int((completed/total)*100)}%")


if __name__ == "__main__":
    main()
