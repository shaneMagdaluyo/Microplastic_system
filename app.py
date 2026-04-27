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
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .subsection-header {
        font-size: 1.4rem;
        font-weight: 500;
        color: #34495e;
        margin-top: 0.8rem;
    }
    .stButton > button {
        width: 100%;
        background-color: #1f77b4;
        color: white;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background-color: #2980b9;
        border-color: #2980b9;
    }
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

def load_dataset(uploaded_file):
    """Load dataset from uploaded file with encoding fix."""
    try:
        if uploaded_file.name.endswith('.csv'):
            try:
                data = pd.read_csv(uploaded_file, encoding='utf-8')
            except UnicodeDecodeError:
                uploaded_file.seek(0)
                try:
                    data = pd.read_csv(uploaded_file, encoding='latin1')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    data = pd.read_csv(uploaded_file, encoding='cp1252')
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format. Please upload CSV or Excel file.")
            return None
        st.session_state.data = data
        st.success(f"✅ Dataset loaded successfully! Shape: {data.shape}")
        return data
    except Exception as e:
        st.error(f"❌ Error loading file: {str(e)}")
        return None

def generate_sample_data():
    """Generate sample microplastic data for demonstration."""
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Sample_ID': [f'MP_{i:04d}' for i in range(n_samples)],
        'Latitude': np.random.uniform(12.8, 13.0, n_samples),
        'Longitude': np.random.uniform(123.9, 124.1, n_samples),
        'MP_Count_per_L': np.random.poisson(lam=50, size=n_samples),
        'Microplastic_Size_mm': np.random.choice(['0.1-5.0', '5.0-10.0', '0.1-1.0'], n_samples),
        'Density': np.random.choice(['1.3-1.4', '1.2-1.3', '1.0-1.2'], n_samples),
        'Particle_Size_um': np.random.normal(100, 30, n_samples),
        'Polymer_Type': np.random.choice(['PE', 'PP', 'PS', 'PET', 'PVC', 'Nylon'], n_samples),
        'Water_Source': np.random.choice(['River', 'Lake', 'Ocean', 'Groundwater', 'Tap'], n_samples),
        'pH': np.random.normal(7, 0.5, n_samples),
        'Temperature_C': np.random.normal(20, 5, n_samples),
        'Risk_Score': np.random.uniform(0, 100, n_samples),
        'Risk_Level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_samples, 
                                       p=[0.3, 0.35, 0.25, 0.1]),
        'Risk_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples, 
                                     p=[0.5, 0.3, 0.2]),
        'Location': np.random.choice(['Urban', 'Rural', 'Industrial', 'Coastal'], n_samples),
        'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_samples),
        'Author': np.random.choice(['Author_A', 'Author_B', 'Author_C'], n_samples),
        'Source': np.random.choice(['Source_1', 'Source_2', 'Source_3'], n_samples)
    }
    
    df = pd.DataFrame(data)
    for col in df.columns:
        if col != 'Sample_ID' and df[col].dtype in ['float64', 'int64']:
            mask = np.random.random(n_samples) < 0.05
            df.loc[mask, col] = np.nan
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    try:
        df_clean = df.copy()
        for col in df_clean.columns:
            if df_clean[col].isnull().sum() > 0:
                if df_clean[col].dtype in ['float64', 'int64']:
                    median_val = df_clean[col].median()
                    if pd.isna(median_val): median_val = 0
                    df_clean[col].fillna(median_val, inplace=True)
                else:
                    mode_series = df_clean[col].mode()
                    mode_val = mode_series[0] if not mode_series.empty else 'Unknown'
                    df_clean[col].fillna(mode_val, inplace=True)
        return df_clean
    except Exception as e:
        st.error(f"Error handling missing values: {str(e)}")
        return df

def cap_outliers_iqr(df, columns):
    """Cap outliers using IQR method."""
    log_messages = []
    df_capped = df.copy()
    for col in columns:
        if df_capped[col].dtype in ['float64', 'int64']:
            Q1 = df_capped[col].quantile(0.25)
            Q3 = df_capped[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
            log_messages.append(f"Capped outliers in '{col}'")
    return df_capped, log_messages

def one_hot_encode(df):
    """Apply one-hot encoding to categorical columns."""
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        cols_to_encode = [col for col in categorical_cols if 'ID' not in col and 'Sample' not in col]
        if len(cols_to_encode) == 0:
            return df, [], [], df.shape
        df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
        new_cols = [col for col in df_encoded.columns if col not in df.columns]
        encoded_shape = df_encoded.shape
        return df_encoded, new_cols, cols_to_encode, encoded_shape
    except Exception as e:
        st.error(f"Error in one-hot encoding: {str(e)}")
        return df, [], [], df.shape

def calculate_mutual_info(X, y):
    """Calculate Mutual Information scores for features."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({'Feature': X.columns, 'Mutual_Info': mi_scores})
    mi_df = mi_df.sort_values('Mutual_Info', ascending=False)
    return mi_df

def calculate_chi2(X, y):
    """Calculate Chi-squared scores for features."""
    X_scaled = X - X.min() + 1
    chi2_scores, p_values = chi2(X_scaled, y)
    chi2_df = pd.DataFrame({'Feature': X.columns, 'Chi2_Score': chi2_scores, 'P_Value': p_values})
    chi2_df = chi2_df.sort_values('Chi2_Score', ascending=False)
    return chi2_df

def calculate_rf_importance(X, y):
    """Calculate Random Forest feature importance."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_df = pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_})
    rf_df = rf_df.sort_values('Importance', ascending=False)
    return rf_df

def train_and_evaluate_for_target(df, target_col):
    """Train models for a specific target and return results."""
    feature_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
    if target_col in feature_cols: feature_cols.remove(target_col)
    
    X = df[feature_cols].copy()
    y = df[target_col].copy()
    
    mask = y.notna()
    X = X[mask]; y = y[mask]
    if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
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
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
    
    return results

def plot_distribution(data, column, title):
    """Create distribution plot."""
    try:
        clean_data = data[column].dropna()
        if clean_data.empty: return go.Figure()
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
        fig.add_trace(go.Histogram(x=clean_data, name='Distribution', nbinsx=30, marker_color='#3498db'), row=1, col=1)
        fig.add_trace(go.Box(y=clean_data, name='Box Plot', marker_color='#e74c3c'), row=1, col=2)
        fig.update_layout(title_text=title, showlegend=False, height=500)
        return fig
    except Exception as e:
        st.error(f"Error creating distribution plot: {str(e)}")
        return go.Figure()

def main():
    """Main application function."""
    
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
    if st.session_state.trained: st.sidebar.success(f"✅ Models Trained")
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
            
            # Feature Scaling
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
                        st.success(f"✅ {len(cols)} columns scaled!")
                        st.dataframe(sdf.head(), column_config={c: st.column_config.NumberColumn(c,format="%.6f") for c in cols}, use_container_width=True)
            
            # Risk Score vs MP Count
            if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
                st.markdown("---")
                st.markdown("### 🔬 Risk Score vs MP Count per L")
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
                                            trendline='ols', title='MP Count vs Risk Score', opacity=0.7)
                            st.plotly_chart(fig, use_container_width=True)
                        except: st.warning("⚠️ Trendline not available")
            
            # Risk Score by Risk Level
            if 'Risk_Score' in df.columns and 'Risk_Level' in df.columns:
                st.markdown("---")
                st.markdown("### 📊 Risk Score by Risk Level")
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
        
        prep_tab1, prep_tab2, prep_tab3, prep_tab4, prep_tab5 = st.tabs([
            "📏 Feature Scaling", "🔄 Categorical Encoding", "🎯 Outlier Capping", 
            "📊 Skewness & Transform", "📋 Summary & Next Steps"
        ])
        
        with prep_tab1:
            st.markdown("### 📏 Perform Feature Scaling")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_to_scale = [col for col in numeric_cols if 'ID' not in col and 'Sample' not in col]
            
            if st.button("🔧 Apply Feature Scaling (StandardScaler)", type="primary", key="scale_tab"):
                with st.spinner('Applying StandardScaler...'):
                    if len(cols_to_scale) > 0:
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[cols_to_scale].fillna(df[cols_to_scale].median()))
                        scaled_df = pd.DataFrame(scaled_data, columns=cols_to_scale)
                        st.session_state.scaler = scaler
                        st.success(f"✅ {len(cols_to_scale)} columns scaled!")
                        st.dataframe(scaled_df.head(), column_config={col: st.column_config.NumberColumn(col, format="%.6f") for col in cols_to_scale}, use_container_width=True)
        
        with prep_tab2:
            st.markdown("### 🔄 Encode Categorical Variables")
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            cols_to_encode = [col for col in categorical_cols if 'ID' not in col and 'Sample' not in col]
            if len(cols_to_encode) > 0:
                st.markdown(f"**Categorical columns ({len(cols_to_encode)}):** {', '.join(cols_to_encode)}")
            
            if st.button("🔄 Apply One-Hot Encoding", type="primary", key="encode_tab"):
                with st.spinner('Applying...'):
                    if len(cols_to_encode) > 0:
                        encoded_df, new_cols, _, encoded_shape = one_hot_encode(df)
                        st.session_state.encoded_data = encoded_df
                        st.session_state.encoded_shape = encoded_shape
                        st.success(f"✅ Created {len(new_cols)} new columns! Shape: {encoded_shape}")
                        st.dataframe(encoded_df.head(), use_container_width=True)
        
        with prep_tab3:
            st.markdown("### 🎯 Address Outliers")
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_for_outliers = [col for col in numeric_cols if 'ID' not in col and 'Sample' not in col]
            if st.button("🎯 Cap Outliers (IQR Method)", type="primary", key="outlier_tab"):
                if len(cols_for_outliers) > 0:
                    df_capped, cap_logs = cap_outliers_iqr(df, cols_for_outliers)
                    st.session_state.processed_data = df_capped
                    st.success(f"✅ Outliers capped!")
                    for log in cap_logs: st.write(f"- {log}")
        
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
                st.markdown("### 🚀 Next Steps")
                st.markdown("Proceed to **📊 Cross Validation & Evaluation** for model training and comparison.")
    
    # ==================== FEATURE SELECTION & RELEVANCE ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data.copy()
        
        st.markdown("### 📈 Exploratory Data Analysis")
        
        if 'Risk_Score' in df.columns:
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean = df['Risk_Score'].dropna()
            if len(clean) > 0:
                st.plotly_chart(plot_distribution(df, 'Risk_Score', 'Risk Score Distribution'), use_container_width=True)
        
        if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
            st.markdown("---")
            st.markdown("#### 🔬 MP Count vs Risk Score")
            df['MP_Count_per_L'] = pd.to_numeric(df['MP_Count_per_L'], errors='coerce')
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
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
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean = df.dropna(subset=['Risk_Score'])
            clean['Risk_Level'] = clean['Risk_Level'].astype(str)
            if len(clean) > 0:
                fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level', title='Risk Score by Risk Level')
                st.plotly_chart(fig, use_container_width=True)
        
        # Feature Selection
        st.markdown("---")
        st.markdown("### 🎯 Feature Selection Methods")
        
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
                    fig = px.bar(rf_df.head(20), x='Importance', y='Feature', orientation='h',
                               title='Random Forest Importance', color='Importance', 
                               color_continuous_scale='Viridis', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with ft2:
                    fig = px.bar(mi_df.head(20), x='Mutual_Info', y='Feature', orientation='h',
                               title='Mutual Information', color='Mutual_Info',
                               color_continuous_scale='Viridis', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with ft3:
                    fig = px.bar(chi2_df.head(20), x='Chi2_Score', y='Feature', orientation='h',
                               title='Chi-squared Test', color='Chi2_Score',
                               color_continuous_scale='Viridis', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.success(f"✅ Feature selection completed!")
    
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
                rf.fit(X_train, y_train); models['Random Forest'] = rf
            except: pass
            try:
                gb = GradientBoostingClassifier(n_estimators=50, random_state=42)
                gb.fit(X_train, y_train); models['GradientBoostingClassifier'] = gb
            except: pass
            
            if models:
                st.session_state.models = models
                st.session_state.X_test = X_test; st.session_state.y_test = y_test
                st.session_state.trained = True
                st.success(f"✅ {len(models)} models trained!")
                st.balloons()
                
                for name, model in models.items():
                    y_pred = model.predict(X_test)
                    st.markdown(f"**{name}:** Acc={accuracy_score(y_test, y_pred):.4f} | F1={f1_score(y_test, y_pred, average='weighted'):.4f}")
    
    # ==================== CROSS VALIDATION & EVALUATION (COMBINED) ====================
    elif section == "📊 Cross Validation & Evaluation":
        st.markdown('<p class="section-header">📊 Cross Validation & Model Evaluation</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data.copy()
        
        # Tabs for different evaluations
        eval_tab1, eval_tab2, eval_tab3 = st.tabs([
            "📊 Evaluate Models for Risk Type", 
            "📊 Compare Model Performance",
            "🔄 Cross Validation"
        ])
        
        # ===== TAB 1: EVALUATE MODELS FOR RISK TYPE =====
        with eval_tab1:
            st.markdown("### 📊 Evaluate Models for Risk Type")
            st.markdown("*Evaluate the performance of the trained models for 'Risk_Type' using appropriate metrics*")
            
            target_col = 'Risk_Type'
            if target_col not in df.columns:
                st.error(f"❌ '{target_col}' column not found!")
            else:
                if st.button("🚀 Train & Evaluate for Risk_Type", type="primary", key="eval_risk_type"):
                    with st.spinner('Training and evaluating models for Risk_Type...'):
                        results = train_and_evaluate_for_target(df, target_col)
                    
                    if results:
                        st.markdown("---")
                        for name, res in results.items():
                            st.markdown(f"### --- {name} Model Evaluation for '{target_col}' ---")
                            st.markdown(f"**Accuracy:** {res['accuracy']:.4f}")
                            st.markdown(f"**Precision:** {res['precision']:.4f}")
                            st.markdown(f"**Recall:** {res['recall']:.4f}")
                            st.markdown(f"**F1-Score:** {res['f1_score']:.4f}")
                            st.markdown("---")
                            st.markdown("")
                        
                        # Comparison table
                        st.markdown("### 📊 Performance Comparison Table")
                        metrics_data = []
                        for name, res in results.items():
                            metrics_data.append({
                                'Model': name, 'Accuracy': res['accuracy'],
                                'Precision': res['precision'], 'Recall': res['recall'], 'F1-Score': res['f1_score']
                            })
                        metrics_df = pd.DataFrame(metrics_data)
                        st.dataframe(metrics_df, column_config={
                            "Model": "Model",
                            "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                            "Precision": st.column_config.NumberColumn("Precision", format="%.4f"),
                            "Recall": st.column_config.NumberColumn("Recall", format="%.4f"),
                            "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.4f"),
                        }, use_container_width=True, hide_index=True)
                        
                        # Bar chart
                        fig = px.bar(metrics_df, x='Model', y=['Accuracy','Precision','Recall','F1-Score'],
                                    barmode='group', title='Model Performance - Risk_Type',
                                    color_discrete_sequence=['#3498db','#e74c3c','#2ecc71','#f39c12'], height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        best_acc = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
                        best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
                        st.markdown(f"""
                        <div style="background: #d4edda; border: 2px solid #27ae60; border-radius: 10px; padding: 20px; margin: 15px 0;">
                            <p style="margin: 5px 0; color: #155724;">Based on <b>Accuracy</b>, best: <b>{best_acc['Model']}</b> ({best_acc['Accuracy']:.4f})</p>
                            <p style="margin: 5px 0; color: #155724;">Based on <b>F1-Score</b>, best: <b>{best_f1['Model']}</b> ({best_f1['F1-Score']:.4f})</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # ===== TAB 2: COMPARE MODEL PERFORMANCE =====
        with eval_tab2:
            st.markdown("### 📊 Compare Model Performance")
            st.markdown("*Compare the performance of the models for both 'Risk_Type' and 'Risk_Level'*")
            
            if st.button("🚀 Train & Compare for Both Targets", type="primary", key="compare_both"):
                all_comparisons = {}
                
                for target_col in ['Risk_Type', 'Risk_Level']:
                    if target_col not in df.columns: continue
                    
                    with st.spinner(f'Training models for {target_col}...'):
                        results = train_and_evaluate_for_target(df, target_col)
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
                            <p style="margin: 5px 0; color: #155724;">Based on <b>Accuracy</b>, best: <b>{best_acc['Model']}</b> ({best_acc['Accuracy']:.4f})</p>
                            <p style="margin: 5px 0; color: #155724;">Based on <b>F1-Score</b>, best: <b>{best_f1['Model']}</b> ({best_f1['F1-Score']:.4f})</p>
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
                
                # Overall summary
                if len(all_comparisons) > 1:
                    st.markdown("---")
                    st.markdown("## 📊 Overall Summary")
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
            st.info("Evaluate model stability using stratified k-fold cross-validation.")
            
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
                        cv_results.append({'Model':name,'Mean Accuracy':round(acc.mean(),4),'Std Accuracy':round(acc.std(),4),
                                          'Mean F1':round(f1.mean(),4),'Std F1':round(f1.std(),4)})
                    except: pass
                
                if cv_results:
                    cv_df = pd.DataFrame(cv_results)
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
                    best_cv = cv_df.loc[cv_df['Mean F1'].idxmax()]
                    st.success(f"🏆 Best CV Model: **{best_cv['Model']}** (Mean F1: {best_cv['Mean F1']:.4f})")
                    
                    fig_cv = go.Figure()
                    for name, scores in all_scores.items():
                        fig_cv.add_trace(go.Box(y=scores, name=name, boxmean='sd'))
                    fig_cv.update_layout(title=f'CV F1 Scores ({folds}-Fold)', yaxis_title='F1 Score', height=400)
                    st.plotly_chart(fig_cv, use_container_width=True)


if __name__ == "__main__":
    main()
