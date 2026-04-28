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

# Custom CSS - FIXED
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-top: 1rem; margin-bottom: 1rem; }
    .stButton > button { width: 100%; background-color: #1f77b4; color: white; font-weight: 600; border-radius: 8px; padding: 0.5rem 1rem; }
    .stButton > button:hover { background-color: #2980b9; border-color: #2980b9; }
    
    /* Force dark text everywhere */
    .stMarkdown, .stMarkdown p, .stMarkdown li, .stMarkdown div { color: #2c3e50 !important; }
    div[data-testid="stExpander"] p { color: #2c3e50 !important; }
    div[data-testid="stExpander"] li { color: #2c3e50 !important; }
    
    .metric-explain { font-size: 0.9rem; color: #666; font-style: italic; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize all session state variables."""
    if 'data' not in st.session_state: st.session_state.data = None
    if 'processed_data' not in st.session_state: st.session_state.processed_data = None
    if 'models' not in st.session_state: st.session_state.models = {}
    if 'feature_importance' not in st.session_state: st.session_state.feature_importance = None
    if 'mutual_info' not in st.session_state: st.session_state.mutual_info = None
    if 'chi2_scores' not in st.session_state: st.session_state.chi2_scores = None
    if 'X_selected' not in st.session_state: st.session_state.X_selected = None
    if 'best_model' not in st.session_state: st.session_state.best_model = None
    if 'trained' not in st.session_state: st.session_state.trained = False
    if 'X_test' not in st.session_state: st.session_state.X_test = None
    if 'y_test' not in st.session_state: st.session_state.y_test = None
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

# ==================== ALL FUNCTION DEFINITIONS ====================

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
        'Risk_Level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_samples, p=[0.3, 0.35, 0.25, 0.1]),
        'Risk_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples, p=[0.5, 0.3, 0.2]),
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

def one_hot_encode(df):
    """Apply one-hot encoding to categorical columns."""
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        cols_to_encode = [col for col in categorical_cols if 'ID' not in col and 'Sample' not in col]
        if len(cols_to_encode) == 0:
            return df, [], [], df.shape
        df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
        new_cols = [col for col in df_encoded.columns if col not in df.columns]
        return df_encoded, new_cols, cols_to_encode, df_encoded.shape
    except Exception as e:
        st.error(f"Error in one-hot encoding: {str(e)}")
        return df, [], [], df.shape

def detect_outliers(df, columns):
    """Detect outliers using IQR method."""
    outlier_info = {}
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_info[col] = {
                'count': len(outliers),
                'percentage': (len(outliers) / len(df)) * 100 if len(df) > 0 else 0,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound
            }
    return outlier_info

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
            outliers_before = ((df_capped[col] < lower_bound) | (df_capped[col] > upper_bound)).sum()
            df_capped[col] = df_capped[col].clip(lower=lower_bound, upper=upper_bound)
            log_messages.append(f"Capped {outliers_before} outliers in '{col}'")
    return df_capped, log_messages

def analyze_skewness(df, columns):
    """Analyze skewness of numerical columns."""
    skew_info = []
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            skew_val = df[col].skew()
            skew_info.append({
                'Column': col,
                'Skewness': round(skew_val, 4),
                'Abs Skewness': round(abs(skew_val), 4),
                'Skewed (>0.5)': 'Yes' if abs(skew_val) > 0.5 else 'No'
            })
    return pd.DataFrame(skew_info)

def apply_log_transform(df, columns):
    """Apply log transformation to skewed columns."""
    df_transformed = df.copy()
    log_messages = []
    for col in columns:
        if df_transformed[col].dtype in ['float64', 'int64']:
            skew_before = df_transformed[col].skew()
            if abs(skew_before) > 0.5:
                min_val = df_transformed[col].min()
                shift = abs(min_val) + 1 if min_val <= 0 else 0
                df_transformed[col] = np.log1p(df_transformed[col] + shift)
                skew_after = df_transformed[col].skew()
                log_messages.append(f"Log transformed '{col}': Skewness {skew_before:.4f} → {skew_after:.4f}")
    return df_transformed, log_messages

def calculate_mutual_info(X, y):
    """Calculate Mutual Information scores."""
    mi_scores = mutual_info_classif(X, y, random_state=42)
    return pd.DataFrame({'Feature': X.columns, 'Mutual_Info': mi_scores}).sort_values('Mutual_Info', ascending=False)

def calculate_chi2(X, y):
    """Calculate Chi-squared scores."""
    X_scaled = X - X.min() + 1
    chi2_scores, p_values = chi2(X_scaled, y)
    return pd.DataFrame({'Feature': X.columns, 'Chi2_Score': chi2_scores, 'P_Value': p_values}).sort_values('Chi2_Score', ascending=False)

def calculate_rf_importance(X, y):
    """Calculate Random Forest feature importance."""
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    return pd.DataFrame({'Feature': X.columns, 'Importance': rf.feature_importances_}).sort_values('Importance', ascending=False)

def train_and_evaluate_detailed(df, target_col):
    """Train models and return detailed metrics."""
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
        'X_train_shape': X_train.shape, 'X_test_shape': X_test.shape,
        'y_train_shape': y_train.shape, 'y_test_shape': y_test.shape,
        'target_col': target_col, 'total_samples': len(df),
        'train_pct': round(len(X_train)/len(X)*100, 1), 'test_pct': round(len(X_test)/len(X)*100, 1)
    }
    
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
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, zero_division=0)
        }
    
    return results, data_split_info

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
    except: return go.Figure()

# ==================== MAIN APPLICATION ====================

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
    if st.session_state.trained: st.sidebar.success(f"✅ Models Trained")
    else: st.sidebar.warning("⚠️ Models Not Trained")
    
    # ==================== HOME ====================
    if section == "🏠 Home":
        st.markdown('<p class="section-header">🏠 Home - Upload Dataset</p>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ About this section", expanded=False):
            st.markdown("""
            **Purpose of the Home Page:**
            This is the starting point of your analysis. Here you can:
            - **Upload your dataset** (CSV or Excel format)
            - **Generate sample data** to explore the dashboard's capabilities
            - **Preview your data** to understand its structure
            - **Apply initial scaling** to see how StandardScaler transforms numerical columns
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
            st.markdown("### 📋 Dataset Preview")
            st.info("**📖 Understanding the Dataset Preview:**\n• **Samples:** Number of rows (observations)\n• **Features:** Number of columns (variables)\n• **Missing Values:** Gaps in data that need preprocessing")
            
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Samples", df.shape[0])
            with c2: st.metric("Features", df.shape[1])
            with c3: st.metric("Missing", df.isnull().sum().sum())
            st.dataframe(df.head(10), use_container_width=True)
            
            # Feature Scaling Preview
            st.markdown("---")
            st.markdown("### 📏 Feature Scaling Preview")
            st.info("**📖 Why Feature Scaling?**\nStandardScaler transforms numerical columns to **mean=0** and **std=1**. This is essential because ML algorithms perform better when features are on the same scale.")
            
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
                        st.success(f"✅ {len(cols)} columns scaled! Mean=0, Std=1")
                        st.dataframe(sdf.head(), column_config={c: st.column_config.NumberColumn(c,format="%.6f") for c in cols}, use_container_width=True)
            
            # Risk Score vs MP Count
            if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
                st.markdown("---")
                st.markdown("### 🔬 Risk Score vs MP Count per L")
                st.info("**📖 Why analyze this?**\nExplores relationship between **MP Count per Liter** and **Risk Score** to determine if MP Count predicts Risk Score.")
                
                df['MP_Count_per_L'] = pd.to_numeric(df['MP_Count_per_L'], errors='coerce')
                df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                clean = df.dropna(subset=['MP_Count_per_L','Risk_Score'])
                if len(clean) > 0:
                    try:
                        fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                        color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                        trendline='ols', title='MP Count vs Risk Score', opacity=0.7)
                    except:
                        fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score', title='MP Count vs Risk Score', opacity=0.7)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Risk Score by Risk Level
            if 'Risk_Score' in df.columns and 'Risk_Level' in df.columns:
                st.markdown("---")
                st.markdown("### 📊 Risk Score by Risk Level")
                st.info("**📖 Why investigate this?**\nBox plots show Risk Score distribution across Risk Level categories, revealing if different levels have distinct score ranges.")
                
                clean = df.dropna(subset=['Risk_Score'])
                clean['Risk_Level'] = clean['Risk_Level'].astype(str)
                if len(clean) > 0:
                    fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                                title='Risk Score by Risk Level', points='outliers')
                    st.plotly_chart(fig, use_container_width=True)
            
            # Quality Check
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
        
        with st.expander("ℹ️ About Preprocessing", expanded=False):
            st.markdown("""
            **Purpose of Preprocessing:**
            Data preprocessing transforms raw data into a clean, structured format suitable for machine learning.
            **Key Steps:** Feature Scaling, Categorical Encoding, Outlier Capping, Skewness Transformation.
            """)
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            return
        
        df = st.session_state.data.copy()
        
        prep_tab1, prep_tab2, prep_tab3, prep_tab4, prep_tab5 = st.tabs([
            "📏 Feature Scaling", "🔄 Categorical Encoding", "🎯 Outlier Capping", 
            "📊 Skewness & Transform", "📋 Summary"
        ])
        
        with prep_tab1:
            st.markdown("### 📏 Perform Feature Scaling")
            st.info("**📖 StandardScaler** transforms features to **mean=0, std=1** preventing features with larger ranges from dominating the model.")
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_to_scale = [col for col in numeric_cols if 'ID' not in col and 'Sample' not in col]
            
            if st.button("🔧 Apply Feature Scaling (StandardScaler)", type="primary", key="scale_tab"):
                with st.spinner('Applying...'):
                    if len(cols_to_scale) > 0:
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[cols_to_scale].fillna(df[cols_to_scale].median()))
                        scaled_df = pd.DataFrame(scaled_data, columns=cols_to_scale)
                        st.session_state.scaler = scaler
                        st.session_state.scaled_columns = cols_to_scale
                        st.session_state.scaled_data = scaled_df
                        st.success(f"✅ {len(cols_to_scale)} columns scaled!")
                        st.dataframe(scaled_df.head(), column_config={col: st.column_config.NumberColumn(col, format="%.6f") for col in cols_to_scale}, use_container_width=True)
        
        with prep_tab2:
            st.markdown("### 🔄 Encode Categorical Variables")
            st.info("**📖 One-Hot Encoding** creates binary columns for each category. ML models require numerical input.")
            
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
            st.info("**📖 Outlier Capping** limits extreme values using IQR method (Q1-1.5×IQR to Q3+1.5×IQR).")
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_for_outliers = [col for col in numeric_cols if 'ID' not in col and 'Sample' not in col]
            if len(cols_for_outliers) > 0:
                outlier_info = detect_outliers(df, cols_for_outliers)
                outlier_summary = [{'Column':col,'Outliers':info['count'],'Percentage':f"{info['percentage']:.1f}%"} for col,info in outlier_info.items()]
                st.dataframe(pd.DataFrame(outlier_summary), use_container_width=True, hide_index=True)
            
            if st.button("🎯 Cap Outliers (IQR Method)", type="primary", key="outlier_tab"):
                if len(cols_for_outliers) > 0:
                    df_capped, cap_logs = cap_outliers_iqr(df, cols_for_outliers)
                    st.session_state.processed_data = df_capped
                    st.success(f"✅ Outliers capped!")
                    for log in cap_logs: st.write(f"- {log}")
        
        with prep_tab4:
            st.markdown("### 📊 Skewness Analysis & Log Transformation")
            st.info("**📖 Skewness** measures distribution asymmetry. **Log transformation** reduces right-skewness.")
            
            numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
            cols_for_skew = [col for col in numeric_cols if 'ID' not in col and 'Sample' not in col]
            if len(cols_for_skew) > 0:
                skew_df = analyze_skewness(df, cols_for_skew)
                st.dataframe(skew_df, use_container_width=True, hide_index=True)
                skewed_cols = skew_df[skew_df['Abs Skewness'] > 0.5]['Column'].tolist()
                if len(skewed_cols) > 0: st.markdown(f"**Skewed columns:** {', '.join(skewed_cols)}")
            
            if st.button("📊 Apply Log Transformation", type="primary", key="skew_tab"):
                if len(cols_for_skew) > 0:
                    df_transformed, _ = apply_log_transform(df, cols_for_skew)
                    st.session_state.processed_data = df_transformed
                    st.success(f"✅ Log transformation applied!")
        
        with prep_tab5:
            st.markdown("### 📋 Summary & Next Steps")
            actions = []
            if st.session_state.get('scaled_data') is not None: actions.append("✅ Feature Scaling applied")
            if st.session_state.get('encoded_data') is not None: actions.append("✅ Categorical Encoding applied")
            if st.session_state.get('processed_data') is not None: actions.append("✅ Outliers capped")
            if actions:
                for a in actions: st.markdown(a)
                st.markdown("---\n### 🚀 Next Steps\nProceed to **🛠️ Feature Selection & Relevance** or **📊 Cross Validation & Evaluation**.")
    
    # ==================== FEATURE SELECTION & RELEVANCE ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        
        with st.expander("ℹ️ About Feature Selection", expanded=False):
            st.markdown("""
            **Purpose:** Identifies most important variables for prediction using:
            1. **Mutual Information** - Measures dependency between features and target
            2. **Chi-squared Test** - Tests statistical relationship
            3. **Random Forest Importance** - Shows contribution to predictions
            """)
        
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
                st.session_state.selected_features = rf_df.head(10)['Feature'].tolist()
                
                ft1, ft2, ft3 = st.tabs(["🌲 Random Forest", "📊 Mutual Information", "🔢 Chi-squared"])
                
                with ft1:
                    st.markdown("**Top 20 features - RandomForest Feature Importances:**")
                    fig = px.bar(rf_df.head(20), x='Importance', y='Feature', orientation='h',
                               title='Top 20 Features - Random Forest', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with ft2:
                    st.markdown("**Top 20 features - Mutual Information:**")
                    fig = px.bar(mi_df.head(20), x='Mutual_Info', y='Feature', orientation='h',
                               title='Top 20 Features - Mutual Information', height=500)
                    st.plotly_chart(fig, use_container_width=True)
                
                with ft3:
                    st.markdown("**Top 20 features - Chi-squared Test:**")
                    fig = px.bar(chi2_df.head(20), x='Chi2_Score', y='Feature', orientation='h',
                               title='Top 20 Features - Chi-squared Test', height=500)
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
        
        with st.expander("ℹ️ About Model Evaluation", expanded=False):
            st.markdown("""
            **Purpose of Model Evaluation:**
            - **Accuracy**: Overall correct predictions
            - **Precision**: Correct positive predictions
            - **Recall**: Actual positives found
            - **F1-Score**: Balance of precision & recall
            - **Cross Validation**: Tests model stability across data splits
            """)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data.copy()
        
        eval_tab1, eval_tab2, eval_tab3, eval_tab4 = st.tabs([
            "📊 Evaluate Models", 
            "📊 Compare Both Targets",
            "🔄 Cross Validation",
            "📋 Overall Pipeline Summary"
        ])
        
        # ===== TAB 1: EVALUATE MODELS =====
        with eval_tab1:
            st.markdown("### 📊 Evaluate the Models")
            st.info("**📖 Trains models on 80% data, evaluates on 20% held-out test data using weighted averaging for multi-class metrics.**")
            
            target_col = 'Risk_Type'
            if target_col not in df.columns:
                st.error(f"❌ '{target_col}' column not found!")
            else:
                if st.button("🚀 Evaluate Models", type="primary", key="eval_detail"):
                    with st.spinner('Training and evaluating models...'):
                        results, split_info = train_and_evaluate_detailed(df, target_col)
                        st.session_state.evaluation_ran = True
                        st.session_state.last_eval_results = results
                    
                    if results:
                        st.markdown("---")
                        st.markdown("### 📊 Data Split Information")
                        st.info("**📖 Training Set (80%)** teaches the model. **Testing Set (20%)** evaluates generalization. Prevents overfitting.")
                        
                        st.markdown(f"**Target:** {split_info['target_col']} | **X_train:** {split_info['X_train_shape']} | **X_test:** {split_info['X_test_shape']}")
                        
                        st.markdown("---")
                        
                        model_descriptions = {
                            'Logistic Regression': 'Linear model estimating probabilities. Works well for linearly separable data.',
                            'RandomForestClassifier': 'Ensemble of decision trees. Reduces overfitting by averaging multiple trees.',
                            'GradientBoostingClassifier': 'Builds trees sequentially, each correcting errors of previous trees. High accuracy.'
                        }
                        
                        for model_name in ['Logistic Regression', 'RandomForestClassifier', 'GradientBoostingClassifier']:
                            if model_name in results:
                                res = results[model_name]
                                st.markdown(f"### # Evaluate {model_name} Model")
                                st.markdown(f"*{model_descriptions.get(model_name, '')}*")
                                st.markdown(f"**Accuracy:** {res['accuracy']:.4f} — Overall correct predictions")
                                st.markdown(f"**Precision:** {res['precision']:.4f} — Correct positive predictions")
                                st.markdown(f"**Recall:** {res['recall']:.4f} — Actual positives found")
                                st.markdown(f"**F1-Score:** {res['f1_score']:.4f} — Balance of precision & recall")
                                st.markdown("---")
                                st.markdown("")
                        
                        # Comparison Table
                        st.markdown("### Model Performance Comparison")
                        st.info("**📖 Higher values = better. Accuracy** best for balanced classes. **F1-Score** better for imbalanced classes.")
                        
                        metrics_data = []
                        for name, res in results.items():
                            metrics_data.append({
                                'Model': name, 'Accuracy': res['accuracy'],
                                'Precision': res['precision'], 'Recall': res['recall'], 'F1-Score': res['f1_score']
                            })
                        metrics_df = pd.DataFrame(metrics_data)
                        
                        st.dataframe(metrics_df, column_config={
                            "Model": "Model",
                            "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.6f"),
                            "Precision": st.column_config.NumberColumn("Precision", format="%.6f"),
                            "Recall": st.column_config.NumberColumn("Recall", format="%.6f"),
                            "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.6f"),
                        }, use_container_width=True)
                        
                        fig = px.bar(metrics_df, x='Model', y=['Accuracy','Precision','Recall','F1-Score'],
                                    barmode='group', title='Model Performance Metrics',
                                    color_discrete_sequence=['#3498db','#e74c3c','#2ecc71','#f39c12'], height=400)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        best_acc = metrics_df.loc[metrics_df['Accuracy'].idxmax()]
                        best_f1 = metrics_df.loc[metrics_df['F1-Score'].idxmax()]
                        
                        st.markdown(f"""
                        <div style="background-color: #d4edda; border-left: 5px solid #27ae60; padding: 15px 20px; margin: 15px 0; border-radius: 5px;">
                            <p style="margin: 5px 0; color: #155724;">Based on <b>Accuracy</b>, best: <b>{best_acc['Model']}</b> ({best_acc['Accuracy']:.4f})</p>
                            <p style="margin: 5px 0; color: #155724;">Based on <b>F1-Score</b>, best: <b>{best_f1['Model']}</b> ({best_f1['F1-Score']:.4f})</p>
                        </div>
                        """, unsafe_allow_html=True)
        
        # ===== TAB 2: COMPARE BOTH TARGETS =====
        with eval_tab2:
            st.markdown("### 📊 Compare Model Performance (Both Targets)")
            st.info("**📖 Compares Risk_Type and Risk_Level to see which prediction task is more feasible.**")
            
            if st.button("🚀 Train & Compare for Both Targets", type="primary", key="compare_both"):
                all_comparisons = {}
                
                for target_col in ['Risk_Type', 'Risk_Level']:
                    if target_col not in df.columns: continue
                    with st.spinner(f'Training for {target_col}...'):
                        results, _ = train_and_evaluate_detailed(df, target_col)
                        all_comparisons[target_col] = results
                
                st.session_state.comparison_ran = True
                st.session_state.last_comparison = all_comparisons
                
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
                        <div style="background-color: #d4edda; border-left: 5px solid #27ae60; padding: 15px 20px; margin: 15px 0; border-radius: 5px;">
                            <p style="margin: 5px 0; color: #155724;">Best <b>Accuracy</b>: <b>{best_acc['Model']}</b> ({best_acc['Accuracy']:.4f})</p>
                            <p style="margin: 5px 0; color: #155724;">Best <b>F1-Score</b>: <b>{best_f1['Model']}</b> ({best_f1['F1-Score']:.4f})</p>
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
            st.info("**📖 K-Fold CV evaluates model stability. Lower Std = more consistent performance.**")
            
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
                
                st.session_state.cv_ran = True
                st.session_state.last_cv_results = cv_results
                
                if cv_results:
                    cv_df = pd.DataFrame(cv_results)
                    st.markdown("#### 📊 Cross Validation Results")
                    st.markdown("*Lower Std = more consistent performance across data splits*")
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
                    best_cv = cv_df.loc[cv_df['Mean F1'].idxmax()]
                    st.success(f"🏆 Best CV Model: **{best_cv['Model']}** (Mean F1: {best_cv['Mean F1']:.4f} ±{best_cv['Std F1']:.4f})")
                    
                    fig_cv = go.Figure()
                    for name, scores in all_scores.items():
                        fig_cv.add_trace(go.Box(y=scores, name=name, boxmean='sd'))
                    fig_cv.update_layout(
                        title=f'Cross Validation F1 Scores ({folds}-Fold)<br><sub>Box shows spread of scores across folds</sub>',
                        yaxis_title='F1 Score', height=400)
                    st.plotly_chart(fig_cv, use_container_width=True)
        
        # ===== TAB 4: OVERALL PIPELINE SUMMARY =====
        with eval_tab4:
            st.markdown("### 📋 Overall Pipeline Summary")
            st.info("**📖 This table provides a comprehensive overview of ALL processing steps performed across the entire dashboard, tracking what was done, results, and key findings.**")
            
            if st.button("🔄 Generate Pipeline Summary", type="primary", key="pipeline_summary", use_container_width=True):
                
                pipeline_data = []
                
                # STAGE 1: DATA LOADING
                if st.session_state.data is not None:
                    df = st.session_state.data
                    pipeline_data.append({
                        'Stage': '1. Data Loading',
                        'Step': 'Dataset Loaded',
                        'Status': '✅ Completed',
                        'Details': f'Shape: {df.shape[0]} rows × {df.shape[1]} columns',
                        'Key Findings': f'{df.shape[0]} samples with {df.shape[1]} features'
                    })
                    missing_pct = (df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100
                    pipeline_data.append({
                        'Stage': '1. Data Loading',
                        'Step': 'Missing Values Analysis',
                        'Status': '✅ Completed',
                        'Details': f'{df.isnull().sum().sum()} missing ({missing_pct:.2f}%)',
                        'Key Findings': 'Significant missing data' if missing_pct > 5 else 'Minimal missing data'
                    })
                    pipeline_data.append({
                        'Stage': '1. Data Loading',
                        'Step': 'Data Types Identified',
                        'Status': '✅ Completed',
                        'Details': f'Numeric: {len(df.select_dtypes(include=["float64","int64"]).columns)}, Categorical: {len(df.select_dtypes(include=["object"]).columns)}',
                        'Key Findings': f'{len(df.select_dtypes(include=["object"]).columns)} columns need encoding'
                    })
                else:
                    pipeline_data.append({
                        'Stage': '1. Data Loading',
                        'Step': 'Dataset',
                        'Status': '❌ Not Loaded',
                        'Details': 'No data available',
                        'Key Findings': 'Upload or generate data first'
                    })
                
                # STAGE 2: PREPROCESSING
                pipeline_data.append({
                    'Stage': '2. Preprocessing',
                    'Step': 'Feature Scaling (StandardScaler)',
                    'Status': '✅ Completed' if st.session_state.get('scaled_data') is not None else '⬜ Not Run',
                    'Details': f'{len(st.session_state.get("scaled_columns", []))} columns scaled (mean=0, std=1)' if st.session_state.get('scaled_data') is not None else 'Not yet applied',
                    'Key Findings': 'Features normalized' if st.session_state.get('scaled_data') is not None else 'Run to normalize features'
                })
                
                pipeline_data.append({
                    'Stage': '2. Preprocessing',
                    'Step': 'Categorical Encoding (One-Hot)',
                    'Status': '✅ Completed' if st.session_state.get('encoded_data') is not None else '⬜ Not Run',
                    'Details': f'Shape after encoding: {st.session_state.get("encoded_shape", "N/A")}' if st.session_state.get('encoded_data') is not None else 'Not yet applied',
                    'Key Findings': 'Categories converted to binary columns' if st.session_state.get('encoded_data') is not None else 'Run to convert text to numbers'
                })
                
                pipeline_data.append({
                    'Stage': '2. Preprocessing',
                    'Step': 'Outlier Capping (IQR)',
                    'Status': '✅ Completed' if st.session_state.get('processed_data') is not None and st.session_state.get('scaled_data') is None else '⬜ Not Run',
                    'Details': 'Extreme values capped to IQR bounds' if st.session_state.get('processed_data') is not None else 'Not yet applied',
                    'Key Findings': 'Outliers reduced' if st.session_state.get('processed_data') is not None else 'Run to handle extreme values'
                })
                
                # STAGE 3: FEATURE ENGINEERING
                pipeline_data.append({
                    'Stage': '3. Feature Selection',
                    'Step': 'Feature Importance Analysis',
                    'Status': '✅ Completed' if st.session_state.get('feature_importance') is not None else '⬜ Not Run',
                    'Details': f'Top features identified via MI, Chi2, RF' if st.session_state.get('feature_importance') is not None else 'Not yet run',
                    'Key Findings': f'{len(st.session_state.get("selected_features", []))} top features selected' if st.session_state.get('selected_features') is not None else 'Run to identify important features'
                })
                
                # STAGE 4: MODEL TRAINING
                pipeline_data.append({
                    'Stage': '4. Model Training',
                    'Step': 'Models Trained',
                    'Status': '✅ Completed' if st.session_state.get('trained') else '⬜ Not Run',
                    'Details': f'{len(st.session_state.get("models", {}))} models trained' if st.session_state.get('trained') else 'Not yet trained',
                    'Key Findings': 'LR, RF, GBC trained' if st.session_state.get('trained') else 'Run Modeling section'
                })
                
                # STAGE 5: EVALUATION
                pipeline_data.append({
                    'Stage': '5. Model Evaluation',
                    'Step': 'Individual Model Evaluation',
                    'Status': '✅ Completed' if st.session_state.get('evaluation_ran') else '⬜ Not Run',
                    'Details': 'Accuracy, Precision, Recall, F1 calculated' if st.session_state.get('evaluation_ran') else 'Not yet evaluated',
                    'Key Findings': 'See Evaluate Models tab for results' if st.session_state.get('evaluation_ran') else 'Run Evaluate Models'
                })
                
                pipeline_data.append({
                    'Stage': '5. Model Evaluation',
                    'Step': 'Target Comparison (Risk_Type vs Risk_Level)',
                    'Status': '✅ Completed' if st.session_state.get('comparison_ran') else '⬜ Not Run',
                    'Details': 'Both targets evaluated' if st.session_state.get('comparison_ran') else 'Not yet compared',
                    'Key Findings': 'See Compare Both Targets tab' if st.session_state.get('comparison_ran') else 'Run comparison'
                })
                
                pipeline_data.append({
                    'Stage': '5. Model Evaluation',
                    'Step': 'Cross Validation',
                    'Status': '✅ Completed' if st.session_state.get('cv_ran') else '⬜ Not Run',
                    'Details': 'K-Fold CV with stability metrics' if st.session_state.get('cv_ran') else 'Not yet run',
                    'Key Findings': 'See Cross Validation tab' if st.session_state.get('cv_ran') else 'Run CV for stability check'
                })
                
                # Display pipeline table
                pipeline_df = pd.DataFrame(pipeline_data)
                
                st.markdown("### 📊 Complete Analysis Pipeline Overview")
                
                # Color-coded status
                def color_status(val):
                    if '✅' in str(val):
                        return 'background-color: #d4edda; color: #155724; font-weight: bold'
                    elif '❌' in str(val):
                        return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                    return 'background-color: #fff3cd; color: #856404'
                
                styled_df = pipeline_df.style.applymap(color_status, subset=['Status'])
                
                st.dataframe(
                    styled_df,
                    column_config={
                        "Stage": st.column_config.TextColumn("Stage", width="small"),
                        "Step": st.column_config.TextColumn("Step", width="medium"),
                        "Status": st.column_config.TextColumn("Status", width="small"),
                        "Details": st.column_config.TextColumn("Details", width="large"),
                        "Key Findings": st.column_config.TextColumn("Key Findings", width="large"),
                    },
                    use_container_width=True,
                    height=500,
                )
                
                # Summary counts
                completed = sum(1 for d in pipeline_data if '✅' in d['Status'])
                not_run = sum(1 for d in pipeline_data if '⬜' in d['Status'])
                failed = sum(1 for d in pipeline_data if '❌' in d['Status'])
                
                st.markdown("---")
                st.markdown("### 📊 Pipeline Progress")
                c1,c2,c3 = st.columns(3)
                with c1: st.metric("✅ Completed", completed)
                with c2: st.metric("⬜ Pending", not_run)
                with c3: st.metric("❌ Failed", failed)
                
                # Progress bar
                total_steps = len(pipeline_data)
                progress_pct = (completed / total_steps) * 100 if total_steps > 0 else 0
                st.progress(int(progress_pct), text=f"Overall Progress: {progress_pct:.0f}% ({completed}/{total_steps} steps completed)")
                
                if not_run > 0:
                    st.info(f"💡 **{not_run} step(s)** still pending. Run the corresponding sections to complete the full analysis pipeline.")
                elif completed == total_steps:
                    st.success("🎉 **All pipeline steps completed!** The analysis is ready for interpretation and reporting.")


if __name__ == "__main__":
    main()
