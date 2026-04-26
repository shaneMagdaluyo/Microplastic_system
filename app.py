"""
Microplastic Risk Analysis Dashboard
A comprehensive Streamlit application for analyzing microplastic risk data,
featuring data preprocessing, EDA, model training, and evaluation.
Enhanced with professional styled tables and optimized performance.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE
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
    if 'data' not in st.session_state:
        st.session_state.data = None
    if 'processed_data' not in st.session_state:
        st.session_state.processed_data = None
    if 'models' not in st.session_state:
        st.session_state.models = {}
    if 'feature_importance' not in st.session_state:
        st.session_state.feature_importance = None
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    if 'preprocessing_log' not in st.session_state:
        st.session_state.preprocessing_log = []
    if 'trained' not in st.session_state:
        st.session_state.trained = False
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    if 'encoders' not in st.session_state:
        st.session_state.encoders = {}
    if 'scaler' not in st.session_state:
        st.session_state.scaler = None
    if 'target_encoder' not in st.session_state:
        st.session_state.target_encoder = None
    if 'selected_features' not in st.session_state:
        st.session_state.selected_features = None

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
        'MP_Count_per_L': np.random.poisson(lam=50, size=n_samples),
        'Particle_Size_um': np.random.normal(100, 30, n_samples),
        'Polymer_Type': np.random.choice(['PE', 'PP', 'PS', 'PET', 'PVC', 'Nylon'], n_samples),
        'Water_Source': np.random.choice(['River', 'Lake', 'Ocean', 'Groundwater', 'Tap'], n_samples),
        'pH': np.random.normal(7, 0.5, n_samples),
        'Temperature_C': np.random.normal(20, 5, n_samples),
        'Turbidity_NTU': np.random.exponential(10, n_samples),
        'Dissolved_O2_mgL': np.random.normal(8, 2, n_samples),
        'Conductivity_uScm': np.random.normal(500, 150, n_samples),
        'Risk_Score': np.random.uniform(0, 100, n_samples),
        'Risk_Level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n_samples, 
                                       p=[0.3, 0.35, 0.25, 0.1]),
        'Risk_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples, 
                                     p=[0.5, 0.3, 0.2]),
        'Location': np.random.choice(['Urban', 'Rural', 'Industrial', 'Coastal'], n_samples),
        'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_samples)
    }
    
    df = pd.DataFrame(data)
    for col in df.columns:
        if col != 'Sample_ID' and df[col].dtype in ['float64', 'int64']:
            mask = np.random.random(n_samples) < 0.05
            df.loc[mask, col] = np.nan
    
    return df

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].isnull().sum() > 0:
            if df_clean[col].dtype in ['float64', 'int64']:
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df_clean[col].fillna(median_val, inplace=True)
            else:
                mode_series = df_clean[col].mode()
                mode_val = mode_series[0] if not mode_series.empty else 'Unknown'
                df_clean[col].fillna(mode_val, inplace=True)
    return df_clean

def encode_categorical(df):
    """Encode categorical variables."""
    df_encoded = df.copy()
    categorical_cols = df_encoded.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col not in ['Sample_ID']:
            le = LabelEncoder()
            df_encoded[f'{col}_Encoded'] = le.fit_transform(df_encoded[col].astype(str))
    return df_encoded

def scale_features(df, feature_cols):
    """Scale numerical features."""
    df_scaled = df.copy()
    scaler = StandardScaler()
    numeric_cols = df_scaled[feature_cols].select_dtypes(include=['float64', 'int64']).columns
    if len(numeric_cols) > 0:
        df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
        st.session_state.scaler = scaler
    return df_scaled

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

def plot_distribution(data, column, title):
    """Create distribution plot."""
    clean_data = data[column].dropna()
    if clean_data.empty:
        return go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
    fig.add_trace(go.Histogram(x=clean_data, name='Distribution', nbinsx=30, marker_color='#3498db'), row=1, col=1)
    fig.add_trace(go.Box(y=clean_data, name='Box Plot', marker_color='#e74c3c'), row=1, col=2)
    fig.update_layout(title_text=title, showlegend=False, height=500)
    return fig

def plot_correlation_heatmap(df, columns):
    """Create correlation heatmap."""
    numeric_df = df[columns].select_dtypes(include=['float64', 'int64', 'int32'])
    if numeric_df.shape[1] < 2:
        return go.Figure(), None
    numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
    corr_matrix = numeric_df.corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=corr_matrix.columns.tolist(),
        y=corr_matrix.index.tolist(),
        colorscale='RdBu',
        zmin=-1,
        zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        showscale=True
    ))
    fig.update_layout(title='Feature Correlation Heatmap', height=600)
    return fig, corr_matrix

def prepare_modeling_data(df, feature_cols, target_col):
    """Prepare data for modeling with enhanced error handling."""
    X = df[feature_cols].select_dtypes(include=['float64', 'int64', 'int32'])
    if X.shape[1] == 0:
        st.error("❌ No numeric features selected. Please select at least one numeric feature.")
        return None, None
    y = df[target_col]
    valid_mask = y.notna()
    X = X[valid_mask]
    y = y[valid_mask]
    if len(y) == 0:
        st.error("❌ No valid target values found.")
        return None, None
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
        st.session_state.target_encoder = le
        class_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
        st.info(f"📋 Target Encoding Mapping: {class_mapping}")
    if X.isnull().sum().sum() > 0:
        st.warning(f"⚠️ Found {X.isnull().sum().sum()} missing values in features. Filling with median...")
        X = X.fillna(X.median())
    if len(X) < 10:
        st.error(f"❌ Only {len(X)} samples available. At least 10 samples are recommended for training.")
        return None, None
    return X, y

def train_models_fast(X_train, X_test, y_train, y_test):
    """Train classification models with optimized fast performance."""
    models = {}
    training_times = {}
    n_samples = X_train.shape[0]
    
    # Logistic Regression
    start_time = time.time()
    try:
        lr_model = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', solver='lbfgs', n_jobs=-1)
        lr_model.fit(X_train, y_train)
        models['Logistic Regression'] = lr_model
        training_times['Logistic Regression'] = time.time() - start_time
    except:
        try:
            lr_model = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', solver='saga', n_jobs=-1)
            lr_model.fit(X_train, y_train)
            models['Logistic Regression'] = lr_model
            training_times['Logistic Regression'] = time.time() - start_time
        except:
            pass
    
    # Random Forest
    start_time = time.time()
    try:
        n_estimators = min(80, max(30, n_samples // 5))
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=42,
                                          class_weight='balanced', max_depth=min(12, n_samples // 30),
                                          n_jobs=-1)
        rf_model.fit(X_train, y_train)
        models['Random Forest'] = rf_model
        training_times['Random Forest'] = time.time() - start_time
    except:
        try:
            rf_model = RandomForestClassifier(n_estimators=30, random_state=42, max_depth=8, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            models['Random Forest'] = rf_model
            training_times['Random Forest'] = time.time() - start_time
        except:
            pass
    
    # Decision Tree
    start_time = time.time()
    try:
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=min(10, max(3, n_samples // 30)),
                                          min_samples_split=max(2, n_samples // 50), class_weight='balanced')
        dt_model.fit(X_train, y_train)
        models['Decision Tree'] = dt_model
        training_times['Decision Tree'] = time.time() - start_time
    except:
        try:
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=5)
            dt_model.fit(X_train, y_train)
            models['Decision Tree'] = dt_model
            training_times['Decision Tree'] = time.time() - start_time
        except:
            pass
    
    return models, training_times

def train_models_quality(X_train, X_test, y_train, y_test):
    """Train classification models with GridSearch for better quality."""
    models = {}
    training_times = {}
    
    # Logistic Regression with GridSearch
    start_time = time.time()
    try:
        lr_params = {'C': [0.1, 1, 10], 'max_iter': [1000]}
        lr_grid = GridSearchCV(LogisticRegression(random_state=42, class_weight='balanced', n_jobs=-1),
                               lr_params, cv=3, scoring='f1_weighted')
        lr_grid.fit(X_train, y_train)
        models['Logistic Regression'] = lr_grid.best_estimator_
        training_times['Logistic Regression'] = time.time() - start_time
    except:
        pass
    
    # Random Forest
    start_time = time.time()
    try:
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        rf_model.fit(X_train, y_train)
        models['Random Forest'] = rf_model
        training_times['Random Forest'] = time.time() - start_time
    except:
        pass
    
    # Decision Tree
    start_time = time.time()
    try:
        dt_model = DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced')
        dt_model.fit(X_train, y_train)
        models['Decision Tree'] = dt_model
        training_times['Decision Tree'] = time.time() - start_time
    except:
        pass
    
    return models, training_times

def evaluate_models(models, X_test, y_test):
    """Evaluate trained models."""
    evaluation_results = {}
    for name, model in models.items():
        try:
            y_pred = model.predict(X_test)
            evaluation_results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
        except:
            pass
    return evaluation_results

def main():
    """Main application function."""
    
    # Main header
    st.markdown('<p class="main-header">🔬 Microplastic Risk Analysis Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio(
        "Select Section",
        ["📁 Upload Dataset", "🔧 Data Preprocessing", "📈 EDA (Risk Analysis)",
         "🛠️ Feature Engineering", "🤖 Model Training", "📊 Model Evaluation",
         "🎯 Feature Importance", "🧬 Polymer Analysis"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(
        "This dashboard analyzes microplastic risk data to predict risk types "
        "and identify key factors contributing to microplastic pollution."
    )
    
    # Status indicators in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Status")
    if st.session_state.data is not None:
        st.sidebar.success("✅ Data Loaded")
    else:
        st.sidebar.warning("⚠️ No Data")
    
    if st.session_state.trained:
        st.sidebar.success(f"✅ Models Trained ({len(st.session_state.models)})")
    else:
        st.sidebar.warning("⚠️ Models Not Trained")
    
    # Section: Upload Dataset
    if section == "📁 Upload Dataset":
        st.markdown('<p class="section-header">📁 Upload Dataset</p>', unsafe_allow_html=True)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload your microplastic dataset (CSV or Excel)",
                type=['csv', 'xlsx', 'xls']
            )
            if uploaded_file is not None:
                load_dataset(uploaded_file)
        
        with col2:
            st.markdown("#### Quick Start")
            if st.button("Generate Sample Dataset", type="primary"):
                st.session_state.data = generate_sample_data()
                st.success("✅ Sample dataset generated!")
                st.rerun()
        
        if st.session_state.data is not None:
            st.markdown("---")
            st.markdown('<p class="subsection-header">📋 Dataset Preview</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            df = st.session_state.data
            
            with col1:
                st.metric("Number of Samples", df.shape[0])
            with col2:
                st.metric("Number of Features", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            # Enhanced preview with styled HTML table
            preview_df = df.head(10)
            html_preview = '<div style="overflow-x: auto;"><table style="width:100%; border-collapse: collapse; font-size: 0.85rem;">'
            html_preview += '<tr style="background-color: #1f77b4; color: white;">'
            for col in preview_df.columns:
                html_preview += f'<th style="padding: 8px; text-align: left;">{col}</th>'
            html_preview += '</tr>'
            
            for i, (_, row) in enumerate(preview_df.iterrows()):
                bg = '#f8f9fa' if i % 2 == 0 else '#ffffff'
                html_preview += f'<tr style="background-color: {bg};">'
                for val in row:
                    if pd.isna(val):
                        html_preview += '<td style="padding: 6px; color: #e74c3c; font-style: italic;">NULL</td>'
                    elif isinstance(val, float):
                        html_preview += f'<td style="padding: 6px;">{val:.2f}</td>'
                    else:
                        html_preview += f'<td style="padding: 6px;">{val}</td>'
                html_preview += '</tr>'
            
            html_preview += '</table></div>'
            st.markdown(html_preview, unsafe_allow_html=True)
            
            st.markdown("---")
            st.markdown("#### Dataset Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Types:**")
                st.write(df.dtypes)
            with col2:
                st.write("**Basic Statistics:**")
                st.write(df.describe())
    
    # Section: Data Preprocessing
    elif section == "🔧 Data Preprocessing":
        st.markdown('<p class="section-header">🔧 Data Preprocessing</p>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            return
        
        df = st.session_state.data.copy()
        
        preprocessing_options = st.multiselect(
            "Select Preprocessing Steps",
            ["Handle Missing Values", "Encode Categorical Variables",
             "Detect Outliers", "Scale Features"],
            default=["Handle Missing Values", "Encode Categorical Variables"]
        )
        
        if st.button("Run Preprocessing", type="primary"):
            processed_df = df.copy()
            
            if "Handle Missing Values" in preprocessing_options:
                st.markdown("### 🔍 Missing Value Analysis")
                with st.spinner('Handling missing values...'):
                    processed_df = handle_missing_values(processed_df)
                    st.success("✅ Missing values handled")
            
            if "Encode Categorical Variables" in preprocessing_options:
                st.markdown("### 🔄 Categorical Encoding")
                with st.spinner('Encoding categorical variables...'):
                    processed_df = encode_categorical(processed_df)
                    encoded_cols = [col for col in processed_df.columns if col.endswith('_Encoded')]
                    st.write("**Encoded columns added:**", encoded_cols)
            
            if "Detect Outliers" in preprocessing_options:
                st.markdown("### 🎯 Outlier Detection")
                numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
                outlier_info = detect_outliers(processed_df, numeric_cols)
                for col, info in outlier_info.items():
                    if info['count'] > 0:
                        st.write(f"**{col}**: {info['count']} outliers ({info['percentage']:.1f}%)")
            
            if "Scale Features" in preprocessing_options:
                st.markdown("### 📏 Feature Scaling")
                numeric_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
                processed_df = scale_features(processed_df, numeric_cols)
                st.success("✅ Features scaled using StandardScaler")
            
            st.session_state.processed_data = processed_df
            st.success("✅ Preprocessing completed!")
        
        if st.session_state.processed_data is not None:
            st.markdown("---")
            st.markdown("### 📋 Processed Data Preview")
            st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)
    
    # Section: EDA
    elif section == "📈 EDA (Risk Analysis)":
        st.markdown('<p class="section-header">📈 Exploratory Data Analysis</p>', unsafe_allow_html=True)
        
        data_to_use = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        if data_to_use is None:
            st.warning("⚠️ Please load and preprocess data first!")
            return
        
        df = data_to_use.copy()
        
        st.markdown("### 📊 Risk Score Distribution")
        
        if 'Risk_Score' in df.columns:
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean_risk = df['Risk_Score'].dropna()
            
            if len(clean_risk) > 0:
                fig_dist = plot_distribution(df, 'Risk_Score', 'Risk Score Distribution')
                st.plotly_chart(fig_dist, use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Mean Risk Score", f"{clean_risk.mean():.2f}")
                with col2:
                    st.metric("Median Risk Score", f"{clean_risk.median():.2f}")
                with col3:
                    st.metric("Max Risk Score", f"{clean_risk.max():.2f}")
                with col4:
                    st.metric("Min Risk Score", f"{clean_risk.min():.2f}")
            else:
                st.warning("⚠️ No valid Risk Score data")
        else:
            st.warning("⚠️ 'Risk_Score' column not found in dataset")
        
        st.markdown("---")
        st.markdown("### 🔬 MP Count vs Risk Score")
        
        if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
            df['MP_Count_per_L'] = pd.to_numeric(df['MP_Count_per_L'], errors='coerce')
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean_df = df.dropna(subset=['MP_Count_per_L', 'Risk_Score'])
            
            if not clean_df.empty:
                try:
                    fig_scatter = px.scatter(clean_df, x='MP_Count_per_L', y='Risk_Score',
                                            color='Risk_Level' if 'Risk_Level' in clean_df.columns else None,
                                            trendline='ols', title='Microplastic Count vs Risk Score')
                except:
                    fig_scatter = px.scatter(clean_df, x='MP_Count_per_L', y='Risk_Score',
                                            color='Risk_Level' if 'Risk_Level' in clean_df.columns else None,
                                            title='Microplastic Count vs Risk Score')
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.warning("⚠️ No valid numeric data for plotting.")
        else:
            st.warning("⚠️ Required columns not found")
        
        st.markdown("---")
        st.markdown("### 📊 Risk Score by Risk Level")
        
        if 'Risk_Level' in df.columns and 'Risk_Score' in df.columns:
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean_df = df.dropna(subset=['Risk_Score'])
            
            if len(clean_df) > 0:
                fig_box = px.box(clean_df, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                                title='Risk Score Distribution by Risk Level')
                fig_box.update_layout(height=500)
                st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.warning("⚠️ No valid data for box plot")
        else:
            st.warning("⚠️ Required columns not found")
        
        st.markdown("---")
        st.markdown("### 📈 Additional Analysis")
        
        eda_options = st.multiselect("Select variables to analyze", df.columns.tolist(),
                                     default=['Risk_Score'] if 'Risk_Score' in df.columns else [])
        if eda_options:
            for col in eda_options:
                if df[col].dtype in ['float64', 'int64']:
                    if len(df[col].dropna()) > 0:
                        fig = px.histogram(df, x=col, title=f'Distribution of {col}')
                        st.plotly_chart(fig, use_container_width=True)
    
    # Section: Feature Engineering
    elif section == "🛠️ Feature Engineering":
        st.markdown('<p class="section-header">🛠️ Feature Engineering & Selection</p>', 
                   unsafe_allow_html=True)
        
        data_to_use = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        if data_to_use is None:
            st.warning("⚠️ Please load and preprocess data first!")
            return
        
        df = data_to_use
        
        st.markdown("### 🎯 Target Variable Selection")
        target_col = st.selectbox("Select Target Variable", df.columns.tolist(),
                                  index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        
        st.markdown("### 🔍 Feature Selection")
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        st.markdown("#### 📊 Correlation Analysis")
        if len(numeric_cols) > 1:
            with st.spinner('Computing correlation matrix...'):
                fig_corr, corr_matrix = plot_correlation_heatmap(df, numeric_cols)
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("⚠️ Not enough numeric features for correlation analysis")
        
        st.markdown("#### 🌲 Random Forest Feature Importance")
        if st.button("Calculate Feature Importance", type="primary"):
            try:
                X = df[numeric_cols].copy()
                y = df[target_col].copy()
                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)
                X = X.fillna(X.median()).dropna(axis=1, how='any')
                if X.shape[1] == 0:
                    st.error("No valid features after cleaning")
                    return
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(X, y)
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=True)
                st.session_state.feature_importance = importance_df
                fig_imp = px.bar(importance_df.tail(15), x='importance', y='feature',
                                orientation='h', title='Top 15 Feature Importance (Random Forest)')
                fig_imp.update_layout(height=500)
                st.plotly_chart(fig_imp, use_container_width=True)
                top_features = importance_df.nlargest(10, 'importance')['feature'].tolist()
                st.session_state.selected_features = top_features
                st.success(f"✅ Selected top {len(top_features)} features")
                st.write("**Selected Features:**", top_features)
            except Exception as e:
                st.error(f"Error calculating feature importance: {str(e)}")
    
    # Section: Model Training
    elif section == "🤖 Model Training":
        st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)
        
        data_to_use = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        if data_to_use is None:
            st.warning("⚠️ Please load and preprocess data first!")
            return
        
        df = data_to_use
        
        st.markdown("### 🎯 Model Configuration")
        target_col = st.selectbox("Select Target Variable", df.columns.tolist(), key='train_target')
        all_features = [col for col in df.columns if col != target_col]
        default_features = st.session_state.get('selected_features', 
                          df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:5])
        default_features = [f for f in default_features if f in all_features]
        feature_cols = st.multiselect("Select Features", all_features, default=default_features)
        
        st.markdown("### ⚙️ Model Parameters")
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", 0, 100, 42)
        with col2:
            use_smote = st.checkbox("Use SMOTE for class imbalance", value=True)
            fast_mode = st.checkbox("⚡ Fast Training Mode", value=True)
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if len(feature_cols) == 0:
                st.error("❌ Please select at least one feature!")
                return
            
            X, y = prepare_modeling_data(df, feature_cols, target_col)
            if X is None or y is None:
                return
            
            class_counts = pd.Series(y).value_counts()
            st.info("### 📊 Class Distribution")
            st.write(class_counts)
            
            use_stratify = len(class_counts) > 1 and class_counts.min() >= 2
            try:
                if use_stratify:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                                        random_state=random_state, stratify=y)
                else:
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                                        random_state=random_state)
            except:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, 
                                                                    random_state=random_state)
            
            st.info(f"📊 Training set: {X_train.shape[0]} samples, Test set: {X_test.shape[0]} samples")
            
            if use_smote:
                train_class_counts = pd.Series(y_train).value_counts()
                if train_class_counts.min() >= 2:
                    try:
                        smote = SMOTE(random_state=random_state, k_neighbors=min(5, train_class_counts.min()-1))
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        st.success("✅ SMOTE applied successfully!")
                    except:
                        st.warning("⚠️ SMOTE failed. Training without SMOTE...")
            
            total_start = time.time()
            if fast_mode:
                with st.spinner('⚡ Training models in FAST mode...'):
                    models, training_times = train_models_fast(X_train, X_test, y_train, y_test)
            else:
                with st.spinner('🔬 Training models in QUALITY mode...'):
                    models, training_times = train_models_quality(X_train, X_test, y_train, y_test)
            total_time = time.time() - total_start
            
            if models and len(models) > 0:
                st.session_state.models = models
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.X_train = X_train
                st.session_state.y_train = y_train
                st.session_state.trained = True
                
                st.success(f"✅ Successfully trained {len(models)} models in {total_time:.2f} seconds!")
                if fast_mode:
                    st.balloons()
                
                st.markdown("### 📊 Quick Performance Overview")
                cols = st.columns(len(models))
                for idx, (name, model) in enumerate(models.items()):
                    try:
                        train_score = model.score(X_train, y_train)
                        test_score = model.score(X_test, y_test)
                        with cols[idx]:
                            st.markdown(f"**{name}**")
                            st.metric("Train Score", f"{train_score:.3f}")
                            st.metric("Test Score", f"{test_score:.3f}")
                    except:
                        pass
                
                st.info("👉 Go to **'📊 Model Evaluation'** in the sidebar to see detailed results!")
            else:
                st.error("❌ No models were successfully trained. Please check your data and try again.")
    
    # ==================== MODEL EVALUATION (ENHANCED TABLES) ====================
    elif section == "📊 Model Evaluation":
        st.markdown('<p class="section-header">📊 Model Evaluation</p>', unsafe_allow_html=True)
        
        if not st.session_state.get('trained', False) or len(st.session_state.get('models', {})) == 0:
            st.warning("⚠️ No trained models found!")
            st.info("👉 Please go to **'🤖 Model Training'** section to train your models first.")
            return
        
        models = st.session_state.models
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        if X_test is None or y_test is None:
            st.error("❌ Test data is missing. Please re-run the Model Training step.")
            return
        
        st.success(f"✅ Found {len(models)} trained model(s)")
        
        with st.spinner('Evaluating models...'):
            evaluation_results = evaluate_models(models, X_test, y_test)
        
        if evaluation_results and len(evaluation_results) > 0:
            metrics_data = {}
            for name, results in evaluation_results.items():
                metrics_data[name] = {
                    'Accuracy': results['accuracy'],
                    'F1 Score': results['f1_score']
                }
            metrics_df = pd.DataFrame(metrics_data).T
            
            # ==================== STYLED METRICS TABLE ====================
            st.markdown("#### 📈 Model Performance Comparison")
            
            styled_df = metrics_df.copy().round(4)
            styled_df['Model'] = styled_df.index
            styled_df['Accuracy %'] = (styled_df['Accuracy'] * 100).round(2)
            styled_df['F1 Score %'] = (styled_df['F1 Score'] * 100).round(2)
            styled_df['Rank'] = styled_df['F1 Score'].rank(ascending=False).astype(int)
            
            best_acc_idx = styled_df['Accuracy'].idxmax()
            best_f1_idx = styled_df['F1 Score'].idxmax()
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                st.markdown("**📊 Performance Metrics**")
                
                html_table = '<div style="overflow-x: auto;"><table style="width:100%; border-collapse: collapse; font-family: Arial;">'
                html_table += '<tr style="background-color: #1f77b4; color: white;">'
                html_table += '<th style="padding: 12px; text-align: left;">Model</th>'
                html_table += '<th style="padding: 12px; text-align: center;">Accuracy</th>'
                html_table += '<th style="padding: 12px; text-align: center;">F1 Score</th>'
                html_table += '<th style="padding: 12px; text-align: center;">Rank</th></tr>'
                
                for i, (model, row) in enumerate(styled_df.iterrows()):
                    bg = '#f8f9fa' if i % 2 == 0 else '#ffffff'
                    if model == best_f1_idx:
                        bg = '#d4edda'
                    
                    html_table += f'<tr style="background-color: {bg};">'
                    html_table += f'<td style="padding: 10px; font-weight: bold;">{model}</td>'
                    html_table += f'<td style="padding: 10px; text-align: center;">{row["Accuracy"]:.4f}</td>'
                    html_table += f'<td style="padding: 10px; text-align: center;">{row["F1 Score"]:.4f}</td>'
                    
                    rank = row['Rank']
                    if rank == 1: medal = '🥇'
                    elif rank == 2: medal = '🥈'
                    elif rank == 3: medal = '🥉'
                    else: medal = f'#{rank}'
                    html_table += f'<td style="padding: 10px; text-align: center; font-size: 1.2rem;">{medal}</td>'
                    html_table += '</tr>'
                
                html_table += '</table></div>'
                st.markdown(html_table, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**🏆 Best Scores**")
                st.metric("Accuracy", f"{best_acc_idx}", 
                         delta=f"{styled_df.loc[best_acc_idx, 'Accuracy %']:.1f}%")
                st.metric("F1 Score", f"{best_f1_idx}", 
                         delta=f"{styled_df.loc[best_f1_idx, 'F1 Score %']:.1f}%")
                
                st.markdown("**📊 Score Distribution**")
                for model in styled_df.index:
                    st.write(f"**{model}**")
                    st.progress(float(styled_df.loc[model, 'Accuracy']))
            
            with col3:
                st.markdown("**📈 Quick Stats**")
                st.metric("Avg Accuracy", f"{styled_df['Accuracy'].mean():.3f}")
                st.metric("Avg F1 Score", f"{styled_df['F1 Score'].mean():.3f}")
                st.metric("Best Margin", 
                         f"{(styled_df['F1 Score'].max() - styled_df['F1 Score'].min()):.3f}")
            
            # ==================== CONFUSION MATRIX ====================
            st.markdown("---")
            st.markdown("#### 📊 Confusion Matrices")
            
            selected_model = st.selectbox("Select model to view", list(evaluation_results.keys()))
            
            if selected_model:
                cm = evaluation_results[selected_model].get('confusion_matrix')
                if cm is not None and cm.size > 0:
                    n_classes = cm.shape[0]
                    
                    col_a, col_b = st.columns([1, 1])
                    
                    with col_a:
                        st.markdown(f"**{selected_model} - Confusion Matrix**")
                        
                        html_cm = '<div style="overflow-x: auto;"><table style="width:100%; border-collapse: collapse; font-family: monospace;">'
                        html_cm += '<tr style="background-color: #2c3e50; color: white;">'
                        html_cm += '<th style="padding: 10px;"></th>'
                        for j in range(n_classes):
                            html_cm += f'<th style="padding: 10px; text-align: center;">Pred {j}</th>'
                        html_cm += '<th style="padding: 10px; text-align: center;">Total</th></tr>'
                        
                        for i in range(n_classes):
                            row_sum = cm[i, :].sum()
                            html_cm += '<tr>'
                            html_cm += f'<td style="padding: 10px; font-weight: bold; background-color: #34495e; color: white;">Actual {i}</td>'
                            for j in range(n_classes):
                                val = cm[i, j]
                                bg = '#d4edda' if i == j else '#f8d7da' if val > 0 else '#ffffff'
                                html_cm += f'<td style="padding: 10px; text-align: center; background-color: {bg};">{val}</td>'
                            html_cm += f'<td style="padding: 10px; text-align: center; font-weight: bold;">{row_sum}</td>'
                            html_cm += '</tr>'
                        
                        html_cm += '</table></div>'
                        st.markdown(html_cm, unsafe_allow_html=True)
                    
                    with col_b:
                        st.markdown(f"**Per-Class Metrics**")
                        
                        metrics_list = []
                        for i in range(n_classes):
                            tp = int(cm[i, i])
                            fp = int(cm[:, i].sum()) - tp
                            fn = int(cm[i, :].sum()) - tp
                            prec = tp/(tp+fp) if (tp+fp) > 0 else 0
                            rec = tp/(tp+fn) if (tp+fn) > 0 else 0
                            f1_val = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
                            support = int(cm[i, :].sum())
                            metrics_list.append({
                                'Class': f'Class {i}',
                                'Precision': f'{prec:.3f}',
                                'Recall': f'{rec:.3f}',
                                'F1-Score': f'{f1_val:.3f}',
                                'Support': support
                            })
                        
                        html_metrics = '<div style="overflow-x: auto;"><table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">'
                        html_metrics += '<tr style="background-color: #1f77b4; color: white;">'
                        for col in ['Class', 'Precision', 'Recall', 'F1-Score', 'Support']:
                            html_metrics += f'<th style="padding: 8px; text-align: center;">{col}</th>'
                        html_metrics += '</tr>'
                        
                        for i, row in enumerate(metrics_list):
                            bg = '#f8f9fa' if i % 2 == 0 else '#ffffff'
                            html_metrics += f'<tr style="background-color: {bg};">'
                            html_metrics += f'<td style="padding: 8px; text-align: center; font-weight: bold;">{row["Class"]}</td>'
                            html_metrics += f'<td style="padding: 8px; text-align: center;">{row["Precision"]}</td>'
                            html_metrics += f'<td style="padding: 8px; text-align: center;">{row["Recall"]}</td>'
                            html_metrics += f'<td style="padding: 8px; text-align: center;">{row["F1-Score"]}</td>'
                            html_metrics += f'<td style="padding: 8px; text-align: center;">{row["Support"]}</td>'
                            html_metrics += '</tr>'
                        
                        html_metrics += '</table></div>'
                        st.markdown(html_metrics, unsafe_allow_html=True)
                    
                    if st.checkbox("📈 Show Heatmap Chart", key='show_heatmap'):
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=[f'Predicted {i}' for i in range(n_classes)],
                            y=[f'Actual {i}' for i in range(n_classes)],
                            colorscale='Blues',
                            text=[[str(int(val)) for val in row] for row in cm],
                            texttemplate="%{text}",
                            textfont={"size": 14},
                            showscale=True
                        ))
                        fig_cm.update_layout(title=f'{selected_model} - Confusion Matrix', height=400)
                        st.plotly_chart(fig_cm, use_container_width=True)
            
            # ==================== CLASSIFICATION REPORT ====================
            st.markdown("---")
            st.markdown("#### 📋 Classification Report")
            
            model_for_report = st.selectbox("Select model for report", 
                                           list(evaluation_results.keys()), key='report')
            if model_for_report in evaluation_results:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{model_for_report} - Detailed Report**")
                    st.code(evaluation_results[model_for_report]['classification_report'])
                with col2:
                    acc = evaluation_results[model_for_report]['accuracy']
                    f1 = evaluation_results[model_for_report]['f1_score']
                    
                    st.markdown("**Model Summary**")
                    st.metric("Accuracy", f"{acc:.4f}")
                    st.metric("F1 Score", f"{f1:.4f}")
                    
                    if acc >= 0.90: badge = "🌟 Excellent"
                    elif acc >= 0.80: badge = "👍 Good"
                    elif acc >= 0.70: badge = "📊 Fair"
                    else: badge = "⚠️ Needs Work"
                    st.markdown(f"**Performance: {badge}**")
            
            # ==================== BEST MODEL & RANKING ====================
            st.markdown("---")
            best_model_name = metrics_df['F1 Score'].idxmax()
            
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, #1f77b4, #2c3e50); 
                        padding: 20px; border-radius: 15px; text-align: center;">
                <h2 style="color: white; margin: 0;">🏆 Best Performing Model</h2>
                <h1 style="color: #ffd700; font-size: 2.5rem; margin: 10px 0;">{best_model_name}</h1>
                <p style="color: white;">Accuracy: {evaluation_results[best_model_name]['accuracy']:.3f} | 
                   F1 Score: {evaluation_results[best_model_name]['f1_score']:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 📊 Model Ranking")
            ranked = metrics_df.sort_values('F1 Score', ascending=False)
            for i, (name, row) in enumerate(ranked.iterrows(), 1):
                medal = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                bar_width = int(row['F1 Score'] * 100)
                st.markdown(f"""
                <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 8px;">
                    <span style="font-size: 1.5rem;">{medal}</span>
                    <span style="font-weight: bold; width: 180px;">{name}</span>
                    <div style="flex-grow: 1; background: #e9ecef; border-radius: 5px; height: 25px;">
                        <div style="width: {bar_width}%; background: linear-gradient(90deg, #1f77b4, #3498db); 
                                    height: 100%; border-radius: 5px; display: flex; align-items: center; padding-left: 10px;">
                            <span style="color: white; font-size: 0.8rem; font-weight: bold;">Acc: {row['Accuracy']:.3f} | F1: {row['F1 Score']:.3f}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        else:
            st.warning("⚠️ No evaluation results available.")
    
    # Section: Feature Importance
    elif section == "🎯 Feature Importance":
        st.markdown('<p class="section-header">🎯 Feature Importance Analysis</p>', 
                   unsafe_allow_html=True)
        
        if st.session_state.feature_importance is not None:
            importance_df = st.session_state.feature_importance
            
            st.markdown("### 🌲 Feature Importance from Random Forest")
            
            fig = px.bar(
                importance_df.nlargest(20, 'importance'),
                x='importance',
                y='feature',
                orientation='h',
                title='Top 20 Most Important Features',
                color='importance',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 Feature Importance Table")
            
            col1, col2 = st.columns([3, 1])
            with col1:
                display_df = importance_df.copy()
                display_df['importance'] = display_df['importance'].round(4)
                display_df['percentage'] = (display_df['importance'] / display_df['importance'].sum() * 100).round(2)
                display_df = display_df.sort_values('importance', ascending=False)
                display_df['Rank'] = range(1, len(display_df) + 1)
                display_df['Bar'] = display_df['importance'].apply(lambda x: '█' * int(x * 100))
                
                html_imp = '<div style="overflow-x: auto;"><table style="width:100%; border-collapse: collapse; font-size: 0.9rem;">'
                html_imp += '<tr style="background-color: #1f77b4; color: white;">'
                for col in ['Rank', 'Feature', 'Importance', 'Percentage', 'Visualization']:
                    html_imp += f'<th style="padding: 10px; text-align: left;">{col}</th>'
                html_imp += '</tr>'
                
                for i, (_, row) in enumerate(display_df.iterrows()):
                    bg = '#f8f9fa' if i % 2 == 0 else '#ffffff'
                    if i < 3: bg = '#d4edda'
                    
                    html_imp += f'<tr style="background-color: {bg};">'
                    html_imp += f'<td style="padding: 8px; font-weight: bold;">#{row["Rank"]}</td>'
                    html_imp += f'<td style="padding: 8px;">{row["feature"]}</td>'
                    html_imp += f'<td style="padding: 8px;">{row["importance"]:.4f}</td>'
                    html_imp += f'<td style="padding: 8px;">{row["percentage"]:.1f}%</td>'
                    html_imp += f'<td style="padding: 8px; font-family: monospace; color: #1f77b4;">{row["Bar"]}</td>'
                    html_imp += '</tr>'
                
                html_imp += '</table></div>'
                st.markdown(html_imp, unsafe_allow_html=True)
            
            with col2:
                st.markdown("**Top 5 Features**")
                top5 = importance_df.nlargest(5, 'importance')
                for idx, (_, row) in enumerate(top5.iterrows(), 1):
                    st.metric(f"#{idx} {row['feature'][:20]}", f"{row['importance']:.3f}")
            
            st.markdown("---")
            st.markdown("### 💡 Feature Importance Interpretation")
            top_3 = importance_df.nlargest(3, 'importance')
            for idx, (index, row) in enumerate(top_3.iterrows()):
                st.markdown(f"**{idx+1}. {row['feature']}** (Importance: {row['importance']:.4f})")
                st.progress(float(row['importance'] / importance_df['importance'].max()))
            
            csv = importance_df.to_csv(index=False)
            st.download_button("📥 Download Feature Importance (CSV)", csv, "feature_importance.csv", "text/csv")
        else:
            st.warning("⚠️ Please calculate feature importance in the Feature Engineering section first!")
    
    # Section: Polymer Analysis
    elif section == "🧬 Polymer Analysis":
        st.markdown('<p class="section-header">🧬 Polymer Type Analysis</p>', unsafe_allow_html=True)
        
        data_to_use = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        if data_to_use is None:
            st.warning("⚠️ Please load data first!")
            return
        
        df = data_to_use
        
        st.markdown("### 🧪 Polymer Type Distribution")
        
        if 'Polymer_Type' in df.columns:
            polymer_counts = df['Polymer_Type'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 📊 Bar Chart")
                fig_bar = px.bar(x=polymer_counts.index, y=polymer_counts.values,
                                title='Polymer Type Distribution',
                                color=polymer_counts.values, color_continuous_scale='Viridis')
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                st.markdown("#### 🥧 Pie Chart")
                fig_pie = px.pie(values=polymer_counts.values, names=polymer_counts.index,
                                title='Polymer Type Distribution')
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("---")
            st.markdown("### 📊 Polymer Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Polymer Types", len(polymer_counts))
            with col2:
                st.metric("Most Common", polymer_counts.index[0])
            with col3:
                st.metric("Most Common Count", polymer_counts.values[0])
            
            st.markdown("---")
            st.markdown("### 🔬 Polymer Type vs Risk Level")
            
            if 'Risk_Level' in df.columns:
                fig_cross = px.histogram(df, x='Polymer_Type', color='Risk_Level',
                                        title='Polymer Type Distribution by Risk Level', barmode='group')
                fig_cross.update_layout(height=500)
                st.plotly_chart(fig_cross, use_container_width=True)
        else:
            st.warning("⚠️ 'Polymer_Type' column not found in dataset")
            st.write("Available columns in dataset:")
            st.write(df.columns.tolist())


if __name__ == "__main__":
    main()
