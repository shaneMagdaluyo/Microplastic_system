# app.py - Microplastic Pollution Risk Prediction System (PERSISTENT DATA)
# Fixed: Data persists across navigation pages
# Researchers: Matthew Joseph Viernes & Shane Mark R. Magdaluyo
# ASSCAT - March to December 2025

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, zscore, f_oneway
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_auc_score, 
                             roc_curve, mean_squared_error, r2_score)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE

# For handling imbalanced data (optional)
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

# For saving models
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import os

# Page configuration
st.set_page_config(
    page_title="Microplastic Risk Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-high {
        background-color: #ff6b6b;
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .risk-medium {
        background-color: #ffd93d;
        padding: 0.5rem;
        border-radius: 5px;
        color: #333;
        text-align: center;
    }
    .risk-low {
        background-color: #6bcb77;
        padding: 0.5rem;
        border-radius: 5px;
        color: white;
        text-align: center;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton > button {
        background-color: #2a5298;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #1e3c72;
        color: white;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .info-message {
        background-color: #d1ecf1;
        color: #0c5460;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# INITIALIZE SESSION STATE WITH PERSISTENT STORAGE
# =============================================================================

# Initialize all session state variables for persistence
if 'risk_data' not in st.session_state:
    st.session_state.risk_data = None
if 'risk_data_original' not in st.session_state:
    st.session_state.risk_data_original = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'data_filename' not in st.session_state:
    st.session_state.data_filename = None
    
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'feature_encoders' not in st.session_state:
    st.session_state.feature_encoders = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def save_uploaded_data(df, filename=None):
    """Save uploaded data to session state"""
    st.session_state.risk_data = df.copy()
    st.session_state.risk_data_original = df.copy()
    st.session_state.data_loaded = True
    if filename:
        st.session_state.data_filename = filename
    st.session_state.data_source = "upload"

def load_sample_data():
    """Load sample data"""
    df = generate_sample_data(1000)
    st.session_state.risk_data = df.copy()
    st.session_state.risk_data_original = df.copy()
    st.session_state.data_loaded = True
    st.session_state.data_source = "sample"
    st.session_state.data_filename = "sample_data.csv"
    return df

def generate_sample_data(n_samples=1000):
    """Generate synthetic microplastic pollution data"""
    np.random.seed(42)
    
    locations = ['Coastal Area', 'River Delta', 'Urban Runoff', 'Industrial Zone', 
                 'Agricultural Area', 'Marine Reserve', 'Estuary', 'Beach', 
                 'Open Ocean', 'Harbor']
    species = ['Fish_A', 'Fish_B', 'Mollusk', 'Crustacean', 'Bird', 'Mammal']
    habitat = ['Marine', 'Freshwater', 'Estuary', 'Coastal']
    risk_types = ['Ecological Risk', 'Human Health Risk', 'Chemical Hazard', 
                  'Food Chain Contamination', 'Low Risk', 'Medium Risk', 'High Risk']
    
    data = {
        'Study_Location': np.random.choice(locations, n_samples),
        'Species_Name': np.random.choice(species, n_samples),
        'Habitat_Type': np.random.choice(habitat, n_samples),
        'MP_Presence': np.random.choice(['Yes', 'No'], n_samples, p=[0.85, 0.15]),
        'MP_Concentration': np.random.uniform(0.1, 500, n_samples),
        'Particle_Size_mm': np.random.uniform(0.01, 5.0, n_samples),
        'Water_Temperature_C': np.random.uniform(10, 35, n_samples),
        'pH_Level': np.random.uniform(6.0, 8.5, n_samples),
        'Dissolved_Oxygen_mgL': np.random.uniform(2, 12, n_samples),
        'Turbidity_NTU': np.random.uniform(1, 100, n_samples),
        'Population_Density': np.random.uniform(10, 10000, n_samples),
        'Industrial_Score': np.random.uniform(0, 1, n_samples),
        'Waste_Management_Score': np.random.uniform(0, 1, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate risk score based on features
    df['Risk_Score'] = (
        df['MP_Concentration'] / 500 * 30 +
        df['Industrial_Score'] * 25 +
        (1 - df['Waste_Management_Score']) * 20 +
        np.where(df['Particle_Size_mm'] < 0.5, 15, 0) +
        np.where(df['Population_Density'] > 5000, 10, 0)
    )
    df['Risk_Score'] = df['Risk_Score'].clip(0, 100)
    
    # Assign risk level
    df['Risk_Level'] = pd.cut(df['Risk_Score'], bins=[0, 33, 66, 100], 
                               labels=['Low', 'Medium', 'High'])
    
    # Assign dominant risk type
    df['Dominant_Risk_Type'] = np.random.choice(risk_types, n_samples, p=[0.2, 0.2, 0.15, 0.15, 0.1, 0.1, 0.1])
    
    return df

def check_data_loaded():
    """Check if data is loaded and show appropriate message"""
    if not st.session_state.data_loaded or st.session_state.risk_data is None:
        st.warning("⚠️ No data loaded. Please upload a file or load sample data in the 'Data Upload & Preprocessing' page.")
        return False
    return True

def preprocess_for_training(df, features, target):
    """Properly preprocess data for training"""
    df_processed = df.copy()
    
    # Store encoders for later use
    encoders = {}
    
    # Separate numeric and categorical features
    numeric_features = []
    categorical_features = []
    
    for feature in features:
        if feature in df_processed.columns:
            if df_processed[feature].dtype in ['int64', 'float64']:
                numeric_features.append(feature)
            else:
                categorical_features.append(feature)
    
    # Handle missing values for numeric features
    for feature in numeric_features:
        if df_processed[feature].isnull().any():
            df_processed[feature].fillna(df_processed[feature].median(), inplace=True)
    
    # Handle missing values for categorical features
    for feature in categorical_features:
        if df_processed[feature].isnull().any():
            df_processed[feature].fillna(df_processed[feature].mode()[0] if len(df_processed[feature].mode()) > 0 else 'Unknown', inplace=True)
    
    # Encode categorical features
    for feature in categorical_features:
        le = LabelEncoder()
        df_processed[feature + '_encoded'] = le.fit_transform(df_processed[feature].astype(str))
        encoders[feature] = le
        numeric_features.append(feature + '_encoded')
    
    # Handle target column
    if target in df_processed.columns:
        if df_processed[target].dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(df_processed[target].astype(str))
            return df_processed[numeric_features], y, encoders, target_encoder
        else:
            y = df_processed[target].values
            return df_processed[numeric_features], y, encoders, None
    
    return df_processed[numeric_features], None, encoders, None

def reset_model_state():
    """Reset all model-related session state variables"""
    st.session_state.model_trained = False
    st.session_state.models = None
    st.session_state.results = None
    st.session_state.scaler = None
    st.session_state.selected_features = []
    st.session_state.task_type = None
    st.session_state.target_encoder = None
    st.session_state.feature_encoders = {}

# Title and header
st.markdown("""
<div class="main-header">
    <h1>🌊 Microplastic Pollution Risk Prediction System</h1>
    <p>Data Mining-Based Predictive Risk Modeling for Environmental Microplastic Contamination</p>
    <p>Viernes, M.J. & Magdaluyo, S.M.R. | ASSCAT 2025</p>
</div>
""", unsafe_allow_html=True)

# Display current data status in sidebar
st.sidebar.title("📊 Navigation")

# Show data status
if st.session_state.data_loaded:
    st.sidebar.success(f"✅ Data Loaded: {st.session_state.data_filename if st.session_state.data_filename else 'Custom Data'}")
    if st.session_state.risk_data is not None:
        st.sidebar.info(f"📊 Shape: {st.session_state.risk_data.shape[0]} rows, {st.session_state.risk_data.shape[1]} cols")
else:
    st.sidebar.warning("⚠️ No Data Loaded")

st.sidebar.markdown("---")

page = st.sidebar.radio("Go to", [
    "🏠 Dashboard",
    "📁 Data Upload & Preprocessing",
    "🤖 Model Training",
    "📈 Risk Prediction",
    "🗺️ Risk Mapping & Visualization",
    "📊 Model Evaluation",
    "📄 Generate Report"
])

# Sidebar info
st.sidebar.markdown("---")
st.sidebar.info("""
**System Features:**
- Persistent Data Storage
- Data Preprocessing
- Multiple ML Models
- Risk Prediction
- Report Generation
""")

# Option to clear data
if st.session_state.data_loaded:
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Loaded Data", type="secondary"):
        reset_model_state()
        st.session_state.risk_data = None
        st.session_state.risk_data_original = None
        st.session_state.data_loaded = False
        st.session_state.data_filename = None
        st.session_state.data_source = None
        st.rerun()

# =============================================================================
# PAGE: DASHBOARD
# =============================================================================

if page == "🏠 Dashboard":
    st.header("🏠 Dashboard")
    
    if st.session_state.data_loaded and st.session_state.risk_data is not None:
        df = st.session_state.risk_data
        
        # Key metrics
        st.subheader("📊 Dataset Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Features", len(numeric_cols))
        with col4:
            cat_cols = df.select_dtypes(include=['object']).columns
            st.metric("Categorical Features", len(cat_cols))
        
        # Quick visualizations
        st.subheader("📈 Quick Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Risk level distribution if available
            if 'Risk_Level' in df.columns:
                risk_counts = df['Risk_Level'].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                             title="Risk Level Distribution",
                             color_discrete_map={'Low': '#6bcb77', 'Medium': '#ffd93d', 'High': '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)
            elif 'Dominant_Risk_Type' in df.columns:
                risk_counts = df['Dominant_Risk_Type'].value_counts().head(5)
                fig = px.bar(x=risk_counts.values, y=risk_counts.index, orientation='h',
                            title="Top 5 Risk Types")
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Risk_Score' in df.columns:
                fig = px.histogram(df, x='Risk_Score', nbins=30, 
                                  title="Risk Score Distribution",
                                  color_discrete_sequence=['#2a5298'])
                st.plotly_chart(fig, use_container_width=True)
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))
        
        # System status
        st.subheader("⚙️ System Status")
        
        status_cols = st.columns(3)
        with status_cols[0]:
            if st.session_state.model_trained:
                st.success("✅ Model Trained")
            else:
                st.info("⏳ Model Not Trained")
        
        with status_cols[1]:
            if st.session_state.selected_features:
                st.success(f"✅ {len(st.session_state.selected_features)} Features Selected")
            else:
                st.info("⏳ No Features Selected")
        
        with status_cols[2]:
            if st.session_state.task_type:
                st.success(f"✅ Task: {st.session_state.task_type}")
            else:
                st.info("⏳ Task Not Set")
    
    else:
        st.info("👋 Welcome to the Microplastic Pollution Risk Prediction System!")
        st.markdown("""
        ### Getting Started:
        
        1. Go to **📁 Data Upload & Preprocessing** to load your data
        2. Or click the button below to load sample data
        
        ### Features:
        - **Persistent Data**: Your uploaded data stays loaded across all pages
        - **Multiple ML Models**: Random Forest, Logistic Regression, Decision Tree, and more
        - **Risk Prediction**: Predict microplastic pollution risk levels
        - **Visualizations**: Interactive charts and graphs
        - **Report Generation**: Download comprehensive reports
        """)
        
        if st.button("🚀 Load Sample Data to Get Started", type="primary"):
            load_sample_data()
            st.rerun()

# =============================================================================
# PAGE: DATA UPLOAD & PREPROCESSING (WITH PERSISTENCE)
# =============================================================================

elif page == "📁 Data Upload & Preprocessing":
    st.header("📁 Data Upload and Preprocessing")
    
    # Show current data status
    if st.session_state.data_loaded:
        st.markdown(f"""
        <div class="success-message">
            ✅ <strong>Data Already Loaded</strong><br>
            File: {st.session_state.data_filename if st.session_state.data_filename else 'Custom Data'}<br>
            Shape: {st.session_state.risk_data.shape[0]} rows × {st.session_state.risk_data.shape[1]} columns<br>
            Source: {st.session_state.data_source}
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload New Dataset")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.write("**Data Preview:**")
                st.dataframe(df.head(5))
                
                if st.button("✅ Load This Dataset", type="primary"):
                    save_uploaded_data(df, uploaded_file.name)
                    reset_model_state()  # Reset models when new data is loaded
                    st.success(f"✅ Dataset '{uploaded_file.name}' loaded successfully!")
                    st.rerun()
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with col2:
        st.subheader("📊 Or Use Sample Data")
        st.write("Use the built-in sample dataset to test the system:")
        st.write("- 1,000 samples of microplastic pollution data")
        st.write("- Multiple features including location, species, water quality")
        st.write("- Includes risk scores and risk levels")
        
        if st.button("📊 Load Sample Dataset", type="secondary"):
            load_sample_data()
            reset_model_state()
            st.success("✅ Sample dataset loaded successfully!")
            st.rerun()
    
    # ========================================================================
    # DATA PREPROCESSING SECTION (Only show if data is loaded)
    # ========================================================================
    
    if st.session_state.data_loaded and st.session_state.risk_data is not None:
        st.markdown("---")
        st.subheader("🔧 Data Preprocessing")
        
        df = st.session_state.risk_data
        
        # Data information tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "📈 Statistics", "🔍 Missing Values", "🏷️ Data Types"])
        
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Rows", f"{len(df):,}")
            with col2:
                st.metric("Total Columns", len(df.columns))
            with col3:
                st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            
            st.write("**Data Preview (First 10 rows):**")
            st.dataframe(df.head(10))
        
        with tab2:
            # Show statistics for numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.write("**Numeric Columns Statistics:**")
                st.dataframe(df[numeric_cols].describe())
            else:
                st.info("No numeric columns found.")
        
        with tab3:
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
                'Missing Count': df.isnull().sum().values,
                'Missing Percentage': (df.isnull().sum().values / len(df)) * 100
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if len(missing_df) > 0:
                st.dataframe(missing_df)
            else:
                st.success("✅ No missing values found in the dataset!")
        
        with tab4:
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Unique Values': [df[col].nunique() for col in df.columns],
                'Sample Values': [str(df[col].dropna().iloc[0])[:50] if len(df[col].dropna()) > 0 else 'N/A' for col in df.columns]
            })
            st.dataframe(dtype_df)
        
        # Preprocessing options
        st.subheader("🛠️ Preprocessing Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Handle Missing Values"):
                df_clean = df.copy()
                for col in df_clean.columns:
                    if df_clean[col].dtype in ['float64', 'int64']:
                        df_clean[col].fillna(df_clean[col].median(), inplace=True)
                    else:
                        mode_val = df_clean[col].mode()
                        if len(mode_val) > 0:
                            df_clean[col].fillna(mode_val[0], inplace=True)
                        else:
                            df_clean[col].fillna('Unknown', inplace=True)
                st.session_state.risk_data = df_clean
                st.success("✅ Missing values handled successfully!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Remove Duplicates"):
                original_len = len(df)
                df_clean = df.drop_duplicates()
                st.session_state.risk_data = df_clean
                st.success(f"✅ Removed {original_len - len(df_clean)} duplicate rows!")
                st.rerun()
        
        with col3:
            if st.button("📊 Encode Categorical Variables"):
                df_encoded = df.copy()
                categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
                for col in categorical_cols:
                    le = LabelEncoder()
                    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                    st.session_state.feature_encoders[col] = le
                st.session_state.risk_data = df_encoded
                st.success(f"✅ Encoded {len(categorical_cols)} categorical variables!")
                st.rerun()
        
        # Show categorical columns info
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.markdown(f"""
            <div class="info-message">
                📝 <strong>Categorical Columns Found:</strong> {', '.join(categorical_cols)}<br>
                These columns will be automatically encoded when training models.
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("💡 No data loaded yet. Please upload a CSV/Excel file or load the sample dataset above.")

# =============================================================================
# PAGE: MODEL TRAINING (Rest of the pages remain similar but with persistence)
# =============================================================================

# [The rest of the pages (Model Training, Risk Prediction, etc.) 
#  remain the same as in the previous version, just make sure they use 
#  st.session_state.risk_data which now persists]

elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if not check_data_loaded():
        st.stop()
    
    df = st.session_state.risk_data
    
    st.subheader("Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_options = df.columns.tolist()
        st.session_state.target_column = st.selectbox("Select Target Column (Risk Level/Risk Score/Risk Type)", target_options)
        
        if df[st.session_state.target_column].dtype in ['int64', 'float64'] and df[st.session_state.target_column].nunique() > 10:
            task_type = st.radio("Task Type", ["Classification", "Regression"], index=1)
        else:
            task_type = st.radio("Task Type", ["Classification", "Regression"], index=0)
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
        cv_folds = st.slider("K-Fold Cross Validation Folds", 5, 10, 5)
    
    st.subheader("Select Features for Training")
    
    feature_cols = [col for col in df.columns if col != st.session_state.target_column]
    
    st.info(f"📊 {len(feature_cols)} features available. Categorical features will be automatically encoded.")
    
    selected_features = st.multiselect("Select Features", feature_cols, default=feature_cols[:5] if len(feature_cols) > 5 else feature_cols)
    
    if st.button("🚀 Train Models", type="primary"):
        if len(selected_features) == 0:
            st.error("Please select at least one feature for training.")
        else:
            with st.spinner("Processing data and training models..."):
                try:
                    X_processed, y, encoders, target_encoder = preprocess_for_training(df, selected_features, st.session_state.target_column)
                    
                    X_processed = X_processed.fillna(X_processed.median())
                    
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_processed)
                    
                    if y is None:
                        st.error("Could not process target column. Please check your data.")
                        st.stop()
                    
                    if task_type == "Classification":
                        unique_classes = np.unique(y)
                        if len(unique_classes) < 2:
                            st.error(f"Target column '{st.session_state.target_column}' has only one unique value.")
                            st.stop()
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42, 
                        stratify=y if task_type == "Classification" else None
                    )
                    
                    if task_type == "Classification":
                        models = {
                            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
                            'Decision Tree': DecisionTreeClassifier(random_state=42),
                            'Gradient Boosting': GradientBoostingClassifier(random_state=42),
                            'SVM': SVC(random_state=42, probability=True),
                            'KNN': KNeighborsClassifier()
                        }
                    else:
                        models = {
                            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                            'Decision Tree': DecisionTreeRegressor(random_state=42),
                        }
                    
                    results = {}
                    trained_models = {}
                    
                    progress_bar = st.progress(0)
                    for idx, (name, model) in enumerate(models.items()):
                        st.write(f"Training {name}...")
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        trained_models[name] = model
                        
                        kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                        
                        if task_type == "Classification":
                            cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy')
                            results[name] = {
                                'Accuracy': accuracy_score(y_test, y_pred),
                                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                                'CV Mean': cv_scores.mean(),
                                'CV Std': cv_scores.std()
                            }
                        else:
                            cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='r2')
                            results[name] = {
                                'R2 Score': r2_score(y_test, y_pred),
                                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                                'MSE': mean_squared_error(y_test, y_pred),
                                'CV Mean': cv_scores.mean(),
                                'CV Std': cv_scores.std()
                            }
                        
                        progress_bar.progress((idx + 1) / len(models))
                    
                    st.session_state.models = trained_models
                    st.session_state.results = results
                    st.session_state.scaler = scaler
                    st.session_state.selected_features = selected_features
                    st.session_state.task_type = task_type
                    st.session_state.model_trained = True
                    st.session_state.target_encoder = target_encoder
                    
                    st.success("✅ Models trained successfully!")
                    
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df.style.highlight_max(axis=0))
                    
                    best_model = max(results, key=lambda x: results[x][list(results[x].keys())[0]])
                    st.success(f"🏆 Best Model: **{best_model}**")
                    
                except Exception as e:
                    st.error(f"Error during training: {str(e)}")

# =============================================================================
# PAGE: RISK PREDICTION
# =============================================================================

elif page == "📈 Risk Prediction":
    st.header("📈 Risk Prediction")
    
    if not check_data_loaded():
        st.stop()
    
    if st.session_state.model_trained and st.session_state.models:
        st.subheader("Make New Predictions")
        
        st.info("Enter values for each feature to predict the risk level/score.")
        
        input_data = {}
        cols = st.columns(2)
        
        for idx, feature in enumerate(st.session_state.selected_features):
            with cols[idx % 2]:
                if feature in st.session_state.risk_data.columns:
                    if st.session_state.risk_data[feature].dtype == 'object':
                        unique_vals = st.session_state.risk_data[feature].dropna().unique().tolist()
                        input_data[feature] = st.selectbox(f"{feature}", unique_vals)
                    else:
                        min_val = float(st.session_state.risk_data[feature].min())
                        max_val = float(st.session_state.risk_data[feature].max())
                        input_data[feature] = st.number_input(f"{feature}", value=float(min_val), min_value=min_val, max_value=max_val)
                else:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        if st.button("🔮 Predict Risk", type="primary"):
            input_df = pd.DataFrame([input_data])
            
            try:
                for feature, encoder in st.session_state.feature_encoders.items():
                    if feature in input_df.columns:
                        if input_df[feature].iloc[0] in encoder.classes_:
                            input_df[feature + '_encoded'] = encoder.transform([input_df[feature].iloc[0]])[0]
                
                X_input = input_df[st.session_state.selected_features].copy()
                for col in X_input.columns:
                    if X_input[col].dtype == 'object':
                        X_input[col] = pd.Categorical(X_input[col]).codes
                
                X_input = X_input.fillna(0)
                input_scaled = st.session_state.scaler.transform(X_input)
                
                st.subheader("Prediction Results")
                
                if st.session_state.task_type == "Classification":
                    results_cols = st.columns(len(st.session_state.models))
                    for idx, (name, model) in enumerate(st.session_state.models.items()):
                        prediction = model.predict(input_scaled)[0]
                        if st.session_state.target_encoder:
                            pred_label = st.session_state.target_encoder.inverse_transform([prediction])[0]
                        else:
                            pred_label = str(prediction)
                        
                        color_class = "low"
                        if "High" in str(pred_label):
                            color_class = "high"
                        elif "Medium" in str(pred_label):
                            color_class = "medium"
                        
                        with results_cols[idx]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{name}</h3>
                                <div class="risk-{color_class}">
                                    <h2>{pred_label}</h2>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                else:
                    results_cols = st.columns(len(st.session_state.models))
                    for idx, (name, model) in enumerate(st.session_state.models.items()):
                        prediction = model.predict(input_scaled)[0]
                        with results_cols[idx]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{name}</h3>
                                <h2>{prediction:.2f}</h2>
                                <p>Risk Score</p>
                            </div>
                            """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
    else:
        st.warning("⚠️ Please train models first in the Model Training section.")

# =============================================================================
# PAGE: RISK MAPPING & VISUALIZATION
# =============================================================================

elif page == "🗺️ Risk Mapping & Visualization":
    st.header("🗺️ Risk Mapping & Visualization")
    
    if not check_data_loaded():
        st.stop()
    
    df = st.session_state.risk_data
    
    st.subheader("Interactive Risk Visualizations")
    
    viz_tabs = st.tabs(["📊 Distribution", "📈 Relationships", "🔬 Correlations", "🎯 Risk Analysis"])
    
    with viz_tabs[0]:
        col1, col2 = st.columns(2)
        with col1:
            risk_cols = [col for col in df.columns if 'risk' in col.lower() or 'Risk' in col]
            if risk_cols:
                for risk_col in risk_cols[:2]:
                    if df[risk_col].dtype == 'object':
                        fig = px.pie(df, names=risk_col, title=f"{risk_col} Distribution")
                        st.plotly_chart(fig, use_container_width=True)
        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select column to visualize", numeric_cols)
                fig = px.box(df, y=selected_col, title=f"{selected_col} Box Plot")
                st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[1]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
            y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
            fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}",
                           opacity=0.6, trendline="ols")
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[2]:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                           title="Feature Correlation Matrix",
                           color_continuous_scale='RdBu')
            st.plotly_chart(fig, use_container_width=True)
    
    with viz_tabs[3]:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols and 'Risk_Score' in df.columns:
            cat_col = st.selectbox("Group by", categorical_cols)
            risk_by_cat = df.groupby(cat_col)['Risk_Score'].agg(['mean', 'std', 'count']).reset_index()
            risk_by_cat = risk_by_cat.sort_values('mean', ascending=False).head(10)
            fig = px.bar(risk_by_cat, x=cat_col, y='mean', error_y='std',
                        title=f"Average Risk Score by {cat_col}")
            st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# PAGE: MODEL EVALUATION
# =============================================================================

elif page == "📊 Model Evaluation":
    st.header("📊 Model Evaluation")
    
    if not check_data_loaded():
        st.stop()
    
    if st.session_state.model_trained and st.session_state.results:
        st.subheader("Model Performance Metrics")
        
        results_df = pd.DataFrame(st.session_state.results).T
        st.dataframe(results_df.style.highlight_max(axis=0))
        
        st.subheader("Performance Visualization")
        
        if st.session_state.task_type == "Classification":
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            fig = go.Figure()
            for metric in metrics:
                fig.add_trace(go.Bar(name=metric, x=results_df.index, y=results_df[metric]))
            fig.update_layout(title="Model Performance Comparison", barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("Cross-Validation Scores")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="CV Mean", x=results_df.index, y=results_df['CV Mean'],
                                 error_y=dict(type='data', array=results_df['CV Std'])))
            fig2.update_layout(title="5-Fold Cross Validation Results")
            st.plotly_chart(fig2, use_container_width=True)
        
        best_model = max(st.session_state.results, key=lambda x: st.session_state.results[x][list(st.session_state.results[x].keys())[0]])
        st.metric("🏆 Best Model", best_model)
    else:
        st.warning("⚠️ Please train models first in the Model Training section.")

# =============================================================================
# PAGE: GENERATE REPORT
# =============================================================================

elif page == "📄 Generate Report":
    st.header("📄 Generate Environmental Risk Report")
    
    if not check_data_loaded():
        st.stop()
    
    df = st.session_state.risk_data
    
    st.subheader("Report Configuration")
    
    report_title = st.text_input("Report Title", "Microplastic Pollution Risk Assessment Report")
    author_name = st.text_input("Author/Organization", "Viernes, M.J. & Magdaluyo, S.M.R.")
    
    if st.button("📄 Generate Report", type="primary"):
        report = f"""
{'='*80}
{report_title.upper()}
{author_name}
ASSCAT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

EXECUTIVE SUMMARY
{'-'*40}
- Total Records: {len(df):,}
- Total Features: {len(df.columns)}
- Data Shape: {df.shape}

DATA OVERVIEW
{'-'*40}
{df.describe().to_string() if len(df.select_dtypes(include=[np.number]).columns) > 0 else 'No numeric columns'}

MISSING VALUES
{'-'*40}
{df.isnull().sum().to_string()}

"""
        if st.session_state.model_trained and st.session_state.results:
            report += f"""
MODEL PERFORMANCE
{'-'*40}
{pd.DataFrame(st.session_state.results).T.to_string()}
"""
        
        report += f"\n{'='*80}\nEnd of Report\n{'='*80}"
        
        st.text_area("Generated Report", report, height=400)
        
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"microplastic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌊 Microplastic Pollution Risk Prediction System | ASSCAT 2025 | Viernes, M.J. & Magdaluyo, S.M.R.</p>
</div>
""", unsafe_allow_html=True)
