

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis, shapiro, pearsonr
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder, PowerTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_auc_score, 
                             roc_curve, mean_squared_error, r2_score, mean_absolute_error,
                             silhouette_score, calinski_harabasz_score)
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE, SelectFromModel
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# For handling imbalanced data
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    from imblearn.combine import SMOTEENN
    IMBALANCE_AVAILABLE = True
except ImportError:
    IMBALANCE_AVAILABLE = False

# For saving models
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import base64
import json
import time
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="MP-RAS | Microplastic Risk Modeling",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS WITH SYSTEM BRANDING
# ============================================================================

st.markdown("""
<style>
    /* Main header with system name */
    .main-header {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
        animation: shimmer 3s infinite;
    }
    @keyframes shimmer {
        100% { left: 100%; }
    }
    .main-header h1 {
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
        letter-spacing: 2px;
    }
    .main-header .subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
    }
    .main-header .version {
        font-size: 0.9rem;
        opacity: 0.7;
        font-family: monospace;
    }
    .main-header .researchers {
        font-size: 0.9rem;
        opacity: 0.8;
        margin-top: 0.5rem;
        border-top: 1px solid rgba(255,255,255,0.3);
        display: inline-block;
        padding-top: 0.5rem;
    }
    
    /* System badge */
    .system-badge {
        position: fixed;
        bottom: 20px;
        right: 20px;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-size: 0.8rem;
        font-weight: bold;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
        cursor: pointer;
        transition: transform 0.3s;
    }
    .system-badge:hover {
        transform: scale(1.05);
    }
    
    /* Sidebar header */
    .sidebar-header {
        text-align: center;
        padding: 1rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #667eea;
    }
    .sidebar-header h3 {
        color: #667eea;
        margin: 0;
    }
    .sidebar-header small {
        color: #666;
    }
    
    /* Risk level cards */
    .risk-critical {
        background: linear-gradient(135deg, #ff6b6b, #ee5a52);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .risk-high {
        background: linear-gradient(135deg, #ff9a44, #ff6b6b);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffd93d, #ffc107);
        padding: 1rem;
        border-radius: 10px;
        color: #333;
        text-align: center;
    }
    .risk-low {
        background: linear-gradient(135deg, #6bcb77, #4caf50);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .risk-verylow {
        background: linear-gradient(135deg, #4caf50, #45a049);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        transition: transform 0.3s;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
    }
    .metric-card:hover {
        transform: translateY(-5px);
    }
    .metric-card h3 {
        font-size: 2rem;
        margin: 0;
    }
    .metric-card p {
        margin: 0;
        opacity: 0.9;
    }
    
    /* Welcome card */
    .welcome-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 1rem 0;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .success-box {
        background: linear-gradient(135deg, #00b4db15 0%, #0083b015 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #00b4db;
    }
    .warning-box {
        background: linear-gradient(135deg, #ff9a4415 0%, #ff6b6b15 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #ff6b6b;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1rem;
        font-weight: 600;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
    }
    
    /* System name in sidebar */
    .system-name-sidebar {
        text-align: center;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea20, #764ba220);
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .system-name-sidebar h2 {
        color: #667eea;
        margin: 0;
        font-size: 1.2rem;
    }
    .system-name-sidebar p {
        margin: 0;
        font-size: 0.8rem;
        color: #666;
    }
    
    /* Animation for loading */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    .loading {
        animation: pulse 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# SYSTEM HEADER WITH BRANDING
# ============================================================================

st.markdown("""
<div class="main-header">
    <h1>🌊 MP-risk modeling</h1>
    <div class="subtitle">Microplastic Predictive Risk Modeling</div>
    <div class="version">Version 2.0 | Data Mining-Based Predictive Risk Modeling</div>
    <div class="researchers">Viernes, M.J. & Magdaluyo, S.M.R. | ASSCAT 2025</div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'risk_data': None,
        'data_loaded': False,
        'data_filename': None,
        'data_source': None,
        'data_stats': {},
        
        'model_trained': False,
        'trained_models': {},
        'model_results': {},
        'best_model': None,
        'best_model_name': None,
        
        'scaler': None,
        'feature_encoders': {},
        'target_encoder': None,
        'selected_features': [],
        'task_type': None,
        'target_column': None,
        
        'training_history': [],
        'predictions_history': [],
        
        'experiment_results': {},
        
        'eda_results': {},
        'correlation_matrix': None,
        
        'pca_components': None,
        'cluster_labels': None,
    }
    
    for key, default in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default

init_session_state()

# ============================================================================
# SIDEBAR WITH SYSTEM BRANDING
# ============================================================================

with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <div class="system-name-sidebar">
            <h2>🌊 MP-RAS</h2>
            <p>Microplastic Risk<br>Assessment System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Navigation
    st.markdown("### 📍 Navigation")
    page = st.radio("", [
        "🏠 Dashboard",
        "📁 Data Upload", 
        "📊 EDA",
        "🤖 Model Training",
        "📈 Prediction",
        "📊 Results"
    ])
    
    st.markdown("---")
    
    # System info in sidebar
    st.markdown("""
    <div class="info-box">
        <small>🔬 System Features:</small><br>
        <small>• Multiple ML Models</small><br>
        <small>• Cross-Validation</small><br>
        <small>• Hyperparameter Tuning</small><br>
        <small>• Real-time Prediction</small><br>
        <small>• Report Generation</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Data status
    if st.session_state.data_loaded and st.session_state.risk_data is not None:
        st.markdown(f"""
        <div class="success-box">
            ✅ <strong>Data Ready</strong><br>
            <small>{st.session_state.risk_data.shape[0]} rows<br>
            {st.session_state.risk_data.shape[1]} columns</small>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="warning-box">
            ⚠️ <strong>No Data Loaded</strong><br>
            <small>Please upload data</small>
        </div>
        """, unsafe_allow_html=True)
    
    if st.session_state.model_trained:
        st.markdown(f"""
        <div class="success-box">
            🎯 <strong>Model Ready</strong><br>
            <small>Best: {st.session_state.best_model_name}</small>
        </div>
        """, unsafe_allow_html=True)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def generate_enhanced_sample_data(n_samples=2000):
    """Generate enhanced synthetic microplastic pollution data"""
    np.random.seed(42)
    
    locations = ['Coastal Area', 'River Delta', 'Urban Runoff', 'Industrial Zone', 
                 'Agricultural Area', 'Marine Reserve', 'Estuary', 'Beach', 
                 'Open Ocean', 'Harbor', 'Mangrove Forest', 'Coral Reef']
    
    species = ['Mugil cephalus', 'Oreochromis niloticus', 'Mytilus edulis', 
               'Penaeus monodon', 'Clarias gariepinus', 'Lates calcarifer',
               'Scomber japonicus', 'Sardinella spp.', 'Decapterus spp.']
    
    habitat = ['Marine', 'Freshwater', 'Estuary', 'Coastal', 'Benthic', 'Pelagic']
    
    polymer_types = ['Polyethylene (PE)', 'Polypropylene (PP)', 'Polystyrene (PS)', 
                     'Polyethylene Terephthalate (PET)', 'Polyvinyl Chloride (PVC)',
                     'Polyamide (PA)', 'Polyester', 'Acrylic']
    
    shapes = ['Fiber', 'Fragment', 'Sphere', 'Film', 'Foam', 'Microbead']
    
    colors = ['Transparent', 'White', 'Blue', 'Green', 'Red', 'Yellow', 'Black']
    
    data = {
        'Study_Location': np.random.choice(locations, n_samples),
        'Species_Name': np.random.choice(species, n_samples),
        'Habitat_Type': np.random.choice(habitat, n_samples),
        'Polymer_Type': np.random.choice(polymer_types, n_samples),
        'Particle_Shape': np.random.choice(shapes, n_samples),
        'Particle_Color': np.random.choice(colors, n_samples),
        'MP_Presence': np.random.choice(['Yes', 'No'], n_samples, p=[0.85, 0.15]),
        'MP_Concentration': np.random.gamma(2, 50, n_samples).astype(int),
        'Particle_Size_mm': np.random.uniform(0.01, 5.0, n_samples),
        'Water_Temperature_C': np.random.normal(25, 5, n_samples),
        'pH_Level': np.random.normal(7.5, 0.8, n_samples),
        'Dissolved_Oxygen_mgL': np.random.normal(8, 2, n_samples),
        'Turbidity_NTU': np.random.gamma(2, 10, n_samples),
        'Salinity_psu': np.random.gamma(2, 15, n_samples),
        'Nitrate_mgL': np.random.gamma(2, 2, n_samples),
        'Phosphate_mgL': np.random.gamma(1, 0.5, n_samples),
        'Population_Density_km2': np.random.gamma(2, 2000, n_samples),
        'Industrial_Score': np.random.uniform(0, 1, n_samples),
        'Waste_Management_Score': np.random.uniform(0, 1, n_samples),
        'Distance_to_Coast_km': np.random.gamma(2, 30, n_samples),
        'Water_Depth_m': np.random.gamma(2, 50, n_samples),
        'Sampling_Season': np.random.choice(['Dry', 'Wet', 'Transition'], n_samples),
        'Year': np.random.choice([2020, 2021, 2022, 2023, 2024], n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate comprehensive risk score
    df['Risk_Score'] = (
        (df['MP_Concentration'] / df['MP_Concentration'].max()) * 35 +
        df['Industrial_Score'] * 25 +
        (1 - df['Waste_Management_Score']) * 20 +
        np.where(df['Particle_Size_mm'] < 0.5, 10, 0) +
        np.where(df['Particle_Shape'] == 'Fiber', 5, 0) +
        np.where(df['Population_Density_km2'] > df['Population_Density_km2'].median(), 5, 0) +
        np.where(df['Nitrate_mgL'] > df['Nitrate_mgL'].median(), 5, 0)
    )
    df['Risk_Score'] = df['Risk_Score'].clip(0, 100)
    
    # Assign risk level
    df['Risk_Level'] = pd.cut(df['Risk_Score'], bins=[0, 20, 40, 60, 80, 101], 
                               labels=['Very Low', 'Low', 'Medium', 'High', 'Critical'])
    
    # Assign risk category
    df['Risk_Category'] = pd.cut(df['Risk_Score'], bins=[0, 33, 66, 100], 
                                  labels=['Low Risk', 'Medium Risk', 'High Risk'])
    
    # Assign risk type based on pattern
    def assign_risk_type(row):
        if row['Industrial_Score'] > 0.7:
            return 'Industrial Contamination'
        elif row['Population_Density_km2'] > 5000:
            return 'Urban Runoff'
        elif row['Particle_Size_mm'] < 0.1:
            return 'Nanoplastic Risk'
        elif row['Polymer_Type'] in ['Polyethylene (PE)', 'Polypropylene (PP)']:
            return 'Food Web Contamination'
        elif row['Habitat_Type'] in ['Estuary', 'Coastal']:
            return 'Ecosystem Impact'
        else:
            return 'General Environmental Risk'
    
    df['Risk_Type'] = df.apply(assign_risk_type, axis=1)
    
    # Additional derived features
    df['MP_Concentration_Log'] = np.log1p(df['MP_Concentration'])
    df['Size_Category'] = pd.cut(df['Particle_Size_mm'], bins=[0, 0.1, 0.5, 1, 5], 
                                  labels=['Nano (<0.1)', 'Small (0.1-0.5)', 'Medium (0.5-1)', 'Large (1-5)'])
    
    return df

def advanced_preprocessing(df, features, target):
    """Advanced preprocessing with multiple strategies"""
    df_proc = df.copy()
    encoders = {}
    
    # Separate features
    numeric_features = []
    categorical_features = []
    
    for feat in features:
        if feat in df_proc.columns:
            if df_proc[feat].dtype in ['int64', 'float64']:
                numeric_features.append(feat)
            else:
                categorical_features.append(feat)
    
    # Handle missing values
    for feat in numeric_features:
        if df_proc[feat].isnull().any():
            df_proc[feat].fillna(df_proc[feat].median(), inplace=True)
    
    for feat in categorical_features:
        if df_proc[feat].isnull().any():
            mode_val = df_proc[feat].mode()
            if len(mode_val) > 0:
                df_proc[feat].fillna(mode_val[0], inplace=True)
            else:
                df_proc[feat].fillna('Unknown', inplace=True)
    
    # Encode categorical (multiple methods)
    for feat in categorical_features:
        le = LabelEncoder()
        df_proc[feat + '_enc'] = le.fit_transform(df_proc[feat].astype(str))
        encoders[feat] = le
        numeric_features.append(feat + '_enc')
    
    X = df_proc[numeric_features]
    
    # Remove infinite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Power transform for skewed features
    skewness = X.skew()
    skewed = skewness[abs(skewness) > 1].index.tolist()
    for col in skewed[:10]:  # Limit to first 10
        if (X[col] >= 0).all():
            X[col] = np.log1p(X[col])
        else:
            pt = PowerTransformer(method='yeo-johnson')
            X[col] = pt.fit_transform(X[[col]]).flatten()
    
    # Target processing
    if df_proc[target].dtype == 'object':
        target_enc = LabelEncoder()
        y = target_enc.fit_transform(df_proc[target].astype(str))
        return X, y, encoders, target_enc
    else:
        y = df_proc[target].values
        return X, y, encoders, None

def get_model_with_params(model_name, task_type, random_state=42):
    """Get model with optimized parameters"""
    models = {
        'Random Forest': {
            'Classification': RandomForestClassifier(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=random_state, n_jobs=-1
            ),
            'Regression': RandomForestRegressor(
                n_estimators=200, max_depth=20, min_samples_split=5,
                min_samples_leaf=2, random_state=random_state, n_jobs=-1
            )
        },
        'Gradient Boosting': {
            'Classification': GradientBoostingClassifier(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                random_state=random_state
            ),
            'Regression': GradientBoostingRegressor(
                n_estimators=150, learning_rate=0.1, max_depth=5,
                random_state=random_state
            )
        },
        'Logistic Regression': {
            'Classification': LogisticRegression(
                C=1.0, max_iter=1000, random_state=random_state,
                class_weight='balanced'
            ),
            'Regression': None
        },
        'Decision Tree': {
            'Classification': DecisionTreeClassifier(
                max_depth=10, min_samples_split=5, random_state=random_state
            ),
            'Regression': DecisionTreeRegressor(
                max_depth=10, min_samples_split=5, random_state=random_state
            )
        },
        'SVM': {
            'Classification': SVC(
                C=1.0, kernel='rbf', probability=True, random_state=random_state
            ),
            'Regression': SVR(kernel='rbf', C=1.0)
        },
        'KNN': {
            'Classification': KNeighborsClassifier(n_neighbors=5, weights='distance'),
            'Regression': KNeighborsRegressor(n_neighbors=5, weights='distance')
        },
        'AdaBoost': {
            'Classification': AdaBoostClassifier(n_estimators=100, random_state=random_state),
            'Regression': None
        },
        'Naive Bayes': {
            'Classification': GaussianNB(),
            'Regression': None
        }
    }
    
    model_info = models.get(model_name, {})
    return model_info.get(task_type, None)

# ============================================================================
# PAGE FUNCTIONS
# ============================================================================

def page_home():
    st.header("🏠 Dashboard")
    
    if st.session_state.data_loaded and st.session_state.risk_data is not None:
        df = st.session_state.risk_data
        
        # Welcome message with system name
        st.markdown(f"""
        <div class="success-box">
            <strong>🎉 Welcome to MP-RAS (Microplastic Risk Assessment System)</strong><br>
            System Ready | Data Loaded: {len(df):,} records | {len(df.columns)} features
        </div>
        """, unsafe_allow_html=True)
        
        # Key metrics in styled cards
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <p>📊 Total Records</p>
                <h3>{len(df):,}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <p>📋 Features</p>
                <h3>{len(df.columns)}</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            missing_pct = (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100
            st.markdown(f"""
            <div class="metric-card">
                <p>🔍 Data Quality</p>
                <h3>{100 - missing_pct:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            if st.session_state.model_trained:
                st.markdown(f"""
                <div class="metric-card">
                    <p>🎯 Model Status</p>
                    <h3>{st.session_state.best_model_name[:15]}...</h3>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <p>🎯 Model Status</p>
                    <h3>Not Trained</h3>
                </div>
                """, unsafe_allow_html=True)
        
        # Risk distribution
        st.subheader("🎯 Risk Distribution")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Risk_Level' in df.columns:
                risk_counts = df['Risk_Level'].value_counts()
                colors = {'Critical': '#ff4444', 'High': '#ff6b6b', 'Medium': '#ffd93d', 
                         'Low': '#6bcb77', 'Very Low': '#4caf50'}
                fig = px.pie(
                    values=risk_counts.values, 
                    names=risk_counts.index,
                    title="Risk Level Distribution",
                    color=risk_counts.index,
                    color_discrete_map=colors
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Risk_Score' in df.columns:
                fig = px.histogram(
                    df, x='Risk_Score', nbins=40,
                    title="Risk Score Distribution",
                    color_discrete_sequence=['#667eea'],
                    marginal='box'
                )
                fig.add_vline(x=df['Risk_Score'].mean(), line_dash="dash", 
                             line_color="red", annotation_text=f"Mean: {df['Risk_Score'].mean():.1f}")
                st.plotly_chart(fig, use_container_width=True)
        
        # Data preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))
        
    else:
        # Welcome screen with system branding
        st.markdown("""
        <div class="welcome-card">
            <h2>🌊 Welcome to MP-RAS</h2>
            <h3>Microplastic Risk Assessment System</h3>
            <p>An intelligent system for predicting microplastic pollution risks using advanced data mining techniques</p>
            <hr>
            <p style="color: #667eea;">Built by: Viernes, M.J. & Magdaluyo, S.M.R. | ASSCAT 2025</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✨ System Features
            
            - **Data Upload & Preprocessing**
              - Support for CSV/Excel files
              - Automatic data cleaning
              - Feature encoding and scaling
            
            - **Exploratory Data Analysis**
              - Interactive visualizations
              - Statistical summaries
              - Correlation analysis
            
            - **Multiple ML Models**
              - Random Forest
              - Gradient Boosting
              - Logistic Regression
              - SVM, KNN, and more
            """)
        
        with col2:
            st.markdown("""
            ### 🚀 Getting Started
            
            1. Go to **📁 Data Upload** to load your data
            2. Explore your data in **📊 EDA**
            3. Train models in **🤖 Model Training**
            4. Make predictions in **📈 Prediction**
            
            ### 📊 Sample Data
            
            Click the button below to load sample microplastic data.
            """)
            
            if st.button("🚀 Load Sample Data", use_container_width=True):
                df = generate_enhanced_sample_data(2000)
                st.session_state.risk_data = df
                st.session_state.data_loaded = True
                st.session_state.data_source = "sample"
                st.session_state.data_filename = "microplastic_sample_data.csv"
                st.success("✅ Sample data loaded successfully!")
                st.rerun()
        
        # System capabilities
        st.markdown("---")
        st.subheader("📈 MP-RAS Capabilities")
        
        cap_cols = st.columns(4)
        capabilities = [
            ("🎯", "Multiple ML Models", "6+ classification & regression models"),
            ("📊", "Interactive Viz", "Real-time charts and graphs"),
            ("🔄", "Cross-Validation", "K-Fold CV for robust evaluation"),
            ("📄", "Report Generation", "Export analysis reports")
        ]
        
        for idx, (icon, title, desc) in enumerate(capabilities):
            with cap_cols[idx]:
                st.markdown(f"""
                <div class="info-box" style="text-align: center;">
                    <h2>{icon}</h2>
                    <h4>{title}</h4>
                    <small>{desc}</small>
                </div>
                """, unsafe_allow_html=True)

# ============================================================================
# DATA UPLOAD PAGE
# ============================================================================

def page_data_upload():
    st.header("📁 Data Upload & Preprocessing")
    
    # Show current data status
    if st.session_state.data_loaded:
        st.markdown(f"""
        <div class="success-box">
            ✅ <strong>Data Loaded Successfully</strong><br>
            File: {st.session_state.data_filename}<br>
            Shape: {st.session_state.risk_data.shape[0]} rows × {st.session_state.risk_data.shape[1]} columns
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Your Data")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.write("**Data Preview:**")
                st.dataframe(df.head(5))
                st.write(f"**Shape:** {df.shape}")
                
                if st.button("✅ Load Dataset", type="primary"):
                    st.session_state.risk_data = df
                    st.session_state.data_loaded = True
                    st.session_state.data_filename = uploaded_file.name
                    st.session_state.data_source = "upload"
                    st.session_state.model_trained = False
                    st.success(f"✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with col2:
        st.subheader("📊 Sample Data")
        st.write("Use MP-RAS built-in sample dataset:")
        st.write("- 2,000 samples of microplastic data")
        st.write("- 25+ environmental and biological features")
        st.write("- Pre-calculated risk scores and levels")
        
        if st.button("📊 Load MP-RAS Sample Data", use_container_width=True):
            df = generate_enhanced_sample_data(2000)
            st.session_state.risk_data = df
            st.session_state.data_loaded = True
            st.session_state.data_source = "sample"
            st.session_state.data_filename = "MP-RAS_sample_data.csv"
            st.session_state.model_trained = False
            st.success("✅ Sample data loaded successfully!")
            st.rerun()
    
    # Data preprocessing section
    if st.session_state.data_loaded and st.session_state.risk_data is not None:
        st.markdown("---")
        st.subheader("🛠️ Data Preprocessing")
        
        df = st.session_state.risk_data
        
        # Data quality metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", f"{len(df):,}")
        with col2:
            st.metric("Total Columns", len(df.columns))
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("Missing Values", missing)
        with col4:
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)
        
        # Data info tabs
        tab1, tab2, tab3 = st.tabs(["📊 Data Overview", "📈 Statistics", "🔍 Missing Values"])
        
        with tab1:
            st.dataframe(df.head(10))
        
        with tab2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe())
        
        with tab3:
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing %': (df.isnull().sum().values / len(df) * 100).round(2)
            })
            missing_df = missing_df[missing_df['Missing Count'] > 0]
            if len(missing_df) > 0:
                st.dataframe(missing_df)
                if st.button("🧹 Fill Missing Values"):
                    for col in df.columns:
                        if df[col].dtype in ['int64', 'float64']:
                            df[col].fillna(df[col].median(), inplace=True)
                        else:
                            df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
                    st.session_state.risk_data = df
                    st.success("Missing values filled!")
                    st.rerun()
            else:
                st.success("✅ No missing values found!")

# ============================================================================
# EDA PAGE
# ============================================================================

def page_eda():
    st.header("📊 Exploratory Data Analysis")
    
    if not st.session_state.data_loaded or st.session_state.risk_data is None:
        st.warning("⚠️ Please load data first in the Data Upload page.")
        return
    
    df = st.session_state.risk_data
    
    tab1, tab2, tab3 = st.tabs(["📈 Distribution Analysis", "🔗 Correlation Analysis", "🎯 Risk Analysis"])
    
    with tab1:
        st.subheader("Distribution Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        selected_col = st.selectbox("Select column to analyze", numeric_cols)
        
        if selected_col:
            fig = make_subplots(rows=2, cols=2, 
                                subplot_titles=('Histogram', 'Box Plot', 'Density Plot', 'Q-Q Plot'))
            
            fig.add_trace(go.Histogram(x=df[selected_col].dropna(), nbinsx=30, 
                                       marker_color='#667eea'), row=1, col=1)
            fig.add_trace(go.Box(y=df[selected_col].dropna(), marker_color='#764ba2'), row=1, col=2)
            fig.add_trace(go.Histogram(x=df[selected_col].dropna(), histnorm='probability density',
                                       marker_color='#6bcb77'), row=2, col=1)
            
            from scipy import stats
            qq_data = stats.probplot(df[selected_col].dropna(), dist="norm")
            fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers',
                                     marker_color='#ff6b6b'), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            data = df[selected_col].dropna()
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{data.mean():.3f}")
            with col2:
                st.metric("Median", f"{data.median():.3f}")
            with col3:
                st.metric("Std Dev", f"{data.std():.3f}")
            with col4:
                st.metric("Skewness", f"{data.skew():.3f}")
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                aspect="auto",
                title="Feature Correlation Matrix",
                color_continuous_scale='RdBu',
                zmin=-1, zmax=1
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Risk_Level' in df.columns:
                risk_counts = df['Risk_Level'].value_counts()
                colors = {'Critical': '#ff4444', 'High': '#ff6b6b', 'Medium': '#ffd93d', 
                         'Low': '#6bcb77', 'Very Low': '#4caf50'}
                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                            title="Risk Level Distribution",
                            color=risk_counts.index,
                            color_discrete_map=colors)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Risk_Score' in df.columns:
                fig = px.histogram(df, x='Risk_Score', nbins=40,
                                  title="Risk Score Distribution",
                                  color_discrete_sequence=['#667eea'])
                st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# MODEL TRAINING PAGE
# ============================================================================

def page_model_training():
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded or st.session_state.risk_data is None:
        st.warning("⚠️ Please load data first in the Data Upload page.")
        return
    
    df = st.session_state.risk_data
    
    st.subheader("1. Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        risk_related = [col for col in df.columns if 'risk' in col.lower() or 'Risk' in col]
        target_options = risk_related if risk_related else df.columns.tolist()
        
        target_col = st.selectbox("🎯 Target Column", target_options)
        st.session_state.target_column = target_col
        
        unique_vals = df[target_col].nunique()
        if df[target_col].dtype in ['int64', 'float64'] and unique_vals > 15:
            task_type = st.radio("📊 Task Type", ["Regression", "Classification"], index=0)
        else:
            task_type = st.radio("📊 Task Type", ["Classification", "Regression"], index=0)
        
        st.session_state.task_type = task_type
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        random_state = st.number_input("Random Seed", value=42, step=1)
    
    st.subheader("2. Feature Selection")
    
    feature_cols = [col for col in df.columns if col != target_col]
    selected_features = st.multiselect("Select features for training", feature_cols,
                                       default=feature_cols[:8] if len(feature_cols) > 8 else feature_cols)
    st.session_state.selected_features = selected_features
    
    if len(selected_features) == 0:
        st.error("Please select at least one feature.")
        return
    
    st.subheader("3. Model Selection")
    
    if task_type == "Classification":
        model_options = st.multiselect("Choose models to train",
            ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 
             'Decision Tree', 'SVM', 'KNN'],
            default=['Random Forest', 'Gradient Boosting'])
    else:
        model_options = st.multiselect("Choose models to train",
            ['Random Forest', 'Gradient Boosting', 'Decision Tree'],
            default=['Random Forest', 'Gradient Boosting'])
    
    if st.button("🚀 START TRAINING", type="primary", use_container_width=True):
        if len(model_options) == 0:
            st.error("Please select at least one model.")
        else:
            with st.spinner("Training models..."):
                try:
                    X, y, encoders, target_enc = advanced_preprocessing(df, selected_features, target_col)
                    st.session_state.feature_encoders = encoders
                    st.session_state.target_encoder = target_enc
                    
                    scaler = RobustScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=random_state,
                        stratify=y if task_type == "Classification" else None
                    )
                    
                    results = {}
                    trained_models = {}
                    
                    progress_bar = st.progress(0)
                    
                    for i, model_name in enumerate(model_options):
                        status_text = st.empty()
                        status_text.text(f"Training {model_name}... ({i+1}/{len(model_options)})")
                        
                        model = get_model_with_params(model_name, task_type, random_state)
                        if model is None:
                            continue
                        
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        trained_models[model_name] = model
                        
                        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds,
                                                    scoring='accuracy' if task_type == "Classification" else 'r2')
                        
                        if task_type == "Classification":
                            results[model_name] = {
                                'Accuracy': accuracy_score(y_test, y_pred),
                                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                                'CV_Mean': cv_scores.mean(),
                                'CV_Std': cv_scores.std()
                            }
                        else:
                            results[model_name] = {
                                'R2 Score': r2_score(y_test, y_pred),
                                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                                'MAE': mean_absolute_error(y_test, y_pred),
                                'CV_Mean': cv_scores.mean(),
                                'CV_Std': cv_scores.std()
                            }
                        
                        progress_bar.progress((i + 1) / len(model_options))
                        status_text.empty()
                    
                    st.session_state.trained_models = trained_models
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
                    
                    if task_type == "Classification":
                        st.session_state.best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
                    else:
                        st.session_state.best_model_name = max(results, key=lambda x: results[x]['R2 Score'])
                    
                    st.session_state.best_model = trained_models[st.session_state.best_model_name]
                    
                    st.success("✅ Training complete!")
                    
                    results_df = pd.DataFrame(results).T
                    if task_type == "Classification":
                        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
                    else:
                        display_cols = ['R2 Score', 'RMSE', 'MAE', 'CV_Mean']
                    
                    st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0))
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")

# ============================================================================
# PREDICTION PAGE
# ============================================================================

def page_prediction():
    st.header("📈 Make Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the Model Training page.")
        return
    
    st.subheader("Enter Input Values")
    
    input_data = {}
    cols = st.columns(2)
    
    df = st.session_state.risk_data
    
    for i, feature in enumerate(st.session_state.selected_features):
        with cols[i % 2]:
            if feature in df.columns:
                if df[feature].dtype == 'object':
                    values = df[feature].dropna().unique().tolist()
                    input_data[feature] = st.selectbox(f"📊 {feature}", values)
                else:
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    input_data[feature] = st.number_input(f"📈 {feature}", value=mean_val,
                                                          min_value=min_val, max_value=max_val)
            else:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
    
    if st.button("🔮 PREDICT", type="primary", use_container_width=True):
        try:
            input_df = pd.DataFrame([input_data])
            
            for feat, encoder in st.session_state.feature_encoders.items():
                if feat in input_df.columns:
                    val = input_df[feat].iloc[0]
                    if val in encoder.classes_:
                        input_df[feat + '_enc'] = encoder.transform([val])[0]
            
            X_input = []
            for feat in st.session_state.selected_features:
                if feat + '_enc' in input_df.columns:
                    X_input.append(input_df[feat + '_enc'].iloc[0])
                elif feat in input_df.columns and input_df[feat].dtype in ['int64', 'float64']:
                    X_input.append(input_df[feat].iloc[0])
            
            X_input = np.array(X_input).reshape(1, -1)
            X_input_scaled = st.session_state.scaler.transform(X_input)
            
            best_pred = st.session_state.best_model.predict(X_input_scaled)[0]
            
            st.subheader("🎯 Prediction Result")
            
            if st.session_state.task_type == "Classification":
                if st.session_state.target_encoder:
                    best_label = st.session_state.target_encoder.inverse_transform([int(best_pred)])[0]
                else:
                    best_label = str(best_pred)
                
                risk_level = str(best_label).lower()
                if 'critical' in risk_level:
                    card_class = "risk-critical"
                elif 'high' in risk_level:
                    card_class = "risk-high"
                elif 'medium' in risk_level:
                    card_class = "risk-medium"
                elif 'low' in risk_level:
                    card_class = "risk-low"
                else:
                    card_class = "risk-verylow"
                
                st.markdown(f"""
                <div class="{card_class}" style="padding: 2rem; margin: 1rem 0;">
                    <h2 style="margin: 0;">{best_label}</h2>
                    <p style="margin: 0;">Predicted by MP-RAS using {st.session_state.best_model_name}</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h2>Risk Score: {best_pred:.2f}</h2>
                    <p>Predicted by MP-RAS using {st.session_state.best_model_name}</p>
                </div>
                """, unsafe_allow_html=True)
                
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ============================================================================
# RESULTS PAGE
# ============================================================================

def page_results():
    st.header("📊 Results & Reports")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the Model Training page.")
        return
    
    st.subheader("Model Performance Summary")
    
    results_df = pd.DataFrame(st.session_state.model_results).T
    
    if st.session_state.task_type == "Classification":
        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
    else:
        display_cols = ['R2 Score', 'RMSE', 'MAE', 'CV_Mean']
    
    st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0))
    
    st.subheader(f"🏆 Best Model: {st.session_state.best_model_name}")
    
    best_scores = st.session_state.model_results[st.session_state.best_model_name]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.session_state.task_type == "Classification":
            st.metric("Accuracy", f"{best_scores['Accuracy']:.4f}")
        else:
            st.metric("R² Score", f"{best_scores['R2 Score']:.4f}")
    with col2:
        if st.session_state.task_type == "Classification":
            st.metric("F1-Score", f"{best_scores['F1-Score']:.4f}")
        else:
            st.metric("RMSE", f"{best_scores['RMSE']:.4f}")
    with col3:
        st.metric("CV Mean", f"{best_scores['CV Mean']:.4f}")
    
    # Generate report
    st.subheader("📄 Generate MP-RAS Report")
    
    if st.button("Generate Full Report", type="primary"):
        report = f"""
================================================================================
                    MP-RAS (Microplastic Risk Assessment System)
                                SYSTEM REPORT
================================================================================

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
System Version: 2.0
Researchers: Matthew Joseph Viernes & Shane Mark R. Magdaluyo
Institution: Agusan del Sur State College of Agriculture and Technology (ASSCAT)

================================================================================
1. SYSTEM OVERVIEW
================================================================================

MP-RAS is a data mining-based predictive risk modeling system for 
microplastic pollution assessment in aquatic environments.

Dataset Shape: {st.session_state.risk_data.shape[0]} rows × {st.session_state.risk_data.shape[1]} columns
Data Source: {st.session_state.data_source}
Target Column: {st.session_state.target_column}
Task Type: {st.session_state.task_type}
Selected Features: {len(st.session_state.selected_features)}

================================================================================
2. MODEL PERFORMANCE
================================================================================

{results_df[display_cols].to_string()}

================================================================================
3. BEST MODEL RESULTS
================================================================================

Best Model: {st.session_state.best_model_name}

"""
        for key, value in best_scores.items():
            report += f"- {key}: {value:.4f}\n"

        report += f"""
================================================================================
4. SYSTEM CONCLUSIONS
================================================================================

The MP-RAS system successfully developed a predictive model for 
microplastic risk assessment with {best_scores[display_cols[0]]:.4f} {display_cols[0]}.

================================================================================
                            END OF MP-RAS REPORT
================================================================================
"""
        
        st.download_button(
            label="📥 Download MP-RAS Report",
            data=report,
            file_name=f"MP-RAS_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )

# ============================================================================
# MAIN PAGE ROUTING
# ============================================================================

# Page routing dictionary
pages = {
    "🏠 Dashboard": page_home,
    "📁 Data Upload": page_data_upload,
    "📊 EDA": page_eda,
    "🤖 Model Training": page_model_training,
    "📈 Prediction": page_prediction,
    "📊 Results": page_results,
}

# Display selected page
if page in pages:
    pages[page]()

# ============================================================================
# FLOATING SYSTEM BADGE
# ============================================================================

st.markdown("""
<div class="system-badge">
    🌊 MP-RAS v2.0
</div>
""", unsafe_allow_html=True)

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("""
<div class="footer">
    <p>🌊 MP-RAS | Microplastic Risk Modeling </p>
    <p>Viernes, M.J. & Magdaluyo, S.M.R. | Data Mining-Based Predictive Risk Modeling</p>
</div>
""", unsafe_allow_html=True)
