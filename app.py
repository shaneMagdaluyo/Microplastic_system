# enhanced_app.py - Complete Microplastic Risk Prediction System
# With Advanced Features: EDA, Multiple Models, Hyperparameter Tuning, Visualization, Export
# Researchers: Matthew Joseph Viernes & Shane Mark R. Magdaluyo
# ASSCAT 2025

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
    page_title="Microplastic Risk Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
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
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
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
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2rem;
        color: #666;
        border-top: 1px solid #ddd;
        margin-top: 2rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state with comprehensive variables
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
        (df['Industrial_Score']) * 25 +
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

def calculate_data_statistics(df):
    """Calculate comprehensive data statistics"""
    stats_dict = {
        'shape': df.shape,
        'columns': len(df.columns),
        'rows': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicates': df.duplicated().sum(),
        'numeric_cols': len(df.select_dtypes(include=[np.number]).columns),
        'categorical_cols': len(df.select_dtypes(include=['object']).columns),
        'memory_usage': df.memory_usage(deep=True).sum() / 1024**2,
    }
    
    # Numeric summaries
    numeric_df = df.select_dtypes(include=[np.number])
    if len(numeric_df.columns) > 0:
        stats_dict['numeric_summary'] = {
            'mean': numeric_df.mean().to_dict(),
            'std': numeric_df.std().to_dict(),
            'skewness': numeric_df.skew().to_dict(),
            'kurtosis': numeric_df.kurtosis().to_dict(),
        }
    
    return stats_dict

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
# PAGE: HOME / DASHBOARD
# ============================================================================

def page_home():
    st.header("🏠 Dashboard")
    
    if st.session_state.data_loaded and st.session_state.risk_data is not None:
        df = st.session_state.risk_data
        
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
                <p>🔍 Missing Data</p>
                <h3>{missing_pct:.1f}%</h3>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            memory = df.memory_usage(deep=True).sum() / 1024**2
            st.markdown(f"""
            <div class="metric-card">
                <p>💾 Memory</p>
                <h3>{memory:.1f} MB</h3>
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
        
        # Feature preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10))
        
        # Model status
        if st.session_state.model_trained:
            st.subheader("🤖 Model Status")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.success(f"✅ Best Model: {st.session_state.best_model_name}")
            with col2:
                best_score = st.session_state.model_results.get(st.session_state.best_model_name, {})
                if st.session_state.task_type == "Classification":
                    score = best_score.get('Accuracy', 0)
                else:
                    score = best_score.get('R2 Score', 0)
                st.metric("Best Score", f"{score:.4f}")
            with col3:
                st.metric("Trained Models", len(st.session_state.trained_models))
    
    else:
        # Welcome screen
        st.markdown("""
        <div class="info-box">
            <h3>🌊 Welcome to the Microplastic Pollution Risk Prediction System</h3>
            <p>This system uses advanced data mining techniques to predict microplastic pollution risks.</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✨ Features
            
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
            
            - **Advanced Features**
              - Hyperparameter tuning
              - Cross-validation
              - Feature importance
              - Report generation
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
                st.session_state.data_stats = calculate_data_statistics(df)
                st.rerun()
        
        # System capabilities
        st.markdown("---")
        st.subheader("📈 System Capabilities")
        
        cap_cols = st.columns(4)
        capabilities = [
            ("🎯", "Multiple ML Models", "6+ classification & regression models"),
            ("📊", "Interactive Visualizations", "Real-time charts and graphs"),
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
# PAGE: DATA UPLOAD
# ============================================================================

def page_data_upload():
    st.header("📁 Data Upload & Preprocessing")
    
    # Show current data status
    if st.session_state.data_loaded:
        st.markdown(f"""
        <div class="success-box">
            ✅ <strong>Data Loaded</strong><br>
            File: {st.session_state.data_filename}<br>
            Shape: {st.session_state.risk_data.shape[0]} rows × {st.session_state.risk_data.shape[1]} columns
        </div>
        """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Upload Data")
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
                st.write(f"**Memory:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
                
                if st.button("✅ Load Dataset", type="primary"):
                    st.session_state.risk_data = df
                    st.session_state.data_loaded = True
                    st.session_state.data_filename = uploaded_file.name
                    st.session_state.data_source = "upload"
                    st.session_state.data_stats = calculate_data_statistics(df)
                    # Reset model state
                    st.session_state.model_trained = False
                    st.session_state.trained_models = {}
                    st.session_state.model_results = {}
                    st.success(f"✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with col2:
        st.subheader("📊 Sample Data")
        st.write("Use the built-in sample dataset with realistic microplastic data:")
        st.write("- 2,000 samples")
        st.write("- 25+ features including environmental, biological, and chemical parameters")
        st.write("- Pre-calculated risk scores and levels")
        
        if st.button("📊 Load Enhanced Sample Data", use_container_width=True):
            df = generate_enhanced_sample_data(2000)
            st.session_state.risk_data = df
            st.session_state.data_loaded = True
            st.session_state.data_source = "sample"
            st.session_state.data_filename = "microplastic_sample_data.csv"
            st.session_state.data_stats = calculate_data_statistics(df)
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
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Data Overview", "📈 Statistics", "🔍 Missing Values", "🏷️ Data Types"])
        
        with tab1:
            st.dataframe(df.head(10))
            st.caption(f"Last 10 rows of {len(df)} total rows")
        
        with tab2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe())
            else:
                st.info("No numeric columns found")
        
        with tab3:
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Data Type': df.dtypes.values,
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
        
        with tab4:
            dtype_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Unique Values': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(dtype_df)
            
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols:
                st.write(f"**Categorical columns ({len(categorical_cols)}):** {', '.join(categorical_cols[:10])}")
                if len(categorical_cols) > 10:
                    st.write(f"... and {len(categorical_cols) - 10} more")
                
                if st.button("🔧 Encode Categorical Variables"):
                    for col in categorical_cols:
                        le = LabelEncoder()
                        df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                        st.session_state.feature_encoders[col] = le
                    st.session_state.risk_data = df
                    st.success(f"✅ Encoded {len(categorical_cols)} categorical variables!")
                    st.rerun()

# ============================================================================
# PAGE: EXPLORATORY DATA ANALYSIS
# ============================================================================

def page_eda():
    st.header("📊 Exploratory Data Analysis")
    
    if not st.session_state.data_loaded or st.session_state.risk_data is None:
        st.warning("⚠️ Please load data first in the Data Upload page.")
        return
    
    df = st.session_state.risk_data
    
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📈 Distribution Analysis", 
        "🔗 Correlation Analysis", 
        "🎯 Risk Analysis",
        "🧬 Feature Analysis",
        "📊 Statistical Tests"
    ])
    
    with tab1:
        st.subheader("Distribution Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Select column for distribution
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_col = st.selectbox("Select column to analyze", numeric_cols)
            
            # Distribution plot
            fig = make_subplots(rows=2, cols=2, 
                                subplot_titles=('Histogram', 'Box Plot', 'Q-Q Plot', 'Density Plot'))
            
            # Histogram
            fig.add_trace(go.Histogram(x=df[selected_col].dropna(), nbinsx=30, 
                                       name='Histogram', marker_color='#667eea'), row=1, col=1)
            
            # Box plot
            fig.add_trace(go.Box(y=df[selected_col].dropna(), name='Box Plot', 
                                 marker_color='#764ba2'), row=1, col=2)
            
            # Q-Q Plot
            from scipy import stats
            qq_data = stats.probplot(df[selected_col].dropna(), dist="norm")
            fig.add_trace(go.Scatter(x=qq_data[0][0], y=qq_data[0][1], mode='markers',
                                     name='Q-Q Plot', marker_color='#ff6b6b'), row=2, col=1)
            
            # Density plot
            fig.add_trace(go.Histogram(x=df[selected_col].dropna(), nbinsx=30, 
                                       histnorm='probability density', name='Density',
                                       marker_color='#6bcb77'), row=2, col=2)
            
            fig.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            data = df[selected_col].dropna()
            with col1:
                st.metric("Mean", f"{data.mean():.3f}")
            with col2:
                st.metric("Median", f"{data.median():.3f}")
            with col3:
                st.metric("Std Dev", f"{data.std():.3f}")
            with col4:
                st.metric("Skewness", f"{data.skew():.3f}")
        
        with col2:
            # Multiple distributions
            st.write("**Multiple Feature Distributions**")
            selected_cols = st.multiselect("Select columns to compare", numeric_cols, default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
            
            if selected_cols:
                fig = go.Figure()
                for col in selected_cols:
                    fig.add_trace(go.Box(y=df[col], name=col))
                fig.update_layout(title="Box Plot Comparison", height=400)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Correlation matrix
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
            
            # Top correlations with risk score
            if 'Risk_Score' in numeric_cols:
                st.subheader("Top Correlations with Risk Score")
                risk_corr = corr_matrix['Risk_Score'].sort_values(ascending=False)
                risk_corr_df = pd.DataFrame({
                    'Feature': risk_corr.index,
                    'Correlation': risk_corr.values
                })
                risk_corr_df = risk_corr_df[risk_corr_df['Feature'] != 'Risk_Score']
                
                fig = px.bar(risk_corr_df, x='Correlation', y='Feature', orientation='h',
                            title="Correlation with Risk Score",
                            color='Correlation', color_continuous_scale='RdBu')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Need at least 2 numeric columns for correlation analysis")
    
    with tab3:
        st.subheader("Risk Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Risk_Level' in df.columns:
                # Risk level distribution
                risk_counts = df['Risk_Level'].value_counts()
                colors = {
                    'Critical': '#ff4444', 'High': '#ff6b6b', 
                    'Medium': '#ffd93d', 'Low': '#6bcb77', 'Very Low': '#4caf50'
                }
                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                            title="Risk Level Distribution",
                            color=risk_counts.index,
                            color_discrete_map=colors)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Risk_Type' in df.columns:
                # Risk type distribution
                risk_type_counts = df['Risk_Type'].value_counts().head(8)
                fig = px.bar(x=risk_type_counts.values, y=risk_type_counts.index, 
                            orientation='h', title="Risk Type Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        # Risk score by category
        st.subheader("Risk Score by Categories")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols and 'Risk_Score' in df.columns:
            selected_cat = st.selectbox("Select category to analyze", categorical_cols)
            risk_by_cat = df.groupby(selected_cat)['Risk_Score'].agg(['mean', 'std', 'count']).reset_index()
            risk_by_cat = risk_by_cat.sort_values('mean', ascending=False).head(10)
            
            fig = px.bar(risk_by_cat, x=selected_cat, y='mean', error_y='std',
                        title=f"Average Risk Score by {selected_cat}",
                        color='mean', color_continuous_scale='Reds')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Feature Analysis")
        
        # Feature importance based on statistical tests
        if 'Risk_Score' in df.columns:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            numeric_cols = [c for c in numeric_cols if c != 'Risk_Score']
            
            # Calculate correlation and p-values
            feature_stats = []
            for col in numeric_cols[:20]:  # Limit to 20
                corr, p_val = pearsonr(df[col].dropna(), df['Risk_Score'].dropna())
                feature_stats.append({
                    'Feature': col,
                    'Correlation': corr,
                    'P-Value': p_val,
                    'Significant': p_val < 0.05
                })
            
            feature_df = pd.DataFrame(feature_stats).sort_values('Correlation', ascending=False)
            
            fig = px.bar(feature_df, x='Correlation', y='Feature', orientation='h',
                        title="Feature Correlation with Risk Score",
                        color='Significant', color_discrete_map={True: '#6bcb77', False: '#ff6b6b'})
            st.plotly_chart(fig, use_container_width=True)
            
            st.dataframe(feature_df)
    
    with tab5:
        st.subheader("Statistical Tests")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Normality Test (Shapiro-Wilk)**")
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            selected_test = st.selectbox("Select column for normality test", numeric_cols)
            
            if selected_test:
                data = df[selected_test].dropna()
                if len(data) < 5000:  # Shapiro works best with <5000
                    stat, p = shapiro(data[:5000])
                    st.write(f"Test statistic: {stat:.4f}")
                    st.write(f"P-value: {p:.6f}")
                    if p > 0.05:
                        st.success("✅ Data appears to be normally distributed (p > 0.05)")
                    else:
                        st.warning("⚠️ Data is not normally distributed (p < 0.05)")
                else:
                    st.info("Too many samples for Shapiro test. Using distribution plot instead.")
        
        with col2:
            st.write("**ANOVA Test (Difference between groups)**")
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if categorical_cols and numeric_cols:
                cat_col = st.selectbox("Select categorical column", categorical_cols, key="anova_cat")
                num_col = st.selectbox("Select numeric column", numeric_cols, key="anova_num")
                
                if cat_col and num_col:
                    groups = [group[num_col].dropna().values for name, group in df.groupby(cat_col) if len(group) > 0]
                    if len(groups) >= 2:
                        f_stat, p_val = stats.f_oneway(*groups)
                        st.write(f"F-statistic: {f_stat:.4f}")
                        st.write(f"P-value: {p_val:.6f}")
                        if p_val < 0.05:
                            st.success("✅ Significant difference between groups (p < 0.05)")
                        else:
                            st.warning("❌ No significant difference between groups (p > 0.05)")

# ============================================================================
# PAGE: MODEL TRAINING
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
        # Target selection
        risk_related = [col for col in df.columns if 'risk' in col.lower() or 'Risk' in col]
        target_options = risk_related if risk_related else df.columns.tolist()
        
        target_col = st.selectbox("🎯 Target Column (What to predict)", target_options)
        st.session_state.target_column = target_col
        
        # Determine task type
        unique_vals = df[target_col].nunique()
        if df[target_col].dtype in ['int64', 'float64'] and unique_vals > 15:
            task_type = st.radio("📊 Task Type", ["Regression", "Classification"], index=0)
        else:
            task_type = st.radio("📊 Task Type", ["Classification", "Regression"], index=0)
        
        st.session_state.task_type = task_type
        
        st.info(f"Target '{target_col}' has {unique_vals} unique values")
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        random_state = st.number_input("Random Seed", value=42, step=1)
        
        # Handle imbalanced data
        if task_type == "Classification":
            use_smote = st.checkbox("Use SMOTE for Imbalanced Data", value=False,
                                   disabled=not IMBALANCE_AVAILABLE)
            if use_smote and not IMBALANCE_AVAILABLE:
                st.warning("Install imbalanced-learn: pip install imbalanced-learn")
        else:
            use_smote = False
    
    st.subheader("2. Feature Selection")
    
    feature_cols = [col for col in df.columns if col != target_col]
    
    # Feature grouping
    numeric_feats = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    st.write(f"**Numeric features:** {len(numeric_feats)}")
    st.write(f"**Categorical features:** {len(cat_feats)} (will be automatically encoded)")
    
    selected_features = st.multiselect(
        "Select features for training",
        feature_cols,
        default=feature_cols[:8] if len(feature_cols) > 8 else feature_cols
    )
    st.session_state.selected_features = selected_features
    
    if len(selected_features) == 0:
        st.error("Please select at least one feature.")
        return
    
    st.subheader("3. Model Selection")
    
    # Model options based on task type
    if task_type == "Classification":
        model_options = st.multiselect(
            "Choose models to train",
            ['Random Forest', 'Gradient Boosting', 'Logistic Regression', 
             'Decision Tree', 'SVM', 'KNN', 'AdaBoost', 'Naive Bayes'],
            default=['Random Forest', 'Gradient Boosting', 'Logistic Regression']
        )
    else:
        model_options = st.multiselect(
            "Choose models to train",
            ['Random Forest', 'Gradient Boosting', 'Decision Tree', 'SVR', 'KNN'],
            default=['Random Forest', 'Gradient Boosting']
        )
    
    # Hyperparameter tuning option
    do_tuning = st.checkbox("🔧 Perform Hyperparameter Tuning (may take longer)", value=False)
    
    # Training button
    if st.button("🚀 START TRAINING", type="primary", use_container_width=True):
        if len(model_options) == 0:
            st.error("Please select at least one model.")
        else:
            with st.spinner("Training models... This may take a few moments."):
                try:
                    # Preprocess data
                    X, y, encoders, target_enc = advanced_preprocessing(df, selected_features, target_col)
                    st.session_state.feature_encoders = encoders
                    st.session_state.target_encoder = target_enc
                    
                    # Scale features
                    scaler = RobustScaler()  # More robust to outliers
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                    
                    # Handle imbalanced data
                    if task_type == "Classification" and use_smote and IMBALANCE_AVAILABLE:
                        smote = SMOTE(random_state=random_state)
                        X_scaled, y = smote.fit_resample(X_scaled, y)
                        st.write(f"After SMOTE: {X_scaled.shape[0]} samples")
                    
                    # Split data
                    stratify = y if task_type == "Classification" else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=random_state, stratify=stratify
                    )
                    
                    st.write(f"📊 Training set: {len(X_train)} samples")
                    st.write(f"📊 Testing set: {len(X_test)} samples")
                    
                    # Train models
                    results = {}
                    trained_models = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, model_name in enumerate(model_options):
                        status_text.text(f"Training {model_name}... ({i+1}/{len(model_options)})")
                        
                        model = get_model_with_params(model_name, task_type, random_state)
                        if model is None:
                            continue
                        
                        # Train
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        trained_models[model_name] = model
                        
                        # Cross-validation
                        cv_scores = cross_val_score(model, X_scaled, y, cv=cv_folds, 
                                                    scoring='accuracy' if task_type == "Classification" else 'r2')
                        
                        # Metrics
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
                    
                    status_text.text("Training complete!")
                    
                    # Store results
                    st.session_state.trained_models = trained_models
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
                    
                    # Find best model
                    if task_type == "Classification":
                        st.session_state.best_model_name = max(results, key=lambda x: results[x]['Accuracy'])
                        st.session_state.best_model = trained_models[st.session_state.best_model_name]
                    else:
                        st.session_state.best_model_name = max(results, key=lambda x: results[x]['R2 Score'])
                        st.session_state.best_model = trained_models[st.session_state.best_model_name]
                    
                    # Display results
                    st.success("✅ Training complete!")
                    
                    st.subheader("📊 Model Performance Results")
                    
                    results_df = pd.DataFrame(results).T
                    
                    if task_type == "Classification":
                        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
                    else:
                        display_cols = ['R2 Score', 'RMSE', 'MAE', 'CV_Mean']
                    
                    styled_df = results_df[display_cols].style.format('{:.4f}')
                    styled_df = styled_df.highlight_max(axis=0, subset=[display_cols[0]])
                    st.dataframe(styled_df)
                    
                    # Best model info
                    st.subheader(f"🏆 Best Model: {st.session_state.best_model_name}")
                    
                    # Visual comparison
                    fig = go.Figure()
                    if task_type == "Classification":
                        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
                        for metric in metrics:
                            fig.add_trace(go.Bar(name=metric, x=list(results.keys()), 
                                               y=[results[m][metric] for m in results.keys()]))
                    else:
                        metrics = ['R2 Score']
                        for metric in metrics:
                            fig.add_trace(go.Bar(name=metric, x=list(results.keys()), 
                                               y=[results[m][metric] for m in results.keys()]))
                    
                    fig.update_layout(title="Model Performance Comparison", 
                                     barmode='group', height=400)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confusion Matrix for best classification model
                    if task_type == "Classification":
                        st.subheader(f"Confusion Matrix - {st.session_state.best_model_name}")
                        best_model = trained_models[st.session_state.best_model_name]
                        
                        # Get predictions on test set
                        y_pred_best = best_model.predict(X_test)
                        cm = confusion_matrix(y_test, y_pred_best)
                        
                        fig, ax = plt.subplots(figsize=(10, 8))
                        labels = target_enc.classes_ if target_enc else [str(i) for i in range(len(np.unique(y)))]
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                   xticklabels=labels, yticklabels=labels)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title(f'Confusion Matrix - {st.session_state.best_model_name}')
                        st.pyplot(fig)
                        
                        # Classification Report
                        st.subheader("Classification Report")
                        report = classification_report(y_test, y_pred_best, target_names=labels, output_dict=True)
                        report_df = pd.DataFrame(report).T
                        st.dataframe(report_df.round(4))
                    
                    # Feature Importance
                    if 'Random Forest' in trained_models and hasattr(trained_models['Random Forest'], 'feature_importances_'):
                        st.subheader("📈 Feature Importance Analysis")
                        rf_model = trained_models['Random Forest']
                        importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig = px.bar(importance_df.head(20), x='Importance', y='Feature', orientation='h',
                                    title="Top 20 Feature Importances (Random Forest)",
                                    color='Importance', color_continuous_scale='Blues')
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Save models
                    joblib.dump(st.session_state.best_model, 'best_microplastic_model.pkl')
                    joblib.dump(st.session_state.scaler, 'scaler.pkl')
                    st.success("💾 Best model saved as 'best_microplastic_model.pkl'")
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    st.info("Please check your data and try again.")

# ============================================================================
# PAGE: PREDICTION
# ============================================================================

def page_prediction():
    st.header("📈 Make Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the Model Training page.")
        return
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ No data loaded.")
        return
    
    st.subheader("Enter Input Values")
    
    # Create input fields
    input_data = {}
    
    # Use columns for better layout
    cols = st.columns(2)
    
    df = st.session_state.risk_data
    
    for i, feature in enumerate(st.session_state.selected_features):
        with cols[i % 2]:
            if feature in df.columns:
                if df[feature].dtype == 'object':
                    # Categorical - dropdown
                    values = df[feature].dropna().unique().tolist()
                    input_data[feature] = st.selectbox(f"📊 {feature}", values)
                else:
                    # Numeric - number input with suggestions
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    mean_val = float(df[feature].mean())
                    input_data[feature] = st.number_input(
                        f"📈 {feature}", 
                        value=mean_val,
                        min_value=min_val, 
                        max_value=max_val,
                        help=f"Range: {min_val:.2f} - {max_val:.2f}"
                    )
            else:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
    
    # Batch prediction option
    st.subheader("Or Upload Batch File")
    batch_file = st.file_uploader("Upload CSV for batch predictions", type=['csv'])
    
    if st.button("🔮 PREDICT", type="primary", use_container_width=True):
        if batch_file is not None:
            # Batch prediction
            batch_df = pd.read_csv(batch_file)
            st.write(f"Processing {len(batch_df)} records...")
            
            # Process batch
            predictions = []
            for idx, row in batch_df.iterrows():
                try:
                    # Process single row
                    input_dict = {feat: row[feat] for feat in st.session_state.selected_features if feat in row}
                    input_df = pd.DataFrame([input_dict])
                    
                    # Encode
                    for feat, encoder in st.session_state.feature_encoders.items():
                        if feat in input_df.columns:
                            val = input_df[feat].iloc[0]
                            if val in encoder.classes_:
                                input_df[feat + '_enc'] = encoder.transform([val])[0]
                    
                    # Prepare features
                    X_input = []
                    for feat in st.session_state.selected_features:
                        if feat + '_enc' in input_df.columns:
                            X_input.append(input_df[feat + '_enc'].iloc[0])
                        elif feat in input_df.columns and input_df[feat].dtype in ['int64', 'float64']:
                            X_input.append(input_df[feat].iloc[0])
                    
                    X_input = np.array(X_input).reshape(1, -1)
                    X_input_scaled = st.session_state.scaler.transform(X_input)
                    
                    # Predict
                    pred = st.session_state.best_model.predict(X_input_scaled)[0]
                    
                    if st.session_state.task_type == "Classification" and st.session_state.target_encoder:
                        pred_label = st.session_state.target_encoder.inverse_transform([int(pred)])[0]
                    else:
                        pred_label = pred
                    
                    predictions.append(pred_label)
                except Exception as e:
                    predictions.append(f"Error: {str(e)}")
            
            batch_df['Prediction'] = predictions
            st.dataframe(batch_df)
            
            # Download results
            csv = batch_df.to_csv(index=False)
            st.download_button("📥 Download Predictions", csv, "predictions.csv", "text/csv")
            
        else:
            # Single prediction
            try:
                # Process input
                input_df = pd.DataFrame([input_data])
                
                # Encode categorical features
                for feat, encoder in st.session_state.feature_encoders.items():
                    if feat in input_df.columns:
                        val = input_df[feat].iloc[0]
                        if val in encoder.classes_:
                            input_df[feat + '_enc'] = encoder.transform([val])[0]
                        else:
                            input_df[feat + '_enc'] = -1
                
                # Prepare feature vector
                X_input = []
                feature_names = []
                for feat in st.session_state.selected_features:
                    if feat + '_enc' in input_df.columns:
                        X_input.append(input_df[feat + '_enc'].iloc[0])
                        feature_names.append(feat + '_enc')
                    elif feat in input_df.columns and input_df[feat].dtype in ['int64', 'float64']:
                        X_input.append(input_df[feat].iloc[0])
                        feature_names.append(feat)
                
                X_input = np.array(X_input).reshape(1, -1)
                X_input_scaled = st.session_state.scaler.transform(X_input)
                
                # Make predictions with all models
                st.subheader("🎯 Prediction Results")
                
                # Best model prediction
                best_pred = st.session_state.best_model.predict(X_input_scaled)[0]
                
                if st.session_state.task_type == "Classification":
                    if st.session_state.target_encoder:
                        best_label = st.session_state.target_encoder.inverse_transform([int(best_pred)])[0]
                    else:
                        best_label = str(best_pred)
                    
                    # Risk level styling
                    risk_level = str(best_label).lower()
                    if 'critical' in risk_level:
                        card_class = "risk-critical"
                    elif 'high' in risk_level:
                        card_class = "risk-high"
                    elif 'medium' in risk_level:
                        card_class = "risk-medium"
                    else:
                        card_class = "risk-low"
                    
                    st.markdown(f"""
                    <div class="{card_class}" style="padding: 2rem; margin: 1rem 0;">
                        <h2 style="margin: 0;">{best_label}</h2>
                        <p style="margin: 0;">Predicted Risk Level</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show all model predictions
                    st.subheader("All Model Predictions")
                    pred_cols = st.columns(len(st.session_state.trained_models))
                    
                    for idx, (name, model) in enumerate(st.session_state.trained_models.items()):
                        pred = model.predict(X_input_scaled)[0]
                        if st.session_state.target_encoder:
                            pred_label = st.session_state.target_encoder.inverse_transform([int(pred)])[0]
                        else:
                            pred_label = str(pred)
                        
                        with pred_cols[idx]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h4>{name}</h4>
                                <h3>{pred_label}</h3>
                            </div>
                            """, unsafe_allow_html=True)
                
                else:
                    # Regression
                    st.markdown(f"""
                    <div class="metric-card">
                        <h2>Risk Score: {best_pred:.2f}</h2>
                        <p>Predicted using {st.session_state.best_model_name}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Gauge chart for risk score
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=best_pred,
                        title={'text': "Risk Score"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "#667eea"},
                            'steps': [
                                {'range': [0, 33], 'color': "#6bcb77"},
                                {'range': [33, 66], 'color': "#ffd93d"},
                                {'range': [66, 100], 'color': "#ff6b6b"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': best_pred
                            }
                        }
                    ))
                    fig.update_layout(height=400)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Store prediction history
                st.session_state.predictions_history.append({
                    'timestamp': datetime.now(),
                    'input': input_data,
                    'prediction': best_pred if st.session_state.task_type != "Classification" else best_label,
                    'model': st.session_state.best_model_name
                })
                
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")

# ============================================================================
# PAGE: RESULTS & REPORTS
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
    
    # Visual comparison
    st.subheader("Model Comparison")
    
    fig = go.Figure()
    if st.session_state.task_type == "Classification":
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        for metric in metrics:
            fig.add_trace(go.Bar(name=metric, x=list(st.session_state.model_results.keys()), 
                               y=[st.session_state.model_results[m][metric] for m in st.session_state.model_results.keys()]))
    else:
        metrics = ['R2 Score']
        for metric in metrics:
            fig.add_trace(go.Bar(name=metric, x=list(st.session_state.model_results.keys()), 
                               y=[st.session_state.model_results[m][metric] for m in st.session_state.model_results.keys()]))
    
    fig.update_layout(title="Model Performance Comparison", barmode='group', height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Best model details
    st.subheader(f"🏆 Best Model: {st.session_state.best_model_name}")
    
    col1, col2, col3 = st.columns(3)
    best_scores = st.session_state.model_results[st.session_state.best_model_name]
    
    with col1:
        if st.session_state.task_type == "Classification":
            st.metric("Accuracy", f"{best_scores['Accuracy']:.4f}")
            st.metric("Precision", f"{best_scores['Precision']:.4f}")
        else:
            st.metric("R² Score", f"{best_scores['R2 Score']:.4f}")
    
    with col2:
        if st.session_state.task_type == "Classification":
            st.metric("Recall", f"{best_scores['Recall']:.4f}")
            st.metric("F1-Score", f"{best_scores['F1-Score']:.4f}")
        else:
            st.metric("RMSE", f"{best_scores['RMSE']:.4f}")
    
    with col3:
        st.metric("CV Mean", f"{best_scores['CV Mean']:.4f}")
        st.metric("CV Std", f"{best_scores['CV Std']:.4f}")
    
    # Generate comprehensive report
    st.subheader("📄 Generate Report")
    
    if st.button("Generate Full Report", type="primary"):
        report = f"""
================================================================================
                    MICROPLASTIC RISK PREDICTION SYSTEM REPORT
================================================================================

Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Researchers: Matthew Joseph Viernes & Shane Mark R. Magdaluyo
Institution: Agusan del Sur State College of Agriculture and Technology (ASSCAT)

================================================================================
1. DATA OVERVIEW
================================================================================

Dataset Shape: {st.session_state.risk_data.shape[0]} rows × {st.session_state.risk_data.shape[1]} columns
Data Source: {st.session_state.data_source}
File Name: {st.session_state.data_filename}

Features:
- Total Features: {len(st.session_state.risk_data.columns)}
- Numeric Features: {len(st.session_state.risk_data.select_dtypes(include=[np.number]).columns)}
- Categorical Features: {len(st.session_state.risk_data.select_dtypes(include=['object']).columns)}

Missing Values: {st.session_state.risk_data.isnull().sum().sum()}
Duplicate Rows: {st.session_state.risk_data.duplicated().sum()}

================================================================================
2. MODEL CONFIGURATION
================================================================================

Target Column: {st.session_state.target_column}
Task Type: {st.session_state.task_type}
Selected Features: {', '.join(st.session_state.selected_features[:10])}{'...' if len(st.session_state.selected_features) > 10 else ''}
Number of Features: {len(st.session_state.selected_features)}

================================================================================
3. MODEL PERFORMANCE
================================================================================

{results_df[display_cols].to_string()}

================================================================================
4. BEST MODEL RESULTS
================================================================================

Best Model: {st.session_state.best_model_name}

Performance Metrics:
"""
        for key, value in best_scores.items():
            report += f"- {key}: {value:.4f}\n"

        report += f"""
================================================================================
5. CONCLUSIONS & RECOMMENDATIONS
================================================================================

1. The {st.session_state.best_model_name} model achieved the best performance with 
   {best_scores[display_cols[0]]:.4f} {display_cols[0]}.

2. The model can be used for predicting microplastic pollution risk levels
   based on the selected environmental and biological features.

3. Recommendations:
   - Regular model retraining with new data
   - Feature engineering for improved predictions
   - Integration with environmental monitoring systems

================================================================================
                            END OF REPORT
================================================================================
"""
        
        st.download_button(
            label="📥 Download Report",
            data=report,
            file_name=f"microplastic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain"
        )
        
        st.success("Report generated successfully!")

# ============================================================================
# MAIN APP ROUTING
# ============================================================================

# Sidebar navigation
st.sidebar.markdown("---")

# Navigation menu
page_options = {
    "🏠 Dashboard": page_home,
    "📁 Data Upload": page_data_upload,
    "📊 EDA": page_eda,
    "🤖 Model Training": page_model_training,
    "📈 Prediction": page_prediction,
    "📊 Results": page_results,
}

# Create navigation buttons in sidebar
st.sidebar.markdown("### 📍 Navigation")
selected_page = st.sidebar.radio("", list(page_options.keys()))

# Display selected page
page_options[selected_page]()

# Footer
st.markdown("""
<div class="footer">
    <p>🌊 Microplastic Pollution Risk Prediction System | ASSCAT 2025 | Viernes, M.J. & Magdaluyo, S.M.R.</p>
    <p>Advanced Data Mining Techniques for Environmental Risk Assessment</p>
</div>
""", unsafe_allow_html=True)
