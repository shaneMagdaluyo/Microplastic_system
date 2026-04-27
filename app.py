"""
Microplastic Risk Analysis Dashboard
A comprehensive Streamlit application for analyzing microplastic risk data,
featuring data preprocessing, EDA, model training, and cross validation.
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
    .success-message {
        color: #27ae60;
        font-weight: 600;
    }
    .warning-message {
        color: #e67e22;
        font-weight: 600;
    }
    .error-message {
        color: #e74c3c;
        font-weight: 600;
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
    if 'scaled_columns' not in st.session_state:
        st.session_state.scaled_columns = None
    if 'scaled_data' not in st.session_state:
        st.session_state.scaled_data = None

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
    try:
        df_clean = df.copy()
        log_messages = []
        missing_before = df_clean.isnull().sum().sum()
        if missing_before > 0:
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    if df_clean[col].dtype in ['float64', 'int64']:
                        median_val = df_clean[col].median()
                        if pd.isna(median_val):
                            median_val = 0
                        df_clean[col].fillna(median_val, inplace=True)
                        log_messages.append(f"Filled missing values in '{col}' with median ({median_val:.2f})")
                    else:
                        mode_series = df_clean[col].mode()
                        mode_val = mode_series[0] if not mode_series.empty else 'Unknown'
                        df_clean[col].fillna(mode_val, inplace=True)
                        log_messages.append(f"Filled missing values in '{col}' with mode ({mode_val})")
        log_messages.append(f"Total missing values handled: {missing_before}")
        st.session_state.preprocessing_log = log_messages
        return df_clean
    except Exception as e:
        st.error(f"Error handling missing values: {str(e)}")
        return df

def encode_categorical(df):
    """Encode categorical variables."""
    try:
        df_encoded = df.copy()
        encoders = {}
        categorical_cols = df_encoded.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col not in ['Sample_ID']:
                le = LabelEncoder()
                df_encoded[f'{col}_Encoded'] = le.fit_transform(df_encoded[col].astype(str))
                encoders[col] = le
        st.session_state.encoders = encoders
        return df_encoded
    except Exception as e:
        st.error(f"Error encoding categorical variables: {str(e)}")
        return df

def scale_features(df, feature_cols):
    """Scale numerical features."""
    try:
        df_scaled = df.copy()
        scaler = StandardScaler()
        numeric_cols = df_scaled[feature_cols].select_dtypes(include=['float64', 'int64']).columns
        if len(numeric_cols) > 0:
            df_scaled[numeric_cols] = scaler.fit_transform(df_scaled[numeric_cols])
            st.session_state.scaler = scaler
        return df_scaled
    except Exception as e:
        st.error(f"Error scaling features: {str(e)}")
        return df

def detect_outliers(df, columns):
    """Detect outliers using IQR method."""
    try:
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
    except Exception as e:
        st.error(f"Error detecting outliers: {str(e)}")
        return {}

def plot_distribution(data, column, title):
    """Create distribution plot."""
    try:
        clean_data = data[column].dropna()
        if clean_data.empty:
            return go.Figure()
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
        fig.add_trace(go.Histogram(x=clean_data, name='Distribution', nbinsx=30, marker_color='#3498db'), row=1, col=1)
        fig.add_trace(go.Box(y=clean_data, name='Box Plot', marker_color='#e74c3c'), row=1, col=2)
        fig.update_layout(title_text=title, showlegend=False, height=500)
        return fig
    except Exception as e:
        st.error(f"Error creating distribution plot: {str(e)}")
        return go.Figure()

def plot_correlation_heatmap(df, columns):
    """Create correlation heatmap."""
    try:
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
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}',
            textfont={"size": 10},
            showscale=True
        ))
        fig.update_layout(title='Feature Correlation Heatmap', height=600)
        return fig, corr_matrix
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        return go.Figure(), None

def prepare_modeling_data(df, feature_cols, target_col):
    """Prepare data for modeling with enhanced error handling."""
    try:
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
    except Exception as e:
        st.error(f"Error preparing modeling data: {str(e)}")
        return None, None

def train_models_fast(X_train, X_test, y_train, y_test):
    """Train classification models with optimized fast performance."""
    models = {}
    training_times = {}
    try:
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
                                              class_weight='balanced', max_depth=min(12, n_samples // 30), n_jobs=-1)
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
    except Exception as e:
        st.error(f"Error in model training pipeline: {str(e)}")
        return {}, {}

def train_models_quality(X_train, X_test, y_train, y_test):
    """Train classification models with GridSearch for better quality."""
    models = {}
    training_times = {}
    try:
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
    except Exception as e:
        st.error(f"Error in model training pipeline: {str(e)}")
        return {}, {}

def evaluate_models(models, X_test, y_test):
    """Evaluate trained models."""
    evaluation_results = {}
    try:
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                evaluation_results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred)
                }
            except Exception as e:
                st.error(f"Error evaluating {name}: {str(e)}")
        return evaluation_results
    except Exception as e:
        st.error(f"Error evaluating models: {str(e)}")
        return {}

def main():
    """Main application function."""
    
    # Main header
    st.markdown('<p class="main-header">🔬 Microplastic Risk Analysis Dashboard</p>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation - 5 SECTIONS
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio(
        "Select Section",
        ["🏠 Home", "🔧 Preprocessing", "🛠️ Feature Selection & Relevance", 
         "🤖 Modeling", "📊 Cross Validation"]
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
    
    # ==================== HOME ====================
    if section == "🏠 Home":
        st.markdown('<p class="section-header">🏠 Home - Upload Dataset</p>', unsafe_allow_html=True)
        
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
            df = st.session_state.data
            
            st.markdown("---")
            st.markdown('<p class="subsection-header">📋 Dataset Preview</p>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Number of Samples", df.shape[0])
            with col2:
                st.metric("Number of Features", df.shape[1])
            with col3:
                st.metric("Missing Values", df.isnull().sum().sum())
            
            st.dataframe(df.head(10), use_container_width=True)
            
            # ===== FEATURE SCALING PREVIEW =====
            st.markdown("---")
            st.markdown("### 📏 Feature Scaling Preview")
            st.markdown("*Apply StandardScaler to numerical columns*")
            
            if st.button("🔧 Apply Feature Scaling (StandardScaler)", type="primary"):
                with st.spinner('Applying StandardScaler to numerical columns...'):
                    
                    # Select numerical columns
                    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                    
                    # Remove ID columns if present
                    cols_to_scale = [col for col in numeric_cols if 'ID' not in col and 'Sample' not in col]
                    
                    if len(cols_to_scale) > 0:
                        # Apply StandardScaler
                        scaler = StandardScaler()
                        scaled_data = scaler.fit_transform(df[cols_to_scale].fillna(df[cols_to_scale].median()))
                        scaled_df = pd.DataFrame(scaled_data, columns=cols_to_scale)
                        
                        # Store in session state
                        st.session_state.scaler = scaler
                        st.session_state.scaled_columns = cols_to_scale
                        st.session_state.scaled_data = scaled_df
                        
                        st.success(f"✅ Feature scaling applied to {len(cols_to_scale)} numerical columns!")
                        
                        # Display first 5 rows
                        st.markdown("**First 5 rows of scaled numerical data:**")
                        st.dataframe(
                            scaled_df.head(),
                            column_config={
                                col: st.column_config.NumberColumn(col, format="%.6f") 
                                for col in cols_to_scale
                            },
                            use_container_width=True,
                        )
                        
                        # Show scaling statistics
                        with st.expander("📊 Scaling Statistics (Before vs After)"):
                            stats_list = []
                            for col in cols_to_scale[:10]:  # Limit to 10 columns
                                stats_list.append({
                                    'Column': col,
                                    'Mean (Before)': f"{df[col].mean():.4f}",
                                    'Std (Before)': f"{df[col].std():.4f}",
                                    'Mean (After)': f"{scaled_df[col].mean():.6f}",
                                    'Std (After)': f"{scaled_df[col].std():.6f}",
                                })
                            st.dataframe(pd.DataFrame(stats_list), use_container_width=True, hide_index=True)
                    else:
                        st.warning("⚠️ No numerical columns found to scale.")
            
            # ===== DATASET INFORMATION =====
            st.markdown("---")
            st.markdown("#### Dataset Information")
            
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Data Types:**")
                st.write(df.dtypes)
            with col2:
                st.write("**Basic Statistics:**")
                st.write(df.describe())
            
            # ===== QUICK DATA QUALITY CHECK =====
            st.markdown("---")
            st.markdown("#### 🔍 Quick Data Quality Check")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                st.metric("Missing Data %", f"{missing_pct:.2f}%")
            with col2:
                duplicate_rows = df.duplicated().sum()
                st.metric("Duplicate Rows", duplicate_rows)
            with col3:
                num_cols_count = len(df.select_dtypes(include=['float64', 'int64']).columns)
                st.metric("Numeric Columns", num_cols_count)
            with col4:
                cat_cols_count = len(df.select_dtypes(include=['object']).columns)
                st.metric("Categorical Columns", cat_cols_count)
    
    # ==================== PREPROCESSING ====================
    elif section == "🔧 Preprocessing":
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
                    missing_before = processed_df.isnull().sum()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Missing values before:**")
                        st.write(missing_before[missing_before > 0])
                    processed_df = handle_missing_values(processed_df)
                    with col2:
                        st.write("**Missing values after:**")
                        st.write(processed_df.isnull().sum())
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
    
    # ==================== FEATURE SELECTION & RELEVANCE ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        
        data_to_use = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        if data_to_use is None:
            st.warning("⚠️ Please load and preprocess data first!")
            return
        
        df = data_to_use.copy()
        
        # EDA Section
        st.markdown("### 📈 Exploratory Data Analysis")
        
        st.markdown("#### 📊 Risk Score Distribution")
        if 'Risk_Score' in df.columns:
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean_risk = df['Risk_Score'].dropna()
            if len(clean_risk) > 0:
                fig_dist = plot_distribution(df, 'Risk_Score', 'Risk Score Distribution')
                st.plotly_chart(fig_dist, use_container_width=True)
                
                # Enhanced Statistics
                q1 = clean_risk.quantile(0.25)
                q3 = clean_risk.quantile(0.75)
                iqr = q3 - q1
                skewness = clean_risk.skew()
                kurtosis = clean_risk.kurtosis()
                
                low_risk = (clean_risk < 25).sum()
                medium_risk = ((clean_risk >= 25) & (clean_risk < 50)).sum()
                high_risk = ((clean_risk >= 50) & (clean_risk < 75)).sum()
                critical_risk = (clean_risk >= 75).sum()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**📈 Descriptive Statistics**")
                    stats_data = [
                        ('Count', f'{len(clean_risk):,}'),
                        ('Mean', f'{clean_risk.mean():.4f}'),
                        ('Median', f'{clean_risk.median():.4f}'),
                        ('Std Dev', f'{clean_risk.std():.4f}'),
                        ('Min', f'{clean_risk.min():.4f}'),
                        ('25% (Q1)', f'{q1:.4f}'),
                        ('75% (Q3)', f'{q3:.4f}'),
                        ('IQR', f'{iqr:.4f}'),
                        ('Max', f'{clean_risk.max():.4f}'),
                        ('Range', f'{clean_risk.max() - clean_risk.min():.4f}'),
                        ('Skewness', f'{skewness:.4f}'),
                        ('Kurtosis', f'{kurtosis:.4f}'),
                    ]
                    stats_df = pd.DataFrame(stats_data, columns=['Statistic', 'Value'])
                    st.dataframe(stats_df, use_container_width=True, hide_index=True)
                
                with col2:
                    st.markdown("**🎯 Risk Category Distribution**")
                    categories = [
                        ('🟢 Low Risk', '0 - 25', low_risk, (low_risk/len(clean_risk))*100, '#27ae60'),
                        ('🟡 Medium Risk', '25 - 50', medium_risk, (medium_risk/len(clean_risk))*100, '#f39c12'),
                        ('🟠 High Risk', '50 - 75', high_risk, (high_risk/len(clean_risk))*100, '#e67e22'),
                        ('🔴 Critical Risk', '75 - 100', critical_risk, (critical_risk/len(clean_risk))*100, '#e74c3c'),
                    ]
                    for cat, rng, count, pct, color in categories:
                        st.markdown(f"**{cat}** ({rng}): {count:,} ({pct:.1f}%)")
                        st.progress(int(pct))
        
        st.markdown("---")
        st.markdown("#### 🔬 MP Count vs Risk Score")
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
        
        st.markdown("---")
        st.markdown("#### 📊 Risk Score by Risk Level")
        if 'Risk_Level' in df.columns and 'Risk_Score' in df.columns:
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean_df = df.dropna(subset=['Risk_Score'])
            if len(clean_df) > 0:
                fig_box = px.box(clean_df, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                                title='Risk Score Distribution by Risk Level')
                fig_box.update_layout(height=500)
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Risk Score by Risk Level table
                st.markdown("**📊 Risk Score Statistics by Risk Level**")
                risk_level_stats = df.groupby('Risk_Level')['Risk_Score'].agg([
                    'count', 'mean', 'median', 'std', 'min', 'max'
                ]).round(2)
                risk_level_stats.columns = ['Count', 'Mean', 'Median', 'Std Dev', 'Min', 'Max']
                st.dataframe(risk_level_stats, use_container_width=True)
        
        # Feature Engineering Section
        st.markdown("---")
        st.markdown("### 🎯 Feature Engineering")
        
        target_col = st.selectbox("Select Target Variable", df.columns.tolist(),
                                  index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        
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
    
    # ==================== MODELING ====================
    elif section == "🤖 Modeling":
        st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)
        
        if st.session_state.get('trained', False) and len(st.session_state.get('models', {})) > 0:
            st.success(f"✅ Models are already trained! ({len(st.session_state.models)} models available)")
        
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
            fast_mode = st.checkbox("⚡ Fast Training Mode", value=True, 
                                   help="Enable for faster training (recommended)")
            if fast_mode:
                st.success("⚡ Fast mode enabled: Training will be optimized for speed")
            else:
                st.info("🔬 Quality mode: Training will use GridSearchCV for better results (slower)")
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if len(feature_cols) == 0:
                st.error("❌ Please select at least one feature!")
                return
            
            try:
                X, y = prepare_modeling_data(df, feature_cols, target_col)
                if X is None or y is None:
                    return
                
                class_counts = pd.Series(y).value_counts()
                st.info("### 📊 Class Distribution")
                st.write(class_counts)
                
                use_stratify = len(class_counts) > 1 and class_counts.min() >= 2
                try:
                    if use_stratify:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state, stratify=y)
                        st.success("✅ Data split with stratification")
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=test_size, random_state=random_state)
                        st.info("ℹ️ Data split without stratification")
                except:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=random_state)
                
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
                    
                    # ==================== MODEL PERFORMANCE RESULTS ====================
                    st.markdown("---")
                    st.markdown("## 📊 Model Performance Results")
                    
                    eval_results = evaluate_models(models, X_test, y_test)
                    
                    if eval_results:
                        # Calculate averages
                        all_acc = [r['accuracy'] for r in eval_results.values()]
                        all_f1 = [r['f1_score'] for r in eval_results.values()]
                        avg_acc = np.mean(all_acc)
                        avg_f1 = np.mean(all_f1)
                        
                        # Average Score Banner
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #1f77b4, #2c3e50); 
                                    padding: 25px; border-radius: 15px; margin: 20px 0; text-align: center;">
                            <h2 style="color: white; margin: 0;">📊 Average Model Performance</h2>
                            <div style="display: flex; justify-content: center; gap: 40px; margin-top: 15px;">
                                <div>
                                    <p style="color: #ffd700; margin: 0; font-size: 1rem;">Average Accuracy</p>
                                    <p style="color: white; font-size: 2.5rem; font-weight: bold; margin: 5px 0;">{avg_acc:.4f}</p>
                                    <p style="color: #ccc; font-size: 0.9rem;">({avg_acc*100:.1f}%)</p>
                                </div>
                                <div style="border-left: 2px solid rgba(255,255,255,0.3); padding-left: 40px;">
                                    <p style="color: #ffd700; margin: 0; font-size: 1rem;">Average F1 Score</p>
                                    <p style="color: white; font-size: 2.5rem; font-weight: bold; margin: 5px 0;">{avg_f1:.4f}</p>
                                    <p style="color: #ccc; font-size: 0.9rem;">({avg_f1*100:.1f}%)</p>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        target_name = target_col
                        st.markdown(f"**From your evaluation for {target_name} prediction:**")
                        st.markdown("")
                        
                        for name, results in eval_results.items():
                            f1 = results['f1_score']
                            acc = results['accuracy']
                            st.markdown(f"**{name}:** F1-Score = **{f1:.4f}** (weighted average) | Accuracy = **{acc:.4f}**")
                            st.markdown("")
                        
                        best_model_name = max(eval_results.items(), key=lambda x: x[1]['f1_score'])[0]
                        best_f1 = max(eval_results.items(), key=lambda x: x[1]['f1_score'])[1]['f1_score']
                        
                        st.markdown("---")
                        st.markdown(f"""
                        <div style="background: #d4edda; border: 2px solid #27ae60; border-radius: 10px; padding: 20px; margin: 15px 0;">
                            <p style="font-size: 1.1rem; margin: 0; color: #155724;">
                                ✅ The <b>{best_model_name}</b> performed best for <b>{target_name}</b> prediction 
                                with an F1-Score of <b>{best_f1:.4f}</b> (weighted average).
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Performance Comparison Table with Average
                        st.markdown("### 📊 Performance Comparison Table")
                        summary_data = []
                        for name, results in eval_results.items():
                            summary_data.append({
                                'Model': name,
                                'Accuracy': results['accuracy'],
                                'F1-Score': results['f1_score']
                            })
                        summary_data.append({'Model': '📊 AVERAGE', 'Accuracy': avg_acc, 'F1-Score': avg_f1})
                        st.dataframe(pd.DataFrame(summary_data), 
                                    column_config={
                                        "Model": "Model",
                                        "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.4f"),
                                        "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.4f"),
                                    }, use_container_width=True, hide_index=True)
                        
                        # Performance Chart
                        st.markdown("### 📈 Performance Metrics Comparison")
                        metrics_dict = {}
                        for name, results in eval_results.items():
                            metrics_dict[name] = {
                                'Accuracy': results['accuracy'],
                                'F1 Score': results['f1_score']
                            }
                        metrics_df = pd.DataFrame(metrics_dict).T
                        
                        fig_metrics = px.bar(
                            metrics_df.reset_index(),
                            x='index',
                            y=['Accuracy', 'F1 Score'],
                            barmode='group',
                            title='Model Performance Comparison',
                            labels={'index': 'Model', 'value': 'Score'},
                            color_discrete_sequence=['#3498db', '#e74c3c']
                        )
                        fig_metrics.add_hline(y=avg_acc, line_dash="dash", line_color="#3498db", 
                                             annotation_text=f"Avg Acc: {avg_acc:.3f}")
                        fig_metrics.add_hline(y=avg_f1, line_dash="dash", line_color="#e74c3c", 
                                             annotation_text=f"Avg F1: {avg_f1:.3f}")
                        fig_metrics.update_layout(height=400)
                        st.plotly_chart(fig_metrics, use_container_width=True)
                        
                        # Confusion Matrix
                        st.markdown("---")
                        st.markdown("### 🧩 Confusion Matrix")
                        selected_model = st.selectbox("Select model to view Confusion Matrix", 
                                                     list(eval_results.keys()), key='cm_model')
                        
                        if selected_model:
                            cm = eval_results[selected_model].get('confusion_matrix')
                            if cm is not None and cm.size > 0:
                                n_classes = cm.shape[0]
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
                        
                        # Classification Report
                        st.markdown("---")
                        st.markdown("### 📋 Classification Report")
                        report_model = st.selectbox("Select model for detailed report", 
                                                   list(eval_results.keys()), key='report_model')
                        if report_model:
                            st.code(eval_results[report_model]['classification_report'])
                        
                        # Final Summary
                        st.success(f"🏆 Best: **{best_model_name}** | 📊 Avg Acc: **{avg_acc:.4f}** | 📊 Avg F1: **{avg_f1:.4f}**")
                        
                else:
                    st.error("❌ No models were successfully trained. Please check your data and try again.")
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
                st.info("💡 Tip: Try reducing the number of features or use a different target variable.")
    
    # ==================== CROSS VALIDATION ====================
    elif section == "📊 Cross Validation":
        st.markdown('<p class="section-header">📊 Cross Validation</p>', unsafe_allow_html=True)
        
        data_to_use = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        
        if data_to_use is None:
            st.warning("⚠️ Please load and preprocess data first!")
            return
        
        df = data_to_use
        
        st.markdown("### 🔄 Cross Validation Analysis")
        st.info("Evaluate model stability using stratified k-fold cross-validation.")
        
        target_col = st.selectbox(
            "Select Target Variable for CV",
            df.columns.tolist(),
            index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0
        )
        
        feature_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        if target_col in feature_cols:
            feature_cols.remove(target_col)
        
        cv_folds = st.slider("Number of CV Folds", 3, 10, 5)
        
        if st.button("🔄 Run Cross Validation", type="primary", use_container_width=True):
            with st.spinner('Running cross-validation...'):
                
                X = df[feature_cols].copy()
                y = df[target_col].copy()
                
                mask = y.notna()
                X = X[mask]
                y = y[mask]
                
                if y.dtype == 'object':
                    y = LabelEncoder().fit_transform(y)
                
                X = X.fillna(X.median())
                
                cv_models = {
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', n_jobs=-1),
                    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1),
                    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8, class_weight='balanced')
                }
                
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
                
                cv_results = []
                all_scores = {}
                
                for name, model in cv_models.items():
                    try:
                        acc_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
                        
                        all_scores[name] = f1_scores
                        
                        cv_results.append({
                            'Model': name,
                            'Mean Accuracy': round(acc_scores.mean(), 4),
                            'Std Accuracy': round(acc_scores.std(), 4),
                            'Mean F1 Score': round(f1_scores.mean(), 4),
                            'Std F1 Score': round(f1_scores.std(), 4),
                            'Min F1': round(f1_scores.min(), 4),
                            'Max F1': round(f1_scores.max(), 4)
                        })
                    except Exception as e:
                        st.error(f"Error with {name}: {str(e)}")
                
                if cv_results:
                    cv_df = pd.DataFrame(cv_results)
                    
                    st.markdown("#### 📊 Cross Validation Results")
                    st.dataframe(cv_df, use_container_width=True, hide_index=True)
                    
                    best_cv = cv_df.loc[cv_df['Mean F1 Score'].idxmax()]
                    st.success(f"""
                    🏆 **Best CV Model:** {best_cv['Model']}
                    - Mean F1 Score: **{best_cv['Mean F1 Score']:.4f}** (±{best_cv['Std F1 Score']:.4f})
                    - Mean Accuracy: **{best_cv['Mean Accuracy']:.4f}** (±{best_cv['Std Accuracy']:.4f})
                    """)
                    
                    st.markdown("#### 📈 Cross Validation F1 Scores Distribution")
                    fig_cv = go.Figure()
                    for name, scores in all_scores.items():
                        fig_cv.add_trace(go.Box(y=scores, name=name, boxmean='sd'))
                    fig_cv.update_layout(
                        title=f'Cross Validation F1 Scores ({cv_folds}-Fold Stratified)',
                        yaxis_title='F1 Score (Weighted)',
                        height=450
                    )
                    st.plotly_chart(fig_cv, use_container_width=True)


if __name__ == "__main__":
    main()
