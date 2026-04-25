"""
Microplastic Risk Analysis Dashboard
A comprehensive Streamlit application for analyzing microplastic risk data,
featuring data preprocessing, EDA, model training, and evaluation.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             classification_report)
from imblearn.over_sampling import SMOTE
import warnings
import io
import base64
from datetime import datetime

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
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

def load_dataset(uploaded_file):
    """Load dataset from uploaded file with encoding fix."""
    try:
        if uploaded_file.name.endswith('.csv'):
            # Try multiple encodings (prevents utf-8 crash)
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
        'Risk_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n_samples),
        'Location': np.random.choice(['Urban', 'Rural', 'Industrial', 'Coastal'], n_samples),
        'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_samples)
    }
    
    # Add some missing values for demonstration
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
        
        # Check missing values
        missing_before = df_clean.isnull().sum().sum()
        
        if missing_before > 0:
            for col in df_clean.columns:
                if df_clean[col].isnull().sum() > 0:
                    if df_clean[col].dtype in ['float64', 'int64']:
                        # Numeric: fill with median
                        median_val = df_clean[col].median()
                        df_clean[col].fillna(median_val, inplace=True)
                        log_messages.append(f"Filled {df_clean[col].isnull().sum()} missing values in '{col}' with median ({median_val:.2f})")
                    else:
                        # Categorical: fill with mode
                        mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
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
        
        # Only scale numeric columns that are not encoded
        numeric_cols = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns
        df_scaled[feature_cols] = scaler.fit_transform(df_scaled[feature_cols])
        
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
                    'percentage': (len(outliers) / len(df)) * 100,
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
        fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
        
        # Histogram
        fig.add_trace(
            go.Histogram(x=data[column], name='Distribution', nbinsx=30,
                        marker_color='#3498db'),
            row=1, col=1
        )
        
        # Box plot
        fig.add_trace(
            go.Box(y=data[column], name='Box Plot', marker_color='#e74c3c'),
            row=1, col=2
        )
        
        fig.update_layout(title_text=title, showlegend=False, height=500)
        return fig
        
    except Exception as e:
        st.error(f"Error creating distribution plot: {str(e)}")
        return go.Figure()


def plot_correlation_heatmap(df, columns):
    """Create correlation heatmap."""
    try:
        corr_matrix = df[columns].corr()
        
        fig = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns.tolist(),
            y=corr_matrix.index.tolist(),
            colorscale='RdBu',
            showscale=True
        )
        
        fig.update_layout(title='Feature Correlation Heatmap', height=600)
        return fig, corr_matrix
        
    except Exception as e:
        st.error(f"Error creating correlation heatmap: {str(e)}")
        return go.Figure(), None


def prepare_modeling_data(df, feature_cols, target_col):
    """Prepare data for modeling."""
    try:
        # Select features and target
        X = df[feature_cols].select_dtypes(include=['float64', 'int64', 'int32'])
        y = df[target_col]
        
        # Encode target if categorical
        if y.dtype == 'object':
            le = LabelEncoder()
            y = le.fit_transform(y)
            st.session_state.target_encoder = le
        
        # Handle any remaining missing values
        if X.isnull().sum().sum() > 0:
            X = X.fillna(X.median())
        
        return X, y
        
    except Exception as e:
        st.error(f"Error preparing modeling data: {str(e)}")
        return None, None


def train_models(X_train, X_test, y_train, y_test):
    """Train classification models."""
    models = {}
    
    try:
        with st.spinner('Training Logistic Regression...'):
            # Logistic Regression with GridSearchCV
            lr_params = {'C': [0.1, 1, 10], 'max_iter': [1000]}
            lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=3, scoring='f1_weighted')
            lr_grid.fit(X_train, y_train)
            models['Logistic Regression'] = lr_grid.best_estimator_
            
        with st.spinner('Training Random Forest...'):
            rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
            rf_model.fit(X_train, y_train)
            models['Random Forest'] = rf_model
            
        with st.spinner('Training Decision Tree...'):
            dt_model = DecisionTreeClassifier(random_state=42, max_depth=10)
            dt_model.fit(X_train, y_train)
            models['Decision Tree'] = dt_model
            
        return models
        
    except Exception as e:
        st.error(f"Error training models: {str(e)}")
        return {}


def evaluate_models(models, X_test, y_test):
    """Evaluate trained models."""
    evaluation_results = {}
    
    try:
        for name, model in models.items():
            y_pred = model.predict(X_test)
            
            evaluation_results[name] = {
                'accuracy': accuracy_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_test, y_pred),
                'classification_report': classification_report(y_test, y_pred)
            }
        
        return evaluation_results
        
    except Exception as e:
        st.error(f"Error evaluating models: {str(e)}")
        return {}


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
                data = load_dataset(uploaded_file)
                st.session_state.data = data
        
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
            
            st.dataframe(df.head(10), use_container_width=True)
            
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
        
        # Display preprocessing steps
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
                    
                    st.info(f"✅ Missing values handled")
            
            if "Encode Categorical Variables" in preprocessing_options:
                st.markdown("### 🔄 Categorical Encoding")
                with st.spinner('Encoding categorical variables...'):
                    processed_df = encode_categorical(processed_df)
                    
                    st.write("**Encoded columns added:**")
                    encoded_cols = [col for col in processed_df.columns if col.endswith('_Encoded')]
                    st.write(encoded_cols)
            
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
        
        df = data_to_use
        
        # Risk Score Analysis
        st.markdown("### 📊 Risk Score Distribution")
        
        if 'Risk_Score' in df.columns:
            fig_dist = plot_distribution(df, 'Risk_Score', 'Risk Score Distribution')
            st.plotly_chart(fig_dist, use_container_width=True)
            
            # Risk Score statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean Risk Score", f"{df['Risk_Score'].mean():.2f}")
            with col2:
                st.metric("Median Risk Score", f"{df['Risk_Score'].median():.2f}")
            with col3:
                st.metric("Max Risk Score", f"{df['Risk_Score'].max():.2f}")
            with col4:
                st.metric("Min Risk Score", f"{df['Risk_Score'].min():.2f}")
        else:
            st.warning("⚠️ 'Risk_Score' column not found in dataset")
        
        # MP Count vs Risk Score
        st.markdown("---")
        st.markdown("### 🔬 MP Count vs Risk Score")
        
        if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
            # Convert to numeric (fix ValueError)
            df['MP_Count_per_L'] = pd.to_numeric(df['MP_Count_per_L'], errors='coerce')
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            
            # Remove invalid rows
            clean_df = df.dropna(subset=['MP_Count_per_L', 'Risk_Score'])
            
            if clean_df.empty:
                st.warning("⚠️ No valid numeric data for plotting.")
            else:
                # Create scatter plot with trendline
                try:
                    fig_scatter = px.scatter(
                        clean_df,
                        x='MP_Count_per_L',
                        y='Risk_Score',
                        color='Risk_Level' if 'Risk_Level' in clean_df.columns else None,
                        trendline='ols',
                        title='Microplastic Count vs Risk Score'
                    )
                except:
                    fig_scatter = px.scatter(
                        clean_df,
                        x='MP_Count_per_L',
                        y='Risk_Score',
                        color='Risk_Level' if 'Risk_Level' in clean_df.columns else None,
                        title='Microplastic Count vs Risk Score'
                    )
                    st.warning("⚠️ Trendline could not be generated (data issue). Showing scatter only.")
                
                st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.warning("⚠️ Required columns not found")
        
        # Risk Level Analysis
        st.markdown("---")
        st.markdown("### 📊 Risk Score by Risk Level")
        
        if 'Risk_Level' in df.columns and 'Risk_Score' in df.columns:
            fig_box = px.box(
                df, x='Risk_Level', y='Risk_Score',
                color='Risk_Level',
                title='Risk Score Distribution by Risk Level'
            )
            fig_box.update_layout(height=500)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.warning("⚠️ Required columns not found")
        
        # Additional EDA
        st.markdown("---")
        st.markdown("### 📈 Additional Analysis")
        
        eda_options = st.multiselect(
            "Select variables to analyze",
            df.columns.tolist(),
            default=['Risk_Score'] if 'Risk_Score' in df.columns else []
        )
        
        if eda_options:
            for col in eda_options:
                if df[col].dtype in ['float64', 'int64']:
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
        
        # Assume Risk_Type is the target for modeling
        target_col = st.selectbox(
            "Select Target Variable",
            df.columns.tolist(),
            index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0
        )
        
        st.markdown("### 🔍 Feature Selection")
        
        # Identify numeric features
        numeric_cols = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        
        # Correlation Analysis
        st.markdown("#### 📊 Correlation Analysis")
        
        if len(numeric_cols) > 1:
            with st.spinner('Computing correlation matrix...'):
                fig_corr, corr_matrix = plot_correlation_heatmap(df, numeric_cols + [target_col] if target_col in df.select_dtypes(include=['float64', 'int64']).columns else numeric_cols)
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("⚠️ Not enough numeric features for correlation analysis")
        
        # Feature Importance using Random Forest
        st.markdown("#### 🌲 Random Forest Feature Importance")
        
        if st.button("Calculate Feature Importance", type="primary"):
            try:
                X = df[numeric_cols]
                y = df[target_col]
                
                # Encode target if needed
                if y.dtype == 'object':
                    le = LabelEncoder()
                    y = le.fit_transform(y)
                
                # Handle missing values
                X = X.fillna(X.median())
                
                # Train Random Forest
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Get feature importance
                importance_df = pd.DataFrame({
                    'feature': numeric_cols,
                    'importance': rf.feature_importances_
                }).sort_values('importance', ascending=True)
                
                st.session_state.feature_importance = importance_df
                
                # Plot
                fig_imp = px.bar(
                    importance_df.tail(15),
                    x='importance',
                    y='feature',
                    orientation='h',
                    title='Top 15 Feature Importance (Random Forest)'
                )
                fig_imp.update_layout(height=500)
                st.plotly_chart(fig_imp, use_container_width=True)
                
                # Select top features
                top_features = importance_df.nlargest(10, 'importance')['feature'].tolist()
                st.session_state.selected_features = top_features
                
                st.success(f"✅ Selected top {len(top_features)} features")
                st.write("**Selected Features:**")
                st.write(top_features)
                
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
        
        # Target selection
        target_col = st.selectbox(
            "Select Target Variable",
            df.columns.tolist(),
            key='train_target'
        )
        
        # Feature selection
        feature_cols = st.multiselect(
            "Select Features",
            [col for col in df.columns if col != target_col],
            default=st.session_state.get('selected_features', 
                   df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:5]
                   if len(df.select_dtypes(include=['float64', 'int64']).columns) > 5 
                   else df.select_dtypes(include=['float64', 'int64']).columns.tolist())
        )
        
        # Model parameters
        st.markdown("### ⚙️ Model Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.number_input("Random State", 0, 100, 42)
        
        with col2:
            use_smote = st.checkbox("Use SMOTE for class imbalance", value=True)
            cv_folds = st.slider("Cross-validation folds", 2, 10, 3)
        
        if st.button("Train Models", type="primary") and len(feature_cols) > 0:
            try:
                # Prepare data
                X, y = prepare_modeling_data(df, feature_cols, target_col)
                
                if X is None or y is None:
                    st.error("Failed to prepare data for modeling")
                    return
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state, stratify=y if use_smote else None
                )
                
                st.info(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")
                
                # Apply SMOTE
                if use_smote:
                    with st.spinner('Applying SMOTE for class balancing...'):
                        smote = SMOTE(random_state=random_state)
                        X_train, y_train = smote.fit_resample(X_train, y_train)
                        st.success(f"✅ After SMOTE - Training set size: {X_train.shape[0]}")
                
                # Train models
                models = train_models(X_train, X_test, y_train, y_test)
                
                if models:
                    st.session_state.models = models
                    st.session_state.X_test = X_test
                    st.session_state.y_test = y_test
                    st.session_state.trained = True
                    
                    st.success(f"✅ Successfully trained {len(models)} models!")
                    
                    # Quick preview of results
                    st.markdown("### 📊 Quick Performance Overview")
                    for name, model in models.items():
                        train_score = model.score(X_train, y_train)
                        test_score = model.score(X_test, y_test)
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric(f"{name} - Train Score", f"{train_score:.3f}")
                        with col2:
                            st.metric(f"{name} - Test Score", f"{test_score:.3f}")
            
            except Exception as e:
                st.error(f"Error during model training: {str(e)}")
    
    # Section: Model Evaluation
    elif section == "📊 Model Evaluation":
        st.markdown('<p class="section-header">📊 Model Evaluation</p>', unsafe_allow_html=True)
        
        if not st.session_state.get('trained', False):
            st.warning("⚠️ Please train models first!")
            return
        
        models = st.session_state.models
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        st.markdown("### 🎯 Model Performance Comparison")
        
        with st.spinner('Evaluating models...'):
            evaluation_results = evaluate_models(models, X_test, y_test)
        
        if evaluation_results:
            # Performance metrics comparison
            metrics_df = pd.DataFrame({
                name: {
                    'Accuracy': results['accuracy'],
                    'F1 Score': results['f1_score']
                }
                for name, results in evaluation_results.items()
            }).T
            
            st.markdown("#### 📈 Performance Metrics")
            
            # Bar chart comparison
            fig_metrics = px.bar(
                metrics_df.reset_index(),
                x='index',
                y=['Accuracy', 'F1 Score'],
                barmode='group',
                title='Model Performance Comparison',
                labels={'index': 'Model', 'value': 'Score'}
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
            
            # Display metrics table
            st.dataframe(metrics_df.style.highlight_max(axis=0), use_container_width=True)
            
            # Confusion Matrices
            st.markdown("---")
            st.markdown("#### 📊 Confusion Matrices")
            
            for name, results in evaluation_results.items():
                st.markdown(f"**{name}**")
                
                fig_cm = ff.create_annotated_heatmap(
                    z=results['confusion_matrix'],
                    x=[f'Class {i}' for i in range(len(results['confusion_matrix']))],
                    y=[f'Class {i}' for i in range(len(results['confusion_matrix']))],
                    colorscale='Blues',
                    showscale=True
                )
                fig_cm.update_layout(title=f'{name} - Confusion Matrix')
                st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification Reports
            st.markdown("---")
            st.markdown("#### 📋 Classification Reports")
            
            model_for_report = st.selectbox(
                "Select model for detailed report",
                list(evaluation_results.keys())
            )
            
            st.text("Classification Report:")
            st.text(evaluation_results[model_for_report]['classification_report'])
            
            # Select best model
            best_model_name = metrics_df['F1 Score'].idxmax()
            st.session_state.best_model = {
                'name': best_model_name,
                'model': models[best_model_name],
                'metrics': evaluation_results[best_model_name]
            }
            
            st.markdown("---")
            st.success(f"🏆 Best Performing Model: **{best_model_name}**")
            st.metric("Best F1 Score", f"{metrics_df.loc[best_model_name, 'F1 Score']:.3f}")
    
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
            st.dataframe(importance_df.style.background_gradient(cmap='Blues'), use_container_width=True)
            
            # Interpretation
            st.markdown("---")
            st.markdown("### 💡 Feature Importance Interpretation")
            
            top_3 = importance_df.nlargest(3, 'importance')
            
            for idx, row in top_3.iterrows():
                st.markdown(f"**{idx+1}. {row['feature']}** (Importance: {row['importance']:.3f})")
                st.markdown(f"   - This feature has strong predictive power for the target variable")
                st.markdown(f"   - Consider investigating its relationship with microplastic risk")
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
                fig_bar = px.bar(
                    x=polymer_counts.index,
                    y=polymer_counts.values,
                    title='Polymer Type Distribution',
                    labels={'x': 'Polymer Type', 'y': 'Count'},
                    color=polymer_counts.values,
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_bar, use_container_width=True)
            
            with col2:
                st.markdown("#### 🥧 Pie Chart")
                fig_pie = px.pie(
                    values=polymer_counts.values,
                    names=polymer_counts.index,
                    title='Polymer Type Distribution'
                )
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # Polymer statistics
            st.markdown("---")
            st.markdown("### 📊 Polymer Statistics")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Polymer Types", len(polymer_counts))
            with col2:
                st.metric("Most Common", polymer_counts.index[0])
            with col3:
                st.metric("Most Common Count", polymer_counts.values[0])
            
            # Cross-analysis
            st.markdown("---")
            st.markdown("### 🔬 Polymer Type vs Risk Level")
            
            if 'Risk_Level' in df.columns:
                fig_cross = px.histogram(
                    df, x='Polymer_Type', color='Risk_Level',
                    title='Polymer Type Distribution by Risk Level',
                    barmode='group'
                )
                fig_cross.update_layout(height=500)
                st.plotly_chart(fig_cross, use_container_width=True)
        else:
            st.warning("⚠️ 'Polymer_Type' column not found in dataset")
            
            # Display available columns
            st.write("Available columns in dataset:")
            st.write(df.columns.tolist())


if __name__ == "__main__":
    main()
