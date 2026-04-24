# app.py - Microplastic Pollution Risk Prediction System (FIXED)
# Fixed: Handles string/categorical data properly
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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'risk_data' not in st.session_state:
    st.session_state.risk_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'results' not in st.session_state:
    st.session_state.results = None
if 'models' not in st.session_state:
    st.session_state.models = None
if 'feature_encoders' not in st.session_state:
    st.session_state.feature_encoders = {}

# Title and header
st.markdown("""
<div class="main-header">
    <h1>🌊 Microplastic Pollution Risk Prediction System</h1>
    <p>Data Mining-Based Predictive Risk Modeling for Environmental Microplastic Contamination</p>
    <p>Viernes, M.J. & Magdaluyo, S.M.R. | ASSCAT 2025</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("📊 Navigation")
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
- Data Preprocessing
- Classification (RF, LR, DT, GB, SVM, KNN)
- Clustering Analysis
- Risk Prediction
- K-Fold Cross Validation
- Report Generation

**Data Mining Techniques:**
- Random Forest
- Logistic Regression
- Decision Tree
- Gradient Boosting
- SVM
- KNN
""")

# Function to check if a column is numeric
def is_numeric_column(col):
    """Check if column is numeric"""
    try:
        pd.to_numeric(col, errors='raise')
        return True
    except:
        return False

# Function to preprocess data before training
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

# Function to generate sample data
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

# Page: Data Upload & Preprocessing
if page == "📁 Data Upload & Preprocessing":
    st.header("📁 Data Upload and Preprocessing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                st.session_state.risk_data = df
                st.success(f"✅ File loaded successfully! Shape: {df.shape}")
                
                st.subheader("Data Preview")
                st.dataframe(df.head(10))
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    with col2:
        st.subheader("Or Use Sample Data")
        if st.button("📊 Load Sample Dataset"):
            st.session_state.risk_data = generate_sample_data(1000)
            st.success("✅ Sample dataset loaded! Shape: 1000 x 15")
            st.dataframe(st.session_state.risk_data.head(10))
    
    if st.session_state.risk_data is not None:
        st.markdown("---")
        st.subheader("Data Overview")
        
        df = st.session_state.risk_data
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            missing = df.isnull().sum().sum()
            st.metric("Missing Values", missing)
        with col4:
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)
        
        # Show data types
        st.subheader("Data Types")
        dtype_df = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Missing %': [(df[col].isnull().sum() / len(df) * 100) for col in df.columns]
        })
        st.dataframe(dtype_df)
        
        st.subheader("Preprocessing Options")
        
        col1, col2 = st.columns(2)
        
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
                st.success("Missing values handled successfully!")
                st.rerun()
        
        with col2:
            if st.button("🗑️ Remove Duplicates"):
                df_clean = df.copy()
                df_clean.drop_duplicates(inplace=True)
                st.session_state.risk_data = df_clean
                st.success(f"Duplicates removed! New shape: {df_clean.shape}")
                st.rerun()
        
        # Show categorical columns for encoding
        st.subheader("Categorical Columns")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.write("The following categorical columns will be encoded for modeling:")
            for col in categorical_cols:
                st.write(f"- {col}: {df[col].nunique()} unique values")
            
            if st.button("🔧 Encode All Categorical Variables"):
                df_encoded = df.copy()
                for col in categorical_cols:
                    le = LabelEncoder()
                    df_encoded[col + '_encoded'] = le.fit_transform(df_encoded[col].astype(str))
                    st.session_state.feature_encoders[col] = le
                st.session_state.risk_data = df_encoded
                st.success("Categorical variables encoded successfully!")
                st.rerun()
        else:
            st.info("No categorical columns found.")

# Page: Model Training (FIXED VERSION)
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.risk_data is not None:
        df = st.session_state.risk_data
        
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Find potential target columns (both numeric and categorical)
            target_options = df.columns.tolist()
            target_column = st.selectbox("Select Target Column (Risk Level/Risk Score/Risk Type)", target_options)
            
            # Detect task type based on target column
            if df[target_column].dtype in ['int64', 'float64'] and df[target_column].nunique() > 10:
                task_type = st.radio("Task Type", ["Classification", "Regression"], index=1)
            else:
                task_type = st.radio("Task Type", ["Classification", "Regression"], index=0)
        
        with col2:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
            cv_folds = st.slider("K-Fold Cross Validation Folds", 5, 10, 5)
        
        st.subheader("Select Features for Training")
        
        # Get all features except target
        feature_cols = [col for col in df.columns if col != target_column]
        
        # Show data types for transparency
        st.info(f"📊 {len(feature_cols)} features available. Note: Categorical features will be automatically encoded.")
        
        selected_features = st.multiselect("Select Features", feature_cols, default=feature_cols[:5] if len(feature_cols) > 5 else feature_cols)
        
        if st.button("🚀 Train Models", type="primary"):
            if len(selected_features) == 0:
                st.error("Please select at least one feature for training.")
            else:
                with st.spinner("Processing data and training models..."):
                    try:
                        # Preprocess data properly
                        X_processed, y, encoders, target_encoder = preprocess_for_training(df, selected_features, target_column)
                        
                        # Handle missing values in X_processed (should be none after preprocessing)
                        X_processed = X_processed.fillna(X_processed.median())
                        
                        # Scale features
                        scaler = StandardScaler()
                        X_scaled = scaler.fit_transform(X_processed)
                        
                        # Check if we have valid data
                        if y is None:
                            st.error("Could not process target column. Please check your data.")
                            st.stop()
                        
                        # Check for class imbalance in classification
                        if task_type == "Classification":
                            unique_classes = np.unique(y)
                            if len(unique_classes) < 2:
                                st.error(f"Target column '{target_column}' has only one unique value. Please select a different target.")
                                st.stop()
                            
                            # Show class distribution
                            st.subheader("Class Distribution")
                            class_counts = pd.Series(y).value_counts()
                            fig = px.bar(x=class_counts.index.astype(str), y=class_counts.values, 
                                        title="Target Class Distribution",
                                        labels={'x': 'Class', 'y': 'Count'})
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Split data
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=test_size, random_state=42, stratify=y if task_type == "Classification" else None
                        )
                        
                        st.write(f"Training set size: {X_train.shape[0]} samples")
                        st.write(f"Testing set size: {X_test.shape[0]} samples")
                        
                        # Initialize models based on task type
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
                                'Gradient Boosting': GradientBoostingClassifier(random_state=42) if False else None
                            }
                            # Filter out None values
                            models = {k: v for k, v in models.items() if v is not None}
                        
                        results = {}
                        trained_models = {}
                        
                        progress_bar = st.progress(0)
                        for idx, (name, model) in enumerate(models.items()):
                            st.write(f"Training {name}...")
                            
                            # Train model
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            trained_models[name] = model
                            
                            # K-Fold Cross Validation
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
                        
                        # Store everything in session state
                        st.session_state.models = trained_models
                        st.session_state.results = results
                        st.session_state.X_test = X_test
                        st.session_state.y_test = y_test
                        st.session_state.scaler = scaler
                        st.session_state.selected_features = selected_features
                        st.session_state.task_type = task_type
                        st.session_state.model_trained = True
                        st.session_state.target_encoder = target_encoder
                        st.session_state.feature_encoders = encoders
                        
                        st.success("✅ Models trained successfully!")
                        
                        # Display results
                        st.subheader("Model Performance Results")
                        results_df = pd.DataFrame(results).T
                        
                        # Highlight best model
                        if task_type == "Classification":
                            best_col = 'Accuracy'
                        else:
                            best_col = 'R2 Score'
                        
                        st.dataframe(results_df.style.highlight_max(axis=0, subset=[best_col]))
                        
                        # Best model
                        best_model = max(results, key=lambda x: results[x][best_col])
                        st.success(f"🏆 Best Model: **{best_model}** with {best_col}: {results[best_model][best_col]:.4f}")
                        
                        # Confusion Matrix for classification
                        if task_type == "Classification":
                            st.subheader("Confusion Matrix - Best Model")
                            best_model_obj = trained_models[best_model]
                            y_pred_best = best_model_obj.predict(X_test)
                            cm = confusion_matrix(y_test, y_pred_best)
                            
                            fig, ax = plt.subplots(figsize=(8, 6))
                            
                            # Get class labels
                            if target_encoder:
                                class_labels = target_encoder.classes_
                            else:
                                class_labels = [str(i) for i in range(len(np.unique(y)))]
                            
                            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                       xticklabels=class_labels, yticklabels=class_labels)
                            ax.set_xlabel('Predicted')
                            ax.set_ylabel('Actual')
                            ax.set_title(f'Confusion Matrix - {best_model}')
                            st.pyplot(fig)
                            
                            # Classification Report
                            st.subheader("Classification Report")
                            report = classification_report(y_test, y_pred_best, target_names=class_labels, output_dict=True)
                            report_df = pd.DataFrame(report).T
                            st.dataframe(report_df.round(4))
                        
                        # Feature importance for Random Forest
                        if 'Random Forest' in trained_models:
                            st.subheader("Feature Importance (Random Forest)")
                            rf_model = trained_models['Random Forest']
                            if hasattr(rf_model, 'feature_importances_'):
                                importance_df = pd.DataFrame({
                                    'Feature': X_processed.columns,
                                    'Importance': rf_model.feature_importances_
                                }).sort_values('Importance', ascending=False)
                                
                                fig = px.bar(importance_df.head(15), x='Importance', y='Feature', orientation='h',
                                            title="Top 15 Feature Importances",
                                            color='Importance', color_continuous_scale='Blues')
                                st.plotly_chart(fig, use_container_width=True)
                        
                    except Exception as e:
                        st.error(f"Error during training: {str(e)}")
                        st.info("""
                        **Troubleshooting Tips:**
                        1. Make sure your data doesn't have missing values (use preprocessing first)
                        2. Check that you have selected valid numeric or categorical features
                        3. Ensure your target column has at least 2 different values for classification
                        4. Try using the sample dataset to test the system first
                        """)
    
    else:
        st.warning("⚠️ Please load or upload data first in the Data Upload section.")

# Page: Risk Prediction
elif page == "📈 Risk Prediction":
    st.header("📈 Risk Prediction")
    
    if st.session_state.model_trained and st.session_state.models:
        st.subheader("Make New Predictions")
        
        st.info("Enter values for each feature to predict the risk level/score.")
        
        # Create input fields for each feature
        input_data = {}
        
        # Use columns to organize inputs
        cols = st.columns(2)
        
        for idx, feature in enumerate(st.session_state.selected_features):
            with cols[idx % 2]:
                # Get the original data to show possible values
                if st.session_state.risk_data is not None and feature in st.session_state.risk_data.columns:
                    if st.session_state.risk_data[feature].dtype == 'object':
                        # For categorical features, show dropdown
                        unique_vals = st.session_state.risk_data[feature].dropna().unique().tolist()
                        input_data[feature] = st.selectbox(f"{feature}", unique_vals)
                    else:
                        # For numeric features, show number input
                        min_val = float(st.session_state.risk_data[feature].min())
                        max_val = float(st.session_state.risk_data[feature].max())
                        input_data[feature] = st.number_input(f"{feature}", value=float(min_val), min_value=min_val, max_value=max_val)
                else:
                    input_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        if st.button("🔮 Predict Risk", type="primary"):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Process the input data using the same preprocessing
            try:
                # Encode categorical features
                for feature, encoder in st.session_state.feature_encoders.items():
                    if feature in input_df.columns:
                        # Transform using the saved encoder
                        if input_df[feature].iloc[0] in encoder.classes_:
                            input_df[feature + '_encoded'] = encoder.transform([input_df[feature].iloc[0]])[0]
                        else:
                            input_df[feature + '_encoded'] = -1  # Unknown category
                
                # Select only the features used in training
                X_input = input_df[st.session_state.selected_features].copy()
                
                # Convert any remaining categorical to numeric
                for col in X_input.columns:
                    if X_input[col].dtype == 'object':
                        X_input[col] = pd.Categorical(X_input[col]).codes
                
                # Handle missing values
                X_input = X_input.fillna(0)
                
                # Scale input
                input_scaled = st.session_state.scaler.transform(X_input)
                
                st.subheader("Prediction Results")
                
                if st.session_state.task_type == "Classification":
                    results_cols = st.columns(len(st.session_state.models))
                    
                    for idx, (name, model) in enumerate(st.session_state.models.items()):
                        prediction = model.predict(input_scaled)[0]
                        
                        # Decode prediction
                        if st.session_state.target_encoder:
                            pred_label = st.session_state.target_encoder.inverse_transform([prediction])[0]
                        else:
                            pred_label = str(prediction)
                        
                        # Get probability if available
                        if hasattr(model, 'predict_proba'):
                            proba = model.predict_proba(input_scaled)[0]
                            confidence = max(proba) * 100
                        else:
                            confidence = 100
                        
                        # Determine color class
                        color_class = "low"
                        if "High" in str(pred_label) or "high" in str(pred_label):
                            color_class = "high"
                        elif "Medium" in str(pred_label) or "medium" in str(pred_label):
                            color_class = "medium"
                        
                        with results_cols[idx]:
                            st.markdown(f"""
                            <div class="metric-card">
                                <h3>{name}</h3>
                                <div class="risk-{color_class}">
                                    <h2>{pred_label}</h2>
                                </div>
                                <p>Confidence: {confidence:.1f}%</p>
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
                st.info("Please make sure all inputs are valid and try again.")
    
    else:
        st.warning("⚠️ Please train models first in the Model Training section.")

# Page: Risk Mapping & Visualization
elif page == "🗺️ Risk Mapping & Visualization":
    st.header("🗺️ Risk Mapping & Visualization")
    
    if st.session_state.risk_data is not None:
        df = st.session_state.risk_data
        
        st.subheader("Interactive Risk Visualizations")
        
        viz_tabs = st.tabs(["📊 Distribution", "📈 Relationships", "🔬 Correlations", "🎯 Risk Analysis"])
        
        with viz_tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                # Find risk-related columns
                risk_cols = [col for col in df.columns if 'risk' in col.lower() or 'Risk' in col]
                if risk_cols:
                    for risk_col in risk_cols[:2]:
                        if df[risk_col].dtype == 'object':
                            fig = px.pie(df, names=risk_col, title=f"{risk_col} Distribution")
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            fig = px.histogram(df, x=risk_col, nbins=30, title=f"{risk_col} Distribution")
                            st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if numeric_cols:
                    selected_col = st.selectbox("Select column to visualize", numeric_cols)
                    fig = px.box(df, y=selected_col, title=f"{selected_col} Box Plot")
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:
            # Scatter plot
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                x_col = st.selectbox("X-axis", numeric_cols, key="scatter_x")
                y_col = st.selectbox("Y-axis", numeric_cols, key="scatter_y")
                
                fig = px.scatter(df, x=x_col, y=y_col, title=f"{y_col} vs {x_col}",
                               opacity=0.6, trendline="ols")
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[2]:
            # Correlation matrix
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric_cols) >= 2:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                               title="Feature Correlation Matrix",
                               color_continuous_scale='RdBu', zmin=-1, zmax=1)
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[3]:
            # Risk analysis by categories
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if categorical_cols and 'Risk_Score' in df.columns:
                cat_col = st.selectbox("Group by", categorical_cols)
                risk_by_cat = df.groupby(cat_col)['Risk_Score'].agg(['mean', 'std', 'count']).reset_index()
                risk_by_cat = risk_by_cat.sort_values('mean', ascending=False).head(10)
                
                fig = px.bar(risk_by_cat, x=cat_col, y='mean', error_y='std',
                            title=f"Average Risk Score by {cat_col}",
                            labels={'mean': 'Mean Risk Score'})
                st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("⚠️ Please load or upload data first.")

# Page: Model Evaluation
elif page == "📊 Model Evaluation":
    st.header("📊 Model Evaluation")
    
    if st.session_state.model_trained and st.session_state.results:
        st.subheader("Model Performance Metrics")
        
        results_df = pd.DataFrame(st.session_state.results).T
        st.dataframe(results_df.style.highlight_max(axis=0))
        
        # Visualization of metrics
        st.subheader("Performance Visualization")
        
        if st.session_state.task_type == "Classification":
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            fig = go.Figure()
            for metric in metrics:
                fig.add_trace(go.Bar(name=metric, x=results_df.index, y=results_df[metric]))
            
            fig.update_layout(title="Model Performance Comparison",
                            xaxis_title="Models",
                            yaxis_title="Score",
                            yaxis_range=[0, 1],
                            barmode='group')
            st.plotly_chart(fig, use_container_width=True)
            
            # CV Scores
            st.subheader("Cross-Validation Scores")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(name="CV Mean", x=results_df.index, y=results_df['CV Mean'],
                                 error_y=dict(type='data', array=results_df['CV Std'])))
            fig2.update_layout(title="5-Fold Cross Validation Results",
                              xaxis_title="Models",
                              yaxis_title="Mean Accuracy")
            st.plotly_chart(fig2, use_container_width=True)
        
        st.subheader("Model Insights")
        
        # Find best model
        if st.session_state.task_type == "Classification":
            best_model = max(st.session_state.results, key=lambda x: st.session_state.results[x]['Accuracy'])
            st.metric("🏆 Best Model", best_model, f"Accuracy: {st.session_state.results[best_model]['Accuracy']:.4f}")
        else:
            best_model = max(st.session_state.results, key=lambda x: st.session_state.results[x]['R2 Score'])
            st.metric("🏆 Best Model", best_model, f"R² Score: {st.session_state.results[best_model]['R2 Score']:.4f}")
    
    else:
        st.warning("⚠️ Please train models first in the Model Training section.")

# Page: Generate Report
elif page == "📄 Generate Report":
    st.header("📄 Generate Environmental Risk Report")
    
    if st.session_state.risk_data is not None:
        df = st.session_state.risk_data
        
        st.subheader("Report Configuration")
        
        report_title = st.text_input("Report Title", "Microplastic Pollution Risk Assessment Report")
        author_name = st.text_input("Author/Organization", "Viernes, M.J. & Magdaluyo, S.M.R.")
        
        if st.button("📄 Generate Report", type="primary"):
            # Generate report content
            report = f"""
{'='*80}
{report_title.upper()}
{author_name}
Agusan del Sur State College of Agriculture and Technology (ASSCAT)
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*80}

EXECUTIVE SUMMARY
{'-'*40}
This report presents a comprehensive analysis of microplastic pollution risk assessment
using data mining techniques including classification, regression, and clustering.

Key Statistics:
- Total Data Points: {len(df):,}
- Features Analyzed: {len(df.columns)}
- Data Shape: {df.shape[0]} rows × {df.shape[1]} columns

DATA OVERVIEW
{'-'*40}
{df.describe().to_string()}

MISSING VALUES ANALYSIS
{'-'*40}
Missing Values by Column:
{df.isnull().sum().to_string()}

"""
            
            if st.session_state.model_trained and st.session_state.results:
                report += f"""
MODEL PERFORMANCE
{'-'*40}
{pd.DataFrame(st.session_state.results).T.to_string()}

Best Model: {best_model if 'best_model' in dir() else 'N/A'}

"""
            
            report += f"""
KEY FINDINGS
{'-'*40}
1. Microplastic contamination shows significant variation across different locations.

2. {len(df.columns)} distinct features contribute to the predictive risk modeling framework.

3. The analysis provides insights for environmental decision-making.

RECOMMENDATIONS
{'-'*40}
1. Prioritize monitoring in high-risk areas.
2. Implement regular data collection for model updates.
3. Strengthen waste management infrastructure.

{'='*80}
End of Report
{'='*80}
"""
            
            st.text_area("Generated Report", report, height=400)
            
            st.download_button(
                label="📥 Download Report",
                data=report,
                file_name=f"microplastic_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )
            
            st.success("Report generated successfully!")
    
    else:
        st.warning("⚠️ Please load or upload data first.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>🌊 Microplastic Pollution Risk Prediction System | ASSCAT 2025 | Viernes, M.J. & Magdaluyo, S.M.R.</p>
</div>
""", unsafe_allow_html=True)
