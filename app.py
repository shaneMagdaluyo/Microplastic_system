# app.py - Microplastic Pollution Risk Prediction System (COMPLETELY FIXED)
# Fixed: Duplicate column names, classification/regression issues
# Researchers: Matthew Joseph Viernes & Shane Mark R. Magdaluyo
# ASSCAT - March to December 2025

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, mean_squared_error, r2_score)

# For saving models
import joblib
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

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
# INITIALIZE SESSION STATE
# =============================================================================

if 'risk_data' not in st.session_state:
    st.session_state.risk_data = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'data_filename' not in st.session_state:
    st.session_state.data_filename = None
if 'data_source' not in st.session_state:
    st.session_state.data_source = None
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'feature_encoders' not in st.session_state:
    st.session_state.feature_encoders = {}
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def fix_duplicate_columns(df):
    """Fix duplicate column names in dataframe"""
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
    df.columns = cols
    return df

def generate_sample_data(n_samples=1000):
    """Generate synthetic microplastic pollution data"""
    np.random.seed(42)
    
    locations = ['Coastal Area', 'River Delta', 'Urban Runoff', 'Industrial Zone', 
                 'Agricultural Area', 'Marine Reserve', 'Estuary', 'Beach', 
                 'Open Ocean', 'Harbor']
    species = ['Fish_A', 'Fish_B', 'Mollusk', 'Crustacean', 'Bird', 'Mammal']
    habitat = ['Marine', 'Freshwater', 'Estuary', 'Coastal']
    
    data = {
        'Location': np.random.choice(locations, n_samples),
        'Species': np.random.choice(species, n_samples),
        'Habitat': np.random.choice(habitat, n_samples),
        'MP_Concentration': np.random.uniform(0.1, 500, n_samples),
        'Particle_Size_mm': np.random.uniform(0.01, 5.0, n_samples),
        'Temperature_C': np.random.uniform(10, 35, n_samples),
        'pH': np.random.uniform(6.0, 8.5, n_samples),
        'Dissolved_Oxygen': np.random.uniform(2, 12, n_samples),
        'Turbidity': np.random.uniform(1, 100, n_samples),
        'Population_Density': np.random.uniform(10, 10000, n_samples),
        'Industrial_Score': np.random.uniform(0, 1, n_samples),
        'Waste_Score': np.random.uniform(0, 1, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate risk score
    df['Risk_Score'] = (
        df['MP_Concentration'] / 500 * 30 +
        df['Industrial_Score'] * 25 +
        (1 - df['Waste_Score']) * 20 +
        np.where(df['Particle_Size_mm'] < 0.5, 15, 0) +
        np.where(df['Population_Density'] > 5000, 10, 0)
    )
    df['Risk_Score'] = df['Risk_Score'].clip(0, 100)
    
    # Assign risk level
    df['Risk_Level'] = pd.cut(df['Risk_Score'], bins=[0, 33, 66, 100], 
                               labels=['Low', 'Medium', 'High'])
    
    return df

def load_sample_data():
    """Load sample data"""
    df = generate_sample_data(1000)
    df = fix_duplicate_columns(df)
    st.session_state.risk_data = df.copy()
    st.session_state.data_loaded = True
    st.session_state.data_source = "sample"
    st.session_state.data_filename = "sample_data.csv"
    return df

def check_data_loaded():
    if not st.session_state.data_loaded or st.session_state.risk_data is None:
        st.warning("⚠️ No data loaded. Please upload a file or load sample data.")
        return False
    return True

def preprocess_for_training(df, features, target):
    """Preprocess data for training"""
    df_processed = df.copy()
    encoders = {}
    
    numeric_features = []
    categorical_features = []
    
    for feature in features:
        if feature in df_processed.columns:
            if df_processed[feature].dtype in ['int64', 'float64']:
                numeric_features.append(feature)
            else:
                categorical_features.append(feature)
    
    # Handle missing values
    for feature in numeric_features:
        if df_processed[feature].isnull().any():
            df_processed[feature].fillna(df_processed[feature].median(), inplace=True)
    
    for feature in categorical_features:
        if df_processed[feature].isnull().any():
            mode_val = df_processed[feature].mode()
            if len(mode_val) > 0:
                df_processed[feature].fillna(mode_val[0], inplace=True)
            else:
                df_processed[feature].fillna('Unknown', inplace=True)
    
    # Encode categorical features
    for feature in categorical_features:
        le = LabelEncoder()
        df_processed[feature + '_enc'] = le.fit_transform(df_processed[feature].astype(str))
        encoders[feature] = le
        numeric_features.append(feature + '_enc')
    
    X = df_processed[numeric_features]
    
    # Handle target
    if target in df_processed.columns:
        if df_processed[target].dtype == 'object':
            target_encoder = LabelEncoder()
            y = target_encoder.fit_transform(df_processed[target].astype(str))
            return X, y, encoders, target_encoder
        else:
            y = df_processed[target].values
            return X, y, encoders, None
    
    return X, None, encoders, None

# =============================================================================
# HEADER
# =============================================================================

st.markdown("""
<div class="main-header">
    <h1>🌊 Microplastic Pollution Risk Prediction System</h1>
    <p>Data Mining-Based Predictive Risk Modeling for Environmental Microplastic Contamination</p>
    <p>Viernes, M.J. & Magdaluyo, S.M.R. | ASSCAT 2025</p>
</div>
""", unsafe_allow_html=True)

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("📊 Navigation")

if st.session_state.data_loaded:
    st.sidebar.success(f"✅ Data Loaded: {st.session_state.data_filename}")
    if st.session_state.risk_data is not None:
        st.sidebar.info(f"📊 Shape: {st.session_state.risk_data.shape[0]} rows, {st.session_state.risk_data.shape[1]} cols")
else:
    st.sidebar.warning("⚠️ No Data Loaded")

st.sidebar.markdown("---")

page = st.sidebar.radio("Go to", [
    "🏠 Dashboard",
    "📁 Data Upload",
    "🤖 Model Training",
    "📈 Risk Prediction",
    "📊 Results"
])

if st.session_state.data_loaded:
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Clear Data"):
        st.session_state.risk_data = None
        st.session_state.data_loaded = False
        st.session_state.model_trained = False
        st.session_state.models = {}
        st.session_state.results = {}
        st.rerun()

# =============================================================================
# PAGE: DASHBOARD
# =============================================================================

if page == "🏠 Dashboard":
    st.header("🏠 Dashboard")
    
    if st.session_state.data_loaded and st.session_state.risk_data is not None:
        df = st.session_state.risk_data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Total Features", len(df.columns))
        with col3:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            st.metric("Numeric Features", len(numeric_cols))
        with col4:
            if st.session_state.model_trained:
                st.metric("Model Status", "Trained ✅")
            else:
                st.metric("Model Status", "Not Trained")
        
        # Quick visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            if 'Risk_Level' in df.columns:
                risk_counts = df['Risk_Level'].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                             title="Risk Level Distribution",
                             color_discrete_map={'Low': '#6bcb77', 'Medium': '#ffd93d', 'High': '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Risk_Score' in df.columns:
                fig = px.histogram(df, x='Risk_Score', nbins=30, 
                                  title="Risk Score Distribution")
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
    else:
        st.info("👋 Welcome! Go to **Data Upload** to load your data.")
        if st.button("🚀 Load Sample Data"):
            load_sample_data()
            st.rerun()

# =============================================================================
# PAGE: DATA UPLOAD
# =============================================================================

elif page == "📁 Data Upload":
    st.header("📁 Data Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        uploaded_file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                # Fix duplicate column names
                df = fix_duplicate_columns(df)
                
                st.write("**Data Preview:**")
                st.dataframe(df.head(5))
                
                if st.button("✅ Load Dataset"):
                    st.session_state.risk_data = df
                    st.session_state.data_loaded = True
                    st.session_state.data_filename = uploaded_file.name
                    st.session_state.data_source = "upload"
                    st.session_state.model_trained = False
                    st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.write("Or use sample data:")
        if st.button("📊 Load Sample Data"):
            load_sample_data()
            st.session_state.model_trained = False
            st.success("Sample data loaded!")
            st.rerun()
    
    if st.session_state.data_loaded and st.session_state.risk_data is not None:
        st.markdown("---")
        st.subheader("Data Info")
        df = st.session_state.risk_data
        
        tab1, tab2 = st.tabs(["📊 Columns", "📈 Statistics"])
        
        with tab1:
            col_info = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes.values,
                'Unique': [df[col].nunique() for col in df.columns]
            })
            st.dataframe(col_info)
        
        with tab2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if numeric_cols:
                st.dataframe(df[numeric_cols].describe())

# =============================================================================
# PAGE: MODEL TRAINING
# =============================================================================

elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if not check_data_loaded():
        st.stop()
    
    df = st.session_state.risk_data
    
    col1, col2 = st.columns(2)
    
    with col1:
        target_options = df.columns.tolist()
        target = st.selectbox("🎯 Target Column", target_options)
        st.session_state.target_column = target
        
        # Check target type
        unique_count = df[target].nunique()
        is_numeric = df[target].dtype in ['int64', 'float64']
        
        if is_numeric and unique_count > 10:
            task_type = "Regression"
            st.info(f"📊 Target has {unique_count} values → Using **REGRESSION**")
        else:
            task_type = "Classification"
            st.info(f"🏷️ Target has {unique_count} categories → Using **CLASSIFICATION**")
        
        st.session_state.task_type = task_type
    
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        cv_folds = st.slider("CV Folds", 5, 10, 10)
    
    st.subheader("Select Features")
    feature_cols = [c for c in df.columns if c != target]
    selected = st.multiselect("Features", feature_cols, default=feature_cols[:5] if len(feature_cols) > 5 else feature_cols)
    st.session_state.selected_features = selected
    
    if len(selected) == 0:
        st.error("Select at least one feature")
        st.stop()
    
    # Model selection
    st.subheader("Models")
    if task_type == "Classification":
        col1, col2, col3 = st.columns(3)
        with col1:
            use_rf = st.checkbox("Random Forest", value=True)
        with col2:
            use_lr = st.checkbox("Logistic Regression", value=True)
        with col3:
            use_dt = st.checkbox("Decision Tree", value=True)
        use_rfr = False
        use_dtr = False
    else:
        col1, col2 = st.columns(2)
        with col1:
            use_rfr = st.checkbox("Random Forest", value=True)
        with col2:
            use_dtr = st.checkbox("Decision Tree", value=True)
        use_rf = False
        use_lr = False
        use_dt = False
    
    if st.button("🚀 TRAIN MODELS", type="primary", use_container_width=True):
        with st.spinner("Training..."):
            try:
                X, y, encoders, target_enc = preprocess_for_training(df, selected, target)
                st.session_state.feature_encoders = encoders
                st.session_state.target_encoder = target_enc
                
                # Scale
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                st.session_state.scaler = scaler
                
                # Split
                if task_type == "Classification" and len(np.unique(y)) > 1:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42, stratify=y
                    )
                else:
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42
                    )
                
                kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                results = {}
                models = {}
                
                if task_type == "Classification":
                    # Random Forest
                    if use_rf:
                        rf = RandomForestClassifier(n_estimators=100, random_state=42)
                        rf.fit(X_train, y_train)
                        y_pred = rf.predict(X_test)
                        models['Random Forest'] = rf
                        cv_scores = cross_val_score(rf, X_scaled, y, cv=kfold, scoring='accuracy')
                        results['Random Forest'] = {
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                            'CV Mean': cv_scores.mean(),
                            'CV Std': cv_scores.std()
                        }
                        st.write(f"✅ Random Forest: {results['Random Forest']['Accuracy']:.4f}")
                    
                    # Logistic Regression
                    if use_lr:
                        lr = LogisticRegression(max_iter=1000, random_state=42)
                        lr.fit(X_train, y_train)
                        y_pred = lr.predict(X_test)
                        models['Logistic Regression'] = lr
                        cv_scores = cross_val_score(lr, X_scaled, y, cv=kfold, scoring='accuracy')
                        results['Logistic Regression'] = {
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                            'CV Mean': cv_scores.mean(),
                            'CV Std': cv_scores.std()
                        }
                        st.write(f"✅ Logistic Regression: {results['Logistic Regression']['Accuracy']:.4f}")
                    
                    # Decision Tree
                    if use_dt:
                        dt = DecisionTreeClassifier(random_state=42)
                        dt.fit(X_train, y_train)
                        y_pred = dt.predict(X_test)
                        models['Decision Tree'] = dt
                        cv_scores = cross_val_score(dt, X_scaled, y, cv=kfold, scoring='accuracy')
                        results['Decision Tree'] = {
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                            'CV Mean': cv_scores.mean(),
                            'CV Std': cv_scores.std()
                        }
                        st.write(f"✅ Decision Tree: {results['Decision Tree']['Accuracy']:.4f}")
                
                else:  # Regression
                    if use_rfr:
                        rfr = RandomForestRegressor(n_estimators=100, random_state=42)
                        rfr.fit(X_train, y_train)
                        y_pred = rfr.predict(X_test)
                        models['Random Forest'] = rfr
                        cv_scores = cross_val_score(rfr, X_scaled, y, cv=kfold, scoring='r2')
                        results['Random Forest'] = {
                            'R2 Score': r2_score(y_test, y_pred),
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'CV Mean': cv_scores.mean(),
                            'CV Std': cv_scores.std()
                        }
                        st.write(f"✅ Random Forest: R2={results['Random Forest']['R2 Score']:.4f}")
                    
                    if use_dtr:
                        dtr = DecisionTreeRegressor(random_state=42)
                        dtr.fit(X_train, y_train)
                        y_pred = dtr.predict(X_test)
                        models['Decision Tree'] = dtr
                        cv_scores = cross_val_score(dtr, X_scaled, y, cv=kfold, scoring='r2')
                        results['Decision Tree'] = {
                            'R2 Score': r2_score(y_test, y_pred),
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'CV Mean': cv_scores.mean(),
                            'CV Std': cv_scores.std()
                        }
                        st.write(f"✅ Decision Tree: R2={results['Decision Tree']['R2 Score']:.4f}")
                
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.model_trained = True
                
                st.success("✅ Training complete!")
                
                # Display results
                st.subheader("Results")
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df.style.format('{:.4f}').highlight_max(axis=0))
                
                if task_type == "Classification":
                    best = max(results, key=lambda x: results[x]['Accuracy'])
                    st.success(f"🏆 Best: {best} (Accuracy: {results[best]['Accuracy']:.4f})")
                    
                    # Confusion Matrix
                    if best in models:
                        st.subheader(f"Confusion Matrix - {best}")
                        best_model = models[best]
                        y_pred_best = best_model.predict(X_test)
                        cm = confusion_matrix(y_test, y_pred_best)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        if target_enc:
                            labels = target_enc.classes_
                        else:
                            labels = [str(i) for i in range(len(np.unique(y)))]
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                   xticklabels=labels, yticklabels=labels)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                else:
                    best = max(results, key=lambda x: results[x]['R2 Score'])
                    st.success(f"🏆 Best: {best} (R2: {results[best]['R2 Score']:.4f})")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")

# =============================================================================
# PAGE: RISK PREDICTION
# =============================================================================

elif page == "📈 Risk Prediction":
    st.header("📈 Risk Prediction")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first")
        st.stop()
    
    st.subheader("Enter Values")
    
    input_data = {}
    cols = st.columns(2)
    df = st.session_state.risk_data
    
    for i, feat in enumerate(st.session_state.selected_features):
        with cols[i % 2]:
            if feat in df.columns:
                if df[feat].dtype == 'object':
                    vals = df[feat].dropna().unique().tolist()
                    input_data[feat] = st.selectbox(f"{feat}", vals)
                else:
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    mean_val = float(df[feat].mean())
                    input_data[feat] = st.number_input(f"{feat}", value=mean_val, 
                                                       min_value=min_val, max_value=max_val)
            else:
                input_data[feat] = st.number_input(f"{feat}", value=0.0)
    
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
                elif feat in input_df.columns:
                    try:
                        X_input.append(float(input_df[feat].iloc[0]))
                    except:
                        X_input.append(0)
            
            X_input = np.array(X_input).reshape(1, -1)
            X_scaled = st.session_state.scaler.transform(X_input)
            
            st.subheader("Results")
            pred_cols = st.columns(len(st.session_state.models))
            
            for idx, (name, model) in enumerate(st.session_state.models.items()):
                pred = model.predict(X_scaled)[0]
                
                if st.session_state.task_type == "Classification":
                    if st.session_state.target_encoder:
                        label = st.session_state.target_encoder.inverse_transform([int(pred)])[0]
                    else:
                        label = str(pred)
                    
                    label_lower = str(label).lower()
                    if 'high' in label_lower:
                        color = "#ff6b6b"
                    elif 'medium' in label_lower:
                        color = "#ffd93d"
                    else:
                        color = "#6bcb77"
                    
                    with pred_cols[idx]:
                        st.markdown(f"""
                        <div style="background: {color}; padding: 1rem; border-radius: 10px; text-align: center;">
                            <h3>{name}</h3>
                            <h2>{label}</h2>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with pred_cols[idx]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                            <h3>{name}</h3>
                            <h2>{pred:.2f}</h2>
                        </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error: {str(e)}")

# =============================================================================
# PAGE: RESULTS
# =============================================================================

elif page == "📊 Results":
    st.header("📊 Results")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first")
        st.stop()
    
    results_df = pd.DataFrame(st.session_state.results).T
    st.dataframe(results_df.style.format('{:.4f}').highlight_max(axis=0))
    
    if st.session_state.task_type == "Classification":
        best = max(st.session_state.results, key=lambda x: st.session_state.results[x]['Accuracy'])
        st.success(f"🏆 Best Model: {best}")
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df.index, results_df['Accuracy'], color=['#667eea', '#764ba2', '#f093fb'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    else:
        best = max(st.session_state.results, key=lambda x: st.session_state.results[x]['R2 Score'])
        st.success(f"🏆 Best Model: {best}")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df.index, results_df['R2 Score'], color=['#667eea', '#764ba2'])
        ax.set_ylim(-1, 1)
        ax.set_ylabel('R2 Score')
        ax.set_title('Model R2 Comparison')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    # Download report
    report = f"""
    ========================================
    MICROPLASTIC RISK PREDICTION REPORT
    ========================================
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    Task: {st.session_state.task_type}
    Target: {st.session_state.target_column}
    
    RESULTS:
    {results_df.to_string()}
    
    BEST MODEL: {best}
    ========================================
    """
    
    st.download_button("📥 Download Report", report, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

# =============================================================================
# FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌊 Microplastic Pollution Risk Prediction System | ASSCAT 2025 | Viernes, M.J. & Magdaluyo, S.M.R.</p>
</div>
""", unsafe_allow_html=True)
