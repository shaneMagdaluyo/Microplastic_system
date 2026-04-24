# app.py - Microplastic Pollution Risk Prediction System (FULLY FUNCTIONAL)
# Complete working model training
# Researchers: Matthew Joseph Viernes & Shane Mark R. Magdaluyo
# ASSCAT - March to December 2025

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, mean_squared_error, 
                             r2_score)

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'risk_data' not in st.session_state:
    st.session_state.risk_data = None
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {}
if 'model_results' not in st.session_state:
    st.session_state.model_results = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = []
if 'task_type' not in st.session_state:
    st.session_state.task_type = None
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None
if 'feature_encoders' not in st.session_state:
    st.session_state.feature_encoders = {}
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# Title header
st.markdown("""
<div class="main-header">
    <h1>🌊 Microplastic Pollution Risk Prediction System</h1>
    <p>Data Mining-Based Predictive Risk Modeling for Environmental Microplastic Contamination</p>
    <p>Viernes, M.J. & Magdaluyo, S.M.R. | ASSCAT 2025</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.title("📊 Navigation")

# Data status
if st.session_state.data_loaded and st.session_state.risk_data is not None:
    st.sidebar.success(f"✅ Data: {st.session_state.risk_data.shape[0]} rows")
else:
    st.sidebar.warning("⚠️ No Data Loaded")

st.sidebar.markdown("---")

page = st.sidebar.radio("Go to", [
    "🏠 Dashboard",
    "📁 Data Upload",
    "🤖 Model Training",
    "📈 Prediction",
    "📊 Results"
])

# Helper functions
def generate_sample_data():
    """Generate sample microplastic data"""
    np.random.seed(42)
    n = 1000
    
    data = {
        'Location': np.random.choice(['Coastal', 'River', 'Urban', 'Industrial', 'Agricultural'], n),
        'Water_Type': np.random.choice(['Freshwater', 'Marine', 'Estuary'], n),
        'Polymer_Type': np.random.choice(['PE', 'PP', 'PS', 'PET', 'PVC'], n),
        'MP_Concentration': np.random.uniform(0.1, 500, n),
        'Particle_Size': np.random.uniform(0.01, 5, n),
        'Temperature': np.random.uniform(10, 35, n),
        'pH': np.random.uniform(6, 8.5, n),
        'DO_mgL': np.random.uniform(2, 12, n),
        'Turbidity': np.random.uniform(1, 100, n),
        'Population_Density': np.random.uniform(10, 10000, n),
        'Industrial_Score': np.random.uniform(0, 1, n),
        'Waste_Score': np.random.uniform(0, 1, n),
    }
    df = pd.DataFrame(data)
    
    # Calculate risk score
    df['Risk_Score'] = (
        df['MP_Concentration'] / 500 * 35 +
        df['Industrial_Score'] * 30 +
        (1 - df['Waste_Score']) * 20 +
        np.where(df['Particle_Size'] < 0.5, 15, 0)
    )
    df['Risk_Score'] = df['Risk_Score'].clip(0, 100)
    df['Risk_Level'] = pd.cut(df['Risk_Score'], bins=[0, 33, 66, 100], 
                               labels=['Low', 'Medium', 'High'])
    return df

def preprocess_data(df, features, target):
    """Preprocess data for training"""
    df_proc = df.copy()
    encoders = {}
    
    # Process features
    numeric_features = []
    for feat in features:
        if feat in df_proc.columns:
            if df_proc[feat].dtype in ['int64', 'float64']:
                numeric_features.append(feat)
            else:
                # Encode categorical
                le = LabelEncoder()
                df_proc[feat + '_enc'] = le.fit_transform(df_proc[feat].astype(str))
                encoders[feat] = le
                numeric_features.append(feat + '_enc')
    
    X = df_proc[numeric_features].fillna(0)
    
    # Process target
    if df_proc[target].dtype == 'object':
        target_enc = LabelEncoder()
        y = target_enc.fit_transform(df_proc[target].astype(str))
        return X, y, encoders, target_enc
    else:
        y = df_proc[target].values
        return X, y, encoders, None

# ============================================================================
# DASHBOARD PAGE
# ============================================================================
if page == "🏠 Dashboard":
    st.header("🏠 Dashboard")
    
    if st.session_state.data_loaded and st.session_state.risk_data is not None:
        df = st.session_state.risk_data
        
        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(df):,}")
        with col2:
            st.metric("Features", len(df.columns))
        with col3:
            if 'Risk_Level' in df.columns:
                high_risk = len(df[df['Risk_Level'] == 'High'])
                st.metric("High Risk Areas", high_risk)
        with col4:
            if st.session_state.model_trained:
                st.metric("Model Status", "Trained ✅")
            else:
                st.metric("Model Status", "Not Trained")
        
        # Preview
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
        
        # Quick viz
        if 'Risk_Level' in df.columns:
            fig = px.pie(df, names='Risk_Level', title='Risk Level Distribution',
                        color_discrete_map={'Low': '#6bcb77', 'Medium': '#ffd93d', 'High': '#ff6b6b'})
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("👋 Welcome! Please go to 'Data Upload' to load your data.")

# ============================================================================
# DATA UPLOAD PAGE
# ============================================================================
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
                
                st.write("**Preview:**")
                st.dataframe(df.head(5))
                
                if st.button("✅ Load Dataset", type="primary"):
                    st.session_state.risk_data = df.copy()
                    st.session_state.data_loaded = True
                    st.session_state.model_trained = False
                    st.success(f"Loaded {len(df)} rows, {len(df.columns)} columns!")
                    st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.write("**Or use sample data:**")
        if st.button("📊 Load Sample Data"):
            df = generate_sample_data()
            st.session_state.risk_data = df
            st.session_state.data_loaded = True
            st.session_state.model_trained = False
            st.success("Sample data loaded!")
            st.rerun()
        
        if st.session_state.data_loaded:
            st.markdown("---")
            st.write("**Current Data:**")
            st.write(f"Rows: {st.session_state.risk_data.shape[0]}")
            st.write(f"Columns: {st.session_state.risk_data.shape[1]}")
            st.write(f"Columns: {list(st.session_state.risk_data.columns)}")

# ============================================================================
# MODEL TRAINING PAGE (FULLY FUNCTIONAL)
# ============================================================================
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if not st.session_state.data_loaded or st.session_state.risk_data is None:
        st.warning("⚠️ Please load data first in the Data Upload page.")
        st.stop()
    
    df = st.session_state.risk_data
    
    st.subheader("1. Configure Training")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Target selection
        target_options = [col for col in df.columns if 'risk' in col.lower() or 'Risk' in col or 'level' in col.lower()]
        if not target_options:
            target_options = df.columns.tolist()
        
        target_col = st.selectbox("🎯 Target Column (What to predict)", target_options, key="target_select")
        st.session_state.target_column = target_col
        
        # Determine task type
        unique_vals = df[target_col].nunique()
        if df[target_col].dtype in ['int64', 'float64'] and unique_vals > 10:
            task_type = st.radio("📊 Task Type", ["Regression (Predict Score)", "Classification (Predict Category)"], index=0)
            task_type = "Regression" if "Regression" in task_type else "Classification"
        else:
            task_type = st.radio("📊 Task Type", ["Classification (Predict Category)", "Regression (Predict Score)"], index=0)
            task_type = "Classification" if "Classification" in task_type else "Regression"
        
        st.session_state.task_type = task_type
        
        # Show target info
        st.info(f"Target '{target_col}' has {unique_vals} unique values")
        if task_type == "Classification" and unique_vals > 20:
            st.warning(f"⚠️ {unique_vals} unique values. Consider using Regression or reducing categories.")
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        random_state = st.number_input("Random Seed", value=42, step=1)
    
    st.subheader("2. Select Features")
    
    # Feature selection
    feature_cols = [col for col in df.columns if col != target_col]
    
    st.info(f"📊 {len(feature_cols)} features available")
    
    # Show feature types
    numeric_feats = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_feats = df[feature_cols].select_dtypes(include=['object']).columns.tolist()
    
    st.write(f"**Numeric features:** {len(numeric_feats)}")
    st.write(f"**Categorical features:** {len(cat_feats)} (will be encoded)")
    
    selected_features = st.multiselect(
        "Select features for training",
        feature_cols,
        default=feature_cols[:5] if len(feature_cols) > 5 else feature_cols
    )
    st.session_state.selected_features = selected_features
    
    if len(selected_features) == 0:
        st.error("Please select at least one feature.")
        st.stop()
    
    st.subheader("3. Select Models")
    
    if task_type == "Classification":
        model_options = st.multiselect(
            "Choose models to train",
            ['Random Forest', 'Logistic Regression', 'Decision Tree', 'Gradient Boosting', 'SVM', 'KNN'],
            default=['Random Forest', 'Logistic Regression', 'Decision Tree']
        )
    else:
        model_options = st.multiselect(
            "Choose models to train",
            ['Random Forest', 'Decision Tree', 'Gradient Boosting'],
            default=['Random Forest', 'Decision Tree']
        )
    
    # Model mapping
    def get_model(name, random_state):
        models = {
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state) if task_type == "Classification" else RandomForestRegressor(n_estimators=100, random_state=random_state),
            'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
            'Decision Tree': DecisionTreeClassifier(random_state=random_state) if task_type == "Classification" else DecisionTreeRegressor(random_state=random_state),
            'Gradient Boosting': GradientBoostingClassifier(random_state=random_state) if task_type == "Classification" else None,
            'SVM': SVC(random_state=random_state, probability=True),
            'KNN': KNeighborsClassifier()
        }
        return models.get(name)
    
    # Training button
    if st.button("🚀 TRAIN MODELS", type="primary", use_container_width=True):
        if len(model_options) == 0:
            st.error("Please select at least one model.")
        else:
            with st.spinner("Training models... This may take a moment."):
                try:
                    # Preprocess data
                    X, y, encoders, target_enc = preprocess_data(df, selected_features, target_col)
                    st.session_state.feature_encoders = encoders
                    st.session_state.target_encoder = target_enc
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                    
                    # Split data
                    stratify = y if task_type == "Classification" else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=random_state, stratify=stratify
                    )
                    
                    st.write(f"📊 Training set: {len(X_train)} samples")
                    st.write(f"📊 Testing set: {len(X_test)} samples")
                    
                    # Train each model
                    results = {}
                    trained_models = {}
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, model_name in enumerate(model_options):
                        status_text.text(f"Training {model_name}... ({i+1}/{len(model_options)})")
                        
                        model = get_model(model_name, random_state)
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
                                'CV_Mean': cv_scores.mean(),
                                'CV_Std': cv_scores.std()
                            }
                        
                        progress_bar.progress((i + 1) / len(model_options))
                    
                    status_text.text("Training complete!")
                    
                    # Store results
                    st.session_state.trained_models = trained_models
                    st.session_state.model_results = results
                    st.session_state.model_trained = True
                    
                    # Display results
                    st.success("✅ Training complete!")
                    
                    st.subheader("📊 Model Performance")
                    
                    results_df = pd.DataFrame(results).T
                    
                    if task_type == "Classification":
                        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
                    else:
                        display_cols = ['R2 Score', 'RMSE', 'CV_Mean']
                    
                    st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0))
                    
                    # Best model
                    if task_type == "Classification":
                        best = max(results, key=lambda x: results[x]['Accuracy'])
                        best_score = results[best]['Accuracy']
                    else:
                        best = max(results, key=lambda x: results[x]['R2 Score'])
                        best_score = results[best]['R2 Score']
                    
                    st.success(f"🏆 Best Model: **{best}** with score: {best_score:.4f}")
                    
                    # Confusion Matrix for best classification model
                    if task_type == "Classification" and best in trained_models:
                        st.subheader(f"Confusion Matrix - {best}")
                        best_model = trained_models[best]
                        y_pred_best = best_model.predict(X_test)
                        
                        cm = confusion_matrix(y_test, y_pred_best)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        
                        # Get labels
                        if target_enc:
                            labels = target_enc.classes_
                        else:
                            labels = [str(i) for i in range(len(np.unique(y)))]
                        
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                   xticklabels=labels, yticklabels=labels)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title(f'Confusion Matrix - {best}')
                        st.pyplot(fig)
                        
                        # Classification report
                        st.subheader("Classification Report")
                        report = classification_report(y_test, y_pred_best, target_names=labels, output_dict=True)
                        report_df = pd.DataFrame(report).T
                        st.dataframe(report_df.round(4))
                    
                    # Feature importance for Random Forest
                    if 'Random Forest' in trained_models:
                        st.subheader("📈 Feature Importance (Random Forest)")
                        rf_model = trained_models['Random Forest']
                        if hasattr(rf_model, 'feature_importances_'):
                            importance_df = pd.DataFrame({
                                'Feature': X.columns,
                                'Importance': rf_model.feature_importances_
                            }).sort_values('Importance', ascending=False)
                            
                            fig = px.bar(importance_df.head(15), x='Importance', y='Feature', orientation='h',
                                        title="Top 15 Feature Importances",
                                        color='Importance', color_continuous_scale='Blues')
                            st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    st.info("Please check your data and try again.")

# ============================================================================
# PREDICTION PAGE
# ============================================================================
elif page == "📈 Prediction":
    st.header("📈 Make Predictions")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the Model Training page.")
        st.stop()
    
    if not st.session_state.data_loaded:
        st.warning("⚠️ No data loaded.")
        st.stop()
    
    st.subheader("Enter Values for Prediction")
    
    # Create input fields for each feature
    input_data = {}
    cols = st.columns(2)
    
    df = st.session_state.risk_data
    
    for i, feature in enumerate(st.session_state.selected_features):
        with cols[i % 2]:
            if feature in df.columns:
                if df[feature].dtype == 'object':
                    # Categorical - dropdown
                    values = df[feature].dropna().unique().tolist()
                    input_data[feature] = st.selectbox(f"{feature}", values)
                else:
                    # Numeric - number input
                    min_val = float(df[feature].min())
                    max_val = float(df[feature].max())
                    default = (min_val + max_val) / 2
                    input_data[feature] = st.number_input(f"{feature}", value=default, 
                                                          min_value=min_val, max_value=max_val)
            else:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
    
    if st.button("🔮 PREDICT", type="primary", use_container_width=True):
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
            
            # Select features
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
            
            # Make predictions
            st.subheader("Prediction Results")
            
            results_cols = st.columns(len(st.session_state.trained_models))
            
            for idx, (name, model) in enumerate(st.session_state.trained_models.items()):
                pred = model.predict(X_input_scaled)[0]
                
                if st.session_state.task_type == "Classification":
                    if st.session_state.target_encoder:
                        pred_label = st.session_state.target_encoder.inverse_transform([int(pred)])[0]
                    else:
                        pred_label = str(pred)
                    
                    # Get probability if available
                    confidence = ""
                    if hasattr(model, 'predict_proba'):
                        proba = model.predict_proba(X_input_scaled)[0]
                        confidence = f"<p>Confidence: {max(proba)*100:.1f}%</p>"
                    
                    # Color based on prediction
                    color = "low"
                    if "High" in str(pred_label):
                        color = "high"
                    elif "Medium" in str(pred_label):
                        color = "medium"
                    
                    with results_cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{name}</h3>
                            <div class="risk-{color}">
                                <h2>{pred_label}</h2>
                            </div>
                            {confidence}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with results_cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{name}</h3>
                            <h2>{pred:.2f}</h2>
                            <p>Risk Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ============================================================================
# RESULTS PAGE
# ============================================================================
elif page == "📊 Results":
    st.header("📊 Model Results")
    
    if not st.session_state.model_trained:
        st.warning("⚠️ Please train models first in the Model Training page.")
        st.stop()
    
    st.subheader("Performance Summary")
    
    results_df = pd.DataFrame(st.session_state.model_results).T
    
    if st.session_state.task_type == "Classification":
        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
    else:
        display_cols = ['R2 Score', 'RMSE', 'CV_Mean']
    
    st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0))
    
    # Visualization
    st.subheader("Visual Comparison")
    
    if st.session_state.task_type == "Classification":
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        fig = go.Figure()
        for metric in metrics:
            fig.add_trace(go.Bar(name=metric, x=results_df.index, y=results_df[metric]))
        fig.update_layout(title="Model Performance Comparison", barmode='group', height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Best model details
    st.subheader("Best Model Details")
    
    if st.session_state.task_type == "Classification":
        best_model = max(st.session_state.model_results, key=lambda x: st.session_state.model_results[x]['Accuracy'])
        best_score = st.session_state.model_results[best_model]['Accuracy']
    else:
        best_model = max(st.session_state.model_results, key=lambda x: st.session_state.model_results[x]['R2 Score'])
        best_score = st.session_state.model_results[best_model]['R2 Score']
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Best Model", best_model)
    with col2:
        st.metric("Best Score", f"{best_score:.4f}")
    with col3:
        st.metric("CV Folds", "5-Fold")
    
    # Download report
    st.subheader("Download Report")
    
    report = f"""
    MICROPLASTIC RISK PREDICTION REPORT
    {'='*50}
    Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    
    MODEL PERFORMANCE
    {'='*50}
    {results_df[display_cols].to_string()}
    
    BEST MODEL: {best_model}
    BEST SCORE: {best_score:.4f}
    TASK TYPE: {st.session_state.task_type}
    """
    
    st.download_button(
        label="📥 Download Report",
        data=report,
        file_name=f"model_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌊 Microplastic Risk Prediction System | ASSCAT 2025</p>
</div>
""", unsafe_allow_html=True)
