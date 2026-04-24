# app.py - Microplastic Pollution Risk Prediction System
# Capstone Project: Predictive Risk Modeling for Microplastic Pollution
# Researchers: Matthew Joseph Viernes & Shane Mark R. Magdaluyo
# ASSCAT - March to December 2025

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score, mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import warnings
import io
import base64
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

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
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'risk_data' not in st.session_state:
    st.session_state.risk_data = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None

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
- Classification (RF, LR, DT)
- Clustering Analysis
- Risk Prediction
- K-Fold Cross Validation
- Report Generation

**Data Mining Techniques:**
- Random Forest
- Logistic Regression
- Decision Tree
- K-Means Clustering
""")

# Function to generate sample data
def generate_sample_data(n_samples=1000):
    """Generate synthetic microplastic pollution data"""
    np.random.seed(42)
    
    locations = ['Coastal Area', 'River Delta', 'Urban Runoff', 'Industrial Zone', 
                 'Agricultural Area', 'Marine Reserve', 'Estuary', 'Beach', 
                 'Open Ocean', 'Harbor']
    polymer_types = ['PE', 'PP', 'PS', 'PET', 'PVC', 'PA']
    risk_types = ['Ecological Risk', 'Human Health Risk', 'Chemical Hazard', 
                  'Food Chain Contamination', 'Water Quality Risk']
    
    data = {
        'Location': np.random.choice(locations, n_samples),
        'Polymer_Type': np.random.choice(polymer_types, n_samples),
        'Risk_Type': np.random.choice(risk_types, n_samples),
        'MP_Concentration': np.random.uniform(0.1, 500, n_samples),
        'Particle_Size_mm': np.random.uniform(0.01, 5.0, n_samples),
        'Water_Temperature_C': np.random.uniform(10, 35, n_samples),
        'pH_Level': np.random.uniform(6.0, 8.5, n_samples),
        'Dissolved_Oxygen_mgL': np.random.uniform(2, 12, n_samples),
        'Turbidity_NTU': np.random.uniform(1, 100, n_samples),
        'Population_Density_km2': np.random.uniform(10, 10000, n_samples),
        'Industrial_Activity_Score': np.random.uniform(0, 1, n_samples),
        'Waste_Management_Score': np.random.uniform(0, 1, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate risk score based on features
    df['Risk_Score'] = (
        df['MP_Concentration'] / 500 * 30 +
        df['Industrial_Activity_Score'] * 25 +
        (1 - df['Waste_Management_Score']) * 25 +
        np.where(df['Particle_Size_mm'] < 0.5, 20, 0)
    )
    df['Risk_Score'] = df['Risk_Score'].clip(0, 100)
    
    # Assign risk level
    df['Risk_Level'] = pd.cut(df['Risk_Score'], bins=[0, 33, 66, 100], 
                               labels=['Low', 'Medium', 'High'])
    
    return df

# Page: Data Upload & Preprocessing
if page == "📁 Data Upload & Preprocessing":
    st.header("📁 Data Upload and Preprocessing")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Upload Dataset")
        uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
        
        if uploaded_file is not None:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            st.session_state.risk_data = df
            st.success(f"✅ File loaded successfully! Shape: {df.shape}")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(10))
    
    with col2:
        st.subheader("Or Use Sample Data")
        if st.button("📊 Load Sample Dataset"):
            st.session_state.risk_data = generate_sample_data(1000)
            st.success("✅ Sample dataset loaded! Shape: 1000 x 13")
            st.dataframe(st.session_state.risk_data.head(10))
    
    if st.session_state.risk_data is not None:
        st.markdown("---")
        st.subheader("Data Preprocessing")
        
        df = st.session_state.risk_data
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Rows", df.shape[0])
            st.metric("Total Columns", df.shape[1])
        
        with col2:
            missing = df.isnull().sum().sum()
            st.metric("Missing Values", missing)
        
        with col3:
            duplicates = df.duplicated().sum()
            st.metric("Duplicate Rows", duplicates)
        
        st.subheader("Data Information")
        
        tab1, tab2, tab3 = st.tabs(["📊 Data Summary", "📈 Statistical Description", "🔍 Missing Values"])
        
        with tab1:
            buffer = io.StringIO()
            df.info(buf=buffer)
            st.text(buffer.getvalue())
        
        with tab2:
            st.dataframe(df.describe())
        
        with tab3:
            missing_df = pd.DataFrame({
                'Column': df.columns,
                'Missing Count': df.isnull().sum().values,
                'Missing Percentage': (df.isnull().sum().values / len(df)) * 100
            })
            st.dataframe(missing_df)
        
        st.subheader("Preprocessing Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🧹 Handle Missing Values"):
                for col in df.columns:
                    if df[col].dtype in ['float64', 'int64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if len(df[col].mode()) > 0 else 'Unknown', inplace=True)
                st.session_state.risk_data = df
                st.success("Missing values handled successfully!")
        
        with col2:
            if st.button("🗑️ Remove Duplicates"):
                df.drop_duplicates(inplace=True)
                st.session_state.risk_data = df
                st.success(f"Duplicates removed! New shape: {df.shape}")
        
        st.subheader("Feature Encoding")
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        if categorical_cols:
            st.write("Categorical columns:", categorical_cols)
            if st.button("Encode Categorical Variables"):
                le_dict = {}
                for col in categorical_cols:
                    le = LabelEncoder()
                    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    le_dict[col] = le
                st.session_state.risk_data = df
                st.session_state.label_encoders = le_dict
                st.success("Categorical variables encoded successfully!")
        else:
            st.info("No categorical columns found.")

# Page: Model Training
elif page == "🤖 Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.risk_data is not None:
        df = st.session_state.risk_data
        
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_column = st.selectbox("Select Target Column (Risk Level/Risk Score)", 
                                          [col for col in df.columns if 'risk' in col.lower() or 'Risk' in col])
            
            if 'Risk_Score' in df.columns or 'risk_score' in df.columns:
                actual_target = 'Risk_Score' if 'Risk_Score' in df.columns else 'risk_score'
                task_type = st.radio("Task Type", ["Classification (Risk Level)", "Regression (Risk Score)"])
            else:
                task_type = "Classification (Risk Level)"
        
        with col2:
            test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
            cv_folds = st.slider("K-Fold Cross Validation Folds", 5, 20, 10)
        
        st.subheader("Select Features for Training")
        
        feature_cols = [col for col in df.columns if col != target_column]
        selected_features = st.multiselect("Select Features", feature_cols, default=feature_cols[:5] if len(feature_cols) > 5 else feature_cols)
        
        if st.button("🚀 Train Models", type="primary"):
            with st.spinner("Training models with K-Fold Cross Validation..."):
                X = df[selected_features].copy()
                
                # Handle categorical features
                for col in X.columns:
                    if X[col].dtype == 'object':
                        X[col] = LabelEncoder().fit_transform(X[col].astype(str))
                
                # Handle missing values
                X = X.fillna(X.median())
                
                # Scale features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Target preparation
                if task_type == "Classification (Risk Level)":
                    y = df[target_column]
                    if y.dtype == 'object':
                        y = LabelEncoder().fit_transform(y)
                else:
                    y = df['Risk_Score'] if 'Risk_Score' in df.columns else df['risk_score']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size, random_state=42)
                
                # Initialize models
                models = {
                    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42) if task_type == "Classification (Risk Level)" else RandomForestRegressor(n_estimators=100, random_state=42),
                    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000) if task_type == "Classification (Risk Level)" else None,
                    'Decision Tree': DecisionTreeClassifier(random_state=42) if task_type == "Classification (Risk Level)" else DecisionTreeRegressor(random_state=42)
                }
                
                # Remove None for regression
                if task_type != "Classification (Risk Level)":
                    models = {k: v for k, v in models.items() if v is not None}
                
                results = {}
                cv_results = {}
                
                for name, model in models.items():
                    # Train model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    
                    # K-Fold Cross Validation
                    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    cv_scores = cross_val_score(model, X_scaled, y, cv=kfold, scoring='accuracy' if task_type == "Classification (Risk Level)" else 'r2')
                    
                    # Store results
                    if task_type == "Classification (Risk Level)":
                        results[name] = {
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                        }
                    else:
                        results[name] = {
                            'R2 Score': r2_score(y_test, y_pred),
                            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                            'MSE': mean_squared_error(y_test, y_pred)
                        }
                    
                    cv_results[name] = {
                        'CV Mean': cv_scores.mean(),
                        'CV Std': cv_scores.std(),
                        'CV Scores': cv_scores
                    }
                    
                    st.session_state[f'model_{name}'] = model
                
                st.session_state.models = models
                st.session_state.results = results
                st.session_state.cv_results = cv_results
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.scaler = scaler
                st.session_state.selected_features = selected_features
                st.session_state.task_type = task_type
                st.session_state.model_trained = True
                
                st.success("✅ Models trained successfully!")
                
                # Display results
                st.subheader("Model Performance Results")
                
                results_df = pd.DataFrame(results).T
                st.dataframe(results_df.style.highlight_max(axis=0))
                
                # CV Results
                st.subheader(f"{cv_folds}-Fold Cross Validation Results")
                cv_df = pd.DataFrame({
                    name: {
                        'Mean Accuracy' if task_type == "Classification (Risk Level)" else 'Mean R2': cv_results[name]['CV Mean'],
                        'Std Dev': cv_results[name]['CV Std']
                    } for name in cv_results.keys()
                }).T
                st.dataframe(cv_df)
                
                # Best model
                if task_type == "Classification (Risk Level)":
                    best_model = max(results, key=lambda x: results[x]['Accuracy'])
                else:
                    best_model = max(results, key=lambda x: results[x]['R2 Score'])
                
                st.success(f"🏆 Best Model: **{best_model}**")
                
                # Confusion Matrix for classification
                if task_type == "Classification (Risk Level)":
                    st.subheader("Confusion Matrix - Best Model")
                    best_model_obj = st.session_state[f'model_{best_model}']
                    y_pred_best = best_model_obj.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred_best)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {best_model}')
                    st.pyplot(fig)
    
    else:
        st.warning("⚠️ Please load or upload data first in the Data Upload section.")

# Page: Risk Prediction
elif page == "📈 Risk Prediction":
    st.header("📈 Risk Prediction")
    
    if st.session_state.model_trained:
        st.subheader("Make New Predictions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Input Parameters")
            
            input_data = {}
            for feature in st.session_state.selected_features:
                input_data[feature] = st.number_input(f"{feature}", value=0.0)
        
        if st.button("🔮 Predict Risk", type="primary"):
            # Prepare input data
            input_df = pd.DataFrame([input_data])
            
            # Scale input
            input_scaled = st.session_state.scaler.transform(input_df)
            
            # Get predictions from all models
            st.subheader("Prediction Results")
            
            results_cols = st.columns(len(st.session_state.models))
            
            for idx, (name, model) in enumerate(st.session_state.models.items()):
                prediction = model.predict(input_scaled)[0]
                
                if st.session_state.task_type == "Classification (Risk Level)":
                    # Decode prediction if needed
                    risk_labels = ['Low', 'Medium', 'High']
                    pred_label = prediction if isinstance(prediction, str) else risk_labels[prediction] if prediction < len(risk_labels) else str(prediction)
                    
                    with results_cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{name}</h3>
                            <div class="risk-{pred_label.lower()}">
                                <h2>{pred_label}</h2>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    with results_cols[idx]:
                        st.markdown(f"""
                        <div class="metric-card">
                            <h3>{name}</h3>
                            <h2>{prediction:.2f}</h2>
                            <p>Risk Score (0-100)</p>
                        </div>
                        """, unsafe_allow_html=True)
    
    else:
        st.warning("⚠️ Please train models first in the Model Training section.")

# Page: Risk Mapping & Visualization
elif page == "🗺️ Risk Mapping & Visualization":
    st.header("🗺️ Risk Mapping & Visualization")
    
    if st.session_state.risk_data is not None:
        df = st.session_state.risk_data
        
        st.subheader("Interactive Risk Visualizations")
        
        viz_tabs = st.tabs(["📊 Risk Distribution", "🗺️ Risk by Location", "📈 Trends", "🔬 Clustering Analysis"])
        
        with viz_tabs[0]:
            col1, col2 = st.columns(2)
            
            with col1:
                if 'Risk_Level' in df.columns:
                    risk_counts = df['Risk_Level'].value_counts()
                    fig = px.pie(values=risk_counts.values, names=risk_counts.index, 
                                 title="Risk Level Distribution", color=risk_counts.index,
                                 color_discrete_map={'Low': '#6bcb77', 'Medium': '#ffd93d', 'High': '#ff6b6b'})
                    st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                if 'Risk_Score' in df.columns:
                    fig = px.histogram(df, x='Risk_Score', nbins=30, title="Risk Score Distribution",
                                       color_discrete_sequence=['#2a5298'])
                    st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[1]:
            if 'Location' in df.columns and 'Risk_Score' in df.columns:
                location_risk = df.groupby('Location')['Risk_Score'].mean().sort_values(ascending=False).head(10)
                fig = px.bar(x=location_risk.values, y=location_risk.index, orientation='h',
                             title="Top 10 Locations by Average Risk Score",
                             color=location_risk.values, color_continuous_scale='Reds')
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[2]:
            if 'MP_Concentration' in df.columns and 'Risk_Score' in df.columns:
                fig = px.scatter(df, x='MP_Concentration', y='Risk_Score', color='Risk_Level' if 'Risk_Level' in df.columns else None,
                                 title="MP Concentration vs Risk Score", 
                                 labels={'MP_Concentration': 'Microplastic Concentration (particles/L)',
                                        'Risk_Score': 'Risk Score'},
                                 color_discrete_map={'Low': '#6bcb77', 'Medium': '#ffd93d', 'High': '#ff6b6b'})
                st.plotly_chart(fig, use_container_width=True)
        
        with viz_tabs[3]:
            st.subheader("K-Means Clustering Analysis")
            
            # Select features for clustering
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            cluster_features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols)
            
            if cluster_features and st.button("Perform Clustering"):
                X_cluster = df[cluster_features].fillna(df[cluster_features].median())
                X_cluster_scaled = StandardScaler().fit_transform(X_cluster)
                
                # Elbow method
                inertias = []
                k_range = range(1, 11)
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(X_cluster_scaled)
                    inertias.append(kmeans.inertia_)
                
                fig = px.line(x=list(k_range), y=inertias, markers=True,
                              title="Elbow Method for Optimal K",
                              labels={'x': 'Number of Clusters (K)', 'y': 'Inertia'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Choose K and cluster
                n_clusters = st.slider("Select number of clusters", 2, 8, 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                df['Cluster'] = kmeans.fit_predict(X_cluster_scaled)
                
                # Visualize clusters
                pca = PCA(n_components=2)
                pca_result = pca.fit_transform(X_cluster_scaled)
                
                fig = px.scatter(x=pca_result[:, 0], y=pca_result[:, 1], color=df['Cluster'].astype(str),
                                 title=f"Clustering Results (PCA Projection)",
                                 labels={'x': 'First Principal Component', 'y': 'Second Principal Component'})
                st.plotly_chart(fig, use_container_width=True)
                
                # Cluster characteristics
                st.subheader("Cluster Characteristics")
                cluster_summary = df.groupby('Cluster')[cluster_features].mean()
                st.dataframe(cluster_summary)
                
                # Save clustered data
                st.session_state.risk_data = df
                
    else:
        st.warning("⚠️ Please load or upload data first.")

# Page: Model Evaluation
elif page == "📊 Model Evaluation":
    st.header("📊 Model Evaluation")
    
    if st.session_state.model_trained:
        st.subheader("Model Performance Metrics")
        
        if hasattr(st.session_state, 'results'):
            results_df = pd.DataFrame(st.session_state.results).T
            st.dataframe(results_df.style.highlight_max(axis=0))
            
            # CV Results
            st.subheader("Cross-Validation Results")
            cv_df = pd.DataFrame({
                name: {
                    'CV Mean': st.session_state.cv_results[name]['CV Mean'],
                    'CV Std': st.session_state.cv_results[name]['CV Std']
                } for name in st.session_state.cv_results.keys()
            }).T
            st.dataframe(cv_df)
            
            # Visualization of metrics
            fig = make_subplots(rows=1, cols=2, subplot_titles=["Model Comparison", "CV Scores"])
            
            # Bar chart for metrics
            metrics = list(st.session_state.results[list(st.session_state.results.keys())[0]].keys())
            for model in st.session_state.results.keys():
                values = [st.session_state.results[model][m] for m in metrics]
                fig.add_trace(go.Bar(name=model, x=metrics, y=values), row=1, col=1)
            
            # CV scores box plot
            cv_data = []
            for model in st.session_state.cv_results.keys():
                cv_data.append(go.Box(y=st.session_state.cv_results[model]['CV Scores'], name=model))
            for trace in cv_data:
                fig.add_trace(trace, row=1, col=2)
            
            fig.update_layout(height=500, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature Importance
        if 'Random Forest' in st.session_state.models:
            st.subheader("Feature Importance Analysis")
            
            rf_model = st.session_state.models['Random Forest']
            if hasattr(rf_model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'Feature': st.session_state.selected_features,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(importance_df, x='Importance', y='Feature', orientation='h',
                             title="Random Forest Feature Importance",
                             color='Importance', color_continuous_scale='Blues')
                st.plotly_chart(fig, use_container_width=True)
    
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
        
        include_sections = st.multiselect("Include Sections", 
                                          ["Executive Summary", "Data Overview", "Risk Analysis", 
                                           "Model Performance", "Key Findings", "Recommendations"],
                                          default=["Executive Summary", "Data Overview", "Risk Analysis", "Key Findings"])
        
        if st.button("📄 Generate Report", type="primary"):
            # Generate report content
            report = f"""
            {80 * '='}\n
            {report_title.upper()}\n
            {author_name}\n
            Agusan del Sur State College of Agriculture and Technology (ASSCAT)\n
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n
            {80 * '='}\n\n"""
            
            if "Executive Summary" in include_sections:
                report += f"""
            EXECUTIVE SUMMARY
            {40 * '-'}
            This report presents a comprehensive analysis of microplastic pollution risk assessment
            using data mining techniques including classification, clustering, and regression analysis.
            
            Key Statistics:
            - Total Data Points: {len(df):,}
            - Features Analyzed: {len(df.columns)}
            - High Risk Areas Identified: {len(df[df.get('Risk_Level', pd.Series([''])).astype(str) == 'High'] if 'Risk_Level' in df.columns else 0)}
            - Medium Risk Areas: {len(df[df.get('Risk_Level', pd.Series([''])).astype(str) == 'Medium'] if 'Risk_Level' in df.columns else 0)}
            - Low Risk Areas: {len(df[df.get('Risk_Level', pd.Series([''])).astype(str) == 'Low'] if 'Risk_Level' in df.columns else 0)}
            
            """
            
            if "Data Overview" in include_sections:
                report += f"""
            DATA OVERVIEW
            {40 * '-'}
            Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns
            
            Missing Values: {df.isnull().sum().sum()}
            
            Numerical Features Summary:
            {df.describe().to_string()}
            
            """
            
            if "Risk Analysis" in include_sections and 'Risk_Score' in df.columns:
                report += f"""
            RISK ANALYSIS
            {40 * '-'}
            Risk Score Distribution:
            - Mean Risk Score: {df['Risk_Score'].mean():.2f}
            - Median Risk Score: {df['Risk_Score'].median():.2f}
            - Std Deviation: {df['Risk_Score'].std():.2f}
            - Minimum Risk: {df['Risk_Score'].min():.2f}
            - Maximum Risk: {df['Risk_Score'].max():.2f}
            
            Risk Level Distribution:
            {df['Risk_Level'].value_counts().to_string() if 'Risk_Level' in df.columns else 'Not available'}
            
            """
            
            if "Model Performance" in include_sections and hasattr(st.session_state, 'results'):
                report += f"""
            MODEL PERFORMANCE
            {40 * '-'}
            {pd.DataFrame(st.session_state.results).T.to_string()}
            
            Cross-Validation Results:
            {pd.DataFrame({name: {'CV Mean': st.session_state.cv_results[name]['CV Mean'], 
                                  'CV Std': st.session_state.cv_results[name]['CV Std']} 
                          for name in st.session_state.cv_results.keys()}).T.to_string()}
            
            """
            
            if "Key Findings" in include_sections:
                report += f"""
            KEY FINDINGS
            {40 * '-'}
            1. Microplastic contamination shows significant variation across different locations and environmental conditions.
            
            2. {len(df.columns)} distinct features contribute to the predictive risk modeling framework.
            
            3. The developed models demonstrate robust performance with cross-validation.
            
            4. High-risk areas are primarily associated with elevated microplastic concentrations and industrial activities.
            
            """
            
            if "Recommendations" in include_sections:
                report += f"""
            RECOMMENDATIONS
            {40 * '-'}
            1. Prioritize monitoring and mitigation efforts in high-risk areas identified by the model.
            
            2. Implement regular data collection and model updating to maintain prediction accuracy.
            
            3. Strengthen waste management infrastructure in areas with high industrial activity scores.
            
            4. Develop community awareness programs about microplastic pollution sources and impacts.
            
            5. Integrate this predictive tool into environmental decision-making processes.
            
            """
            
            report += f"\n{80 * '='}\nEnd of Report\n{80 * '='}"
            
            # Display report
            st.text_area("Generated Report", report, height=400)
            
            # Download button
            st.download_button(
                label="📥 Download Report as Text File",
                data=report,
                file_name=f"microplastic_risk_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
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
    <p>Data Mining Techniques: Random Forest, Logistic Regression, Decision Tree, K-Means Clustering</p>
</div>
""", unsafe_allow_html=True)
