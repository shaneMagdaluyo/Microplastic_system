# ============================================================================
# MICROPLASTIC POLLUTION RISK PREDICTION SYSTEM
# Complete Working Version - No SMOTE Dependency
# ============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc,
    ConfusionMatrixDisplay
)
from sklearn.utils.class_weight import compute_class_weight
import warnings
import io
import base64
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Microplastic Pollution Risk Prediction System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
    }
    .main-header h1 {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .main-header p {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .metric-card h3 {
        font-size: 2rem;
        margin: 0;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .risk-low { background-color: #2ECC71; color: white; padding: 5px 10px; border-radius: 20px; text-align: center; }
    .risk-medium { background-color: #F39C12; color: white; padding: 5px 10px; border-radius: 20px; text-align: center; }
    .risk-high { background-color: #E67E22; color: white; padding: 5px 10px; border-radius: 20px; text-align: center; }
    .risk-critical { background-color: #E74C3C; color: white; padding: 5px 10px; border-radius: 20px; text-align: center; }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# DATA GENERATION FUNCTION
# ============================================================================
@st.cache_data
def generate_microplastic_data(n_samples=10000):
    """Generate comprehensive microplastic pollution dataset"""
    np.random.seed(42)
    
    # Location types with varying risk levels
    locations = {
        'Marine Coastal': {'risk_factor': 1.2},
        'Freshwater Lake': {'risk_factor': 1.0},
        'River Estuary': {'risk_factor': 1.4},
        'Open Ocean': {'risk_factor': 0.8},
        'Urban Runoff': {'risk_factor': 1.6},
        'Industrial Area': {'risk_factor': 1.8},
        'Agricultural Runoff': {'risk_factor': 1.1},
        'Wastewater Treatment': {'risk_factor': 1.3}
    }
    
    # Polymer types with toxicity levels
    polymer_types = {
        'Polyethylene (PE)': {'toxicity': 0.3, 'density': 0.95},
        'Polypropylene (PP)': {'toxicity': 0.2, 'density': 0.90},
        'Polystyrene (PS)': {'toxicity': 0.8, 'density': 1.05},
        'Polyethylene Terephthalate (PET)': {'toxicity': 0.4, 'density': 1.38},
        'Polyvinyl Chloride (PVC)': {'toxicity': 0.9, 'density': 1.40},
        'Nylon (PA)': {'toxicity': 0.5, 'density': 1.14},
        'Polyacrylamide (PAM)': {'toxicity': 0.6, 'density': 1.30}
    }
    
    # Generate data
    location_list = list(locations.keys())
    polymer_list = list(polymer_types.keys())
    
    data = {
        'Location': np.random.choice(location_list, n_samples, p=[0.15, 0.12, 0.10, 0.13, 0.18, 0.15, 0.10, 0.07]),
        'Polymer_Type': np.random.choice(polymer_list, n_samples, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07]),
        'Water_Temperature_C': np.random.normal(22, 5, n_samples),
        'pH_Level': np.random.normal(7.5, 0.5, n_samples),
        'Salinity_ppt': np.random.gamma(2, 15, n_samples),
        'Dissolved_Oxygen_mgL': np.random.normal(6.5, 1.5, n_samples),
        'Turbidity_NTU': np.random.exponential(10, n_samples),
        'Population_Density_km2': np.random.lognormal(5, 1.5, n_samples),
        'Industrial_Discharge_m3': np.random.exponential(500, n_samples),
        'Wastewater_Treatment_Efficiency': np.random.uniform(40, 98, n_samples),
        'Distance_to_Shore_km': np.random.exponential(50, n_samples),
        'Water_Flow_Speed_ms': np.random.gamma(1.5, 0.3, n_samples),
        'Rainfall_mm': np.random.exponential(100, n_samples),
        'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n_samples, p=[0.2, 0.3, 0.3, 0.2])
    }
    
    df = pd.DataFrame(data)
    
    # Add polymer toxicity and density
    df['Polymer_Toxicity'] = df['Polymer_Type'].map(lambda x: polymer_types[x]['toxicity'])
    df['Polymer_Density'] = df['Polymer_Type'].map(lambda x: polymer_types[x]['density'])
    
    # Add location risk factor
    df['Location_Risk_Factor'] = df['Location'].map(lambda x: locations[x]['risk_factor'])
    
    # Calculate MP Count per Liter (comprehensive formula)
    df['MP_Count_per_L'] = (
        (df['Industrial_Discharge_m3'] / 1000) * 0.4 +
        (1 - df['Wastewater_Treatment_Efficiency']/100) * 120 +
        df['Population_Density_km2'] / 400 +
        df['Polymer_Toxicity'] * 80 +
        df['Location_Risk_Factor'] * 50 +
        (df['Turbidity_NTU'] / 20) * 10 +
        np.random.normal(0, 25, n_samples)
    )
    df['MP_Count_per_L'] = np.maximum(5, df['MP_Count_per_L'])
    df['MP_Count_per_L'] = np.minimum(500, df['MP_Count_per_L'])
    
    # Calculate Risk Score (comprehensive formula)
    df['Risk_Score'] = (
        (df['MP_Count_per_L'] / 500) * 35 +
        df['Polymer_Toxicity'] * 20 +
        (df['Industrial_Discharge_m3'] / max(df['Industrial_Discharge_m3'])) * 15 +
        (df['Population_Density_km2'] / max(df['Population_Density_km2'])) * 15 +
        (df['Location_Risk_Factor'] / 2) * 10 +
        (1 - df['Wastewater_Treatment_Efficiency']/100) * 5
    ) * 100 / 100
    
    df['Risk_Score'] = df['Risk_Score'] * 100
    df['Risk_Score'] = np.minimum(100, np.maximum(0, df['Risk_Score']))
    
    # Assign Risk Levels
    conditions = [
        df['Risk_Score'] < 30,
        (df['Risk_Score'] >= 30) & (df['Risk_Score'] < 55),
        (df['Risk_Score'] >= 55) & (df['Risk_Score'] < 75),
        df['Risk_Score'] >= 75
    ]
    choices = ['Low', 'Medium', 'High', 'Critical']
    df['Risk_Level'] = np.select(conditions, choices)
    
    # Assign Risk Types
    risk_types = []
    for idx, row in df.iterrows():
        if row['Risk_Score'] > 75 and row['MP_Count_per_L'] > 150:
            risk_types.append('Human Health Risk')
        elif row['Risk_Score'] > 65 and row['Industrial_Discharge_m3'] > 800:
            risk_types.append('Ecosystem Risk')
        elif row['Polymer_Toxicity'] > 0.7:
            risk_types.append('Chemical Hazard Risk')
        elif row['Location'] in ['Urban Runoff', 'Industrial Area']:
            risk_types.append('Source Contamination Risk')
        elif row['Risk_Score'] > 50:
            risk_types.append('Moderate Environmental Risk')
        else:
            risk_types.append('Low Environmental Risk')
    df['Risk_Type'] = risk_types
    
    # Add color distribution
    colors = ['White/Clear', 'Blue', 'Green', 'Red', 'Black', 'Yellow', 'Transparent']
    df['Color_Distribution'] = np.random.choice(colors, n_samples, p=[0.35, 0.20, 0.15, 0.08, 0.10, 0.05, 0.07])
    
    # Add size ranges
    size_ranges = ['<0.1mm', '0.1-0.5mm', '0.5-1mm', '1-2mm', '2-5mm']
    df['Size_Range'] = np.random.choice(size_ranges, n_samples, p=[0.25, 0.35, 0.20, 0.12, 0.08])
    
    return df

# ============================================================================
# DATA PREPROCESSING FUNCTION
# ============================================================================
@st.cache_data
def preprocess_data(df):
    """Preprocess the data for modeling"""
    df_processed = df.copy()
    
    # Encode categorical variables
    encoders = {}
    categorical_cols = ['Location', 'Polymer_Type', 'Risk_Level', 'Risk_Type', 
                        'Color_Distribution', 'Size_Range', 'Season']
    
    for col in categorical_cols:
        le = LabelEncoder()
        df_processed[f'{col}_Encoded'] = le.fit_transform(df_processed[col])
        encoders[col] = le
    
    # Select features for modeling
    feature_cols = [
        'MP_Count_per_L', 'Industrial_Discharge_m3', 'Population_Density_km2',
        'Wastewater_Treatment_Efficiency', 'Water_Temperature_C', 'pH_Level',
        'Salinity_ppt', 'Dissolved_Oxygen_mgL', 'Turbidity_NTU',
        'Distance_to_Shore_km', 'Water_Flow_Speed_ms', 'Rainfall_mm',
        'Polymer_Toxicity', 'Polymer_Density', 'Location_Risk_Factor',
        'Location_Encoded', 'Polymer_Type_Encoded', 'Season_Encoded',
        'Color_Distribution_Encoded', 'Size_Range_Encoded'
    ]
    
    X = df_processed[feature_cols]
    y_level = df_processed['Risk_Level_Encoded']
    y_type = df_processed['Risk_Type_Encoded']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
    
    return X_scaled, y_level, y_type, scaler, encoders, feature_cols

# ============================================================================
# MODEL TRAINING FUNCTION
# ============================================================================
@st.cache_resource
def train_models(X, y):
    """Train multiple models and return the best one"""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Define models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'SVM': SVC(kernel='rbf', random_state=42, probability=True),
        'KNN': KNeighborsClassifier(n_neighbors=5)
    }
    
    # Train and evaluate
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1': f1_score(y_test, y_pred, average='weighted'),
            'model': model,
            'predictions': y_pred
        }
        trained_models[name] = model
    
    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['accuracy'])
    best_model = results[best_model_name]['model']
    
    return best_model, best_model_name, results, X_train, X_test, y_train, y_test

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================
def plot_risk_distribution(df):
    """Plot risk level distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    risk_counts = df['Risk_Level'].value_counts()
    colors_risk = {'Low': '#2ECC71', 'Medium': '#F39C12', 'High': '#E67E22', 'Critical': '#E74C3C'}
    colors_list = [colors_risk[level] for level in risk_counts.index]
    
    bars = ax.bar(risk_counts.index, risk_counts.values, color=colors_list, edgecolor='black', linewidth=2)
    ax.set_xlabel('Risk Level', fontsize=12, fontweight='bold')
    ax.set_ylabel('Count', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Microplastic Pollution Risk Levels', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    for bar, val in zip(bars, risk_counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, str(val), 
                ha='center', fontweight='bold', fontsize=11)
    
    return fig

def plot_risk_score_histogram(df):
    """Plot risk score distribution"""
    fig, ax = plt.subplots(figsize=(10, 6))
    n, bins, patches = ax.hist(df['Risk_Score'], bins=40, edgecolor='black', alpha=0.7, color='#3498DB')
    
    # Color bins by risk level
    for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
        center = (left + right) / 2
        if center < 30:
            patches[i].set_facecolor('#2ECC71')
        elif center < 55:
            patches[i].set_facecolor('#F39C12')
        elif center < 75:
            patches[i].set_facecolor('#E67E22')
        else:
            patches[i].set_facecolor('#E74C3C')
    
    ax.axvline(df['Risk_Score'].mean(), color='blue', linestyle='--', linewidth=2, 
               label=f'Mean: {df["Risk_Score"].mean():.1f}')
    ax.axvline(df['Risk_Score'].median(), color='red', linestyle='--', linewidth=2, 
               label=f'Median: {df["Risk_Score"].median():.1f}')
    ax.set_xlabel('Risk Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
    ax.set_title('Risk Score Distribution with Risk Level Boundaries', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    return fig

def plot_correlation_heatmap(df, feature_cols):
    """Plot correlation heatmap"""
    fig, ax = plt.subplots(figsize=(12, 10))
    correlation_matrix = df[feature_cols[:15]].corr()
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r', 
                center=0, square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    return fig

def plot_location_risk(df):
    """Plot risk score by location"""
    fig, ax = plt.subplots(figsize=(12, 6))
    location_risk = df.groupby('Location')['Risk_Score'].agg(['mean', 'std']).sort_values('mean', ascending=True)
    
    colors = ['#2ECC71' if x < 40 else '#F39C12' if x < 60 else '#E67E22' if x < 75 else '#E74C3C' 
              for x in location_risk['mean']]
    
    ax.barh(location_risk.index, location_risk['mean'], xerr=location_risk['std'], 
            capsize=5, color=colors, edgecolor='black', alpha=0.8)
    ax.set_xlabel('Average Risk Score', fontsize=12, fontweight='bold')
    ax.set_ylabel('Location', fontsize=12, fontweight='bold')
    ax.set_title('Average Risk Score by Location', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    for i, (idx, row) in enumerate(location_risk.iterrows()):
        ax.text(row['mean'] + 1, i, f'{row["mean"]:.1f}', va='center', fontweight='bold')
    
    return fig

def plot_polymer_risk(df):
    """Plot risk by polymer type"""
    fig, ax = plt.subplots(figsize=(10, 6))
    polymer_risk = pd.crosstab(df['Polymer_Type'], df['Risk_Level'], normalize='index') * 100
    polymer_risk.plot(kind='bar', stacked=True, ax=ax, colormap='RdYlGn_r', edgecolor='black')
    ax.set_xlabel('Polymer Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
    ax.set_title('Risk Level Distribution by Polymer Type', fontsize=14, fontweight='bold')
    ax.legend(title='Risk Level', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return fig

def plot_mp_vs_risk(df):
    """Plot MP count vs risk score relationship"""
    fig, ax = plt.subplots(figsize=(10, 6))
    scatter = ax.scatter(df['MP_Count_per_L'], df['Risk_Score'], 
                        c=df['Risk_Score'], cmap='RdYlGn_r', alpha=0.5, s=30)
    ax.set_xlabel('Microplastic Count per Liter', fontsize=12, fontweight='bold')
    ax.set_ylabel('Risk Score', fontsize=12, fontweight='bold')
    ax.set_title('Relationship: Microplastic Count vs Environmental Risk Score', fontsize=14, fontweight='bold')
    
    # Add trend line
    z = np.polyfit(df['MP_Count_per_L'], df['Risk_Score'], 1)
    p = np.poly1d(z)
    ax.plot(np.sort(df['MP_Count_per_L']), p(np.sort(df['MP_Count_per_L'])), 
            "r-", linewidth=2, label=f'Trend (r={df["MP_Count_per_L"].corr(df["Risk_Score"]):.3f})')
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Risk Score', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    return fig

def plot_confusion_matrix(y_test, y_pred, classes):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=classes, yticklabels=classes, cbar_kws={"shrink": 0.8})
    ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
    ax.set_ylabel('Actual', fontsize=12, fontweight='bold')
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    return fig

def plot_feature_importance(model, feature_names, top_n=15):
    """Plot feature importance"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.barh(range(len(indices)), importances[indices], color='#3498DB', edgecolor='black')
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
        return fig
    return None

def plot_model_comparison(results):
    """Plot model performance comparison"""
    fig, ax = plt.subplots(figsize=(12, 6))
    models = list(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    x = np.arange(len(models))
    width = 0.2
    
    for i, metric in enumerate(metrics):
        scores = [results[model][metric] for model in models]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, scores, width, label=metric.capitalize())
        
        # Add value labels
        for bar, score in zip(bars, scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                   f'{score:.3f}', ha='center', fontsize=8)
    
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1.1)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================
def predict_risk(input_data, model, scaler, encoders, feature_cols):
    """Make risk prediction based on input parameters"""
    # Create DataFrame from input
    input_df = pd.DataFrame([input_data])
    
    # Encode categorical variables
    for col, encoder in encoders.items():
        if col in input_df.columns:
            try:
                input_df[f'{col}_Encoded'] = encoder.transform(input_df[col])
            except:
                input_df[f'{col}_Encoded'] = 0
    
    # Prepare feature vector
    feature_vector = []
    for col in feature_cols:
        if col in input_df.columns:
            feature_vector.append(input_df[col].iloc[0])
        else:
            feature_vector.append(0)
    
    # Scale and predict
    feature_scaled = scaler.transform([feature_vector])
    prediction = model.predict(feature_scaled)[0]
    
    return prediction

# ============================================================================
# MAIN APPLICATION
# ============================================================================
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌊 Microplastic Pollution Risk Prediction System</h1>
        <p>Advanced Data Mining and Machine Learning for Environmental Risk Assessment</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner('🔄 Loading environmental data and training models...'):
        df = generate_microplastic_data(10000)
        X, y_level, y_type, scaler, encoders, feature_cols = preprocess_data(df)
        best_model, best_model_name, results, X_train, X_test, y_train, y_test = train_models(X, y_level)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2936/2936886.png", width=100)
        st.markdown("## 📊 Navigation")
        
        section = st.radio(
            "Select Section",
            ["🏠 Dashboard", "📈 Risk Analysis", "🤖 Model Training", 
             "🔮 Risk Prediction", "📋 Data Explorer", "📊 Reports"]
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ System Information")
        st.info(f"""
        - **Samples:** {len(df):,}
        - **Features:** {len(feature_cols)}
        - **Risk Levels:** 4
        - **Best Model:** {best_model_name}
        - **Accuracy:** {results[best_model_name]['accuracy']:.2%}
        """)
        
        st.markdown("---")
        st.markdown("### 👥 Researchers")
        st.markdown("""
        - Matthew Joseph Viernes
        - Shane Mark Magdaluyo
        """)
        
        st.markdown("---")
        st.markdown("### 📅 Study Period")
        st.markdown("March 2025 - December 2025")
    
    # Dashboard Section
    if section == "🏠 Dashboard":
        st.markdown("## 📊 Executive Dashboard")
        
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <p>📊 Total Samples</p>
                <h3>{:,.0f}</h3>
            </div>
            """.format(len(df)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <p>⚠️ Average Risk Score</p>
                <h3>{:.1f}</h3>
            </div>
            """.format(df['Risk_Score'].mean()), unsafe_allow_html=True)
        
        with col3:
            critical_pct = (df['Risk_Level'] == 'Critical').mean() * 100
            st.markdown("""
            <div class="metric-card">
                <p>🔴 Critical Risk Areas</p>
                <h3>{:.1f}%</h3>
            </div>
            """.format(critical_pct), unsafe_allow_html=True)
        
        with col4:
            high_pct = ((df['Risk_Level'] == 'High').mean() + (df['Risk_Level'] == 'Critical').mean()) * 100
            st.markdown("""
            <div class="metric-card">
                <p>🚨 High+Critical Risk</p>
                <h3>{:.1f}%</h3>
            </div>
            """.format(high_pct), unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Charts row 1
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Risk Level Distribution")
            fig = plot_risk_distribution(df)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("📈 Risk Score Distribution")
            fig = plot_risk_score_histogram(df)
            st.pyplot(fig)
            plt.close()
        
        # Charts row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📍 Risk Score by Location")
            fig = plot_location_risk(df)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            st.subheader("🔬 Risk by Polymer Type")
            fig = plot_polymer_risk(df)
            st.pyplot(fig)
            plt.close()
        
        # Correlation heatmap
        st.subheader("📊 Feature Correlation Analysis")
        fig = plot_correlation_heatmap(df, feature_cols)
        st.pyplot(fig)
        plt.close()
        
        # Summary statistics
        with st.expander("📋 View Summary Statistics"):
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Risk Level Distribution**")
                st.write(df['Risk_Level'].value_counts())
            with col2:
                st.write("**Risk Type Distribution**")
                st.write(df['Risk_Type'].value_counts())
    
    # Risk Analysis Section
    elif section == "📈 Risk Analysis":
        st.markdown("## 📈 Comprehensive Risk Analysis")
        
        # Relationship plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔬 MP Count vs Risk Score")
            fig = plot_mp_vs_risk(df)
            st.pyplot(fig)
            plt.close()
            
            # Correlation statistics
            corr = df['MP_Count_per_L'].corr(df['Risk_Score'])
            st.markdown(f"""
            <div class="info-box">
                <strong>📊 Correlation Analysis:</strong><br>
                Pearson Correlation: {corr:.4f}<br>
                Interpretation: {'Strong positive' if corr > 0.7 else 'Moderate positive' if corr > 0.4 else 'Weak'} correlation
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("🌡️ Environmental Factors Impact")
            fig, ax = plt.subplots(figsize=(8, 6))
            factors = ['Industrial_Discharge_m3', 'Population_Density_km2', 
                      'Wastewater_Treatment_Efficiency', 'Polymer_Toxicity']
            correlations = [df[factor].corr(df['Risk_Score']) for factor in factors]
            colors_corr = ['#E74C3C' if c > 0 else '#2ECC71' for c in correlations]
            bars = ax.bar(factors, correlations, color=colors_corr, edgecolor='black')
            ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
            ax.set_xlabel('Factor', fontsize=11)
            ax.set_ylabel('Correlation with Risk Score', fontsize=11)
            ax.set_title('Environmental Factors Impact on Risk', fontsize=12, fontweight='bold')
            plt.xticks(rotation=45, ha='right')
            for bar, val in zip(bars, correlations):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.02 if val > 0 else -0.08), 
                       f'{val:.3f}', ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()
        
        # Polymer analysis
        st.subheader("🔬 Polymer Type Risk Analysis")
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(10, 6))
            polymer_avg_risk = df.groupby('Polymer_Type')['Risk_Score'].mean().sort_values(ascending=True)
            colors = ['#2ECC71' if x < 40 else '#F39C12' if x < 60 else '#E67E22' if x < 75 else '#E74C3C' 
                     for x in polymer_avg_risk.values]
            ax.barh(polymer_avg_risk.index, polymer_avg_risk.values, color=colors, edgecolor='black')
            ax.set_xlabel('Average Risk Score', fontsize=11)
            ax.set_title('Average Risk Score by Polymer Type', fontsize=12, fontweight='bold')
            ax.grid(axis='x', alpha=0.3)
            st.pyplot(fig)
            plt.close()
        
        with col2:
            fig = plot_polymer_risk(df)
            st.pyplot(fig)
            plt.close()
        
        # Seasonal analysis
        st.subheader("🌦️ Seasonal Risk Variation")
        fig, ax = plt.subplots(figsize=(10, 6))
        seasonal_risk = df.groupby('Season')['Risk_Score'].mean().sort_values()
        colors_season = ['#3498DB', '#2ECC71', '#F39C12', '#E74C3C']
        ax.bar(seasonal_risk.index, seasonal_risk.values, color=colors_season, edgecolor='black')
        ax.set_xlabel('Season', fontsize=11)
        ax.set_ylabel('Average Risk Score', fontsize=11)
        ax.set_title('Seasonal Variation in Microplastic Risk', fontsize=12, fontweight='bold')
        for i, (season, score) in enumerate(seasonal_risk.items()):
            ax.text(i, score + 1, f'{score:.1f}', ha='center', fontweight='bold')
        st.pyplot(fig)
        plt.close()
    
    # Model Training Section
    elif section == "🤖 Model Training":
        st.markdown("## 🤖 Machine Learning Model Training")
        
        st.markdown(f"""
        <div class="info-box">
            <strong>🏆 Best Model:</strong> {best_model_name}<br>
            <strong>📈 Overall Accuracy:</strong> {results[best_model_name]['accuracy']:.2%}<br>
            <strong>🎯 F1-Score:</strong> {results[best_model_name]['f1']:.3f}<br>
            <strong>📊 Precision:</strong> {results[best_model_name]['precision']:.3f}<br>
            <strong>📉 Recall:</strong> {results[best_model_name]['recall']:.3f}
        </div>
        """, unsafe_allow_html=True)
        
        # Model comparison
        st.subheader("📊 Model Performance Comparison")
        fig = plot_model_comparison(results)
        st.pyplot(fig)
        plt.close()
        
        # Confusion Matrix
        st.subheader("🎯 Confusion Matrix (Best Model)")
        y_pred_best = results[best_model_name]['predictions']
        classes = ['Low', 'Medium', 'High', 'Critical']
        fig = plot_confusion_matrix(y_test, y_pred_best, classes)
        st.pyplot(fig)
        plt.close()
        
        # Feature Importance
        st.subheader("🔍 Feature Importance Analysis")
        fig = plot_feature_importance(best_model, feature_cols, 15)
        if fig:
            st.pyplot(fig)
            plt.close()
        
        # Classification Report
        with st.expander("📋 Detailed Classification Report"):
            report = classification_report(y_test, y_pred_best, target_names=classes)
            st.code(report)
        
        # Cross-validation results
        with st.expander("🔄 Cross-Validation Results"):
            cv_scores = cross_val_score(best_model, X, y_level, cv=5)
            st.write(f"5-Fold Cross-Validation Scores: {cv_scores}")
            st.write(f"Mean CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Risk Prediction Section
    elif section == "🔮 Risk Prediction":
        st.markdown("## 🔮 Interactive Risk Prediction")
        st.markdown("Enter environmental parameters to predict microplastic pollution risk:")
        
        col1, col2 = st.columns(2)
        
        with col1:
            mp_count = st.number_input("🔬 Microplastic Count per Liter", 
                                       min_value=0.0, max_value=500.0, value=100.0, step=10.0)
            industrial_discharge = st.number_input("🏭 Industrial Discharge (m³/day)", 
                                                   min_value=0.0, max_value=5000.0, value=500.0, step=50.0)
            population_density = st.number_input("👥 Population Density (per km²)", 
                                                 min_value=0.0, max_value=50000.0, value=5000.0, step=500.0)
            treatment_efficiency = st.slider("💧 Wastewater Treatment Efficiency (%)", 
                                             min_value=0, max_value=100, value=70)
            water_temp = st.number_input("🌡️ Water Temperature (°C)", 
                                         min_value=0.0, max_value=40.0, value=22.0, step=1.0)
        
        with col2:
            ph_level = st.number_input("🧪 pH Level", min_value=5.0, max_value=9.0, value=7.5, step=0.1)
            salinity = st.number_input("💧 Salinity (ppt)", min_value=0.0, max_value=50.0, value=25.0, step=1.0)
            dissolved_oxygen = st.number_input("🫧 Dissolved Oxygen (mg/L)", 
                                               min_value=0.0, max_value=15.0, value=6.5, step=0.5)
            turbidity = st.number_input("🌫️ Turbidity (NTU)", min_value=0.0, max_value=100.0, value=20.0, step=5.0)
            location = st.selectbox("📍 Location Type", df['Location'].unique())
            polymer = st.selectbox("🔬 Polymer Type", df['Polymer_Type'].unique())
        
        if st.button("🔮 Predict Risk Level", use_container_width=True):
            input_data = {
                'MP_Count_per_L': mp_count,
                'Industrial_Discharge_m3': industrial_discharge,
                'Population_Density_km2': population_density,
                'Wastewater_Treatment_Efficiency': treatment_efficiency,
                'Water_Temperature_C': water_temp,
                'pH_Level': ph_level,
                'Salinity_ppt': salinity,
                'Dissolved_Oxygen_mgL': dissolved_oxygen,
                'Turbidity_NTU': turbidity,
                'Location': location,
                'Polymer_Type': polymer,
                'Distance_to_Shore_km': 50,
                'Water_Flow_Speed_ms': 1.0,
                'Rainfall_mm': 100,
                'Season': 'Summer',
                'Color_Distribution': 'White/Clear',
                'Size_Range': '0.1-0.5mm'
            }
            
            # Calculate risk score
            risk_score = (
                (mp_count / 500) * 35 +
                (industrial_discharge / 5000) * 15 +
                (population_density / 50000) * 15 +
                ((100 - treatment_efficiency) / 100) * 5
            ) * 100 / 70
            
            risk_score = min(100, max(0, risk_score))
            
            # Determine risk level
            if risk_score < 30:
                risk_level = "Low"
                color = "#2ECC71"
                icon = "🟢"
                recommendation = "Continue regular monitoring. No immediate action required."
            elif risk_score < 55:
                risk_level = "Medium"
                color = "#F39C12"
                icon = "🟡"
                recommendation = "Increase monitoring frequency. Consider preventive measures."
            elif risk_score < 75:
                risk_level = "High"
                color = "#E67E22"
                icon = "🟠"
                recommendation = "Immediate attention required. Implement mitigation strategies."
            else:
                risk_level = "Critical"
                color = "#E74C3C"
                icon = "🔴"
                recommendation = "URGENT! Immediate intervention required. Alert authorities."
            
            # Display results
            st.markdown("---")
            st.markdown("## 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"""
                <div style="background-color:{color}; padding:1rem; border-radius:15px; text-align:center; color:white;">
                    <h3>Risk Level</h3>
                    <h1>{icon} {risk_level}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding:1rem; border-radius:15px; text-align:center; color:white;">
                    <h3>Risk Score</h3>
                    <h1>{risk_score:.1f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div style="background:linear-gradient(135deg, #11998e 0%, #38ef7d 100%); padding:1rem; border-radius:15px; text-align:center; color:white;">
                    <h3>MP Count/L</h3>
                    <h1>{mp_count:.0f}</h1>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div class="info-box">
                <strong>📋 Recommendation:</strong><br>
                {recommendation}
            </div>
            """, unsafe_allow_html=True)
            
            # Gauge chart
            fig, ax = plt.subplots(figsize=(10, 3))
            ax.barh([0], [100], height=0.5, color='lightgray', alpha=0.5)
            ax.barh([0], [risk_score], height=0.5, color=color, alpha=0.8)
            ax.set_xlim(0, 100)
            ax.set_yticks([])
            ax.set_xlabel('Risk Score')
            ax.set_title('Risk Score Gauge')
            ax.axvline(x=30, color='green', linestyle='--', alpha=0.5)
            ax.axvline(x=55, color='orange', linestyle='--', alpha=0.5)
            ax.axvline(x=75, color='red', linestyle='--', alpha=0.5)
            ax.text(risk_score, -0.1, f'{risk_score:.1f}', ha='center', fontweight='bold')
            st.pyplot(fig)
            plt.close()
    
    # Data Explorer Section
    elif section == "📋 Data Explorer":
        st.markdown("## 📋 Data Explorer")
        
        # Filters
        st.subheader("🔍 Filter Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_risk = st.multiselect("Risk Level", df['Risk_Level'].unique(), default=df['Risk_Level'].unique())
        with col2:
            selected_location = st.multiselect("Location", df['Location'].unique(), default=df['Location'].unique())
        with col3:
            selected_polymer = st.multiselect("Polymer Type", df['Polymer_Type'].unique(), default=df['Polymer_Type'].unique())
        
        # Filter data
        filtered_df = df[
            (df['Risk_Level'].isin(selected_risk)) &
            (df['Location'].isin(selected_location)) &
            (df['Polymer_Type'].isin(selected_polymer))
        ]
        
        st.write(f"**Showing {len(filtered_df):,} of {len(df):,} records**")
        
        # Display data
        st.dataframe(filtered_df, use_container_width=True)
        
        # Export option
        if st.button("📥 Export to CSV"):
            csv = filtered_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="microplastic_data.csv">Download CSV File</a>'
            st.markdown(href, unsafe_allow_html=True)
    
    # Reports Section
    elif section == "📊 Reports":
        st.markdown("## 📊 Generate Reports")
        
        report_type = st.selectbox("Select Report Type", 
                                   ["Risk Summary Report", "Location Analysis Report", "Polymer Analysis Report", "Full Report"])
        
        if st.button("📄 Generate Report", use_container_width=True):
            st.markdown("---")
            st.markdown(f"## 📊 {report_type}")
            st.markdown(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            st.markdown("---")
            
            if report_type == "Risk Summary Report":
                st.markdown("### Risk Level Summary")
                risk_summary = df['Risk_Level'].value_counts()
                st.dataframe(risk_summary)
                
                fig = plot_risk_distribution(df)
                st.pyplot(fig)
                plt.close()
                
                st.markdown("### Key Statistics")
                st.write(f"- Total Samples: {len(df):,}")
                st.write(f"- Average Risk Score: {df['Risk_Score'].mean():.2f}")
                st.write(f"- Median Risk Score: {df['Risk_Score'].median():.2f}")
                st.write(f"- Critical Risk Areas: {(df['Risk_Level'] == 'Critical').sum():,}")
                st.write(f"- High Risk Areas: {(df['Risk_Level'] == 'High').sum():,}")
            
            elif report_type == "Location Analysis Report":
                st.markdown("### Risk Score by Location")
                location_summary = df.groupby('Location').agg({
                    'Risk_Score': ['mean', 'min', 'max', 'count'],
                    'MP_Count_per_L': 'mean'
                }).round(2)
                st.dataframe(location_summary)
                
                fig = plot_location_risk(df)
                st.pyplot(fig)
                plt.close()
                
                st.markdown("### Highest Risk Locations")
                high_risk_locs = df.groupby('Location')['Risk_Score'].mean().sort_values(ascending=False).head(5)
                for loc, score in high_risk_locs.items():
                    st.write(f"- {loc}: {score:.1f}")
            
            elif report_type == "Polymer Analysis Report":
                st.markdown("### Polymer Type Risk Assessment")
                polymer_summary = df.groupby('Polymer_Type').agg({
                    'Risk_Score': ['mean', 'std'],
                    'MP_Count_per_L': 'mean'
                }).round(2)
                st.dataframe(polymer_summary)
                
                fig = plot_polymer_risk(df)
                st.pyplot(fig)
                plt.close()
            
            else:  # Full Report
                st.markdown("### Complete Analysis Report")
                
                st.markdown("#### 1. Dataset Overview")
                st.write(f"- Total Records: {len(df):,}")
                st.write(f"- Features: {len(df.columns)}")
                st.write(f"- Locations: {df['Location'].nunique()}")
                st.write(f"- Polymer Types: {df['Polymer_Type'].nunique()}")
                
                st.markdown("#### 2. Risk Analysis")
                st.write(f"- Average Risk Score: {df['Risk_Score'].mean():.2f}")
                st.write(f"- Risk Score Std Dev: {df['Risk_Score'].std():.2f}")
                st.write(f"- Minimum Risk Score: {df['Risk_Score'].min():.2f}")
                st.write(f"- Maximum Risk Score: {df['Risk_Score'].max():.2f}")
                
                st.markdown("#### 3. Model Performance")
                st.write(f"- Best Model: {best_model_name}")
                st.write(f"- Accuracy: {results[best_model_name]['accuracy']:.2%}")
                st.write(f"- F1-Score: {results[best_model_name]['f1']:.3f}")
                
                st.markdown("#### 4. Recommendations")
                st.markdown("""
                1. Prioritize monitoring in Industrial Areas and Urban Runoff locations
                2. Focus on reducing Polystyrene (PS) and PVC emissions
                3. Improve wastewater treatment efficiency in high-risk zones
                4. Implement early warning systems for critical risk areas
                5. Regular monitoring of MP count in water bodies
                """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>© 2025 Microplastic Pollution Risk Prediction System | Agusan del Sur State College of Agriculture and Technology</p>
        <p>Developed by: Matthew Joseph Viernes & Shane Mark Magdaluyo</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APPLICATION
# ============================================================================
if __name__ == "__main__":
    main()
