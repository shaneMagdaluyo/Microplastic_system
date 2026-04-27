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
    if 'data' not in st.session_state: st.session_state.data = None
    if 'processed_data' not in st.session_state: st.session_state.processed_data = None
    if 'models' not in st.session_state: st.session_state.models = {}
    if 'feature_importance' not in st.session_state: st.session_state.feature_importance = None
    if 'best_model' not in st.session_state: st.session_state.best_model = None
    if 'preprocessing_log' not in st.session_state: st.session_state.preprocessing_log = []
    if 'trained' not in st.session_state: st.session_state.trained = False
    if 'X_test' not in st.session_state: st.session_state.X_test = None
    if 'y_test' not in st.session_state: st.session_state.y_test = None
    if 'X_train' not in st.session_state: st.session_state.X_train = None
    if 'y_train' not in st.session_state: st.session_state.y_train = None
    if 'encoders' not in st.session_state: st.session_state.encoders = {}
    if 'scaler' not in st.session_state: st.session_state.scaler = None
    if 'target_encoder' not in st.session_state: st.session_state.target_encoder = None
    if 'selected_features' not in st.session_state: st.session_state.selected_features = None
    if 'scaled_columns' not in st.session_state: st.session_state.scaled_columns = None
    if 'scaled_data' not in st.session_state: st.session_state.scaled_data = None
    if 'encoded_data' not in st.session_state: st.session_state.encoded_data = None

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
                        if pd.isna(median_val): median_val = 0
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
    """Encode categorical variables using LabelEncoder."""
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

def one_hot_encode(df):
    """Apply one-hot encoding to categorical columns."""
    try:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        # Remove ID columns from encoding
        cols_to_encode = [col for col in categorical_cols if 'ID' not in col and 'Sample' not in col]
        
        if len(cols_to_encode) == 0:
            return df, []
        
        # Apply one-hot encoding
        df_encoded = pd.get_dummies(df, columns=cols_to_encode, drop_first=False)
        
        # Get the new column names
        new_cols = [col for col in df_encoded.columns if col not in df.columns]
        
        return df_encoded, new_cols
    except Exception as e:
        st.error(f"Error in one-hot encoding: {str(e)}")
        return df, []

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
        if clean_data.empty: return go.Figure()
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
        if numeric_df.shape[1] < 2: return go.Figure(), None
        numeric_df = numeric_df.loc[:, numeric_df.std() > 0]
        corr_matrix = numeric_df.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=corr_matrix.columns.tolist(), y=corr_matrix.index.tolist(),
            colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}', textfont={"size": 10}, showscale=True
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
            st.error("❌ No numeric features selected.")
            return None, None
        y = df[target_col]
        valid_mask = y.notna()
        X = X[valid_mask]; y = y[valid_mask]
        if len(y) == 0: return None, None
        if y.dtype == 'object':
            le = LabelEncoder()
            y = pd.Series(le.fit_transform(y), index=y.index)
            st.session_state.target_encoder = le
        if X.isnull().sum().sum() > 0: X = X.fillna(X.median())
        return X, y
    except Exception as e:
        st.error(f"Error preparing modeling data: {str(e)}")
        return None, None

def train_models_fast(X_train, X_test, y_train, y_test):
    """Train classification models with optimized fast performance."""
    models = {}; training_times = {}
    try:
        n_samples = X_train.shape[0]
        t0 = time.time()
        try:
            lr = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', solver='lbfgs', n_jobs=-1)
            lr.fit(X_train, y_train)
            models['Logistic Regression'] = lr; training_times['Logistic Regression'] = time.time()-t0
        except:
            try:
                lr = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', solver='saga', n_jobs=-1)
                lr.fit(X_train, y_train)
                models['Logistic Regression'] = lr; training_times['Logistic Regression'] = time.time()-t0
            except: pass
        t0 = time.time()
        try:
            rf = RandomForestClassifier(n_estimators=min(80,max(30,n_samples//5)), random_state=42,
                                        class_weight='balanced', max_depth=min(12,n_samples//30), n_jobs=-1)
            rf.fit(X_train, y_train)
            models['Random Forest'] = rf; training_times['Random Forest'] = time.time()-t0
        except:
            try:
                rf = RandomForestClassifier(n_estimators=30, random_state=42, max_depth=8, n_jobs=-1)
                rf.fit(X_train, y_train)
                models['Random Forest'] = rf; training_times['Random Forest'] = time.time()-t0
            except: pass
        t0 = time.time()
        try:
            dt = DecisionTreeClassifier(random_state=42, max_depth=min(10,max(3,n_samples//30)),
                                        min_samples_split=max(2,n_samples//50), class_weight='balanced')
            dt.fit(X_train, y_train)
            models['Decision Tree'] = dt; training_times['Decision Tree'] = time.time()-t0
        except:
            try:
                dt = DecisionTreeClassifier(random_state=42, max_depth=5)
                dt.fit(X_train, y_train)
                models['Decision Tree'] = dt; training_times['Decision Tree'] = time.time()-t0
            except: pass
        return models, training_times
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return {}, {}

def train_models_quality(X_train, X_test, y_train, y_test):
    """Train classification models with GridSearch for better quality."""
    models = {}; training_times = {}
    try:
        t0 = time.time()
        try:
            grid = GridSearchCV(LogisticRegression(random_state=42, class_weight='balanced', n_jobs=-1),
                               {'C':[0.1,1,10], 'max_iter':[1000]}, cv=3, scoring='f1_weighted')
            grid.fit(X_train, y_train)
            models['Logistic Regression'] = grid.best_estimator_; training_times['Logistic Regression'] = time.time()-t0
        except: pass
        t0 = time.time()
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
            rf.fit(X_train, y_train)
            models['Random Forest'] = rf; training_times['Random Forest'] = time.time()-t0
        except: pass
        t0 = time.time()
        try:
            dt = DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced')
            dt.fit(X_train, y_train)
            models['Decision Tree'] = dt; training_times['Decision Tree'] = time.time()-t0
        except: pass
        return models, training_times
    except Exception as e:
        st.error(f"Error in model training: {str(e)}")
        return {}, {}

def evaluate_models(models, X_test, y_test):
    """Evaluate trained models."""
    results = {}
    try:
        for name, model in models.items():
            try:
                y_pred = model.predict(X_test)
                results[name] = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred, average='weighted'),
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred)
                }
            except: pass
        return results
    except: return {}

def main():
    """Main application function."""
    
    st.markdown('<p class="main-header">🔬 Microplastic Risk Analysis Dashboard</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio("Select Section", [
        "🏠 Home", "🔧 Preprocessing", "🛠️ Feature Selection & Relevance", 
        "🤖 Modeling", "📊 Cross Validation"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard analyzes microplastic risk data.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Status")
    if st.session_state.data is not None: st.sidebar.success("✅ Data Loaded")
    else: st.sidebar.warning("⚠️ No Data")
    if st.session_state.trained: st.sidebar.success(f"✅ Models Trained ({len(st.session_state.models)})")
    else: st.sidebar.warning("⚠️ Models Not Trained")
    
    # ==================== HOME ====================
    if section == "🏠 Home":
        st.markdown('<p class="section-header">🏠 Home - Upload Dataset</p>', unsafe_allow_html=True)
        
        c1, c2 = st.columns([2, 1])
        with c1:
            f = st.file_uploader("Upload dataset (CSV/Excel)", type=['csv','xlsx','xls'])
            if f: load_dataset(f)
        with c2:
            if st.button("Generate Sample Dataset", type="primary"):
                st.session_state.data = generate_sample_data()
                st.success("✅ Sample dataset generated!")
                st.rerun()
        
        if st.session_state.data is not None:
            df = st.session_state.data
            st.markdown("---")
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Samples", df.shape[0])
            with c2: st.metric("Features", df.shape[1])
            with c3: st.metric("Missing", df.isnull().sum().sum())
            st.dataframe(df.head(10), use_container_width=True)
            
            # Feature Scaling
            st.markdown("---")
            st.markdown("### 📏 Feature Scaling Preview")
            if st.button("🔧 Apply StandardScaler", type="primary", key="scale_home"):
                with st.spinner('Scaling...'):
                    nums = df.select_dtypes(include=['float64','int64']).columns.tolist()
                    cols = [c for c in nums if 'ID' not in c and 'Sample' not in c]
                    if len(cols) > 0:
                        scaler = StandardScaler()
                        sd = scaler.fit_transform(df[cols].fillna(df[cols].median()))
                        sdf = pd.DataFrame(sd, columns=cols)
                        st.session_state.scaler = scaler
                        st.success(f"✅ {len(cols)} columns scaled!")
                        st.dataframe(sdf.head(), column_config={c: st.column_config.NumberColumn(c,format="%.6f") for c in cols}, use_container_width=True)
            
            # Risk Score vs MP Count
            if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
                st.markdown("---")
                st.markdown("### 🔬 Risk Score vs MP Count per L")
                df['MP_Count_per_L'] = pd.to_numeric(df['MP_Count_per_L'], errors='coerce')
                df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                clean = df.dropna(subset=['MP_Count_per_L','Risk_Score'])
                if len(clean) > 0:
                    tab1, tab2 = st.tabs(["📊 Scatter", "📈 Trendline"])
                    with tab1:
                        fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                        color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                        title='MP Count vs Risk Score', opacity=0.7)
                        st.plotly_chart(fig, use_container_width=True)
                    with tab2:
                        try:
                            fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                            color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                            trendline='ols', title='MP Count vs Risk Score with Trendline', opacity=0.7)
                            st.plotly_chart(fig, use_container_width=True)
                        except: st.warning("⚠️ Trendline not available")
            
            # Risk Score by Risk Level
            if 'Risk_Score' in df.columns and 'Risk_Level' in df.columns:
                st.markdown("---")
                st.markdown("### 📊 Risk Score by Risk Level")
                df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
                clean = df.dropna(subset=['Risk_Score'])
                clean['Risk_Level'] = clean['Risk_Level'].astype(str)
                if len(clean) > 0:
                    tab1, tab2 = st.tabs(["📦 Box Plot", "📊 Stats"])
                    with tab1:
                        fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                                    title='Risk Score by Risk Level', points='outliers')
                        st.plotly_chart(fig, use_container_width=True)
                    with tab2:
                        stats = clean.groupby('Risk_Level')['Risk_Score'].agg(['count','mean','median','std','min','max']).round(2)
                        stats.columns = ['Count','Mean','Median','Std Dev','Min','Max']
                        st.dataframe(stats, use_container_width=True)
            
            # Dataset Info
            st.markdown("---")
            c1,c2 = st.columns(2)
            with c1: st.write("**Data Types:**", df.dtypes)
            with c2: st.write("**Statistics:**", df.describe())
            
            # Quality Check
            st.markdown("---")
            c1,c2,c3,c4 = st.columns(4)
            with c1: st.metric("Missing %", f"{(df.isnull().sum().sum()/(df.shape[0]*df.shape[1]))*100:.2f}%")
            with c2: st.metric("Duplicates", df.duplicated().sum())
            with c3: st.metric("Numeric Cols", len(df.select_dtypes(include=['float64','int64']).columns))
            with c4: st.metric("Categorical Cols", len(df.select_dtypes(include=['object']).columns))
    
    # ==================== PREPROCESSING ====================
    elif section == "🔧 Preprocessing":
        st.markdown('<p class="section-header">🔧 Data Preprocessing</p>', unsafe_allow_html=True)
        
        if st.session_state.data is None:
            st.warning("⚠️ Please upload a dataset first!")
            return
        
        df = st.session_state.data.copy()
        
        # ===== FEATURE SCALING =====
        st.markdown("### 📏 Perform Feature Scaling")
        st.markdown("*Apply StandardScaler to the numerical columns*")
        
        if st.button("🔧 Apply Feature Scaling (StandardScaler)", type="primary", key="scale_prep"):
            with st.spinner('Applying StandardScaler...'):
                numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
                cols_to_scale = [col for col in numeric_cols if 'ID' not in col and 'Sample' not in col]
                
                if len(cols_to_scale) > 0:
                    scaler = StandardScaler()
                    scaled_data = scaler.fit_transform(df[cols_to_scale].fillna(df[cols_to_scale].median()))
                    scaled_df = pd.DataFrame(scaled_data, columns=cols_to_scale)
                    
                    st.session_state.scaler = scaler
                    st.session_state.scaled_columns = cols_to_scale
                    st.session_state.scaled_data = scaled_df
                    
                    st.success(f"✅ Feature scaling applied to {len(cols_to_scale)} numerical columns!")
                    
                    st.markdown("**First 5 rows of scaled numerical data:**")
                    st.dataframe(
                        scaled_df.head(),
                        column_config={col: st.column_config.NumberColumn(col, format="%.6f") for col in cols_to_scale},
                        use_container_width=True,
                    )
                    
                    with st.expander("📊 Scaling Statistics (Before vs After)"):
                        stats_list = []
                        for col in cols_to_scale[:10]:
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
        
        # ===== CATEGORICAL ENCODING =====
        st.markdown("---")
        st.markdown("### 🔄 Encode Categorical Variables")
        st.markdown("*Apply One-Hot Encoding to the categorical columns*")
        
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        cols_to_encode = [col for col in categorical_cols if 'ID' not in col and 'Sample' not in col]
        
        if len(cols_to_encode) > 0:
            st.markdown(f"**Categorical columns found ({len(cols_to_encode)}):** {', '.join(cols_to_encode)}")
        else:
            st.info("No categorical columns found to encode.")
        
        if st.button("🔄 Apply One-Hot Encoding", type="primary", key="encode_prep"):
            with st.spinner('Applying One-Hot Encoding...'):
                if len(cols_to_encode) > 0:
                    # Apply one-hot encoding
                    encoded_df, new_cols = one_hot_encode(df)
                    
                    st.session_state.encoded_data = encoded_df
                    
                    st.success(f"✅ One-Hot Encoding applied! Created {len(new_cols)} new columns.")
                    st.markdown(f"**Original shape:** {df.shape} → **Encoded shape:** {encoded_df.shape}")
                    
                    st.markdown("**First 5 rows of the DataFrame after one-hot encoding:**")
                    # Show first 5 rows with all columns (limit display to avoid overflow)
                    display_cols = list(df.columns[:5]) + new_cols[:10] if len(new_cols) > 10 else list(df.columns[:5]) + new_cols
                    display_cols = [c for c in display_cols if c in encoded_df.columns]
                    st.dataframe(encoded_df[display_cols].head(), use_container_width=True)
                    
                    with st.expander("📋 All New One-Hot Encoded Columns"):
                        st.write(new_cols)
                else:
                    st.warning("⚠️ No categorical columns found to encode.")
        
        # ===== ADDITIONAL PREPROCESSING =====
        st.markdown("---")
        st.markdown("### 🔧 Additional Preprocessing Steps")
        
        preprocessing_options = st.multiselect(
            "Select Preprocessing Steps",
            ["Handle Missing Values", "Label Encode Categorical Variables",
             "Detect Outliers", "Scale Features (Full Dataset)"],
            default=["Handle Missing Values"]
        )
        
        if st.button("Run Preprocessing", type="primary", key="run_prep"):
            processed_df = df.copy()
            
            if "Handle Missing Values" in preprocessing_options:
                with st.spinner('Handling missing values...'):
                    processed_df = handle_missing_values(processed_df)
                    st.success("✅ Missing values handled")
            
            if "Label Encode Categorical Variables" in preprocessing_options:
                with st.spinner('Encoding...'):
                    processed_df = encode_categorical(processed_df)
                    st.success("✅ Label encoding applied")
            
            if "Detect Outliers" in preprocessing_options:
                numeric_cols = processed_df.select_dtypes(include=['float64','int64']).columns
                outlier_info = detect_outliers(processed_df, numeric_cols)
                for col, info in outlier_info.items():
                    if info['count'] > 0:
                        st.write(f"**{col}**: {info['count']} outliers ({info['percentage']:.1f}%)")
            
            if "Scale Features (Full Dataset)" in preprocessing_options:
                numeric_cols = processed_df.select_dtypes(include=['float64','int64']).columns
                processed_df = scale_features(processed_df, numeric_cols)
                st.success("✅ Features scaled")
            
            st.session_state.processed_data = processed_df
            st.success("✅ Preprocessing completed!")
        
        if st.session_state.processed_data is not None:
            st.markdown("---")
            st.markdown("### 📋 Processed Data Preview")
            st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)
    
    # ==================== FEATURE SELECTION & RELEVANCE ====================
    elif section == "🛠️ Feature Selection & Relevance":
        st.markdown('<p class="section-header">🛠️ Feature Selection & Relevance</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data.copy()
        
        st.markdown("### 📈 Exploratory Data Analysis")
        
        if 'Risk_Score' in df.columns:
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean = df['Risk_Score'].dropna()
            if len(clean) > 0:
                st.plotly_chart(plot_distribution(df, 'Risk_Score', 'Risk Score Distribution'), use_container_width=True)
                c1,c2 = st.columns(2)
                with c1:
                    q1,q3 = clean.quantile(0.25), clean.quantile(0.75)
                    stats = [('Count',f'{len(clean):,}'),('Mean',f'{clean.mean():.4f}'),('Median',f'{clean.median():.4f}'),
                             ('Std Dev',f'{clean.std():.4f}'),('Min',f'{clean.min():.4f}'),('Q1',f'{q1:.4f}'),
                             ('Q3',f'{q3:.4f}'),('IQR',f'{q3-q1:.4f}'),('Max',f'{clean.max():.4f}')]
                    st.dataframe(pd.DataFrame(stats,columns=['Statistic','Value']), use_container_width=True, hide_index=True)
                with c2:
                    cats = [('🟢 Low','0-25',(clean<25).sum()),('🟡 Medium','25-50',((clean>=25)&(clean<50)).sum()),
                            ('🟠 High','50-75',((clean>=50)&(clean<75)).sum()),('🔴 Critical','75-100',(clean>=75).sum())]
                    for cat,rng,cnt in cats:
                        st.markdown(f"**{cat}** ({rng}): {cnt:,} ({(cnt/len(clean))*100:.1f}%)")
                        st.progress(int((cnt/len(clean))*100))
        
        if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
            st.markdown("---")
            st.markdown("#### 🔬 MP Count vs Risk Score")
            df['MP_Count_per_L'] = pd.to_numeric(df['MP_Count_per_L'], errors='coerce')
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean = df.dropna(subset=['MP_Count_per_L','Risk_Score'])
            if not clean.empty:
                try:
                    fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                    color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                    trendline='ols', title='MP Count vs Risk Score')
                except:
                    fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                    color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                    title='MP Count vs Risk Score')
                st.plotly_chart(fig, use_container_width=True)
        
        if 'Risk_Level' in df.columns and 'Risk_Score' in df.columns:
            st.markdown("---")
            st.markdown("#### 📊 Risk Score by Risk Level")
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean = df.dropna(subset=['Risk_Score'])
            clean['Risk_Level'] = clean['Risk_Level'].astype(str)
            if len(clean) > 0:
                fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                            title='Risk Score by Risk Level')
                st.plotly_chart(fig, use_container_width=True)
                stats = clean.groupby('Risk_Level')['Risk_Score'].agg(['count','mean','median','std','min','max']).round(2)
                stats.columns = ['Count','Mean','Median','Std Dev','Min','Max']
                st.dataframe(stats, use_container_width=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Feature Engineering")
        target = st.selectbox("Target Variable", df.columns.tolist(),
                             index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        nums = df.select_dtypes(include=['float64','int64','int32']).columns.tolist()
        if target in nums: nums.remove(target)
        if len(nums) > 1:
            fig,_ = plot_correlation_heatmap(df, nums)
            st.plotly_chart(fig, use_container_width=True)
        if st.button("Calculate Feature Importance", type="primary"):
            X = df[nums].fillna(df[nums].median()).dropna(axis=1)
            y = df[target]
            if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            imp = pd.DataFrame({'feature':X.columns, 'importance':rf.feature_importances_}).sort_values('importance', ascending=True)
            st.session_state.feature_importance = imp
            fig = px.bar(imp.tail(15), x='importance', y='feature', orientation='h', title='Top 15 Features', height=400)
            st.plotly_chart(fig, use_container_width=True)
            top = imp.nlargest(10,'importance')['feature'].tolist()
            st.session_state.selected_features = top
            st.success(f"✅ Top {len(top)} features selected")
    
    # ==================== MODELING ====================
    elif section == "🤖 Modeling":
        st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data
        
        target = st.selectbox("Target", df.columns.tolist(), key='train_target')
        all_f = [c for c in df.columns if c != target]
        default = st.session_state.get('selected_features', df.select_dtypes(include=['float64','int64']).columns.tolist()[:5])
        default = [f for f in default if f in all_f]
        features = st.multiselect("Features", all_f, default=default)
        c1,c2 = st.columns(2)
        with c1: ts = st.slider("Test Size", 0.1, 0.5, 0.2); rs = st.number_input("Random State", 0, 100, 42)
        with c2: use_smote = st.checkbox("Use SMOTE", value=True); fast = st.checkbox("⚡ Fast Mode", value=True)
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if len(features) == 0: st.error("Select features!"); return
            X, y = prepare_modeling_data(df, features, target)
            if X is None: return
            counts = pd.Series(y).value_counts()
            st.write("Class Distribution:", counts)
            try: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
            except: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
            if use_smote:
                tc = pd.Series(y_train).value_counts()
                if tc.min() >= 2:
                    try:
                        X_train, y_train = SMOTE(random_state=rs, k_neighbors=min(5,tc.min()-1)).fit_resample(X_train, y_train)
                        st.success("✅ SMOTE applied!")
                    except: pass
            t0 = time.time()
            if fast: models, times = train_models_fast(X_train, X_test, y_train, y_test)
            else: models, times = train_models_quality(X_train, X_test, y_train, y_test)
            tt = time.time() - t0
            if models:
                st.session_state.models = models
                st.session_state.X_test = X_test; st.session_state.y_test = y_test
                st.session_state.trained = True
                st.success(f"✅ {len(models)} models trained in {tt:.2f}s!")
                st.balloons()
                
                eval_results = evaluate_models(models, X_test, y_test)
                if eval_results:
                    all_acc = [r['accuracy'] for r in eval_results.values()]
                    all_f1 = [r['f1_score'] for r in eval_results.values()]
                    avg_acc = np.mean(all_acc); avg_f1 = np.mean(all_f1)
                    
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #1f77b4, #2c3e50); 
                                padding: 25px; border-radius: 15px; margin: 20px 0; text-align: center;">
                        <h2 style="color: white; margin: 0;">📊 Average Model Performance</h2>
                        <div style="display: flex; justify-content: center; gap: 40px; margin-top: 15px;">
                            <div><p style="color: #ffd700; margin: 0;">Avg Accuracy</p>
                                <p style="color: white; font-size: 2rem; font-weight: bold; margin: 5px 0;">{avg_acc:.4f}</p></div>
                            <div style="border-left: 2px solid rgba(255,255,255,0.3); padding-left: 40px;">
                                <p style="color: #ffd700; margin: 0;">Avg F1 Score</p>
                                <p style="color: white; font-size: 2rem; font-weight: bold; margin: 5px 0;">{avg_f1:.4f}</p></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"**From your evaluation for {target} prediction:**")
                    for name, r in eval_results.items():
                        st.markdown(f"**{name}:** F1-Score = **{r['f1_score']:.4f}** | Accuracy = **{r['accuracy']:.4f}**")
                    
                    best = max(eval_results.items(), key=lambda x: x[1]['f1_score'])
                    st.markdown(f"""
                    <div style="background: #d4edda; border: 2px solid #27ae60; border-radius: 10px; padding: 20px; margin: 15px 0;">
                        <p style="font-size: 1.1rem; margin: 0; color: #155724;">
                            ✅ The <b>{best[0]}</b> performed best with F1-Score of <b>{best[1]['f1_score']:.4f}</b>
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    sd = [{'Model':n,'Accuracy':r['accuracy'],'F1-Score':r['f1_score']} for n,r in eval_results.items()]
                    sd.append({'Model':'📊 AVERAGE','Accuracy':avg_acc,'F1-Score':avg_f1})
                    st.dataframe(pd.DataFrame(sd), column_config={"Model":"Model","Accuracy":st.column_config.NumberColumn("Accuracy",format="%.4f"),"F1-Score":st.column_config.NumberColumn("F1-Score",format="%.4f")}, use_container_width=True, hide_index=True)
                    
                    md = {n:{'Accuracy':r['accuracy'],'F1 Score':r['f1_score']} for n,r in eval_results.items()}
                    mdf = pd.DataFrame(md).T
                    fig = px.bar(mdf.reset_index(), x='index', y=['Accuracy','F1 Score'], barmode='group',
                                title='Model Performance', color_discrete_sequence=['#3498db','#e74c3c'])
                    fig.add_hline(y=avg_acc, line_dash="dash", line_color="#3498db", annotation_text=f"Avg Acc: {avg_acc:.3f}")
                    fig.add_hline(y=avg_f1, line_dash="dash", line_color="#e74c3c", annotation_text=f"Avg F1: {avg_f1:.3f}")
                    st.plotly_chart(fig, use_container_width=True)
                    
                    sel = st.selectbox("Confusion Matrix", list(eval_results.keys()), key='cm')
                    if sel:
                        cm = eval_results[sel].get('confusion_matrix')
                        if cm is not None and cm.size > 0:
                            n = cm.shape[0]
                            fig_cm = go.Figure(data=go.Heatmap(z=cm, x=[f'Pred {i}' for i in range(n)],
                                y=[f'Actual {i}' for i in range(n)], colorscale='Blues',
                                text=[[str(int(v)) for v in row] for row in cm], texttemplate="%{text}", textfont={"size":14}, showscale=True))
                            fig_cm.update_layout(title=f'{sel} - Confusion Matrix', height=400)
                            st.plotly_chart(fig_cm, use_container_width=True)
                    
                    rep = st.selectbox("Classification Report", list(eval_results.keys()), key='rep')
                    if rep: st.code(eval_results[rep]['classification_report'])
                    
                    st.success(f"🏆 Best: **{best[0]}** | 📊 Avg Acc: **{avg_acc:.4f}** | 📊 Avg F1: **{avg_f1:.4f}**")
    
    # ==================== CROSS VALIDATION ====================
    elif section == "📊 Cross Validation":
        st.markdown('<p class="section-header">📊 Cross Validation</p>', unsafe_allow_html=True)
        
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data
        
        target = st.selectbox("Target Variable for CV", df.columns.tolist(),
                             index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        nums = df.select_dtypes(include=['float64','int64','int32']).columns.tolist()
        if target in nums: nums.remove(target)
        folds = st.slider("CV Folds", 3, 10, 5)
        
        if st.button("🔄 Run Cross Validation", type="primary", use_container_width=True):
            X = df[nums].copy(); y = df[target].copy()
            mask = y.notna(); X = X[mask]; y = y[mask]
            if y.dtype == 'object': y = LabelEncoder().fit_transform(y)
            X = X.fillna(X.median())
            
            cv_models = {
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', n_jobs=-1),
                'Random Forest': RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1),
                'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=8, class_weight='balanced')
            }
            cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
            
            cv_results = []; all_scores = {}
            for name, model in cv_models.items():
                try:
                    acc = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
                    f1 = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted', n_jobs=-1)
                    all_scores[name] = f1
                    cv_results.append({'Model':name,'Mean Accuracy':round(acc.mean(),4),'Std Accuracy':round(acc.std(),4),
                                      'Mean F1':round(f1.mean(),4),'Std F1':round(f1.std(),4),
                                      'Min F1':round(f1.min(),4),'Max F1':round(f1.max(),4)})
                except: pass
            
            if cv_results:
                cv_df = pd.DataFrame(cv_results)
                st.dataframe(cv_df, use_container_width=True, hide_index=True)
                best_cv = cv_df.loc[cv_df['Mean F1'].idxmax()]
                st.success(f"🏆 Best CV Model: **{best_cv['Model']}** (Mean F1: {best_cv['Mean F1']:.4f} ±{best_cv['Std F1']:.4f})")
                
                fig_cv = go.Figure()
                for name, scores in all_scores.items():
                    fig_cv.add_trace(go.Box(y=scores, name=name, boxmean='sd'))
                fig_cv.update_layout(title=f'CV F1 Scores ({folds}-Fold)', yaxis_title='F1 Score', height=400)
                st.plotly_chart(fig_cv, use_container_width=True)


if __name__ == "__main__":
    main()
