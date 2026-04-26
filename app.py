"""
Microplastic Risk Analysis Dashboard
A comprehensive Streamlit application for analyzing microplastic risk data,
featuring data preprocessing, EDA, model training, and evaluation.
Enhanced with readable tables and professional heatmaps.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, GridSearchCV
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

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1f77b4; text-align: center; margin-bottom: 2rem; }
    .section-header { font-size: 1.8rem; font-weight: 600; color: #2c3e50; margin-top: 1rem; margin-bottom: 1rem; }
    .subsection-header { font-size: 1.4rem; font-weight: 500; color: #34495e; margin-top: 0.8rem; }
    .stButton > button { width: 100%; background-color: #1f77b4; color: white; font-weight: 600; border-radius: 8px; padding: 0.5rem 1rem; }
    .stButton > button:hover { background-color: #2980b9; border-color: #2980b9; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
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

init_session_state()

def load_dataset(uploaded_file):
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
            st.error("Unsupported file format.")
            return None
        st.session_state.data = data
        st.success(f"✅ Dataset loaded! Shape: {data.shape}")
        return data
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def generate_sample_data():
    np.random.seed(42)
    n = 1000
    data = {
        'Sample_ID': [f'MP_{i:04d}' for i in range(n)],
        'MP_Count_per_L': np.random.poisson(lam=50, size=n),
        'Particle_Size_um': np.random.normal(100, 30, n),
        'Polymer_Type': np.random.choice(['PE', 'PP', 'PS', 'PET', 'PVC', 'Nylon'], n),
        'Water_Source': np.random.choice(['River', 'Lake', 'Ocean', 'Groundwater', 'Tap'], n),
        'pH': np.random.normal(7, 0.5, n),
        'Temperature_C': np.random.normal(20, 5, n),
        'Turbidity_NTU': np.random.exponential(10, n),
        'Dissolved_O2_mgL': np.random.normal(8, 2, n),
        'Conductivity_uScm': np.random.normal(500, 150, n),
        'Risk_Score': np.random.uniform(0, 100, n),
        'Risk_Level': np.random.choice(['Low', 'Medium', 'High', 'Critical'], n, p=[0.3,0.35,0.25,0.1]),
        'Risk_Type': np.random.choice(['Type_A', 'Type_B', 'Type_C'], n, p=[0.5,0.3,0.2]),
        'Location': np.random.choice(['Urban', 'Rural', 'Industrial', 'Coastal'], n),
        'Season': np.random.choice(['Winter', 'Spring', 'Summer', 'Fall'], n)
    }
    df = pd.DataFrame(data)
    for col in df.columns:
        if col != 'Sample_ID' and df[col].dtype in ['float64', 'int64']:
            df.loc[np.random.random(n) < 0.05, col] = np.nan
    return df

def handle_missing_values(df):
    df = df.copy()
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['float64', 'int64']:
                m = df[col].median()
                df[col].fillna(m if not pd.isna(m) else 0, inplace=True)
            else:
                mode = df[col].mode()
                df[col].fillna(mode[0] if not mode.empty else 'Unknown', inplace=True)
    return df

def encode_categorical(df):
    df = df.copy()
    for col in df.select_dtypes(include=['object']).columns:
        if col not in ['Sample_ID']:
            df[f'{col}_Encoded'] = LabelEncoder().fit_transform(df[col].astype(str))
    return df

def scale_features(df, feature_cols):
    df = df.copy()
    scaler = StandardScaler()
    nums = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns
    if len(nums) > 0:
        df[nums] = scaler.fit_transform(df[nums])
        st.session_state.scaler = scaler
    return df

def detect_outliers(df, columns):
    info = {}
    for col in columns:
        if df[col].dtype in ['float64', 'int64']:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
            out = df[(df[col] < lo) | (df[col] > hi)]
            info[col] = {'count': len(out), 'percentage': (len(out)/len(df))*100 if len(df)>0 else 0}
    return info

def plot_distribution(data, column, title):
    clean = data[column].dropna()
    if clean.empty: return go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
    fig.add_trace(go.Histogram(x=clean, nbinsx=30, marker_color='#3498db'), row=1, col=1)
    fig.add_trace(go.Box(y=clean, marker_color='#e74c3c'), row=1, col=2)
    fig.update_layout(title_text=title, showlegend=False, height=500)
    return fig

def plot_correlation_heatmap(df, columns):
    nums = df[columns].select_dtypes(include=['float64', 'int64', 'int32'])
    if nums.shape[1] < 2: return go.Figure(), None
    nums = nums.loc[:, nums.std() > 0]
    corr = nums.corr()
    fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns.tolist(), y=corr.index.tolist(),
                                    colorscale='RdBu', zmin=-1, zmax=1, text=np.round(corr.values,2),
                                    texttemplate='%{text}', textfont={"size":10}, showscale=True))
    fig.update_layout(title='Correlation Heatmap', height=600)
    return fig, corr

def prepare_modeling_data(df, feature_cols, target_col):
    X = df[feature_cols].select_dtypes(include=['float64', 'int64', 'int32'])
    if X.shape[1] == 0: st.error("❌ No numeric features."); return None, None
    y = df[target_col]
    mask = y.notna()
    X, y = X[mask], y[mask]
    if len(y) == 0: return None, None
    if y.dtype == 'object':
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index)
        st.session_state.target_encoder = le
    if X.isnull().sum().sum() > 0: X = X.fillna(X.median())
    return X, y

def train_models_fast(X_train, X_test, y_train, y_test):
    models, times = {}, {}
    ns = X_train.shape[0]
    t0 = time.time()
    try:
        lr = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', solver='lbfgs', n_jobs=-1)
        lr.fit(X_train, y_train)
        models['Logistic Regression'] = lr; times['Logistic Regression'] = time.time()-t0
    except:
        try:
            lr = LogisticRegression(random_state=42, max_iter=500, class_weight='balanced', solver='saga', n_jobs=-1)
            lr.fit(X_train, y_train)
            models['Logistic Regression'] = lr; times['Logistic Regression'] = time.time()-t0
        except: pass
    t0 = time.time()
    try:
        rf = RandomForestClassifier(n_estimators=min(80,max(30,ns//5)), random_state=42,
                                    class_weight='balanced', max_depth=min(12,ns//30), n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf; times['Random Forest'] = time.time()-t0
    except:
        try:
            rf = RandomForestClassifier(n_estimators=30, random_state=42, max_depth=8, n_jobs=-1)
            rf.fit(X_train, y_train)
            models['Random Forest'] = rf; times['Random Forest'] = time.time()-t0
        except: pass
    t0 = time.time()
    try:
        dt = DecisionTreeClassifier(random_state=42, max_depth=min(10,max(3,ns//30)),
                                    min_samples_split=max(2,ns//50), class_weight='balanced')
        dt.fit(X_train, y_train)
        models['Decision Tree'] = dt; times['Decision Tree'] = time.time()-t0
    except:
        try:
            dt = DecisionTreeClassifier(random_state=42, max_depth=5)
            dt.fit(X_train, y_train)
            models['Decision Tree'] = dt; times['Decision Tree'] = time.time()-t0
        except: pass
    return models, times

def train_models_quality(X_train, X_test, y_train, y_test):
    models, times = {}, {}
    t0 = time.time()
    try:
        grid = GridSearchCV(LogisticRegression(random_state=42, class_weight='balanced', n_jobs=-1),
                           {'C':[0.1,1,10], 'max_iter':[1000]}, cv=3, scoring='f1_weighted')
        grid.fit(X_train, y_train)
        models['Logistic Regression'] = grid.best_estimator_; times['Logistic Regression'] = time.time()-t0
    except: pass
    t0 = time.time()
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf; times['Random Forest'] = time.time()-t0
    except: pass
    t0 = time.time()
    try:
        dt = DecisionTreeClassifier(random_state=42, max_depth=10, class_weight='balanced')
        dt.fit(X_train, y_train)
        models['Decision Tree'] = dt; times['Decision Tree'] = time.time()-t0
    except: pass
    return models, times

def evaluate_models(models, X_test, y_test):
    results = {}
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

def main():
    st.markdown('<p class="main-header">🔬 Microplastic Risk Analysis Dashboard</p>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## 📊 Navigation")
    section = st.sidebar.radio("Select Section", [
        "📁 Upload Dataset", "🔧 Data Preprocessing", "📈 EDA (Risk Analysis)",
        "🛠️ Feature Engineering", "🤖 Model Training", "📊 Model Evaluation",
        "🎯 Feature Importance", "🧬 Polymer Analysis"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.info("This dashboard analyzes microplastic risk data.")
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📌 Status")
    if st.session_state.data is not None: st.sidebar.success("✅ Data Loaded")
    else: st.sidebar.warning("⚠️ No Data")
    if st.session_state.trained: st.sidebar.success(f"✅ Models Trained ({len(st.session_state.models)})")
    else: st.sidebar.warning("⚠️ Models Not Trained")
    
    # ==================== UPLOAD ====================
    if section == "📁 Upload Dataset":
        st.markdown('<p class="section-header">📁 Upload Dataset</p>', unsafe_allow_html=True)
        c1, c2 = st.columns([2,1])
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
            st.markdown("#### 📋 Dataset Preview")
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Samples", df.shape[0])
            with c2: st.metric("Features", df.shape[1])
            with c3: st.metric("Missing", df.isnull().sum().sum())
            st.dataframe(df.head(10), use_container_width=True)
            st.markdown("---")
            st.markdown("#### Dataset Information")
            c1,c2 = st.columns(2)
            with c1: st.write("**Data Types:**", df.dtypes)
            with c2: st.write("**Statistics:**", df.describe())
    
    # ==================== PREPROCESSING ====================
    elif section == "🔧 Data Preprocessing":
        st.markdown('<p class="section-header">🔧 Data Preprocessing</p>', unsafe_allow_html=True)
        if st.session_state.data is None: st.warning("⚠️ Upload data first!"); return
        df = st.session_state.data.copy()
        opts = st.multiselect("Select Steps", ["Handle Missing Values","Encode Categorical Variables",
                              "Detect Outliers","Scale Features"], default=["Handle Missing Values","Encode Categorical Variables"])
        if st.button("Run Preprocessing", type="primary"):
            pdf = df.copy()
            if "Handle Missing Values" in opts:
                pdf = handle_missing_values(pdf); st.success("✅ Missing values handled")
            if "Encode Categorical Variables" in opts:
                pdf = encode_categorical(pdf); st.success("✅ Encoded")
            if "Detect Outliers" in opts:
                info = detect_outliers(pdf, pdf.select_dtypes(include=['float64','int64']).columns)
                for c,i in info.items():
                    if i['count']>0: st.write(f"**{c}**: {i['count']} outliers ({i['percentage']:.1f}%)")
            if "Scale Features" in opts:
                pdf = scale_features(pdf, pdf.select_dtypes(include=['float64','int64']).columns)
                st.success("✅ Scaled")
            st.session_state.processed_data = pdf
            st.success("✅ Preprocessing completed!")
        if st.session_state.processed_data is not None:
            st.markdown("### 📋 Processed Data Preview")
            st.dataframe(st.session_state.processed_data.head(10), use_container_width=True)
    
    # ==================== EDA ====================
    elif section == "📈 EDA (Risk Analysis)":
        st.markdown('<p class="section-header">📈 EDA</p>', unsafe_allow_html=True)
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data.copy()
        if 'Risk_Score' in df.columns:
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean = df['Risk_Score'].dropna()
            if len(clean)>0:
                st.plotly_chart(plot_distribution(df,'Risk_Score','Risk Score Distribution'), use_container_width=True)
                c1,c2,c3,c4 = st.columns(4)
                with c1: st.metric("Mean", f"{clean.mean():.2f}")
                with c2: st.metric("Median", f"{clean.median():.2f}")
                with c3: st.metric("Max", f"{clean.max():.2f}")
                with c4: st.metric("Min", f"{clean.min():.2f}")
        if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
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
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean = df.dropna(subset=['Risk_Score'])
            if len(clean)>0:
                fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                           title='Risk Score by Risk Level')
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== FEATURE ENGINEERING ====================
    elif section == "🛠️ Feature Engineering":
        st.markdown('<p class="section-header">🛠️ Feature Engineering</p>', unsafe_allow_html=True)
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data
        target = st.selectbox("Target Variable", df.columns.tolist(),
                             index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        nums = df.select_dtypes(include=['float64','int64','int32']).columns.tolist()
        if target in nums: nums.remove(target)
        if len(nums)>1:
            with st.spinner('Computing...'):
                fig,_ = plot_correlation_heatmap(df, nums)
                st.plotly_chart(fig, use_container_width=True)
        if st.button("Calculate Feature Importance", type="primary"):
            X = df[nums].fillna(df[nums].median()).dropna(axis=1)
            y = df[target]
            if y.dtype=='object': y = LabelEncoder().fit_transform(y)
            rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            imp = pd.DataFrame({'feature':X.columns, 'importance':rf.feature_importances_}).sort_values('importance', ascending=True)
            st.session_state.feature_importance = imp
            fig = px.bar(imp.tail(15), x='importance', y='feature', orientation='h', title='Top 15 Features')
            st.plotly_chart(fig, use_container_width=True)
            top = imp.nlargest(10,'importance')['feature'].tolist()
            st.session_state.selected_features = top
            st.success(f"✅ Top {len(top)} features selected")
    
    # ==================== MODEL TRAINING ====================
    elif section == "🤖 Model Training":
        st.markdown('<p class="section-header">🤖 Model Training</p>', unsafe_allow_html=True)
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data
        target = st.selectbox("Target", df.columns.tolist(), key='train_target')
        all_f = [c for c in df.columns if c!=target]
        default = st.session_state.get('selected_features', df.select_dtypes(include=['float64','int64']).columns.tolist()[:5])
        default = [f for f in default if f in all_f]
        features = st.multiselect("Features", all_f, default=default)
        c1,c2 = st.columns(2)
        with c1: ts = st.slider("Test Size", 0.1, 0.5, 0.2); rs = st.number_input("Random State", 0, 100, 42)
        with c2: use_smote = st.checkbox("Use SMOTE", value=True); fast = st.checkbox("⚡ Fast Mode", value=True)
        
        if st.button("🚀 Train Models", type="primary", use_container_width=True):
            if len(features)==0: st.error("Select features!"); return
            X, y = prepare_modeling_data(df, features, target)
            if X is None: return
            counts = pd.Series(y).value_counts()
            st.write("**Class Distribution:**", counts)
            strat = len(counts)>1 and counts.min()>=2
            try:
                if strat: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs, stratify=y)
                else: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
            except: X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=rs)
            if use_smote:
                tc = pd.Series(y_train).value_counts()
                if tc.min()>=2:
                    try:
                        X_train, y_train = SMOTE(random_state=rs, k_neighbors=min(5,tc.min()-1)).fit_resample(X_train, y_train)
                        st.success("✅ SMOTE applied!")
                    except: pass
            t0 = time.time()
            if fast: models, times = train_models_fast(X_train, X_test, y_train, y_test)
            else: models, times = train_models_quality(X_train, X_test, y_train, y_test)
            tt = time.time()-t0
            if models:
                st.session_state.models = models
                st.session_state.X_test = X_test; st.session_state.y_test = y_test
                st.session_state.trained = True
                st.success(f"✅ {len(models)} models trained in {tt:.2f}s!")
                st.balloons()
                cols = st.columns(len(models))
                for i,(n,m) in enumerate(models.items()):
                    with cols[i]:
                        st.markdown(f"**{n}**")
                        st.metric("Train", f"{m.score(X_train, y_train):.3f}")
                        st.metric("Test", f"{m.score(X_test, y_test):.3f}")
    
    # ==================== MODEL EVALUATION (CLEAN & READABLE) ====================
    elif section == "📊 Model Evaluation":
        st.markdown('<p class="section-header">📊 Model Evaluation</p>', unsafe_allow_html=True)
        
        if not st.session_state.get('trained', False) or len(st.session_state.get('models', {})) == 0:
            st.warning("⚠️ No trained models found!")
            st.info("👉 Please go to **'🤖 Model Training'** section to train your models first.")
            return
        
        models = st.session_state.models
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        if X_test is None or y_test is None:
            st.error("❌ Test data is missing. Please re-run the Model Training step.")
            return
        
        st.success(f"✅ Found {len(models)} trained model(s)")
        
        with st.spinner('Evaluating models...'):
            evaluation_results = evaluate_models(models, X_test, y_test)
        
        if evaluation_results and len(evaluation_results) > 0:
            metrics_data = {}
            for name, results in evaluation_results.items():
                metrics_data[name] = {
                    'Accuracy': results['accuracy'],
                    'F1 Score': results['f1_score']
                }
            metrics_df = pd.DataFrame(metrics_data).T
            
            # ==================== CLEAN COMPARISON TABLE ====================
            st.markdown("#### 📈 Model Performance Comparison")
            
            comparison_df = metrics_df.round(3)
            comparison_df['Model'] = comparison_df.index
            
            st.dataframe(
                comparison_df[['Model', 'Accuracy', 'F1 Score']],
                column_config={
                    "Model": st.column_config.TextColumn("Model", width="medium"),
                    "Accuracy": st.column_config.NumberColumn("Accuracy", format="%.3f"),
                    "F1 Score": st.column_config.NumberColumn("F1 Score", format="%.3f"),
                },
                use_container_width=True,
                hide_index=True,
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                best_f1_model = metrics_df['F1 Score'].idxmax()
                st.metric("🏆 Best Model", best_f1_model)
            with col2:
                st.metric("📈 Best Accuracy", f"{metrics_df['Accuracy'].max():.3f}")
            with col3:
                st.metric("📈 Best F1 Score", f"{metrics_df['F1 Score'].max():.3f}")
            
            # ==================== CONFUSION MATRIX HEATMAP (FULL WIDTH) ====================
            st.markdown("---")
            st.markdown("#### 🧩 Confusion Matrix Heatmap")
            
            selected_model = st.selectbox(
                "Select model to view Confusion Matrix",
                list(evaluation_results.keys())
            )
            
            if selected_model:
                cm = evaluation_results[selected_model].get('confusion_matrix')
                
                if cm is not None and cm.size > 0:
                    n_classes = cm.shape[0]
                    
                    row_sums = cm.sum(axis=1, keepdims=True)
                    row_sums[row_sums == 0] = 1
                    cm_percent = (cm.astype('float') / row_sums * 100)
                    
                    annotations = []
                    for i in range(n_classes):
                        row_ann = []
                        for j in range(n_classes):
                            count = cm[i, j]
                            pct = cm_percent[i, j]
                            row_ann.append(f"<b>{count}</b><br><sub>({pct:.1f}%)</sub>")
                        annotations.append(row_ann)
                    
                    fig_cm = go.Figure(data=go.Heatmap(
                        z=cm,
                        x=[f'<b>Predicted<br>Class {i}</b>' for i in range(n_classes)],
                        y=[f'<b>Actual<br>Class {i}</b>' for i in range(n_classes)],
                        colorscale=[
                            [0.0, '#f7fbff'], [0.25, '#deebf7'], [0.5, '#9ecae1'],
                            [0.75, '#4292c6'], [1.0, '#2171b5']
                        ],
                        text=annotations,
                        texttemplate="%{text}",
                        textfont={"size": 15, "color": "black"},
                        showscale=True,
                        colorbar=dict(
                            title=dict(text="<b>Count</b>", font=dict(size=12)),
                            tickfont=dict(size=10), tickformat="d", len=0.8, thickness=15
                        ),
                        xgap=3, ygap=3,
                        hovertemplate="<b>Actual:</b> %{y}<br><b>Predicted:</b> %{x}<br><b>Count:</b> %{z}<extra></extra>"
                    ))
                    
                    shapes = []
                    for i in range(n_classes):
                        shapes.append(dict(
                            type="rect", x0=i-0.5, y0=i-0.5, x1=i+0.5, y1=i+0.5,
                            line=dict(color="#2ecc71", width=3),
                            fillcolor="rgba(46, 204, 113, 0.1)"
                        ))
                    
                    fig_cm.update_layout(
                        title=dict(
                            text=f"<b>{selected_model}</b><br><sub>Confusion Matrix (Count & Row Percentage)</sub>",
                            font=dict(size=16, color='#2c3e50'), x=0.5
                        ),
                        height=500, shapes=shapes,
                        plot_bgcolor='white', paper_bgcolor='white',
                        xaxis=dict(title="<b>Predicted Label</b>", tickfont=dict(size=12)),
                        yaxis=dict(title="<b>True Label</b>", tickfont=dict(size=12)),
                        margin=dict(l=80, r=50, t=100, b=60)
                    )
                    
                    st.plotly_chart(fig_cm, use_container_width=True)
            
            # ==================== PER-CLASS METRICS (SEPARATE FULL WIDTH) ====================
            st.markdown("---")
            st.markdown("#### 📋 Per-Class Metrics")
            
            if selected_model:
                cm = evaluation_results[selected_model].get('confusion_matrix')
                if cm is not None and cm.size > 0:
                    n_classes = cm.shape[0]
                    
                    per_class_list = []
                    for i in range(n_classes):
                        tp = int(cm[i, i])
                        fp = int(cm[:, i].sum()) - tp
                        fn = int(cm[i, :].sum()) - tp
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1_val = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        
                        per_class_list.append({
                            'Class': f'Class {i}',
                            'TP': tp, 'FP': fp, 'FN': fn,
                            'Precision': round(precision, 3),
                            'Recall': round(recall, 3),
                            'F1-Score': round(f1_val, 3),
                            'Support': int(cm[i, :].sum())
                        })
                    
                    per_class_df = pd.DataFrame(per_class_list)
                    
                    macro_p = per_class_df['Precision'].mean()
                    macro_r = per_class_df['Recall'].mean()
                    macro_f1 = per_class_df['F1-Score'].mean()
                    
                    st.dataframe(
                        per_class_df,
                        column_config={
                            "Class": st.column_config.TextColumn("Class", width="small"),
                            "TP": st.column_config.NumberColumn("True Positive", format="%d", width="small"),
                            "FP": st.column_config.NumberColumn("False Positive", format="%d", width="small"),
                            "FN": st.column_config.NumberColumn("False Negative", format="%d", width="small"),
                            "Precision": st.column_config.NumberColumn("Precision", format="%.3f"),
                            "Recall": st.column_config.NumberColumn("Recall", format="%.3f"),
                            "F1-Score": st.column_config.NumberColumn("F1-Score", format="%.3f"),
                            "Support": st.column_config.NumberColumn("Support", format="%d"),
                        },
                        use_container_width=True,
                        hide_index=True,
                    )
                    
                    st.markdown(f"**{selected_model} - Macro Averages**")
                    col1, col2, col3 = st.columns(3)
                    with col1: st.metric("Macro Avg Precision", f"{macro_p:.3f}")
                    with col2: st.metric("Macro Avg Recall", f"{macro_r:.3f}")
                    with col3: st.metric("Macro Avg F1-Score", f"{macro_f1:.3f}")
            
            # ==================== CLASSIFICATION REPORT ====================
            st.markdown("---")
            st.markdown("#### 📋 Classification Report")
            
            report_model = st.selectbox(
                "Select model for detailed report",
                list(evaluation_results.keys()), key='report_select'
            )
            
            if report_model in evaluation_results:
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown(f"**{report_model} - Detailed Report**")
                    st.code(evaluation_results[report_model]['classification_report'])
                with col2:
                    acc = evaluation_results[report_model]['accuracy']
                    f1 = evaluation_results[report_model]['f1_score']
                    st.markdown("**Model Summary**")
                    st.metric("Accuracy", f"{acc:.3f}")
                    st.metric("F1 Score", f"{f1:.3f}")
                    if acc >= 0.90: st.success("🌟 Excellent")
                    elif acc >= 0.80: st.success("👍 Good")
                    elif acc >= 0.70: st.warning("📊 Fair")
                    else: st.error("⚠️ Needs Improvement")
            
            # ==================== MODEL RANKING ====================
            st.markdown("---")
            st.markdown("#### 🏆 Model Ranking")
            
            ranked = metrics_df.sort_values('F1 Score', ascending=False)
            for i, (name, row) in enumerate(ranked.iterrows(), 1):
                medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else f"  {i}."
                st.markdown(
                    f"{medal} &nbsp; <b>{name}</b> &nbsp; → &nbsp; "
                    f"Accuracy: <code>{row['Accuracy']:.3f}</code> &nbsp; | &nbsp; "
                    f"F1 Score: <code>{row['F1 Score']:.3f}</code>",
                    unsafe_allow_html=True
                )
        
        else:
            st.warning("⚠️ No evaluation results available.")
            if st.button("🔄 Retry Evaluation", use_container_width=True):
                st.rerun()
    
    # ==================== FEATURE IMPORTANCE ====================
    elif section == "🎯 Feature Importance":
        st.markdown('<p class="section-header">🎯 Feature Importance</p>', unsafe_allow_html=True)
        if st.session_state.feature_importance is not None:
            imp = st.session_state.feature_importance
            fig = px.bar(imp.nlargest(20,'importance'), x='importance', y='feature',
                        orientation='h', title='Top 20 Features', color='importance',
                        color_continuous_scale='Viridis', height=600)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown("### 📊 Feature Importance Table")
            disp = imp.copy().sort_values('importance', ascending=False)
            disp['importance'] = disp['importance'].round(4)
            disp['%'] = (disp['importance']/disp['importance'].sum()*100).round(2)
            disp.index = range(1, len(disp)+1)
            disp.index.name = 'Rank'
            st.dataframe(disp[['feature', 'importance', '%']], use_container_width=True)
            top3 = imp.nlargest(3,'importance')
            for i,(_,r) in enumerate(top3.iterrows()):
                st.markdown(f"**{i+1}. {r['feature']}** ({r['importance']:.4f})")
                st.progress(float(r['importance']/imp['importance'].max()))
            csv = imp.to_csv(index=False)
            st.download_button("📥 Download CSV", csv, "features.csv", "text/csv")
        else:
            st.warning("⚠️ Calculate feature importance first!")
    
    # ==================== POLYMER ANALYSIS ====================
    elif section == "🧬 Polymer Analysis":
        st.markdown('<p class="section-header">🧬 Polymer Analysis</p>', unsafe_allow_html=True)
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None: st.warning("⚠️ Load data first!"); return
        df = data
        if 'Polymer_Type' in df.columns:
            counts = df['Polymer_Type'].value_counts()
            c1,c2 = st.columns(2)
            with c1:
                fig = px.bar(x=counts.index, y=counts.values, title='Polymer Distribution',
                            color=counts.values, color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.pie(values=counts.values, names=counts.index, title='Polymer Distribution')
                st.plotly_chart(fig, use_container_width=True)
            c1,c2,c3 = st.columns(3)
            with c1: st.metric("Total Types", len(counts))
            with c2: st.metric("Most Common", counts.index[0])
            with c3: st.metric("Count", counts.values[0])
            if 'Risk_Level' in df.columns:
                fig = px.histogram(df, x='Polymer_Type', color='Risk_Level',
                                  title='Polymer by Risk', barmode='group')
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
