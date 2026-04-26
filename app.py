"""
Microplastic Risk Analysis Dashboard - STABLE VERSION
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
import warnings
import time

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Microplastic Dashboard", page_icon="🔬", layout="wide")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'feature_importance' not in st.session_state:
    st.session_state.feature_importance = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'selected_features' not in st.session_state:
    st.session_state.selected_features = None

def load_dataset(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return None
        st.session_state.data = data
        st.success(f"Dataset loaded! Shape: {data.shape}")
        return data
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def generate_sample_data():
    np.random.seed(42)
    n = 500
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
                df[col].fillna(df[col].median() if not pd.isna(df[col].median()) else 0, inplace=True)
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
    nums = df[feature_cols].select_dtypes(include=['float64', 'int64']).columns
    if len(nums) > 0:
        df[nums] = StandardScaler().fit_transform(df[nums])
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
    if clean.empty:
        return go.Figure()
    fig = make_subplots(rows=1, cols=2, subplot_titles=('Histogram', 'Box Plot'))
    fig.add_trace(go.Histogram(x=clean, nbinsx=20, marker_color='#3498db'), row=1, col=1)
    fig.add_trace(go.Box(y=clean, marker_color='#e74c3c'), row=1, col=2)
    fig.update_layout(title_text=title, showlegend=False, height=350)
    return fig

def prepare_modeling_data(df, feature_cols, target_col):
    X = df[feature_cols].select_dtypes(include=['float64', 'int64', 'int32'])
    if X.shape[1] == 0:
        st.error("No numeric features selected.")
        return None, None
    y = df[target_col]
    mask = y.notna()
    X, y = X[mask], y[mask]
    if len(y) == 0:
        return None, None
    if y.dtype == 'object':
        y = pd.Series(LabelEncoder().fit_transform(y), index=y.index)
    X = X.fillna(X.median())
    return X, y

def train_models(X_train, y_train):
    models = {}
    
    # Logistic Regression
    try:
        lr = LogisticRegression(random_state=42, max_iter=300, class_weight='balanced', n_jobs=-1)
        lr.fit(X_train, y_train)
        models['Logistic Regression'] = lr
    except:
        pass
    
    # Random Forest
    try:
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', max_depth=10, n_jobs=-1)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
    except:
        pass
    
    # Decision Tree
    try:
        dt = DecisionTreeClassifier(random_state=42, max_depth=8, class_weight='balanced')
        dt.fit(X_train, y_train)
        models['Decision Tree'] = dt
    except:
        pass
    
    return models

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
        except:
            pass
    return results

def main():
    st.markdown('<h2 style="text-align:center;color:#1f77b4;">🔬 Microplastic Risk Analysis Dashboard</h2>', unsafe_allow_html=True)
    
    st.sidebar.markdown("## Navigation")
    section = st.sidebar.radio("Select Section", [
        "Upload Dataset", "Data Preprocessing", "EDA",
        "Feature Engineering", "Model Training", "Model Evaluation",
        "Feature Importance", "Polymer Analysis"
    ])
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Status")
    if st.session_state.data is not None:
        st.sidebar.success("Data Loaded")
    if st.session_state.trained:
        st.sidebar.success(f"Models Trained ({len(st.session_state.models)})")
    
    # ==================== UPLOAD ====================
    if section == "Upload Dataset":
        st.markdown("## Upload Dataset")
        c1, c2 = st.columns([2, 1])
        with c1:
            f = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
            if f:
                load_dataset(f)
        with c2:
            if st.button("Generate Sample Data", type="primary"):
                st.session_state.data = generate_sample_data()
                st.success("Sample data generated!")
                st.rerun()
        
        if st.session_state.data is not None:
            df = st.session_state.data
            st.write(f"**Samples:** {df.shape[0]} | **Features:** {df.shape[1]} | **Missing:** {df.isnull().sum().sum()}")
            st.dataframe(df.head(5))
    
    # ==================== PREPROCESSING ====================
    elif section == "Data Preprocessing":
        st.markdown("## Data Preprocessing")
        if st.session_state.data is None:
            st.warning("Upload data first!")
            return
        
        df = st.session_state.data.copy()
        opts = st.multiselect("Select Steps", 
                             ["Handle Missing Values", "Encode Categorical Variables", "Detect Outliers", "Scale Features"],
                             default=["Handle Missing Values", "Encode Categorical Variables"])
        
        if st.button("Run Preprocessing", type="primary"):
            pdf = df.copy()
            if "Handle Missing Values" in opts:
                pdf = handle_missing_values(pdf)
                st.success("Missing values handled")
            if "Encode Categorical Variables" in opts:
                pdf = encode_categorical(pdf)
                st.success("Categorical variables encoded")
            if "Detect Outliers" in opts:
                info = detect_outliers(pdf, pdf.select_dtypes(include=['float64', 'int64']).columns)
                for c, i in info.items():
                    if i['count'] > 0:
                        st.write(f"**{c}**: {i['count']} outliers ({i['percentage']:.1f}%)")
            if "Scale Features" in opts:
                pdf = scale_features(pdf, pdf.select_dtypes(include=['float64', 'int64']).columns)
                st.success("Features scaled")
            
            st.session_state.processed_data = pdf
            st.success("Preprocessing completed!")
        
        if st.session_state.processed_data is not None:
            st.dataframe(st.session_state.processed_data.head(5))
    
    # ==================== EDA ====================
    elif section == "EDA":
        st.markdown("## Exploratory Data Analysis")
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None:
            st.warning("Load data first!")
            return
        
        df = data.copy()
        
        if 'Risk_Score' in df.columns:
            st.markdown("### Risk Score Distribution")
            df['Risk_Score'] = pd.to_numeric(df['Risk_Score'], errors='coerce')
            clean = df['Risk_Score'].dropna()
            if len(clean) > 0:
                st.plotly_chart(plot_distribution(df, 'Risk_Score', 'Risk Score Distribution'), use_container_width=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Mean", f"{clean.mean():.2f}")
                c2.metric("Median", f"{clean.median():.2f}")
                c3.metric("Max", f"{clean.max():.2f}")
                c4.metric("Min", f"{clean.min():.2f}")
        
        if 'MP_Count_per_L' in df.columns and 'Risk_Score' in df.columns:
            st.markdown("### MP Count vs Risk Score")
            clean = df.dropna(subset=['MP_Count_per_L', 'Risk_Score'])
            if not clean.empty:
                fig = px.scatter(clean, x='MP_Count_per_L', y='Risk_Score',
                                color='Risk_Level' if 'Risk_Level' in clean.columns else None,
                                title='MP Count vs Risk Score', height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        if 'Risk_Level' in df.columns and 'Risk_Score' in df.columns:
            st.markdown("### Risk Score by Risk Level")
            clean = df.dropna(subset=['Risk_Score'])
            if len(clean) > 0:
                fig = px.box(clean, x='Risk_Level', y='Risk_Score', color='Risk_Level',
                           title='Risk Score by Risk Level', height=300)
                st.plotly_chart(fig, use_container_width=True)
    
    # ==================== FEATURE ENGINEERING ====================
    elif section == "Feature Engineering":
        st.markdown("## Feature Engineering")
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None:
            st.warning("Load data first!")
            return
        
        df = data
        target = st.selectbox("Target Variable", df.columns.tolist(),
                             index=df.columns.tolist().index('Risk_Type') if 'Risk_Type' in df.columns else 0)
        nums = df.select_dtypes(include=['float64', 'int64', 'int32']).columns.tolist()
        if target in nums:
            nums.remove(target)
        
        if st.button("Calculate Feature Importance", type="primary"):
            X = df[nums].fillna(df[nums].median()).dropna(axis=1)
            y = df[target]
            if y.dtype == 'object':
                y = LabelEncoder().fit_transform(y)
            
            rf = RandomForestClassifier(n_estimators=30, random_state=42, n_jobs=-1)
            rf.fit(X, y)
            imp = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_}).sort_values('importance', ascending=True)
            st.session_state.feature_importance = imp
            
            fig = px.bar(imp.tail(15), x='importance', y='feature', orientation='h', title='Top 15 Features', height=350)
            st.plotly_chart(fig, use_container_width=True)
            
            top = imp.nlargest(10, 'importance')['feature'].tolist()
            st.session_state.selected_features = top
            st.success(f"Top {len(top)} features selected")
    
    # ==================== MODEL TRAINING ====================
    elif section == "Model Training":
        st.markdown("## Model Training")
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None:
            st.warning("Load data first!")
            return
        
        df = data
        target = st.selectbox("Target Variable", df.columns.tolist(), key='train_target')
        all_f = [c for c in df.columns if c != target]
        default = st.session_state.get('selected_features', df.select_dtypes(include=['float64', 'int64']).columns.tolist()[:5])
        default = [f for f in default if f in all_f]
        features = st.multiselect("Features", all_f, default=default)
        
        ts = st.slider("Test Size", 0.1, 0.5, 0.2)
        
        if st.button("Train Models", type="primary", use_container_width=True):
            if len(features) == 0:
                st.error("Select features!")
                return
            
            X, y = prepare_modeling_data(df, features, target)
            if X is None:
                return
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=ts, random_state=42)
            
            # SMOTE
            tc = pd.Series(y_train).value_counts()
            if tc.min() >= 2:
                try:
                    X_train, y_train = SMOTE(random_state=42, k_neighbors=min(3, tc.min()-1)).fit_resample(X_train, y_train)
                    st.success("SMOTE applied!")
                except:
                    pass
            
            t0 = time.time()
            models = train_models(X_train, y_train)
            tt = time.time() - t0
            
            if models:
                st.session_state.models = models
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.trained = True
                
                st.success(f"{len(models)} models trained in {tt:.2f}s!")
                
                for n, m in models.items():
                    st.write(f"**{n}**: Train={m.score(X_train, y_train):.3f}, Test={m.score(X_test, y_test):.3f}")
    
    # ==================== MODEL EVALUATION ====================
    elif section == "Model Evaluation":
        st.markdown("## Model Evaluation")
        
        if not st.session_state.get('trained') or len(st.session_state.get('models', {})) == 0:
            st.warning("No trained models! Train first.")
            return
        
        models = st.session_state.models
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        eval_results = evaluate_models(models, X_test, y_test)
        
        if eval_results:
            st.markdown("### Performance Comparison")
            
            # Simple table
            for n, r in eval_results.items():
                st.write(f"**{n}**: Accuracy={r['accuracy']:.4f}, F1={r['f1_score']:.4f}")
            
            # Best model
            best = max(eval_results.items(), key=lambda x: x[1]['f1_score'])
            st.success(f"🏆 Best: **{best[0]}** (F1: {best[1]['f1_score']:.4f})")
            
            # Confusion Matrix
            st.markdown("### Confusion Matrix")
            selected = st.selectbox("Select model", list(eval_results.keys()))
            if selected:
                cm = eval_results[selected]['confusion_matrix']
                n = cm.shape[0]
                
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=[f'Pred {i}' for i in range(n)],
                    y=[f'Actual {i}' for i in range(n)],
                    colorscale='Blues',
                    text=[[str(int(v)) for v in row] for row in cm],
                    texttemplate="%{text}",
                    textfont={"size": 14},
                    showscale=True
                ))
                fig_cm.update_layout(title=f'{selected} - Confusion Matrix', height=300)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            # Classification Report
            st.markdown("### Classification Report")
            report_model = st.selectbox("Select model for report", list(eval_results.keys()), key='rep')
            if report_model:
                st.code(eval_results[report_model]['classification_report'])
    
    # ==================== FEATURE IMPORTANCE ====================
    elif section == "Feature Importance":
        st.markdown("## Feature Importance")
        if st.session_state.feature_importance is not None:
            imp = st.session_state.feature_importance
            fig = px.bar(imp.nlargest(20, 'importance'), x='importance', y='feature',
                        orientation='h', title='Top 20 Features', height=350,
                        color='importance', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
            
            disp = imp.copy().sort_values('importance', ascending=False)
            disp['importance'] = disp['importance'].round(4)
            st.dataframe(disp[['feature', 'importance']], use_container_width=True)
        else:
            st.warning("Calculate feature importance first!")
    
    # ==================== POLYMER ANALYSIS ====================
    elif section == "Polymer Analysis":
        st.markdown("## Polymer Analysis")
        data = st.session_state.processed_data if st.session_state.processed_data is not None else st.session_state.data
        if data is None:
            st.warning("Load data first!")
            return
        
        df = data
        if 'Polymer_Type' in df.columns:
            counts = df['Polymer_Type'].value_counts()
            c1, c2 = st.columns(2)
            with c1:
                fig = px.bar(x=counts.index, y=counts.values, title='Polymer Distribution', height=250)
                st.plotly_chart(fig, use_container_width=True)
            with c2:
                fig = px.pie(values=counts.values, names=counts.index, title='Polymer Distribution', height=250)
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
