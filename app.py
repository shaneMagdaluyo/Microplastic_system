# app.py - Microplastic Risk Prediction System
# Simple, Complete, Working Version
# Models: Random Forest, Logistic Regression, Decision Tree

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="MP-RAS | Microplastic Risk Assessment System",
    page_icon="🌊",
    layout="wide"
)

# ============================================
# HEADER
# ============================================
st.markdown("""
<div style="background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0;">🌊 MP-RAS</h1>
    <p style="color: white; margin: 0;">Microplastic Risk Assessment System</p>
    <p style="color: white; font-size: 0.8rem; margin-top: 0.5rem;">Random Forest | Logistic Regression | Decision Tree</p>
    <p style="color: white; font-size: 0.7rem;">Viernes, M.J. & Magdaluyo, S.M.R. | ASSCAT 2025</p>
</div>
""", unsafe_allow_html=True)

# ============================================
# SESSION STATE
# ============================================
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trained' not in st.session_state:
    st.session_state.trained = False
if 'models' not in st.session_state:
    st.session_state.models = {}
if 'results' not in st.session_state:
    st.session_state.results = {}
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'encoders' not in st.session_state:
    st.session_state.encoders = {}
if 'target_encoder' not in st.session_state:
    st.session_state.target_encoder = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'target_col' not in st.session_state:
    st.session_state.target_col = None

# ============================================
# SIDEBAR
# ============================================
st.sidebar.markdown("## 📌 Navigation")
page = st.sidebar.radio("", ["📁 Upload Data", "🤖 Train Model", "📈 Predict", "📊 Results"])

st.sidebar.markdown("---")
if st.session_state.data is not None:
    st.sidebar.success(f"✅ Data: {st.session_state.data.shape[0]} rows")
else:
    st.sidebar.warning("⚠️ No data loaded")

if st.session_state.trained:
    st.sidebar.success("✅ Model trained")

# ============================================
# FUNCTION: Generate Sample Data
# ============================================
def generate_sample_data():
    np.random.seed(42)
    n = 1000
    
    data = {
        'Location': np.random.choice(['Coastal', 'River', 'Urban', 'Industrial'], n),
        'MP_Concentration': np.random.uniform(0.1, 500, n),
        'Particle_Size': np.random.uniform(0.01, 5, n),
        'Temperature': np.random.uniform(10, 35, n),
        'pH': np.random.uniform(6, 8.5, n),
        'DO_mgL': np.random.uniform(2, 12, n),
        'Turbidity': np.random.uniform(1, 100, n),
        'Industrial_Score': np.random.uniform(0, 1, n),
        'Waste_Score': np.random.uniform(0, 1, n),
    }
    df = pd.DataFrame(data)
    
    # Calculate risk score
    df['Risk_Score'] = (
        df['MP_Concentration'] / 500 * 40 +
        df['Industrial_Score'] * 30 +
        (1 - df['Waste_Score']) * 30
    )
    df['Risk_Score'] = df['Risk_Score'].clip(0, 100)
    df['Risk_Level'] = pd.cut(df['Risk_Score'], bins=[0, 33, 66, 100], labels=['Low', 'Medium', 'High'])
    
    return df

# ============================================
# FUNCTION: Preprocess Data
# ============================================
def preprocess_data(df, features, target):
    df_proc = df.copy()
    encoders = {}
    
    numeric_features = []
    for feat in features:
        if feat in df_proc.columns:
            if df_proc[feat].dtype in ['int64', 'float64']:
                numeric_features.append(feat)
            else:
                le = LabelEncoder()
                df_proc[feat + '_enc'] = le.fit_transform(df_proc[feat].astype(str))
                encoders[feat] = le
                numeric_features.append(feat + '_enc')
    
    X = df_proc[numeric_features].fillna(0)
    
    if df_proc[target].dtype == 'object':
        target_enc = LabelEncoder()
        y = target_enc.fit_transform(df_proc[target].astype(str))
        return X, y, encoders, target_enc
    else:
        y = df_proc[target].values
        return X, y, encoders, None

# ============================================
# PAGE 1: UPLOAD DATA
# ============================================
if page == "📁 Upload Data":
    st.header("📁 Upload Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx', 'xls'])
        if file:
            try:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                else:
                    df = pd.read_excel(file)
                st.session_state.data = df
                st.success(f"✅ Loaded {df.shape[0]} rows, {df.shape[1]} columns")
                st.dataframe(df.head())
            except Exception as e:
                st.error(f"Error: {e}")
    
    with col2:
        st.write("Or use sample data:")
        if st.button("📊 Load Sample Data"):
            df = generate_sample_data()
            st.session_state.data = df
            st.success("✅ Sample data loaded!")
            st.dataframe(df.head())
    
    if st.session_state.data is not None:
        st.markdown("---")
        st.subheader("Data Info")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", st.session_state.data.shape[0])
        with col2:
            st.metric("Columns", st.session_state.data.shape[1])
        with col3:
            missing = st.session_state.data.isnull().sum().sum()
            st.metric("Missing Values", missing)

# ============================================
# PAGE 2: TRAIN MODEL
# ============================================
elif page == "🤖 Train Model":
    st.header("🤖 Train Model")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        st.stop()
    
    df = st.session_state.data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select target
        target_options = [c for c in df.columns if 'risk' in c.lower() or 'level' in c.lower()]
        if not target_options:
            target_options = df.columns.tolist()
        target = st.selectbox("🎯 Target Column (what to predict)", target_options)
        st.session_state.target_col = target
        
        # Task type
        if df[target].dtype in ['int64', 'float64'] and df[target].nunique() > 10:
            task = st.radio("Task Type", ["Regression", "Classification"])
        else:
            task = st.radio("Task Type", ["Classification", "Regression"])
    
    with col2:
        test_size = st.slider("Test Size", 0.1, 0.4, 0.2)
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
    
    # Feature selection
    st.subheader("Select Features")
    features = [c for c in df.columns if c != target]
    selected = st.multiselect("Features to use", features, default=features[:5] if len(features) > 5 else features)
    st.session_state.features = selected
    
    if len(selected) == 0:
        st.error("Select at least one feature")
        st.stop()
    
    # Models
    st.subheader("Models")
    use_rf = st.checkbox("Random Forest", value=True)
    use_lr = st.checkbox("Logistic Regression", value=True)
    use_dt = st.checkbox("Decision Tree", value=True)
    
    if st.button("🚀 START TRAINING", type="primary", use_container_width=True):
        if not (use_rf or use_lr or use_dt):
            st.error("Select at least one model")
        else:
            with st.spinner("Training..."):
                try:
                    # Preprocess
                    X, y, encoders, target_enc = preprocess_data(df, selected, target)
                    st.session_state.encoders = encoders
                    st.session_state.target_encoder = target_enc
                    
                    # Scale
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                    
                    # Split
                    stratify = y if task == "Classification" else None
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42, stratify=stratify
                    )
                    
                    results = {}
                    models = {}
                    
                    # Random Forest
                    if use_rf:
                        with st.spinner("Training Random Forest..."):
                            if task == "Classification":
                                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                            else:
                                from sklearn.ensemble import RandomForestRegressor
                                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                            rf.fit(X_train, y_train)
                            y_pred = rf.predict(X_test)
                            models['Random Forest'] = rf
                            
                            cv = cross_val_score(rf, X_scaled, y, cv=cv_folds, 
                                                scoring='accuracy' if task == "Classification" else 'r2')
                            
                            if task == "Classification":
                                results['Random Forest'] = {
                                    'Accuracy': accuracy_score(y_test, y_pred),
                                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                    'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                                    'CV_Mean': cv.mean(),
                                    'CV_Std': cv.std()
                                }
                            else:
                                from sklearn.metrics import r2_score, mean_squared_error
                                results['Random Forest'] = {
                                    'R2 Score': r2_score(y_test, y_pred),
                                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                                    'CV_Mean': cv.mean(),
                                    'CV_Std': cv.std()
                                }
                    
                    # Logistic Regression
                    if use_lr and task == "Classification":
                        with st.spinner("Training Logistic Regression..."):
                            lr = LogisticRegression(max_iter=1000, random_state=42)
                            lr.fit(X_train, y_train)
                            y_pred = lr.predict(X_test)
                            models['Logistic Regression'] = lr
                            
                            cv = cross_val_score(lr, X_scaled, y, cv=cv_folds, scoring='accuracy')
                            results['Logistic Regression'] = {
                                'Accuracy': accuracy_score(y_test, y_pred),
                                'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                                'CV_Mean': cv.mean(),
                                'CV_Std': cv.std()
                            }
                    elif use_lr and task == "Regression":
                        st.info("Logistic Regression is for classification only. Skipping.")
                    
                    # Decision Tree
                    if use_dt:
                        with st.spinner("Training Decision Tree..."):
                            if task == "Classification":
                                dt = DecisionTreeClassifier(random_state=42)
                            else:
                                from sklearn.tree import DecisionTreeRegressor
                                dt = DecisionTreeRegressor(random_state=42)
                            dt.fit(X_train, y_train)
                            y_pred = dt.predict(X_test)
                            models['Decision Tree'] = dt
                            
                            cv = cross_val_score(dt, X_scaled, y, cv=cv_folds,
                                                scoring='accuracy' if task == "Classification" else 'r2')
                            
                            if task == "Classification":
                                results['Decision Tree'] = {
                                    'Accuracy': accuracy_score(y_test, y_pred),
                                    'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                                    'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                                    'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                                    'CV_Mean': cv.mean(),
                                    'CV_Std': cv.std()
                                }
                            else:
                                from sklearn.metrics import r2_score, mean_squared_error
                                results['Decision Tree'] = {
                                    'R2 Score': r2_score(y_test, y_pred),
                                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                                    'CV_Mean': cv.mean(),
                                    'CV_Std': cv.std()
                                }
                    
                    # Save to session
                    st.session_state.models = models
                    st.session_state.results = results
                    st.session_state.trained = True
                    st.session_state.task_type = task
                    
                    st.success("✅ Training complete!")
                    
                    # Display results
                    st.subheader("📊 Results")
                    results_df = pd.DataFrame(results).T
                    
                    if task == "Classification":
                        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
                    else:
                        display_cols = ['R2 Score', 'RMSE', 'CV_Mean']
                    
                    st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0))
                    
                    # Best model
                    if task == "Classification":
                        best = max(results, key=lambda x: results[x]['Accuracy'])
                        st.success(f"🏆 Best Model: {best} (Accuracy: {results[best]['Accuracy']:.4f})")
                    else:
                        best = max(results, key=lambda x: results[x]['R2 Score'])
                        st.success(f"🏆 Best Model: {best} (R2: {results[best]['R2 Score']:.4f})")
                    
                    # Confusion Matrix for best classification model
                    if task == "Classification" and best in models:
                        st.subheader(f"Confusion Matrix - {best}")
                        best_model = models[best]
                        y_pred_best = best_model.predict(X_test)
                        cm = confusion_matrix(y_test, y_pred_best)
                        
                        fig, ax = plt.subplots(figsize=(8, 6))
                        labels = target_enc.classes_ if target_enc else [str(i) for i in range(len(np.unique(y)))]
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                                   xticklabels=labels, yticklabels=labels)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        ax.set_title(f'Confusion Matrix - {best}')
                        st.pyplot(fig)
                    
                    # Feature Importance for Random Forest
                    if 'Random Forest' in models and hasattr(models['Random Forest'], 'feature_importances_'):
                        st.subheader("Feature Importance (Random Forest)")
                        rf_model = models['Random Forest']
                        importance_df = pd.DataFrame({
                            'Feature': X.columns,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance_df['Feature'][:10][::-1], importance_df['Importance'][:10][::-1])
                        ax.set_xlabel('Importance')
                        ax.set_title('Top 10 Features')
                        st.pyplot(fig)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================
# PAGE 3: PREDICT
# ============================================
elif page == "📈 Predict":
    st.header("📈 Make Predictions")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train a model first")
        st.stop()
    
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded")
        st.stop()
    
    st.subheader("Enter Values")
    
    input_data = {}
    cols = st.columns(2)
    df = st.session_state.data
    
    for i, feat in enumerate(st.session_state.features):
        with cols[i % 2]:
            if feat in df.columns:
                if df[feat].dtype == 'object':
                    vals = df[feat].dropna().unique().tolist()
                    input_data[feat] = st.selectbox(f"{feat}", vals)
                else:
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    mean_val = float(df[feat].mean())
                    input_data[feat] = st.number_input(f"{feat}", value=mean_val, min_value=min_val, max_value=max_val)
            else:
                input_data[feat] = st.number_input(f"{feat}", value=0.0)
    
    if st.button("🔮 PREDICT", type="primary", use_container_width=True):
        try:
            # Process input
            input_df = pd.DataFrame([input_data])
            
            for feat, encoder in st.session_state.encoders.items():
                if feat in input_df.columns:
                    val = input_df[feat].iloc[0]
                    if val in encoder.classes_:
                        input_df[feat + '_enc'] = encoder.transform([val])[0]
            
            X_input = []
            for feat in st.session_state.features:
                if feat + '_enc' in input_df.columns:
                    X_input.append(input_df[feat + '_enc'].iloc[0])
                elif feat in input_df.columns and input_df[feat].dtype in ['int64', 'float64']:
                    X_input.append(input_df[feat].iloc[0])
            
            X_input = np.array(X_input).reshape(1, -1)
            X_scaled = st.session_state.scaler.transform(X_input)
            
            st.subheader("🎯 Prediction Results")
            
            # Show all model predictions
            pred_cols = st.columns(len(st.session_state.models))
            
            for idx, (name, model) in enumerate(st.session_state.models.items()):
                pred = model.predict(X_scaled)[0]
                
                if st.session_state.task_type == "Classification":
                    if st.session_state.target_encoder:
                        label = st.session_state.target_encoder.inverse_transform([int(pred)])[0]
                    else:
                        label = str(pred)
                    
                    # Color based on risk
                    risk_lower = str(label).lower()
                    if 'high' in risk_lower:
                        color = "#ff6b6b"
                    elif 'medium' in risk_lower:
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
                            <p>Risk Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# ============================================
# PAGE 4: RESULTS
# ============================================
elif page == "📊 Results":
    st.header("📊 Results")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train a model first")
        st.stop()
    
    st.subheader("Model Performance Summary")
    
    results_df = pd.DataFrame(st.session_state.results).T
    
    if st.session_state.task_type == "Classification":
        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
    else:
        display_cols = ['R2 Score', 'RMSE', 'CV_Mean']
    
    st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0))
    
    # Bar chart comparison
    st.subheader("Model Comparison")
    
    if st.session_state.task_type == "Classification":
        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        x = np.arange(len(results_df.index))
        width = 0.2
        
        for i, metric in enumerate(metrics):
            values = [results_df.loc[model, metric] for model in results_df.index]
            ax.bar(x + i*width, values, width, label=metric)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Score')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(results_df.index, rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        models = results_df.index.tolist()
        r2_scores = [results_df.loc[m, 'R2 Score'] for m in models]
        ax.bar(models, r2_scores, color=['#667eea', '#764ba2', '#f093fb'])
        ax.set_xlabel('Models')
        ax.set_ylabel('R2 Score')
        ax.set_title('R2 Score Comparison')
        ax.set_ylim(0, 1)
        st.pyplot(fig)
    
    # Download report
    st.subheader("Download Report")
    
    report = f"""
    ========================================
    MP-RAS SYSTEM REPORT
    ========================================
    Date: {pd.Timestamp.now()}
    
    TASK TYPE: {st.session_state.task_type}
    TARGET: {st.session_state.target_col}
    FEATURES: {', '.join(st.session_state.features)}
    
    PERFORMANCE:
    {results_df[display_cols].to_string()}
    
    BEST MODEL: {max(st.session_state.results, key=lambda x: st.session_state.results[x][display_cols[0]])}
    """
    
    st.download_button("📥 Download Report", report, "mp_ras_report.txt")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌊 MP-RAS | Microplastic Risk Assessment System | ASSCAT 2025</p>
    <p>Models: Random Forest | Logistic Regression | Decision Tree</p>
</div>
""", unsafe_allow_html=True)
