# app.py - Microplastic Risk Prediction System
# Based strictly on the manuscript requirements
# Algorithms: Random Forest, Logistic Regression, Decision Tree
# Validation: 10-Fold Cross Validation
# Metrics: Accuracy, Precision, Recall, F1-score, AUC

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="Microplastic Risk Prediction System",
    page_icon="🌊",
    layout="wide"
)

# ============================================
# HEADER
# ============================================
st.markdown("""
<div style="background: linear-gradient(135deg, #1a2980 0%, #26d0ce 100%); padding: 2rem; border-radius: 15px; text-align: center; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0;">🌊 Microplastic Risk Prediction System</h1>
    <p style="color: white; margin: 0;">Predictive Risk Modeling using Data Mining Techniques</p>
    <p style="color: white; font-size: 0.8rem; margin-top: 0.5rem;">Random Forest | Logistic Regression | Decision Tree | 10-Fold Cross Validation</p>
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
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = []

# ============================================
# SIDEBAR
# ============================================
st.sidebar.markdown("## 📌 Navigation")
page = st.sidebar.radio("", ["📁 Upload Data", "🤖 Train Models", "📈 Predict", "📊 Results"])

st.sidebar.markdown("---")
if st.session_state.data is not None:
    st.sidebar.success(f"✅ Data: {st.session_state.data.shape[0]} rows")
else:
    st.sidebar.warning("⚠️ No data loaded")

if st.session_state.trained:
    st.sidebar.success("✅ Models trained with 10-Fold CV")

# ============================================
# GENERATE SAMPLE DATA
# ============================================
def generate_sample_data():
    np.random.seed(42)
    n = 1000
    
    data = {
        'Location': np.random.choice(['Coastal', 'River', 'Urban', 'Industrial'], n),
        'MP_Concentration': np.random.uniform(0.1, 500, n),
        'Particle_Size_mm': np.random.uniform(0.01, 5, n),
        'Water_Temperature_C': np.random.uniform(10, 35, n),
        'pH_Level': np.random.uniform(6, 8.5, n),
        'Dissolved_Oxygen_mgL': np.random.uniform(2, 12, n),
        'Turbidity_NTU': np.random.uniform(1, 100, n),
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
# PREPROCESS DATA
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
                df_proc[feat + '_encoded'] = le.fit_transform(df_proc[feat].astype(str))
                encoders[feat] = le
                numeric_features.append(feat + '_encoded')
    
    X = df_proc[numeric_features].fillna(0)
    
    if df_proc[target].dtype == 'object':
        target_enc = LabelEncoder()
        y = target_enc.fit_transform(df_proc[target].astype(str))
        return X, y, encoders, target_enc, X.columns.tolist()
    else:
        y = df_proc[target].values
        return X, y, encoders, None, X.columns.tolist()

# ============================================
# PAGE 1: UPLOAD DATA
# ============================================
if page == "📁 Upload Data":
    st.header("📁 Upload Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file = st.file_uploader("Upload CSV or Excel file", type=['csv', 'xlsx', 'xls'])
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
        st.subheader("Data Information")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", st.session_state.data.shape[0])
        with col2:
            st.metric("Columns", st.session_state.data.shape[1])
        with col3:
            numeric = len(st.session_state.data.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Features", numeric)
        with col4:
            categorical = len(st.session_state.data.select_dtypes(include=['object']).columns)
            st.metric("Categorical Features", categorical)

# ============================================
# PAGE 2: TRAIN MODELS
# ============================================
elif page == "🤖 Train Models":
    st.header("🤖 Train Models")
    st.markdown("""
    <div style="background: #e8f4f8; padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
        <p>📋 <strong>Configuration based on your manuscript:</strong></p>
        <ul>
            <li>Algorithms: Random Forest, Logistic Regression, Decision Tree</li>
            <li>Validation: 10-Fold Cross Validation</li>
            <li>Metrics: Accuracy, Precision, Recall, F1-score, AUC</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        st.stop()
    
    df = st.session_state.data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select target column (look for risk-related columns)
        risk_cols = [c for c in df.columns if 'risk' in c.lower() or 'Risk' in c or 'level' in c.lower()]
        if not risk_cols:
            risk_cols = df.columns.tolist()
        target = st.selectbox("🎯 Target Column (what to predict)", risk_cols)
        st.session_state.target_col = target
        
        # Show target distribution
        st.write(f"Target distribution: {df[target].nunique()} unique values")
        if df[target].dtype == 'object':
            st.write(df[target].value_counts())
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
        st.info("K-Fold Cross Validation: **10 folds** (as specified in your manuscript)")
        n_folds = 10  # Fixed at 10 as per manuscript
    
    # Feature selection
    st.subheader("Select Features for Training")
    features = [c for c in df.columns if c != target]
    selected = st.multiselect("Features to use", features, default=features[:5] if len(features) > 5 else features)
    st.session_state.features = selected
    
    if len(selected) == 0:
        st.error("Please select at least one feature")
        st.stop()
    
    # Model selection
    st.subheader("Select Models to Train")
    col1, col2, col3 = st.columns(3)
    with col1:
        use_rf = st.checkbox("Random Forest", value=True)
    with col2:
        use_lr = st.checkbox("Logistic Regression", value=True)
    with col3:
        use_dt = st.checkbox("Decision Tree", value=True)
    
    if st.button("🚀 START TRAINING", type="primary", use_container_width=True):
        if not (use_rf or use_lr or use_dt):
            st.error("Select at least one model")
        else:
            with st.spinner("Training models with 10-Fold Cross Validation..."):
                try:
                    # Preprocess data
                    X, y, encoders, target_enc, X_columns = preprocess_data(df, selected, target)
                    st.session_state.encoders = encoders
                    st.session_state.target_encoder = target_enc
                    st.session_state.X_columns = X_columns
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                    
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X_scaled, y, test_size=test_size, random_state=42, stratify=y
                    )
                    
                    # Initialize K-Fold (10 folds as per manuscript)
                    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                    
                    results = {}
                    models = {}
                    
                    # ========================================
                    # RANDOM FOREST
                    # ========================================
                    if use_rf:
                        st.write("**Training Random Forest...**")
                        rf = RandomForestClassifier(n_estimators=100, random_state=42)
                        
                        # Train
                        rf.fit(X_train, y_train)
                        y_pred = rf.predict(X_test)
                        y_pred_proba = rf.predict_proba(X_test)
                        models['Random Forest'] = rf
                        
                        # Cross Validation (10 folds)
                        cv_accuracy = cross_val_score(rf, X_scaled, y, cv=kfold, scoring='accuracy')
                        cv_precision = cross_val_score(rf, X_scaled, y, cv=kfold, scoring='precision_weighted')
                        cv_recall = cross_val_score(rf, X_scaled, y, cv=kfold, scoring='recall_weighted')
                        cv_f1 = cross_val_score(rf, X_scaled, y, cv=kfold, scoring='f1_weighted')
                        
                        # Calculate AUC (for multi-class, use One-vs-Rest)
                        try:
                            if len(np.unique(y)) == 2:
                                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                                cv_auc = cross_val_score(rf, X_scaled, y, cv=kfold, scoring='roc_auc')
                            else:
                                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                                cv_auc = cross_val_score(rf, X_scaled, y, cv=kfold, scoring='roc_auc_ovr')
                        except:
                            auc = 0
                            cv_auc = np.array([0])
                        
                        results['Random Forest'] = {
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                            'AUC': auc,
                            'CV_Accuracy_Mean': cv_accuracy.mean(),
                            'CV_Accuracy_Std': cv_accuracy.std(),
                            'CV_Precision_Mean': cv_precision.mean(),
                            'CV_Recall_Mean': cv_recall.mean(),
                            'CV_F1_Mean': cv_f1.mean(),
                            'CV_AUC_Mean': cv_auc.mean() if len(cv_auc) > 0 else 0
                        }
                        st.write(f"  ✅ Random Forest - Accuracy: {results['Random Forest']['Accuracy']:.4f}, CV: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
                    
                    # ========================================
                    # LOGISTIC REGRESSION
                    # ========================================
                    if use_lr:
                        st.write("**Training Logistic Regression...**")
                        lr = LogisticRegression(max_iter=1000, random_state=42)
                        
                        # Train
                        lr.fit(X_train, y_train)
                        y_pred = lr.predict(X_test)
                        y_pred_proba = lr.predict_proba(X_test)
                        models['Logistic Regression'] = lr
                        
                        # Cross Validation (10 folds)
                        cv_accuracy = cross_val_score(lr, X_scaled, y, cv=kfold, scoring='accuracy')
                        cv_precision = cross_val_score(lr, X_scaled, y, cv=kfold, scoring='precision_weighted')
                        cv_recall = cross_val_score(lr, X_scaled, y, cv=kfold, scoring='recall_weighted')
                        cv_f1 = cross_val_score(lr, X_scaled, y, cv=kfold, scoring='f1_weighted')
                        
                        # Calculate AUC
                        try:
                            if len(np.unique(y)) == 2:
                                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                                cv_auc = cross_val_score(lr, X_scaled, y, cv=kfold, scoring='roc_auc')
                            else:
                                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                                cv_auc = cross_val_score(lr, X_scaled, y, cv=kfold, scoring='roc_auc_ovr')
                        except:
                            auc = 0
                            cv_auc = np.array([0])
                        
                        results['Logistic Regression'] = {
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                            'AUC': auc,
                            'CV_Accuracy_Mean': cv_accuracy.mean(),
                            'CV_Accuracy_Std': cv_accuracy.std(),
                            'CV_Precision_Mean': cv_precision.mean(),
                            'CV_Recall_Mean': cv_recall.mean(),
                            'CV_F1_Mean': cv_f1.mean(),
                            'CV_AUC_Mean': cv_auc.mean() if len(cv_auc) > 0 else 0
                        }
                        st.write(f"  ✅ Logistic Regression - Accuracy: {results['Logistic Regression']['Accuracy']:.4f}, CV: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
                    
                    # ========================================
                    # DECISION TREE
                    # ========================================
                    if use_dt:
                        st.write("**Training Decision Tree...**")
                        dt = DecisionTreeClassifier(random_state=42)
                        
                        # Train
                        dt.fit(X_train, y_train)
                        y_pred = dt.predict(X_test)
                        y_pred_proba = dt.predict_proba(X_test)
                        models['Decision Tree'] = dt
                        
                        # Cross Validation (10 folds)
                        cv_accuracy = cross_val_score(dt, X_scaled, y, cv=kfold, scoring='accuracy')
                        cv_precision = cross_val_score(dt, X_scaled, y, cv=kfold, scoring='precision_weighted')
                        cv_recall = cross_val_score(dt, X_scaled, y, cv=kfold, scoring='recall_weighted')
                        cv_f1 = cross_val_score(dt, X_scaled, y, cv=kfold, scoring='f1_weighted')
                        
                        # Calculate AUC
                        try:
                            if len(np.unique(y)) == 2:
                                auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                                cv_auc = cross_val_score(dt, X_scaled, y, cv=kfold, scoring='roc_auc')
                            else:
                                auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                                cv_auc = cross_val_score(dt, X_scaled, y, cv=kfold, scoring='roc_auc_ovr')
                        except:
                            auc = 0
                            cv_auc = np.array([0])
                        
                        results['Decision Tree'] = {
                            'Accuracy': accuracy_score(y_test, y_pred),
                            'Precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                            'Recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                            'F1-Score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
                            'AUC': auc,
                            'CV_Accuracy_Mean': cv_accuracy.mean(),
                            'CV_Accuracy_Std': cv_accuracy.std(),
                            'CV_Precision_Mean': cv_precision.mean(),
                            'CV_Recall_Mean': cv_recall.mean(),
                            'CV_F1_Mean': cv_f1.mean(),
                            'CV_AUC_Mean': cv_auc.mean() if len(cv_auc) > 0 else 0
                        }
                        st.write(f"  ✅ Decision Tree - Accuracy: {results['Decision Tree']['Accuracy']:.4f}, CV: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
                    
                    # Save to session
                    st.session_state.models = models
                    st.session_state.results = results
                    st.session_state.trained = True
                    
                    st.success("✅ Training complete with 10-Fold Cross Validation!")
                    
                    # ========================================
                    # DISPLAY RESULTS
                    # ========================================
                    st.subheader("📊 Model Performance Results")
                    
                    # Create comparison DataFrame
                    comparison = []
                    for name in results.keys():
                        comparison.append({
                            'Model': name,
                            'Accuracy': results[name]['Accuracy'],
                            'Precision': results[name]['Precision'],
                            'Recall': results[name]['Recall'],
                            'F1-Score': results[name]['F1-Score'],
                            'AUC': results[name]['AUC'],
                            'CV Accuracy': f"{results[name]['CV_Accuracy_Mean']:.4f} ± {results[name]['CV_Accuracy_Std']:.4f}"
                        })
                    
                    comparison_df = pd.DataFrame(comparison)
                    st.dataframe(comparison_df.style.format({
                        'Accuracy': '{:.4f}', 'Precision': '{:.4f}', 
                        'Recall': '{:.4f}', 'F1-Score': '{:.4f}', 'AUC': '{:.4f}'
                    }).highlight_max(subset=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']))
                    
                    # Best model
                    best_model = max(results, key=lambda x: results[x]['Accuracy'])
                    st.success(f"🏆 **Best Model: {best_model}** with Accuracy: {results[best_model]['Accuracy']:.4f}")
                    
                    # ========================================
                    # CONFUSION MATRIX
                    # ========================================
                    st.subheader(f"Confusion Matrix - {best_model}")
                    best_model_obj = models[best_model]
                    
                    # Get predictions for test set
                    y_pred_best = best_model_obj.predict(X_test)
                    cm = confusion_matrix(y_test, y_pred_best)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    labels = target_enc.classes_ if target_enc else [str(i) for i in range(len(np.unique(y)))]
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                               xticklabels=labels, yticklabels=labels)
                    ax.set_xlabel('Predicted')
                    ax.set_ylabel('Actual')
                    ax.set_title(f'Confusion Matrix - {best_model}')
                    st.pyplot(fig)
                    
                    # ========================================
                    # CLASSIFICATION REPORT
                    # ========================================
                    st.subheader("Classification Report")
                    report = classification_report(y_test, y_pred_best, target_names=labels, output_dict=True)
                    report_df = pd.DataFrame(report).T
                    st.dataframe(report_df.round(4))
                    
                    # ========================================
                    # FEATURE IMPORTANCE (Random Forest)
                    # ========================================
                    if 'Random Forest' in models:
                        st.subheader("Feature Importance (Random Forest)")
                        rf_model = models['Random Forest']
                        importance_df = pd.DataFrame({
                            'Feature': X_columns,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance_df['Feature'][:10][::-1], importance_df['Importance'][:10][::-1])
                        ax.set_xlabel('Importance')
                        ax.set_title('Top 10 Features')
                        st.pyplot(fig)
                    
                    # ========================================
                    # CROSS VALIDATION RESULTS
                    # ========================================
                    st.subheader(f"10-Fold Cross Validation Results")
                    cv_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'CV Accuracy (Mean)': [results[m]['CV_Accuracy_Mean'] for m in results.keys()],
                        'CV Accuracy (Std)': [results[m]['CV_Accuracy_Std'] for m in results.keys()],
                        'CV Precision': [results[m]['CV_Precision_Mean'] for m in results.keys()],
                        'CV Recall': [results[m]['CV_Recall_Mean'] for m in results.keys()],
                        'CV F1-Score': [results[m]['CV_F1_Mean'] for m in results.keys()],
                    })
                    st.dataframe(cv_df.style.format('{:.4f}').highlight_max(subset=['CV Accuracy (Mean)']))
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    st.info("Please check your data and try again")

# ============================================
# PAGE 3: PREDICT
# ============================================
elif page == "📈 Predict":
    st.header("📈 Make Predictions")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train models first")
        st.stop()
    
    if st.session_state.data is None:
        st.warning("⚠️ No data loaded")
        st.stop()
    
    st.subheader("Enter Values for Prediction")
    
    input_data = {}
    cols = st.columns(2)
    df = st.session_state.data
    
    for i, feat in enumerate(st.session_state.features):
        with cols[i % 2]:
            if feat in df.columns:
                if df[feat].dtype == 'object':
                    # For categorical features
                    values = df[feat].dropna().unique().tolist()
                    input_data[feat] = st.selectbox(f"{feat}", values)
                else:
                    # For numeric features
                    min_val = float(df[feat].min())
                    max_val = float(df[feat].max())
                    mean_val = float(df[feat].mean())
                    input_data[feat] = st.number_input(
                        f"{feat}", 
                        value=mean_val,
                        min_value=min_val, 
                        max_value=max_val,
                        step=(max_val - min_val) / 100
                    )
            else:
                input_data[feat] = st.number_input(f"{feat}", value=0.0)
    
    if st.button("🔮 PREDICT", type="primary", use_container_width=True):
        try:
            # Process input
            input_df = pd.DataFrame([input_data])
            
            # Encode categorical features
            for feat, encoder in st.session_state.encoders.items():
                if feat in input_df.columns:
                    val = input_df[feat].iloc[0]
                    if val in encoder.classes_:
                        input_df[feat + '_encoded'] = encoder.transform([val])[0]
                    else:
                        input_df[feat + '_encoded'] = -1
            
            # Build feature vector
            X_input = []
            for feat in st.session_state.features:
                if feat + '_encoded' in input_df.columns:
                    X_input.append(input_df[feat + '_encoded'].iloc[0])
                elif feat in input_df.columns and input_df[feat].dtype in ['int64', 'float64']:
                    X_input.append(input_df[feat].iloc[0])
            
            X_input = np.array(X_input).reshape(1, -1)
            X_scaled = st.session_state.scaler.transform(X_input)
            
            st.subheader("🎯 Prediction Results")
            
            # Show predictions from all models
            pred_cols = st.columns(len(st.session_state.models))
            
            for idx, (name, model) in enumerate(st.session_state.models.items()):
                pred = model.predict(X_scaled)[0]
                
                if st.session_state.target_encoder:
                    label = st.session_state.target_encoder.inverse_transform([int(pred)])[0]
                else:
                    label = str(pred)
                
                # Get confidence/probability if available
                confidence = ""
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0]
                    confidence = f"<p style='font-size: 0.8rem;'>Confidence: {max(proba)*100:.1f}%</p>"
                
                # Color based on risk level
                risk_lower = str(label).lower()
                if 'high' in risk_lower:
                    color = "#ff6b6b"
                    text_color = "white"
                elif 'medium' in risk_lower:
                    color = "#ffd93d"
                    text_color = "#333"
                else:
                    color = "#6bcb77"
                    text_color = "white"
                
                with pred_cols[idx]:
                    st.markdown(f"""
                    <div style="background: {color}; padding: 1rem; border-radius: 10px; text-align: center; color: {text_color};">
                        <h3>{name}</h3>
                        <h2 style="margin: 0.5rem 0;">{label}</h2>
                        {confidence}
                    </div>
                    """, unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please make sure all inputs are valid")

# ============================================
# PAGE 4: RESULTS
# ============================================
elif page == "📊 Results":
    st.header("📊 Results Summary")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train models first")
        st.stop()
    
    st.subheader("Model Performance Comparison")
    
    results_df = pd.DataFrame(st.session_state.results).T
    
    # Main metrics table
    main_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    st.dataframe(results_df[main_metrics].style.format('{:.4f}').highlight_max(axis=0))
    
    # Bar chart comparison
    st.subheader("Visual Comparison")
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(results_df.index))
    width = 0.15
    
    for i, metric in enumerate(main_metrics):
        values = [results_df.loc[model, metric] for model in results_df.index]
        ax.bar(x + i*width, values, width, label=metric)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(results_df.index, rotation=45, ha='right')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)
    st.pyplot(fig)
    
    # Cross Validation Results
    st.subheader("10-Fold Cross Validation Results")
    cv_metrics = ['CV_Accuracy_Mean', 'CV_Precision_Mean', 'CV_Recall_Mean', 'CV_F1_Mean', 'CV_AUC_Mean']
    cv_df = results_df[cv_metrics].copy()
    cv_df.columns = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC']
    st.dataframe(cv_df.style.format('{:.4f}').highlight_max(axis=0))
    
    # Best model summary
    best_model = max(st.session_state.results, key=lambda x: st.session_state.results[x]['Accuracy'])
    st.subheader(f"🏆 Best Model: {best_model}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Accuracy", f"{st.session_state.results[best_model]['Accuracy']:.4f}")
    with col2:
        st.metric("Precision", f"{st.session_state.results[best_model]['Precision']:.4f}")
    with col3:
        st.metric("Recall", f"{st.session_state.results[best_model]['Recall']:.4f}")
    with col4:
        st.metric("F1-Score", f"{st.session_state.results[best_model]['F1-Score']:.4f}")
    with col5:
        st.metric("AUC", f"{st.session_state.results[best_model]['AUC']:.4f}")
    
    # Download report
    st.subheader("Download Report")
    
    report = f"""
    ========================================
    MICROPLASTIC RISK PREDICTION SYSTEM REPORT
    ========================================
    
    Date: {pd.Timestamp.now()}
    Researchers: Viernes, M.J. & Magdaluyo, S.M.R.
    
    CONFIGURATION:
    - Algorithms: Random Forest, Logistic Regression, Decision Tree
    - Validation: 10-Fold Cross Validation
    - Target: {st.session_state.target_col}
    - Features: {', '.join(st.session_state.features)}
    
    PERFORMANCE RESULTS:
    {results_df[main_metrics].to_string()}
    
    10-FOLD CROSS VALIDATION RESULTS:
    {cv_df.to_string()}
    
    BEST MODEL: {best_model}
    - Accuracy: {st.session_state.results[best_model]['Accuracy']:.4f}
    - Precision: {st.session_state.results[best_model]['Precision']:.4f}
    - Recall: {st.session_state.results[best_model]['Recall']:.4f}
    - F1-Score: {st.session_state.results[best_model]['F1-Score']:.4f}
    - AUC: {st.session_state.results[best_model]['AUC']:.4f}
    
    ========================================
    END OF REPORT
    ========================================
    """
    
    st.download_button(
        "📥 Download Report", 
        report, 
        f"microplastic_risk_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
    )

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌊 Microplastic Risk Prediction System | ASSCAT 2025</p>
    <p>Algorithms: Random Forest | Logistic Regression | Decision Tree | 10-Fold Cross Validation</p>
    <p>Metrics: Accuracy | Precision | Recall | F1-Score | AUC</p>
</div>
""", unsafe_allow_html=True)
