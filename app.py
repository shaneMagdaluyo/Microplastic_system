# app.py - Microplastic Risk Prediction System (FIXED)
# Handles both Classification and Regression

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, r2_score, mean_squared_error
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
if 'task_type' not in st.session_state:
    st.session_state.task_type = "Classification"
if 'X_columns' not in st.session_state:
    st.session_state.X_columns = []

# ============================================
# SIDEBAR
# ============================================
st.sidebar.markdown("## 📌 Navigation")
page = st.sidebar.radio("", ["📁 Upload Data", "🤖 Train Models", "📈 Predict", "📊 Results"])

st.sidebar.markdown("---")
if st.session_state.data is not None:
    st.sidebar.success(f"✅ Data: {st.session_state.data.shape[0]} rows, {st.session_state.data.shape[1]} cols")
else:
    st.sidebar.warning("⚠️ No data loaded")

if st.session_state.trained:
    st.sidebar.success("✅ Models trained")

# ============================================
# GENERATE SAMPLE DATA
# ============================================
def generate_sample_data():
    np.random.seed(42)
    n = 1000
    
    data = {
        'Location': np.random.choice(['Coastal', 'River', 'Urban', 'Industrial', 'Agricultural'], n),
        'Water_Type': np.random.choice(['Freshwater', 'Marine', 'Estuary'], n),
        'MP_Concentration': np.random.uniform(0.1, 500, n),
        'Particle_Size_mm': np.random.uniform(0.01, 5, n),
        'Water_Temperature_C': np.random.uniform(10, 35, n),
        'pH_Level': np.random.uniform(6, 8.5, n),
        'Dissolved_Oxygen_mgL': np.random.uniform(2, 12, n),
        'Turbidity_NTU': np.random.uniform(1, 100, n),
        'Industrial_Score': np.random.uniform(0, 1, n),
        'Waste_Management_Score': np.random.uniform(0, 1, n),
        'Population_Density': np.random.uniform(10, 10000, n),
    }
    df = pd.DataFrame(data)
    
    # Calculate risk score (continuous)
    df['Risk_Score'] = (
        df['MP_Concentration'] / 500 * 35 +
        df['Industrial_Score'] * 30 +
        (1 - df['Waste_Management_Score']) * 20 +
        np.where(df['Particle_Size_mm'] < 0.5, 15, 0)
    )
    df['Risk_Score'] = df['Risk_Score'].clip(0, 100)
    
    # Convert to categorical risk level
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
    
    # Check if target is numeric with many unique values
    if df_proc[target].dtype in ['int64', 'float64'] and df_proc[target].nunique() > 10:
        # Use Regression
        y = df_proc[target].values
        return X, y, encoders, None, X.columns.tolist(), "Regression"
    else:
        # Use Classification
        if df_proc[target].dtype == 'object':
            target_enc = LabelEncoder()
            y = target_enc.fit_transform(df_proc[target].astype(str))
            return X, y, encoders, target_enc, X.columns.tolist(), "Classification"
        else:
            # Numeric but few unique values - treat as classification
            y = df_proc[target].values
            return X, y, encoders, None, X.columns.tolist(), "Classification"

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
        
        df = st.session_state.data
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Rows", df.shape[0])
        with col2:
            st.metric("Columns", df.shape[1])
        with col3:
            numeric = len(df.select_dtypes(include=[np.number]).columns)
            st.metric("Numeric Features", numeric)
        with col4:
            categorical = len(df.select_dtypes(include=['object']).columns)
            st.metric("Categorical Features", categorical)
        
        # Show column info
        st.subheader("Column Information")
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Type': df.dtypes.values,
            'Unique Values': [df[col].nunique() for col in df.columns],
            'Sample': [str(df[col].iloc[0])[:30] for col in df.columns]
        })
        st.dataframe(col_info)
        
        # Recommendation for target column
        st.info("💡 **Tip:** For classification, use a column with few unique values (like 'Risk_Level' with Low/Medium/High). For regression, use a numeric column (like 'Risk_Score').")

# ============================================
# PAGE 2: TRAIN MODELS
# ============================================
elif page == "🤖 Train Models":
    st.header("🤖 Train Models")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        st.stop()
    
    df = st.session_state.data
    
    # Show available columns with their unique value counts
    st.subheader("Available Columns")
    col_info = pd.DataFrame({
        'Column': df.columns,
        'Type': df.dtypes.values,
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Recommendation': ['✅ Good for Classification' if df[col].nunique() <= 10 else '📊 Good for Regression' if df[col].dtype in ['int64', 'float64'] else '🔤 Categorical' for col in df.columns]
    })
    st.dataframe(col_info)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select target column
        target = st.selectbox("🎯 Target Column (what to predict)", df.columns.tolist())
        st.session_state.target_col = target
        
        # Auto-detect task type based on target
        unique_count = df[target].nunique()
        is_numeric = df[target].dtype in ['int64', 'float64']
        
        if is_numeric and unique_count > 10:
            st.info(f"📊 Target '{target}' has {unique_count} unique values → Using **REGRESSION**")
            task_type = "Regression"
        elif is_numeric and unique_count <= 10:
            st.info(f"🏷️ Target '{target}' has {unique_count} unique values → Using **CLASSIFICATION**")
            task_type = "Classification"
        else:
            st.info(f"🏷️ Target '{target}' is categorical → Using **CLASSIFICATION**")
            task_type = "Classification"
        
        st.session_state.task_type = task_type
        
        # Show target distribution
        st.write("**Target Distribution:**")
        if task_type == "Classification":
            st.write(df[target].value_counts())
        else:
            st.write(f"Min: {df[target].min():.2f}, Max: {df[target].max():.2f}, Mean: {df[target].mean():.2f}")
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
        st.info("**10-Fold Cross Validation** (as specified in your manuscript)")
        n_folds = 10
    
    # Feature selection
    st.subheader("Select Features for Training")
    features = [c for c in df.columns if c != target]
    selected = st.multiselect("Features to use", features, default=features[:5] if len(features) > 5 else features)
    st.session_state.features = selected
    
    if len(selected) == 0:
        st.error("Please select at least one feature")
        st.stop()
    
    # Model selection (only show appropriate models for task type)
    st.subheader("Select Models to Train")
    
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
            use_rfr = st.checkbox("Random Forest Regressor", value=True)
        with col2:
            use_dtr = st.checkbox("Decision Tree Regressor", value=True)
        use_rf = False
        use_lr = False
        use_dt = False
        
        st.info("Note: Logistic Regression is for classification only. Using Random Forest and Decision Tree for regression.")
    
    if st.button("🚀 START TRAINING", type="primary", use_container_width=True):
        if task_type == "Classification" and not (use_rf or use_lr or use_dt):
            st.error("Select at least one model")
        elif task_type == "Regression" and not (use_rfr or use_dtr):
            st.error("Select at least one model")
        else:
            with st.spinner(f"Training models with {n_folds}-Fold Cross Validation..."):
                try:
                    # Preprocess data
                    X, y, encoders, target_enc, X_columns, detected_task = preprocess_data(df, selected, target)
                    st.session_state.encoders = encoders
                    st.session_state.target_encoder = target_enc
                    st.session_state.X_columns = X_columns
                    
                    # Scale features
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)
                    st.session_state.scaler = scaler
                    
                    # Split data
                    if task_type == "Classification":
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=test_size, random_state=42, stratify=y
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=test_size, random_state=42
                        )
                    
                    # Initialize K-Fold
                    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                    
                    results = {}
                    models = {}
                    
                    # ========================================
                    # CLASSIFICATION MODELS
                    # ========================================
                    if task_type == "Classification":
                        # Random Forest
                        if use_rf:
                            st.write("**Training Random Forest Classifier...**")
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
                                'CV_Mean': cv_scores.mean(),
                                'CV_Std': cv_scores.std()
                            }
                            st.write(f"  ✅ Accuracy: {results['Random Forest']['Accuracy']:.4f}, CV: {cv_scores.mean():.4f}")
                        
                        # Logistic Regression
                        if use_lr:
                            st.write("**Training Logistic Regression...**")
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
                                'CV_Mean': cv_scores.mean(),
                                'CV_Std': cv_scores.std()
                            }
                            st.write(f"  ✅ Accuracy: {results['Logistic Regression']['Accuracy']:.4f}, CV: {cv_scores.mean():.4f}")
                        
                        # Decision Tree
                        if use_dt:
                            st.write("**Training Decision Tree Classifier...**")
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
                                'CV_Mean': cv_scores.mean(),
                                'CV_Std': cv_scores.std()
                            }
                            st.write(f"  ✅ Accuracy: {results['Decision Tree']['Accuracy']:.4f}, CV: {cv_scores.mean():.4f}")
                    
                    # ========================================
                    # REGRESSION MODELS
                    # ========================================
                    else:
                        # Random Forest Regressor
                        if use_rfr:
                            st.write("**Training Random Forest Regressor...**")
                            rfr = RandomForestRegressor(n_estimators=100, random_state=42)
                            rfr.fit(X_train, y_train)
                            y_pred = rfr.predict(X_test)
                            models['Random Forest Regressor'] = rfr
                            
                            cv_scores = cross_val_score(rfr, X_scaled, y, cv=kfold, scoring='r2')
                            
                            results['Random Forest Regressor'] = {
                                'R2 Score': r2_score(y_test, y_pred),
                                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                                'CV_Mean': cv_scores.mean(),
                                'CV_Std': cv_scores.std()
                            }
                            st.write(f"  ✅ R2 Score: {results['Random Forest Regressor']['R2 Score']:.4f}, CV: {cv_scores.mean():.4f}")
                        
                        # Decision Tree Regressor
                        if use_dtr:
                            st.write("**Training Decision Tree Regressor...**")
                            dtr = DecisionTreeRegressor(random_state=42)
                            dtr.fit(X_train, y_train)
                            y_pred = dtr.predict(X_test)
                            models['Decision Tree Regressor'] = dtr
                            
                            cv_scores = cross_val_score(dtr, X_scaled, y, cv=kfold, scoring='r2')
                            
                            results['Decision Tree Regressor'] = {
                                'R2 Score': r2_score(y_test, y_pred),
                                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                                'CV_Mean': cv_scores.mean(),
                                'CV_Std': cv_scores.std()
                            }
                            st.write(f"  ✅ R2 Score: {results['Decision Tree Regressor']['R2 Score']:.4f}, CV: {cv_scores.mean():.4f}")
                    
                    # Save to session
                    st.session_state.models = models
                    st.session_state.results = results
                    st.session_state.trained = True
                    
                    st.success("✅ Training complete!")
                    
                    # Display results
                    st.subheader("📊 Model Performance Results")
                    results_df = pd.DataFrame(results).T
                    
                    if task_type == "Classification":
                        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
                        st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0))
                        
                        # Best model
                        best_model = max(results, key=lambda x: results[x]['Accuracy'])
                        st.success(f"🏆 **Best Model: {best_model}** with Accuracy: {results[best_model]['Accuracy']:.4f}")
                        
                        # Confusion Matrix for best model
                        if best_model in models:
                            st.subheader(f"Confusion Matrix - {best_model}")
                            best_model_obj = models[best_model]
                            y_pred_best = best_model_obj.predict(X_test)
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
                            ax.set_title(f'Confusion Matrix - {best_model}')
                            st.pyplot(fig)
                    
                    else:
                        display_cols = ['R2 Score', 'RMSE', 'CV_Mean']
                        st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0, subset=['R2 Score']))
                        
                        # Best model
                        best_model = max(results, key=lambda x: results[x]['R2 Score'])
                        st.success(f"🏆 **Best Model: {best_model}** with R2 Score: {results[best_model]['R2 Score']:.4f}")
                    
                    # Feature Importance for Random Forest models
                    rf_model_name = 'Random Forest' if 'Random Forest' in models else 'Random Forest Regressor' if 'Random Forest Regressor' in models else None
                    if rf_model_name and rf_model_name in models and hasattr(models[rf_model_name], 'feature_importances_'):
                        st.subheader("Feature Importance")
                        rf_model = models[rf_model_name]
                        importance_df = pd.DataFrame({
                            'Feature': X_columns,
                            'Importance': rf_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(importance_df['Feature'][:10][::-1], importance_df['Importance'][:10][::-1])
                        ax.set_xlabel('Importance')
                        ax.set_title('Top 10 Features')
                        st.pyplot(fig)
                    
                    # Cross Validation Results
                    st.subheader(f"{n_folds}-Fold Cross Validation Results")
                    cv_df = pd.DataFrame({
                        'Model': list(results.keys()),
                        'CV Mean': [results[m]['CV_Mean'] for m in results.keys()],
                        'CV Std': [results[m]['CV_Std'] for m in results.keys()]
                    })
                    st.dataframe(cv_df.style.format('{:.4f}'))
                    
                except Exception as e:
                    st.error(f"Training error: {str(e)}")
                    st.info("""
                    **Troubleshooting Tips:**
                    1. For CLASSIFICATION: Target should have few categories (like 'Low', 'Medium', 'High')
                    2. For REGRESSION: Target should be a continuous number (like 'Risk_Score')
                    3. Check if your target column has at least 2 different values
                    4. Try using the sample data first to test the system
                    """)

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
                    values = df[feat].dropna().unique().tolist()
                    input_data[feat] = st.selectbox(f"{feat}", values)
                else:
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
            input_df = pd.DataFrame([input_data])
            
            for feat, encoder in st.session_state.encoders.items():
                if feat in input_df.columns:
                    val = input_df[feat].iloc[0]
                    if val in encoder.classes_:
                        input_df[feat + '_encoded'] = encoder.transform([val])[0]
                    else:
                        input_df[feat + '_encoded'] = -1
            
            X_input = []
            for feat in st.session_state.features:
                if feat + '_encoded' in input_df.columns:
                    X_input.append(input_df[feat + '_encoded'].iloc[0])
                elif feat in input_df.columns and input_df[feat].dtype in ['int64', 'float64']:
                    X_input.append(input_df[feat].iloc[0])
            
            X_input = np.array(X_input).reshape(1, -1)
            X_scaled = st.session_state.scaler.transform(X_input)
            
            st.subheader("🎯 Prediction Results")
            
            pred_cols = st.columns(len(st.session_state.models))
            
            for idx, (name, model) in enumerate(st.session_state.models.items()):
                pred = model.predict(X_scaled)[0]
                
                if st.session_state.task_type == "Classification":
                    if st.session_state.target_encoder:
                        label = st.session_state.target_encoder.inverse_transform([int(pred)])[0]
                    else:
                        label = str(pred)
                    
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
    st.header("📊 Results Summary")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train models first")
        st.stop()
    
    st.subheader("Model Performance Comparison")
    
    results_df = pd.DataFrame(st.session_state.results).T
    
    if st.session_state.task_type == "Classification":
        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
        st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0))
        
        # Bar chart
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
        display_cols = ['R2 Score', 'RMSE', 'CV_Mean']
        st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0, subset=['R2 Score']))
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(results_df.index, results_df['R2 Score'], color=['#667eea', '#764ba2'])
        ax.set_xlabel('Models')
        ax.set_ylabel('R2 Score')
        ax.set_title('R2 Score Comparison')
        ax.set_ylim(-1, 1)
        st.pyplot(fig)
    
    # Best model summary
    if st.session_state.task_type == "Classification":
        best_model = max(st.session_state.results, key=lambda x: st.session_state.results[x]['Accuracy'])
        best_score = st.session_state.results[best_model]['Accuracy']
    else:
        best_model = max(st.session_state.results, key=lambda x: st.session_state.results[x]['R2 Score'])
        best_score = st.session_state.results[best_model]['R2 Score']
    
    st.subheader(f"🏆 Best Model: {best_model}")
    st.metric("Best Score", f"{best_score:.4f}")
    
    # Download report
    st.subheader("Download Report")
    
    report = f"""
    ========================================
    MICROPLASTIC RISK PREDICTION SYSTEM REPORT
    ========================================
    
    Date: {pd.Timestamp.now()}
    Task Type: {st.session_state.task_type}
    Target: {st.session_state.target_col}
    Features: {', '.join(st.session_state.features)}
    
    PERFORMANCE RESULTS:
    {results_df[display_cols].to_string()}
    
    BEST MODEL: {best_model}
    BEST SCORE: {best_score:.4f}
    
    ========================================
    END OF REPORT
    ========================================
    """
    
    st.download_button("📥 Download Report", report, f"report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌊 Microplastic Risk Prediction System | ASSCAT 2025</p>
    <p>Algorithms: Random Forest | Logistic Regression | Decision Tree | 10-Fold Cross Validation</p>
</div>
""", unsafe_allow_html=True)
