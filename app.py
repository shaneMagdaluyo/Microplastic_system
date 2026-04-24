# app.py - Microplastic Risk Prediction System (FULLY FIXED)
# Fixed: KeyError and ValueError issues

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
    st.sidebar.success(f"✅ Data: {st.session_state.data.shape[0]} rows")
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
    
    # Calculate risk score (continuous - for regression)
    df['Risk_Score'] = (
        df['MP_Concentration'] / 500 * 35 +
        df['Industrial_Score'] * 30 +
        (1 - df['Waste_Management_Score']) * 20 +
        np.where(df['Particle_Size_mm'] < 0.5, 15, 0)
    )
    df['Risk_Score'] = df['Risk_Score'].clip(0, 100)
    
    # Risk Level (categorical - for classification)
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
    if target in df_proc.columns:
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
    else:
        raise ValueError(f"Target column '{target}' not found in data")

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
        })
        st.dataframe(col_info)
        
        st.info("💡 **Tip:** For classification (predict categories), use 'Risk_Level'. For regression (predict score), use 'Risk_Score'.")

# ============================================
# PAGE 2: TRAIN MODELS
# ============================================
elif page == "🤖 Train Models":
    st.header("🤖 Train Models")
    
    if st.session_state.data is None:
        st.warning("⚠️ Please upload data first")
        st.stop()
    
    df = st.session_state.data
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Select target column
        target = st.selectbox("🎯 Target Column (what to predict)", df.columns.tolist())
        st.session_state.target_col = target
        
        # Show target info
        unique_count = df[target].nunique()
        is_numeric = df[target].dtype in ['int64', 'float64']
        
        if is_numeric and unique_count > 10:
            st.info(f"📊 Target has {unique_count} unique values → **REGRESSION**")
            task_type = "Regression"
        else:
            st.info(f"🏷️ Target has {unique_count} unique values → **CLASSIFICATION**")
            task_type = "Classification"
        
        st.session_state.task_type = task_type
    
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2)
        st.info("**10-Fold Cross Validation**")
        n_folds = 10
    
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
    
    if st.button("🚀 START TRAINING", type="primary", use_container_width=True):
        if task_type == "Classification" and not (use_rf or use_lr or use_dt):
            st.error("Select at least one model")
        elif task_type == "Regression" and not (use_rfr or use_dtr):
            st.error("Select at least one model")
        else:
            with st.spinner(f"Training with {n_folds}-Fold Cross Validation..."):
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
                    
                    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                    results = {}
                    models = {}
                    
                    if task_type == "Classification":
                        # Random Forest
                        if use_rf:
                            st.write("Training Random Forest...")
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
                            st.write(f"  ✅ Accuracy: {results['Random Forest']['Accuracy']:.4f}")
                        
                        # Logistic Regression
                        if use_lr:
                            st.write("Training Logistic Regression...")
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
                            st.write(f"  ✅ Accuracy: {results['Logistic Regression']['Accuracy']:.4f}")
                        
                        # Decision Tree
                        if use_dt:
                            st.write("Training Decision Tree...")
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
                            st.write(f"  ✅ Accuracy: {results['Decision Tree']['Accuracy']:.4f}")
                    
                    else:  # Regression
                        # Random Forest Regressor
                        if use_rfr:
                            st.write("Training Random Forest Regressor...")
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
                            st.write(f"  ✅ R2 Score: {results['Random Forest Regressor']['R2 Score']:.4f}")
                        
                        # Decision Tree Regressor
                        if use_dtr:
                            st.write("Training Decision Tree Regressor...")
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
                            st.write(f"  ✅ R2 Score: {results['Decision Tree Regressor']['R2 Score']:.4f}")
                    
                    st.session_state.models = models
                    st.session_state.results = results
                    st.session_state.trained = True
                    
                    st.success("✅ Training complete!")
                    
                    # Display results
                    st.subheader("📊 Results")
                    results_df = pd.DataFrame(results).T
                    
                    if task_type == "Classification":
                        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
                        st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0))
                        best_model = max(results, key=lambda x: results[x]['Accuracy'])
                        st.success(f"🏆 Best: {best_model} (Accuracy: {results[best_model]['Accuracy']:.4f})")
                    else:
                        display_cols = ['R2 Score', 'RMSE', 'CV_Mean']
                        st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0, subset=['R2 Score']))
                        best_model = max(results, key=lambda x: results[x]['R2 Score'])
                        st.success(f"🏆 Best: {best_model} (R2: {results[best_model]['R2 Score']:.4f})")
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")

# ============================================
# PAGE 3: PREDICT (FIXED VERSION)
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
    df_original = st.session_state.data
    
    # Create input fields for each feature
    for i, feat in enumerate(st.session_state.features):
        # Check if feature exists in the original data
        if feat in df_original.columns:
            if df_original[feat].dtype == 'object':
                # Categorical feature - dropdown
                unique_values = df_original[feat].dropna().unique().tolist()
                input_data[feat] = st.selectbox(f"📊 {feat}", unique_values, key=f"select_{feat}")
            else:
                # Numeric feature - number input
                min_val = float(df_original[feat].min())
                max_val = float(df_original[feat].max())
                default_val = float(df_original[feat].mean())
                input_data[feat] = st.number_input(
                    f"📈 {feat}", 
                    value=default_val,
                    min_value=min_val, 
                    max_value=max_val,
                    step=(max_val - min_val) / 100,
                    key=f"num_{feat}"
                )
        else:
            # Feature not found - use default
            input_data[feat] = st.number_input(f"{feat}", value=0.0, key=f"default_{feat}")
    
    if st.button("🔮 PREDICT", type="primary", use_container_width=True):
        try:
            # Create input dataframe
            input_df = pd.DataFrame([input_data])
            
            # Apply encoders for categorical features
            for feat, encoder in st.session_state.encoders.items():
                if feat in input_df.columns:
                    val = input_df[feat].iloc[0]
                    if val in encoder.classes_:
                        input_df[feat + '_encoded'] = encoder.transform([val])[0]
                    else:
                        # Handle unknown category
                        input_df[feat + '_encoded'] = 0
            
            # Build feature vector
            X_input = []
            for feat in st.session_state.features:
                # Check for encoded version first
                if feat + '_encoded' in input_df.columns:
                    X_input.append(input_df[feat + '_encoded'].iloc[0])
                elif feat in input_df.columns:
                    val = input_df[feat].iloc[0]
                    # Try to convert to numeric
                    try:
                        X_input.append(float(val))
                    except:
                        X_input.append(0)
                else:
                    X_input.append(0)
            
            # Reshape and scale
            X_input = np.array(X_input).reshape(1, -1)
            X_scaled = st.session_state.scaler.transform(X_input)
            
            st.subheader("🎯 Prediction Results")
            
            # Show predictions from all trained models
            pred_cols = st.columns(len(st.session_state.models))
            
            for idx, (name, model) in enumerate(st.session_state.models.items()):
                pred = model.predict(X_scaled)[0]
                
                if st.session_state.task_type == "Classification":
                    # Decode prediction
                    if st.session_state.target_encoder:
                        try:
                            label = st.session_state.target_encoder.inverse_transform([int(pred)])[0]
                        except:
                            label = str(pred)
                    else:
                        label = str(pred)
                    
                    # Color based on risk level
                    label_lower = str(label).lower()
                    if 'high' in label_lower:
                        color = "#ff6b6b"
                        text_color = "white"
                    elif 'medium' in label_lower:
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
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    # Regression - show numeric prediction
                    with pred_cols[idx]:
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea, #764ba2); padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                            <h3>{name}</h3>
                            <h2 style="margin: 0.5rem 0;">{pred:.2f}</h2>
                            <p style="margin: 0;">Risk Score</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            st.info("Please check your input values and try again.")

# ============================================
# PAGE 4: RESULTS
# ============================================
elif page == "📊 Results":
    st.header("📊 Results Summary")
    
    if not st.session_state.trained:
        st.warning("⚠️ Please train models first")
        st.stop()
    
    st.subheader("Model Performance")
    
    results_df = pd.DataFrame(st.session_state.results).T
    
    if st.session_state.task_type == "Classification":
        display_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'CV_Mean']
        st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0))
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df.index, results_df['Accuracy'], color=['#667eea', '#764ba2', '#f093fb'])
        ax.set_ylim(0, 1)
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Accuracy Comparison')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
        
    else:
        display_cols = ['R2 Score', 'RMSE', 'CV_Mean']
        st.dataframe(results_df[display_cols].style.format('{:.4f}').highlight_max(axis=0, subset=['R2 Score']))
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(results_df.index, results_df['R2 Score'], color=['#667eea', '#764ba2'])
        ax.set_ylim(-1, 1)
        ax.set_ylabel('R2 Score')
        ax.set_title('Model R2 Score Comparison')
        plt.xticks(rotation=45, ha='right')
        st.pyplot(fig)
    
    # Best model
    if st.session_state.task_type == "Classification":
        best = max(st.session_state.results, key=lambda x: st.session_state.results[x]['Accuracy'])
        best_score = st.session_state.results[best]['Accuracy']
    else:
        best = max(st.session_state.results, key=lambda x: st.session_state.results[x]['R2 Score'])
        best_score = st.session_state.results[best]['R2 Score']
    
    st.success(f"🏆 Best Model: **{best}** (Score: {best_score:.4f})")

# ============================================
# FOOTER
# ============================================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>🌊 MP-RAS | Microplastic Risk Assessment System | ASSCAT 2025</p>
    <p>Random Forest | Logistic Regression | Decision Tree | 10-Fold Cross Validation</p>
</div>
""", unsafe_allow_html=True)
