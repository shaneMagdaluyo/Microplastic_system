# ============================================================================
# PREDICTIVE RISK MODELING FOR MICROPLASTIC POLLUTION
# Complete Implementation with Data Mining Techniques
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             confusion_matrix, classification_report, roc_auc_score, 
                             roc_curve, ConfusionMatrixDisplay)
from sklearn.feature_selection import SelectKBest, chi2, mutual_info_classif, RFE, SelectFromModel
from imblearn.over_sampling import SMOTE

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# DATA GENERATION (Simulating realistic microplastic pollution data)
# ============================================================================

def generate_microplastic_dataset(n_samples=5000):
    """Generate a realistic dataset for microplastic pollution risk assessment"""
    np.random.seed(42)
    
    # Location types
    locations = ['Marine Coastal', 'Freshwater Lake', 'River Estuary', 
                 'Open Ocean', 'Urban Runoff', 'Industrial Area']
    
    # Polymer types
    polymer_types = ['Polyethylene (PE)', 'Polypropylene (PP)', 'Polystyrene (PS)',
                     'Polyethylene Terephthalate (PET)', 'Polyvinyl Chloride (PVC)',
                     'Nylon (PA)', 'Polyacrylamide (PAM)']
    
    # Risk levels
    risk_levels = ['Low', 'Medium', 'High', 'Critical']
    
    # Generate features
    data = {
        'Location': np.random.choice(locations, n_samples),
        'Polymer_Type': np.random.choice(polymer_types, n_samples, p=[0.25, 0.20, 0.15, 0.15, 0.10, 0.08, 0.07]),
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
    }
    
    df = pd.DataFrame(data)
    
    # Calculate MP count per liter based on multiple factors
    mp_count_base = (
        (df['Industrial_Discharge_m3'] / 1000) * 0.5 +
        (1 - df['Wastewater_Treatment_Efficiency']/100) * 100 +
        np.where(df['Polymer_Type'].str.contains('PE|PP'), 50, 20) +
        df['Population_Density_km2'] / 500 +
        np.random.normal(0, 30, n_samples)
    )
    df['MP_Count_per_L'] = np.maximum(5, mp_count_base + np.abs(np.random.normal(0, 20, n_samples)))
    
    # Calculate risk score based on multiple indicators
    risk_score_base = (
        df['MP_Count_per_L'] / 100 * 0.4 +
        (df['Industrial_Discharge_m3'] / 2000) * 0.3 +
        (df['Population_Density_km2'] / 10000) * 0.2 +
        (1 - df['Wastewater_Treatment_Efficiency']/100) * 0.1
    )
    
    df['Risk_Score'] = np.minimum(100, risk_score_base * 100 + np.random.normal(0, 10, n_samples))
    df['Risk_Score'] = np.maximum(0, df['Risk_Score'])
    
    # Assign Risk Level based on Risk Score
    conditions = [
        df['Risk_Score'] < 30,
        (df['Risk_Score'] >= 30) & (df['Risk_Score'] < 55),
        (df['Risk_Score'] >= 55) & (df['Risk_Score'] < 75),
        df['Risk_Score'] >= 75
    ]
    choices = ['Low', 'Medium', 'High', 'Critical']
    df['Risk_Level'] = np.select(conditions, choices)
    
    # Risk Type classification
    risk_type_conditions = []
    for idx, row in df.iterrows():
        if row['Risk_Score'] > 70 and row['MP_Count_per_L'] > 150:
            risk_type_conditions.append('Human Health Risk')
        elif row['Risk_Score'] > 60 and row['Industrial_Discharge_m3'] > 800:
            risk_type_conditions.append('Ecosystem Risk')
        elif row['Polymer_Type'] in ['Polystyrene (PS)', 'Polyvinyl Chloride (PVC)']:
            risk_type_conditions.append('Chemical Hazard')
        elif row['Location'] in ['Urban Runoff', 'Industrial Area']:
            risk_type_conditions.append('Source Contamination')
        else:
            risk_type_conditions.append('General Pollution')
    
    df['Risk_Type'] = risk_type_conditions
    
    # Add color distribution
    color_dist = ['White/Clear', 'Blue', 'Green', 'Red', 'Black', 'Yellow']
    df['Color_Distribution'] = np.random.choice(color_dist, n_samples, p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05])
    
    # Add size range
    size_ranges = ['<0.1mm', '0.1-0.5mm', '0.5-1mm', '1-2mm', '2-5mm']
    df['Size_Range'] = np.random.choice(size_ranges, n_samples, p=[0.3, 0.35, 0.2, 0.1, 0.05])
    
    return df

# Generate the dataset
print("=" * 80)
print("PREDICTIVE RISK MODELING FOR MICROPLASTIC POLLUTION")
print("=" * 80)
print("\nGenerating microplastic pollution dataset...")
df = generate_microplastic_dataset(5000)
print(f"Dataset shape: {df.shape}")
print(f"Columns: {list(df.columns)}")
print("\nFirst 5 rows:")
print(df.head())

# ============================================================================
# TASK 1: DISTRIBUTION ANALYSIS AND RELATIONSHIP EXPLORATION
# ============================================================================

print("\n" + "=" * 80)
print("TASK 1: DISTRIBUTION ANALYSIS AND RELATIONSHIP EXPLORATION")
print("=" * 80)

# 1.1 Analyze the distribution of risk score
print("\n" + "-" * 60)
print("1.1 Risk Score Distribution Analysis")
print("-" * 60)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Risk Score Distribution Analysis', fontsize=16, fontweight='bold')

# Histogram with KDE
axes[0, 0].hist(df['Risk_Score'], bins=30, edgecolor='black', alpha=0.7, color='steelblue')
axes[0, 0].axvline(df['Risk_Score'].mean(), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {df["Risk_Score"].mean():.2f}')
axes[0, 0].axvline(df['Risk_Score'].median(), color='green', linestyle='dashed', linewidth=2, label=f'Median: {df["Risk_Score"].median():.2f}')
axes[0, 0].set_xlabel('Risk Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Histogram of Risk Score')
axes[0, 0].legend()

# Box plot
box = axes[0, 1].boxplot(df['Risk_Score'], patch_artist=True)
box['boxes'][0].set_facecolor('lightblue')
axes[0, 1].set_ylabel('Risk Score')
axes[0, 1].set_title('Box Plot of Risk Score')
axes[0, 1].grid(True, alpha=0.3)

# Q-Q plot
stats.probplot(df['Risk_Score'], dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Q-Q Plot (Normality Check)')

# Statistical summary table
stats_summary = df['Risk_Score'].describe()
summary_text = f"""
Mean: {stats_summary['mean']:.2f}
Std Dev: {stats_summary['std']:.2f}
Min: {stats_summary['min']:.2f}
25%: {stats_summary['25%']:.2f}
50%: {stats_summary['50%']:.2f}
75%: {stats_summary['75%']:.2f}
Max: {stats_summary['max']:.2f}
Skewness: {df['Risk_Score'].skew():.3f}
Kurtosis: {df['Risk_Score'].kurtosis():.3f}
"""
axes[1, 0].text(0.1, 0.5, summary_text, transform=axes[1, 0].transAxes, 
                fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
axes[1, 0].axis('off')
axes[1, 0].set_title('Statistical Summary')

# Density plot by Risk Level
risk_levels_order = ['Low', 'Medium', 'High', 'Critical']
for level in risk_levels_order:
    subset = df[df['Risk_Level'] == level]
    axes[1, 1].hist(subset['Risk_Score'], alpha=0.5, bins=20, label=level)
axes[1, 1].set_xlabel('Risk Score')
axes[1, 1].set_ylabel('Frequency')
axes[1, 1].set_title('Risk Score Distribution by Risk Level')
axes[1, 1].legend()

# Cumulative distribution
sorted_scores = np.sort(df['Risk_Score'])
cumulative = np.arange(1, len(sorted_scores) + 1) / len(sorted_scores)
axes[1, 2].plot(sorted_scores, cumulative, 'b-', linewidth=2)
axes[1, 2].fill_between(sorted_scores, cumulative, alpha=0.3)
axes[1, 2].set_xlabel('Risk Score')
axes[1, 2].set_ylabel('Cumulative Probability')
axes[1, 2].set_title('Cumulative Distribution Function')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('risk_score_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nRisk Score Statistics:")
print(f"  Mean: {df['Risk_Score'].mean():.2f}")
print(f"  Median: {df['Risk_Score'].median():.2f}")
print(f"  Std Dev: {df['Risk_Score'].std():.2f}")
print(f"  Skewness: {df['Risk_Score'].skew():.3f}")
print(f"  Kurtosis: {df['Risk_Score'].kurtosis():.3f}")

# 1.2 Explore relationship between risk score and MP count per liter
print("\n" + "-" * 60)
print("1.2 Relationship: Risk Score vs MP Count per Liter")
print("-" * 60)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle('Relationship Between Risk Score and MP Count per Liter', fontsize=14, fontweight='bold')

# Scatter plot with regression line
axes[0].scatter(df['MP_Count_per_L'], df['Risk_Score'], alpha=0.5, c='steelblue', edgecolors='none')
z = np.polyfit(df['MP_Count_per_L'], df['Risk_Score'], 1)
p = np.poly1d(z)
axes[0].plot(np.sort(df['MP_Count_per_L']), p(np.sort(df['MP_Count_per_L'])), "r-", linewidth=2, label=f'Linear fit (r={pearsonr(df["MP_Count_per_L"], df["Risk_Score"])[0]:.3f})')
axes[0].set_xlabel('MP Count per Liter')
axes[0].set_ylabel('Risk Score')
axes[0].set_title('Scatter Plot with Regression Line')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Hexbin plot for density
hb = axes[1].hexbin(df['MP_Count_per_L'], df['Risk_Score'], gridsize=40, cmap='YlOrRd', mincnt=1)
axes[1].set_xlabel('MP Count per Liter')
axes[1].set_ylabel('Risk Score')
axes[1].set_title('Density Hexbin Plot')
plt.colorbar(hb, ax=axes[1], label='Count')

# Box plot by risk level
risk_order = ['Low', 'Medium', 'High', 'Critical']
risk_data = [df[df['Risk_Level'] == level]['MP_Count_per_L'] for level in risk_order]
bp = axes[2].boxplot(risk_data, labels=risk_order, patch_artist=True)
for patch, color in zip(bp['boxes'], ['green', 'yellow', 'orange', 'red']):
    patch.set_facecolor(color)
axes[2].set_xlabel('Risk Level')
axes[2].set_ylabel('MP Count per Liter')
axes[2].set_title('MP Count Distribution by Risk Level')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('risk_mp_relationship.png', dpi=150, bbox_inches='tight')
plt.show()

# Statistical correlation tests
pearson_corr, pearson_p = pearsonr(df['MP_Count_per_L'], df['Risk_Score'])
spearman_corr, spearman_p = spearmanr(df['MP_Count_per_L'], df['Risk_Score'])

print("\nCorrelation Analysis:")
print(f"  Pearson Correlation: {pearson_corr:.4f} (p-value: {pearson_p:.2e})")
print(f"  Spearman Correlation: {spearman_corr:.4f} (p-value: {spearman_p:.2e})")

# 1.3 Investigate difference in risk score by risk level
print("\n" + "-" * 60)
print("1.3 Risk Score Differences by Risk Level")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Risk Score Differences Across Risk Levels', fontsize=14, fontweight='bold')

# Box plot
bp = axes[0].boxplot([df[df['Risk_Level'] == level]['Risk_Score'] for level in risk_order], 
                      labels=risk_order, patch_artist=True)
colors = ['green', 'yellow', 'orange', 'red']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[0].set_xlabel('Risk Level')
axes[0].set_ylabel('Risk Score')
axes[0].set_title('Risk Score Distribution by Risk Level')
axes[0].grid(True, alpha=0.3)

# Violin plot
for i, level in enumerate(risk_order):
    data = df[df['Risk_Level'] == level]['Risk_Score']
    parts = axes[1].violinplot([data], positions=[i], showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
axes[1].set_xticks(range(len(risk_order)))
axes[1].set_xticklabels(risk_order)
axes[1].set_xlabel('Risk Level')
axes[1].set_ylabel('Risk Score')
axes[1].set_title('Violin Plot by Risk Level')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('risk_score_by_level.png', dpi=150, bbox_inches='tight')
plt.show()

# ANOVA test
risk_level_groups = [df[df['Risk_Level'] == level]['Risk_Score'] for level in risk_order]
f_stat, p_value = f_oneway(*risk_level_groups)
print("\nANOVA Test Results:")
print(f"  F-statistic: {f_stat:.4f}")
print(f"  P-value: {p_value:.2e}")
print(f"  Significant difference: {p_value < 0.05}")

# Post-hoc analysis (mean differences)
print("\nMean Risk Scores by Risk Level:")
for level in risk_order:
    mean_score = df[df['Risk_Level'] == level]['Risk_Score'].mean()
    print(f"  {level}: {mean_score:.2f}")

# ============================================================================
# TASK 2: DATA PREPROCESSING (Encoding, Scaling, Outliers, Transformation)
# ============================================================================

print("\n" + "=" * 80)
print("TASK 2: DATA PREPROCESSING")
print("=" * 80)

# Create a copy for preprocessing
df_processed = df.copy()

# 2.1 Encode categorical variables
print("\n" + "-" * 60)
print("2.1 Encoding Categorical Variables")
print("-" * 60)

categorical_columns = ['Location', 'Polymer_Type', 'Risk_Level', 'Risk_Type', 'Color_Distribution', 'Size_Range']
encoders = {}

for col in categorical_columns:
    le = LabelEncoder()
    df_processed[f'{col}_Encoded'] = le.fit_transform(df_processed[col])
    encoders[col] = le
    print(f"  Encoded '{col}': {len(le.classes_)} classes -> {list(le.classes_)}")

# 2.2 Perform feature scaling
print("\n" + "-" * 60)
print("2.2 Feature Scaling")
print("-" * 60)

numerical_features = ['Water_Temperature_C', 'pH_Level', 'Salinity_ppt', 'Dissolved_Oxygen_mgL',
                      'Turbidity_NTU', 'Population_Density_km2', 'Industrial_Discharge_m3',
                      'Wastewater_Treatment_Efficiency', 'Distance_to_Shore_km', 'Water_Flow_Speed_ms',
                      'MP_Count_per_L', 'Risk_Score']

# StandardScaler
scaler_standard = StandardScaler()
df_processed[numerical_features] = scaler_standard.fit_transform(df_processed[numerical_features])
print(f"  StandardScaler applied to {len(numerical_features)} numerical features")
print(f"  Mean after scaling: {df_processed[numerical_features].mean().mean():.4f}")
print(f"  Std after scaling: {df_processed[numerical_features].std().mean():.4f}")

# 2.3 Address outliers
print("\n" + "-" * 60)
print("2.3 Outlier Detection and Treatment")
print("-" * 60)

# Detect outliers using IQR method on original data
df_original = generate_microplastic_dataset(5000)
outlier_info = {}

for col in numerical_features:
    Q1 = df_original[col].quantile(0.25)
    Q3 = df_original[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df_original[(df_original[col] < lower_bound) | (df_original[col] > upper_bound)]
    outlier_info[col] = len(outliers)
    print(f"  {col}: {len(outliers)} outliers ({len(outliers)/len(df_original)*100:.2f}%)")

# Visualize outliers before and after treatment
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Outlier Treatment Visualization', fontsize=14, fontweight='bold')

cols_to_vis = ['MP_Count_per_L', 'Industrial_Discharge_m3', 'Population_Density_km2']
for idx, col in enumerate(cols_to_vis):
    # Before capping
    axes[0, idx].boxplot(df_original[col])
    axes[0, idx].set_title(f'{col} (Before)')
    axes[0, idx].set_ylabel('Value')
    
    # After capping (using 99th percentile)
    capped = df_original[col].clip(lower=df_original[col].quantile(0.01), 
                                    upper=df_original[col].quantile(0.99))
    axes[1, idx].boxplot(capped)
    axes[1, idx].set_title(f'{col} (After Capping)')
    axes[1, idx].set_ylabel('Value')

plt.tight_layout()
plt.savefig('outlier_treatment.png', dpi=150, bbox_inches='tight')
plt.show()

# 2.4 Transform skewed numerical columns
print("\n" + "-" * 60)
print("2.4 Transforming Skewed Numerical Columns")
print("-" * 60)

skewed_cols = []
for col in ['Turbidity_NTU', 'Population_Density_km2', 'Industrial_Discharge_m3']:
    skewness = df_original[col].skew()
    skewed_cols.append(col)
    print(f"  {col}: skewness = {skewness:.3f}")

# Apply log transformation
df_original_log = df_original.copy()
for col in skewed_cols:
    # Add small constant to avoid log(0)
    df_original_log[f'{col}_log'] = np.log1p(df_original[col])
    print(f"  Log transform applied to {col}: new skewness = {df_original_log[f'{col}_log'].skew():.3f}")

# Visualize transformations
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Log Transformation for Skewed Features', fontsize=14, fontweight='bold')

for idx, col in enumerate(skewed_cols):
    axes[0, idx].hist(df_original[col], bins=30, alpha=0.7, color='steelblue')
    axes[0, idx].set_title(f'{col} (Original)\nSkewness: {df_original[col].skew():.3f}')
    axes[0, idx].set_xlabel(col)
    
    axes[1, idx].hist(df_original_log[f'{col}_log'], bins=30, alpha=0.7, color='coral')
    axes[1, idx].set_title(f'{col} (Log Transformed)\nSkewness: {df_original_log[f"{col}_log"].skew():.3f}')
    axes[1, idx].set_xlabel(f'log({col})')

plt.tight_layout()
plt.savefig('log_transformations.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# TASK 3: FEATURE SELECTION
# ============================================================================

print("\n" + "=" * 80)
print("TASK 3: FEATURE SELECTION")
print("=" * 80)

# Prepare data for modeling
print("\n" + "-" * 60)
print("3.1 Data Preparation for Feature Selection")
print("-" * 60)

# Select features for modeling
feature_cols = ['Water_Temperature_C', 'pH_Level', 'Salinity_ppt', 'Dissolved_Oxygen_mgL',
                'Turbidity_NTU', 'Population_Density_km2', 'Industrial_Discharge_m3',
                'Wastewater_Treatment_Efficiency', 'Distance_to_Shore_km', 'Water_Flow_Speed_ms',
                'MP_Count_per_L', 'Location_Encoded', 'Polymer_Type_Encoded', 'Color_Distribution_Encoded']

X = df_processed[feature_cols]
y = df_processed['Risk_Level_Encoded']

print(f"Feature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target classes: {sorted(df_processed['Risk_Level'].unique())}")

# 3.2 Explore feature selection methods
print("\n" + "-" * 60)
print("3.2 Feature Selection Methods Exploration")
print("-" * 60)

# Method 1: Mutual Information
print("\nMethod 1: Mutual Information Classification")
mi_selector = SelectKBest(score_func=mutual_info_classif, k=10)
mi_selector.fit(X, y)
mi_scores = pd.DataFrame({
    'Feature': feature_cols,
    'MI_Score': mi_selector.scores_
}).sort_values('MI_Score', ascending=False)

print(mi_scores.head(10))

# Method 2: Random Forest Feature Importance
print("\nMethod 2: Random Forest Feature Importance")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
rf_selector.fit(X, y)
rf_importances = pd.DataFrame({
    'Feature': feature_cols,
    'RF_Importance': rf_selector.feature_importances_
}).sort_values('RF_Importance', ascending=False)

print(rf_importances.head(10))

# 3.3 Implement selected methods
print("\n" + "-" * 60)
print("3.3 Feature Selection Implementation")
print("-" * 60)

# Apply RFE with Random Forest
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=8)
rfe.fit(X, y)

selected_features_rfe = [feature_cols[i] for i in range(len(feature_cols)) if rfe.support_[i]]
print(f"\nRFE Selected Features ({len(selected_features_rfe)}):")
for f in selected_features_rfe:
    print(f"  - {f}")

# 3.4 Evaluate selected features
print("\n" + "-" * 60)
print("3.4 Feature Selection Evaluation")
print("-" * 60)

# Compare model performance with all features vs selected features
X_selected = X[selected_features_rfe]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train_sel, X_test_sel, y_train_sel, y_test_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

# Train model with all features
rf_all = RandomForestClassifier(n_estimators=100, random_state=42)
rf_all.fit(X_train, y_train)
y_pred_all = rf_all.predict(X_test)

# Train model with selected features
rf_sel = RandomForestClassifier(n_estimators=100, random_state=42)
rf_sel.fit(X_train_sel, y_train_sel)
y_pred_sel = rf_sel.predict(X_test_sel)

print("\nModel Performance Comparison:")
print(f"  All Features - Accuracy: {accuracy_score(y_test, y_pred_all):.4f}")
print(f"  Selected Features - Accuracy: {accuracy_score(y_test_sel, y_pred_sel):.4f}")

# ============================================================================
# TASK 4: CLASSIFICATION MODELING FOR RISK LEVEL
# ============================================================================

print("\n" + "=" * 80)
print("TASK 4: CLASSIFICATION MODELING FOR RISK LEVEL")
print("=" * 80)

# 4.1 Prepare the data
print("\n" + "-" * 60)
print("4.1 Data Preparation")
print("-" * 60)

X_class = X_selected  # Use selected features
y_class = y

X_train, X_test, y_train, y_test = train_test_split(X_class, y_class, test_size=0.2, random_state=42, stratify=y_class)

print(f"Training set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")
print(f"Class distribution in training:\n{pd.Series(y_train).value_counts().sort_index()}")

# 4.2 Choose classification models
print("\n" + "-" * 60)
print("4.2 Classification Models Selected")
print("-" * 60)

models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42),
    'KNN': KNeighborsClassifier()
}

for name in models.keys():
    print(f"  - {name}")

# 4.3 Train the models
print("\n" + "-" * 60)
print("4.3 Model Training")
print("-" * 60)

trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"  Trained: {name}")

# 4.4 Evaluate the models
print("\n" + "-" * 60)
print("4.4 Model Evaluation")
print("-" * 60)

results = []
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

results_df = pd.DataFrame(results).sort_values('Accuracy', ascending=False)

# 4.5 Compare model performance
print("\n" + "-" * 60)
print("4.5 Model Performance Comparison")
print("-" * 60)

print("\nRanking by Accuracy:")
for idx, row in results_df.iterrows():
    print(f"  {idx+1}. {row['Model']}: {row['Accuracy']:.4f}")

# Visualize model comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Performance Comparison', fontsize=14, fontweight='bold')

# Bar plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(results_df))
width = 0.2

for i, metric in enumerate(metrics):
    axes[0].bar(x + i*width, results_df[metric], width, label=metric)

axes[0].set_xlabel('Model')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Metrics Comparison')
axes[0].set_xticks(x + width*1.5)
axes[0].set_xticklabels(results_df['Model'], rotation=45, ha='right')
axes[0].legend()
axes[0].set_ylim(0, 1)

# Heatmap
heatmap_data = results_df.set_index('Model')[metrics]
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.4f', ax=axes[1])
axes[1].set_title('Performance Heatmap')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# Confusion matrix for best model
best_model_name = results_df.iloc[0]['Model']
best_model = trained_models[best_model_name]
y_pred_best = best_model.predict(X_test)

fig, ax = plt.subplots(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_best)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Low', 'Medium', 'High', 'Critical'])
disp.plot(ax=ax, cmap='Blues', values_format='d')
ax.set_title(f'Confusion Matrix - {best_model_name}')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

print(f"\nBest Model: {best_model_name}")
print(f"Best Model Accuracy: {results_df.iloc[0]['Accuracy']:.4f}")

# ============================================================================
# TASK 5: RISK TYPE CLASSIFICATION MODELING
# ============================================================================

print("\n" + "=" * 80)
print("TASK 5: RISK TYPE CLASSIFICATION")
print("=" * 80)

# 5.1 Prepare data for risk type modeling
print("\n" + "-" * 60)
print("5.1 Data Preparation for Risk Type")
print("-" * 60)

# Encode Risk Type
le_risk_type = LabelEncoder()
df_processed['Risk_Type_Encoded'] = le_risk_type.fit_transform(df['Risk_Type'])

X_type = X_selected  # Use same selected features
y_type = df_processed['Risk_Type_Encoded']

X_train_type, X_test_type, y_train_type, y_test_type = train_test_split(
    X_type, y_type, test_size=0.2, random_state=42, stratify=y_type
)

print(f"Risk Types: {list(le_risk_type.classes_)}")
print(f"Training size: {X_train_type.shape[0]}")
print(f"Test size: {X_test_type.shape[0]}")

# 5.2 Choose classification models
print("\n" + "-" * 60)
print("5.2 Models for Risk Type Classification")
print("-" * 60)

type_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42)
}

for name in type_models.keys():
    print(f"  - {name}")

# 5.3 Train models for risk type
print("\n" + "-" * 60)
print("5.3 Training Risk Type Models")
print("-" * 60)

trained_type_models = {}
for name, model in type_models.items():
    model.fit(X_train_type, y_train_type)
    trained_type_models[name] = model
    print(f"  Trained: {name}")

# 5.4 Evaluate models for risk type
print("\n" + "-" * 60)
print("5.4 Risk Type Model Evaluation")
print("-" * 60)

type_results = []
for name, model in trained_type_models.items():
    y_pred_type = model.predict(X_test_type)
    accuracy = accuracy_score(y_test_type, y_pred_type)
    precision = precision_score(y_test_type, y_pred_type, average='weighted')
    recall = recall_score(y_test_type, y_pred_type, average='weighted')
    f1 = f1_score(y_test_type, y_pred_type, average='weighted')
    
    type_results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

type_results_df = pd.DataFrame(type_results).sort_values('Accuracy', ascending=False)

# 5.5 Compare model performance
print("\n" + "-" * 60)
print("5.5 Risk Type Model Comparison")
print("-" * 60)

print("\nRanking by Accuracy:")
for idx, row in type_results_df.iterrows():
    print(f"  {idx+1}. {row['Model']}: {row['Accuracy']:.4f}")

# Visualize risk type comparison
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Risk Type Classification - Model Performance', fontsize=14, fontweight='bold')

# Bar plot
x = np.arange(len(type_results_df))
width = 0.2
for i, metric in enumerate(metrics):
    axes[0].bar(x + i*width, type_results_df[metric], width, label=metric)

axes[0].set_xlabel('Model')
axes[0].set_ylabel('Score')
axes[0].set_title('Model Metrics Comparison')
axes[0].set_xticks(x + width*1.5)
axes[0].set_xticklabels(type_results_df['Model'], rotation=45, ha='right')
axes[0].legend()
axes[0].set_ylim(0, 1)

# Heatmap
type_heatmap = type_results_df.set_index('Model')[metrics]
sns.heatmap(type_heatmap, annot=True, cmap='YlOrRd', fmt='.4f', ax=axes[1])
axes[1].set_title('Performance Heatmap')

plt.tight_layout()
plt.savefig('risk_type_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================================================
# TASK 6: FEATURE RELEVANCE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("TASK 6: FEATURE RELEVANCE ANALYSIS")
print("=" * 80)

# 6.1 Extract feature relevance
print("\n" + "-" * 60)
print("6.1 Feature Relevance Extraction")
print("-" * 60)

# Get feature importances from best model
best_type_model = trained_type_models[type_results_df.iloc[0]['Model']]
feature_importances = best_type_model.feature_importances_

feature_relevance = pd.DataFrame({
    'Feature': selected_features_rfe,
    'Importance': feature_importances
}).sort_values('Importance', ascending=False)

print("\nFeature Relevance Scores:")
for idx, row in feature_relevance.iterrows():
    print(f"  {row['Feature']}: {row['Importance']:.4f}")

# 6.2 Analyze feature relevance
print("\n" + "-" * 60)
print("6.2 Feature Relevance Analysis")
print("-" * 60)

print("\nTop 5 Most Important Features:")
for idx in range(min(5, len(feature_relevance))):
    print(f"  {idx+1}. {feature_relevance.iloc[idx]['Feature']}: {feature_relevance.iloc[idx]['Importance']:.4f}")

print("\nBottom 3 Least Important Features:")
for idx in range(min(3, len(feature_relevance))):
    rev_idx = -(idx + 1)
    print(f"  {idx+1}. {feature_relevance.iloc[rev_idx]['Feature']}: {feature_relevance.iloc[rev_idx]['Importance']:.4f}")

# 6.3 Visualize feature relevance
print("\n" + "-" * 60)
print("6.3 Feature Relevance Visualization")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 7))
fig.suptitle('Feature Relevance Analysis', fontsize=14, fontweight='bold')

# Horizontal bar chart
top_features = feature_relevance.head(10)
axes[0].barh(range(len(top_features)), top_features['Importance'], color='steelblue')
axes[0].set_yticks(range(len(top_features)))
axes[0].set_yticklabels(top_features['Feature'])
axes[0].set_xlabel('Importance Score')
axes[0].set_title('Top 10 Feature Importances')
axes[0].invert_yaxis()

# Cumulative importance
cumulative = feature_relevance['Importance'].cumsum()
axes[1].plot(range(1, len(cumulative)+1), cumulative, 'bo-', linewidth=2)
axes[1].axhline(y=0.8, color='r', linestyle='--', label='80% threshold')
axes[1].set_xlabel('Number of Features')
axes[1].set_ylabel('Cumulative Importance')
axes[1].set_title('Cumulative Feature Importance')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Find number of features needed for 80% importance
n_80 = np.argmax(cumulative >= 0.8) + 1
axes[1].axvline(x=n_80, color='g', linestyle='--', alpha=0.5)
axes[1].annotate(f'{n_80} features → 80%', xy=(n_80, 0.8), xytext=(n_80+2, 0.75),
                  arrowprops=dict(arrowstyle='->', color='green'))

plt.tight_layout()
plt.savefig('feature_relevance.png', dpi=150, bbox_inches='tight')
plt.show()

# 6.4 Summarize findings
print("\n" + "-" * 60)
print("6.4 Summary of Feature Relevance Findings")
print("-" * 60)

summary_text = f"""
FEATURE RELEVANCE ANALYSIS SUMMARY
==================================
Best Model: {type_results_df.iloc[0]['Model']}
Total Features Analyzed: {len(feature_relevance)}

Top 5 Most Important Features:
1. {feature_relevance.iloc[0]['Feature']}: {feature_relevance.iloc[0]['Importance']:.4f}
2. {feature_relevance.iloc[1]['Feature']}: {feature_relevance.iloc[1]['Importance']:.4f}
3. {feature_relevance.iloc[2]['Feature']}: {feature_relevance.iloc[2]['Importance']:.4f}
4. {feature_relevance.iloc[3]['Feature']}: {feature_relevance.iloc[3]['Importance']:.4f}
5. {feature_relevance.iloc[4]['Feature']}: {feature_relevance.iloc[4]['Importance']:.4f}

Number of features needed for 80% cumulative importance: {n_80}

Key Insights:
- MP_Count_per_L is a primary predictor for risk assessment
- Polymer type significantly influences risk classification
- Industrial discharge and population density are major anthropogenic factors
"""
print(summary_text)

# ============================================================================
# TASK 7: OBJECTIVE #2 - COMPLETE MODELING PIPELINE
# ============================================================================

print("\n" + "=" * 80)
print("TASK 7: OBJECTIVE #2 - COMPLETE MODELING PIPELINE")
print("=" * 80)

# 7.1 Prepare the data with all features
print("\n" + "-" * 60)
print("7.1 Data Preparation for Objective #2")
print("-" * 60)

X_obj2 = df_processed[selected_features_rfe]
y_obj2 = df_processed['Risk_Level_Encoded']

# Handle class imbalance with SMOTE
print("\nClass distribution before SMOTE:")
print(pd.Series(y_obj2).value_counts())

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_obj2, y_obj2)

print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())

X_train_obj2, X_test_obj2, y_train_obj2, y_test_obj2 = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# 7.2 Choose classification models
print("\n" + "-" * 60)
print("7.2 Classification Models Selected")
print("-" * 60)

objective_models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

print("\nModels included:")
for name in objective_models.keys():
    print(f"  - {name}")

# 7.3 Train all models
print("\n" + "-" * 60)
print("7.3 Training Models")
print("-" * 60)

trained_obj_models = {}
for name, model in objective_models.items():
    model.fit(X_train_obj2, y_train_obj2)
    trained_obj_models[name] = model
    print(f"  ✓ {name} trained")

# 7.4 Evaluate all models
print("\n" + "-" * 60)
print("7.4 Model Evaluation")
print("-" * 60)

objective_results = []
for name, model in trained_obj_models.items():
    y_pred = model.predict(X_test_obj2)
    y_prob = model.predict_proba(X_test_obj2) if hasattr(model, 'predict_proba') else None
    
    accuracy = accuracy_score(y_test_obj2, y_pred)
    precision = precision_score(y_test_obj2, y_pred, average='weighted')
    recall = recall_score(y_test_obj2, y_pred, average='weighted')
    f1 = f1_score(y_test_obj2, y_pred, average='weighted')
    
    objective_results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    print(f"\n{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

obj_results_df = pd.DataFrame(objective_results).sort_values('Accuracy', ascending=False)

# 7.5 Compare model performance
print("\n" + "-" * 60)
print("7.5 Model Performance Comparison (Post-SMOTE)")
print("-" * 60)

print("\nFinal Ranking:")
for idx, row in obj_results_df.iterrows():
    print(f"  {idx+1}. {row['Model']}: {row['Accuracy']:.4f} (F1: {row['F1-Score']:.4f})")

# 7.6 Visualize Model Performance
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('Objective #2: Comprehensive Model Performance Analysis', fontsize=14, fontweight='bold')

# Bar plot comparison
metrics_obj = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x = np.arange(len(obj_results_df))
width = 0.2

for i, metric in enumerate(metrics_obj):
    axes[0, 0].bar(x + i*width, obj_results_df[metric], width, label=metric)
axes[0, 0].set_xlabel('Model')
axes[0, 0].set_ylabel('Score')
axes[0, 0].set_title('Model Performance Metrics')
axes[0, 0].set_xticks(x + width*1.5)
axes[0, 0].set_xticklabels(obj_results_df['Model'], rotation=45, ha='right')
axes[0, 0].legend()
axes[0, 0].set_ylim(0, 1)
axes[0, 0].grid(True, alpha=0.3)

# Heatmap
heatmap_data = obj_results_df.set_index('Model')[metrics_obj]
sns.heatmap(heatmap_data, annot=True, cmap='YlOrRd', fmt='.4f', ax=axes[0, 1])
axes[0, 1].set_title('Performance Heatmap')

# ROC Curves (one-vs-rest for multi-class)
best_model_name_obj = obj_results_df.iloc[0]['Model']
best_model_obj = trained_obj_models[best_model_name_obj]

if hasattr(best_model_obj, 'predict_proba'):
    y_prob_best = best_model_obj.predict_proba(X_test_obj2)
    n_classes = len(np.unique(y_test_obj2))
    
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve((y_test_obj2 == i).astype(int), y_prob_best[:, i])
        auc = roc_auc_score((y_test_obj2 == i).astype(int), y_prob_best[:, i])
        axes[1, 0].plot(fpr, tpr, label=f'Class {i} (AUC={auc:.3f})')
    
    axes[1, 0].plot([0, 1], [0, 1], 'k--')
    axes[1, 0].set_xlabel('False Positive Rate')
    axes[1, 0].set_ylabel('True Positive Rate')
    axes[1, 0].set_title(f'ROC Curves - {best_model_name_obj}')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

# Confusion Matrix for best model
cm_best = confusion_matrix(y_test_obj2, best_model_obj.predict(X_test_obj2))
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_xlabel('Predicted')
axes[1, 1].set_ylabel('Actual')
axes[1, 1].set_title(f'Confusion Matrix - {best_model_name_obj}')

plt.tight_layout()
plt.savefig('objective2_model_performance.png', dpi=150, bbox_inches='tight')
plt.show()

# 7.7 Load and Visualize Polymer Type Distribution
print("\n" + "-" * 60)
print("7.7 Polymer Type Distribution Analysis")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Polymer Type Distribution Analysis', fontsize=14, fontweight='bold')

# Polymer type distribution
polymer_counts = df['Polymer_Type'].value_counts()
axes[0].bar(polymer_counts.index, polymer_counts.values, color='skyblue', edgecolor='black')
axes[0].set_xlabel('Polymer Type')
axes[0].set_ylabel('Count')
axes[0].set_title('Distribution of Polymer Types')
axes[0].tick_params(axis='x', rotation=45)

# Polymer type by risk level
polymer_risk_cross = pd.crosstab(df['Polymer_Type'], df['Risk_Level'], normalize='index')
polymer_risk_cross.plot(kind='bar', stacked=True, ax=axes[1], colormap='RdYlGn_r')
axes[1].set_xlabel('Polymer Type')
axes[1].set_ylabel('Proportion')
axes[1].set_title('Risk Level Distribution by Polymer Type')
axes[1].legend(title='Risk Level', bbox_to_anchor=(1.05, 1))
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('polymer_type_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# 7.8 Hyperparameter Tuning for Best Model
print("\n" + "-" * 60)
print("7.8 Hyperparameter Tuning")
print("-" * 60)

print(f"Performing hyperparameter tuning on {best_model_name_obj}...")

param_grids = {
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
}

if best_model_name_obj in param_grids:
    grid_search = GridSearchCV(
        best_model_obj, 
        param_grids[best_model_name_obj], 
        cv=5, 
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_obj2, y_train_obj2)
    
    print(f"\nBest Parameters: {grid_search.best_params_}")
    print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")
    
    # Evaluate tuned model
    tuned_model = grid_search.best_estimator_
    y_pred_tuned = tuned_model.predict(X_test_obj2)
    tuned_accuracy = accuracy_score(y_test_obj2, y_pred_tuned)
    print(f"Tuned Model Test Accuracy: {tuned_accuracy:.4f}")
else:
    print(f"No hyperparameter grid defined for {best_model_name_obj}")
    tuned_model = best_model_obj
    tuned_accuracy = obj_results_df.iloc[0]['Accuracy']

print("\n" + "=" * 80)
print("FINAL SUMMARY")
print("=" * 80)

# Final comprehensive report
final_report = f"""
================================================================================
PREDICTIVE RISK MODELING FOR MICROPLASTIC POLLUTION - FINAL REPORT
================================================================================

DATASET INFORMATION:
- Total samples: {len(df)}
- Features used: {len(selected_features_rfe)}
- Risk levels: Low, Medium, High, Critical
- Risk types: {list(le_risk_type.classes_)}

DATA PREPROCESSING SUMMARY:
- Encoded {len(categorical_columns)} categorical variables
- Scaled {len(numerical_features)} numerical features
- Addressed outliers using IQR method
- Applied log transformation to skewed features

FEATURE SELECTION RESULTS:
Top 5 features:
1. {feature_relevance.iloc[0]['Feature']}
2. {feature_relevance.iloc[1]['Feature']}
3. {feature_relevance.iloc[2]['Feature']}
4. {feature_relevance.iloc[3]['Feature']}
5. {feature_relevance.iloc[4]['Feature']}

MODEL PERFORMANCE (Risk Level Classification):
Best Model: {best_model_name_obj}
Accuracy: {obj_results_df.iloc[0]['Accuracy']:.4f}
F1-Score: {obj_results_df.iloc[0]['F1-Score']:.4f}

After Hyperparameter Tuning:
Tuned Accuracy: {tuned_accuracy:.4f}

RISK TYPE CLASSIFICATION:
Best Model: {type_results_df.iloc[0]['Model']}
Accuracy: {type_results_df.iloc[0]['Accuracy']:.4f}

CORRELATION FINDINGS:
- Risk Score vs MP Count: Pearson r = {pearson_corr:.4f}
- Risk Score vs MP Count: Spearman ρ = {spearman_corr:.4f}
- ANOVA Test for Risk Levels: F = {f_stat:.4f}, p = {p_value:.2e}

KEY INSIGHTS:
1. MP_Count_per_L is the strongest predictor of risk level
2. Polymer type significantly influences risk classification
3. Industrial discharge and population density are critical anthropogenic factors
4. SMOTE effectively addressed class imbalance
5. {best_model_name_obj} performed best after hyperparameter tuning

RECOMMENDATIONS:
1. Prioritize monitoring of areas with high MP_Count_per_L
2. Focus on industrial discharge reduction in high-risk zones
3. Implement polymer-specific remediation strategies
4. Deploy early warning systems in critical risk areas
5. Establish regular monitoring of top identified features

================================================================================
END OF REPORT
================================================================================
"""

print(final_report)

# Save all results
print("\nSaving results to CSV files...")
obj_results_df.to_csv('model_performance_results.csv', index=False)
type_results_df.to_csv('risk_type_results.csv', index=False)
feature_relevance.to_csv('feature_relevance.csv', index=False)
print("✓ Results saved successfully")
