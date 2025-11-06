import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (cross_val_score, cross_validate, 
                                      StratifiedKFold, train_test_split)
from sklearn.metrics import (classification_report, confusion_matrix, 
                             accuracy_score, make_scorer, f1_score)
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: DATA PREPARATION (Same as before)
# ============================================================================

feature_cols = ['az', 'el', 'range_m', 'aspect_deg', 'length_m', 
                'RCSinst_dB', 'SRNinst_dB']

X = mcs[feature_cols].copy()
y = mcs['object']  # Change to 'size' or combined as needed

# Handle missing values
mask = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask]
y = y[mask]

print("="*70)
print("DATASET INFORMATION")
print("="*70)
print(f"Total samples: {len(X)}")
print(f"Number of features: {X.shape[1]}")
print(f"Number of classes: {y.nunique()}")
print(f"\nClass distribution:")
print(y.value_counts().sort_index())

# ============================================================================
# STEP 2: CREATE A PIPELINE (Best Practice!)
# ============================================================================
# Pipeline ensures scaling is done INSIDE each fold (prevents data leakage)

pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Step 1: Scale features
    ('lda', LinearDiscriminantAnalysis(n_components=2))  # Step 2: LDA
])

print("\n" + "="*70)
print("PIPELINE CREATED: StandardScaler → LDA")
print("="*70)

# ============================================================================
# STEP 3: CROSS-VALIDATION WITH SINGLE METRIC
# ============================================================================

# Define cross-validation strategy
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n" + "="*70)
print("METHOD 1: SIMPLE CROSS-VALIDATION (Accuracy only)")
print("="*70)

# Perform cross-validation
cv_scores = cross_val_score(pipeline, X, y, cv=cv_strategy, 
                            scoring='accuracy', n_jobs=-1)

print(f"Cross-Validation Scores (5 folds): {cv_scores}")
print(f"\nMean Accuracy: {cv_scores.mean():.4f}")
print(f"Std Deviation: {cv_scores.std():.4f}")
print(f"95% Confidence Interval: [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, "
      f"{cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

# ============================================================================
# STEP 4: CROSS-VALIDATION WITH MULTIPLE METRICS
# ============================================================================

print("\n" + "="*70)
print("METHOD 2: COMPREHENSIVE CROSS-VALIDATION (Multiple Metrics)")
print("="*70)

# Define multiple scoring metrics
scoring = {
    'accuracy': 'accuracy',
    'precision_macro': 'precision_macro',
    'recall_macro': 'recall_macro',
    'f1_macro': 'f1_macro'
}

# Perform comprehensive cross-validation
cv_results = cross_validate(pipeline, X, y, cv=cv_strategy, 
                            scoring=scoring, n_jobs=-1, 
                            return_train_score=True)

# Create results dataframe
results_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
    'Train Mean': [
        cv_results['train_accuracy'].mean(),
        cv_results['train_precision_macro'].mean(),
        cv_results['train_recall_macro'].mean(),
        cv_results['train_f1_macro'].mean()
    ],
    'Train Std': [
        cv_results['train_accuracy'].std(),
        cv_results['train_precision_macro'].std(),
        cv_results['train_recall_macro'].std(),
        cv_results['train_f1_macro'].std()
    ],
    'Test Mean': [
        cv_results['test_accuracy'].mean(),
        cv_results['test_precision_macro'].mean(),
        cv_results['test_recall_macro'].mean(),
        cv_results['test_f1_macro'].mean()
    ],
    'Test Std': [
        cv_results['test_accuracy'].std(),
        cv_results['test_precision_macro'].std(),
        cv_results['test_recall_macro'].std(),
        cv_results['test_f1_macro'].std()
    ]
})

print("\nCross-Validation Results Summary:")
print(results_df.to_string(index=False))

# ============================================================================
# STEP 5: VISUALIZE CROSS-VALIDATION RESULTS
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Accuracy across folds
ax = axes[0, 0]
folds = np.arange(1, 6)
ax.plot(folds, cv_results['train_accuracy'], 'o-', label='Train', linewidth=2)
ax.plot(folds, cv_results['test_accuracy'], 's-', label='Test', linewidth=2)
ax.fill_between(folds, cv_results['test_accuracy'], alpha=0.2)
ax.set_xlabel('Fold Number')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy Across 5 Folds')
ax.legend()
ax.grid(alpha=0.3)
ax.set_xticks(folds)

# Plot 2: All metrics comparison
ax = axes[0, 1]
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
test_means = results_df['Test Mean'].values
test_stds = results_df['Test Std'].values
x_pos = np.arange(len(metrics))
bars = ax.bar(x_pos, test_means, yerr=test_stds, capsize=5, 
              alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
ax.set_ylabel('Score')
ax.set_title('Average Performance Metrics (Test Set)')
ax.set_xticks(x_pos)
ax.set_xticklabels(metrics, rotation=45)
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, test_means, test_stds)):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=9)

# Plot 3: Train vs Test comparison
ax = axes[1, 0]
x = np.arange(len(metrics))
width = 0.35
bars1 = ax.bar(x - width/2, results_df['Train Mean'], width, 
               label='Train', alpha=0.8)
bars2 = ax.bar(x + width/2, results_df['Test Mean'], width, 
               label='Test', alpha=0.8)
ax.set_ylabel('Score')
ax.set_title('Train vs Test Performance')
ax.set_xticks(x)
ax.set_xticklabels(metrics, rotation=45)
ax.legend()
ax.set_ylim([0, 1])
ax.grid(axis='y', alpha=0.3)

# Plot 4: Variance analysis
ax = axes[1, 1]
ax.bar(metrics, results_df['Test Std'], alpha=0.7, color='coral')
ax.set_ylabel('Standard Deviation')
ax.set_title('Stability of Metrics Across Folds')
ax.set_xticklabels(metrics, rotation=45)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 6: COMPARE DIFFERENT K VALUES
# ============================================================================

print("\n" + "="*70)
print("COMPARING DIFFERENT K VALUES (Number of Folds)")
print("="*70)

k_values = [3, 5, 7, 10]
k_comparison = []

for k in k_values:
    cv_k = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv_k, scoring='accuracy')
    
    k_comparison.append({
        'K': k,
        'Mean Accuracy': scores.mean(),
        'Std Dev': scores.std(),
        'Min Score': scores.min(),
        'Max Score': scores.max()
    })
    
    print(f"K={k}: Mean={scores.mean():.4f}, Std={scores.std():.4f}, "
          f"Range=[{scores.min():.4f}, {scores.max():.4f}]")

k_comparison_df = pd.DataFrame(k_comparison)

# Visualize K comparison
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.errorbar(k_comparison_df['K'], k_comparison_df['Mean Accuracy'], 
             yerr=k_comparison_df['Std Dev'], marker='o', capsize=5, 
             linewidth=2, markersize=8)
plt.xlabel('Number of Folds (K)')
plt.ylabel('Mean Accuracy')
plt.title('Effect of K on Cross-Validation Results')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.bar(k_comparison_df['K'], k_comparison_df['Std Dev'], alpha=0.7, color='orange')
plt.xlabel('Number of Folds (K)')
plt.ylabel('Standard Deviation')
plt.title('Stability vs Number of Folds')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# STEP 7: COMPARE WITH SIMPLE TRAIN-TEST SPLIT
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: Cross-Validation vs Simple Train-Test Split")
print("="*70)

# Simple train-test split (for comparison)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

pipeline_simple = Pipeline([
    ('scaler', StandardScaler()),
    ('lda', LinearDiscriminantAnalysis(n_components=2))
])

pipeline_simple.fit(X_train, y_train)
simple_accuracy = pipeline_simple.score(X_test, y_test)

print(f"\nSimple Train-Test Split (30% test):")
print(f"  Test Accuracy: {simple_accuracy:.4f}")
print(f"  Uses only ONE test set")

print(f"\n5-Fold Cross-Validation:")
print(f"  Mean Test Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"  Uses FIVE different test sets")
print(f"  More reliable estimate!")

# ============================================================================
# STEP 8: TEST DIFFERENT NUMBER OF LDA COMPONENTS WITH CV
# ============================================================================

print("\n" + "="*70)
print("OPTIMIZING NUMBER OF LDA COMPONENTS WITH CROSS-VALIDATION")
print("="*70)

n_classes = y.nunique()
n_features = X.shape[1]
max_components = min(n_features, n_classes - 1)

component_results = []

for n_comp in range(1, max_components + 1):
    pipeline_comp = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis(n_components=n_comp))
    ])
    
    scores = cross_val_score(pipeline_comp, X, y, cv=cv_strategy, 
                            scoring='accuracy')
    
    component_results.append({
        'n_components': n_comp,
        'mean_accuracy': scores.mean(),
        'std_accuracy': scores.std()
    })
    
    print(f"Components={n_comp}: CV Accuracy={scores.mean():.4f} ± {scores.std():.4f}")

component_results_df = pd.DataFrame(component_results)

# Find optimal number of components
best_idx = component_results_df['mean_accuracy'].idxmax()
best_n_comp = component_results_df.loc[best_idx, 'n_components']
best_accuracy = component_results_df.loc[best_idx, 'mean_accuracy']

print(f"\n✓ OPTIMAL: {best_n_comp} components with CV accuracy={best_accuracy:.4f}")

# Visualize component optimization
plt.figure(figsize=(10, 6))
plt.errorbar(component_results_df['n_components'], 
             component_results_df['mean_accuracy'],
             yerr=component_results_df['std_accuracy'],
             marker='o', capsize=5, linewidth=2, markersize=8)
plt.axvline(x=best_n_comp, color='red', linestyle='--', 
            label=f'Optimal: {best_n_comp} components')
plt.xlabel('Number of LDA Components')
plt.ylabel('Cross-Validated Accuracy')
plt.title('LDA Components Optimization with Cross-Validation')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY: WHY CROSS-VALIDATION IS BETTER")
print("="*70)
print("""
✓ Tests model on multiple different data splits
✓ Provides mean and standard deviation (confidence in results)
✓ Reduces bias from lucky/unlucky splits
✓ More reliable estimate of real-world performance
✓ Helps detect overfitting (train vs test scores)
✓ Essential for model selection and hyperparameter tuning
""")