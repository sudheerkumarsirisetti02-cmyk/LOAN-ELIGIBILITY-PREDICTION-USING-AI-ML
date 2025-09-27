import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
from sklearn.model_selection import learning_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

print("Creating comprehensive visualizations for loan eligibility prediction results...")

# Create results directory
import os
os.makedirs('/home/ubuntu/results_visualizations', exist_ok=True)

# Load data and model
final_model = joblib.load('/home/ubuntu/final_loan_model.pkl')
X_train = pd.read_csv('/home/ubuntu/X_train_processed.csv')
y_train = pd.read_csv('/home/ubuntu/y_train.csv').values.ravel()
model_comparison = pd.read_csv('/home/ubuntu/model_comparison_results.csv', index_col=0)
test_predictions = pd.read_csv('/home/ubuntu/test_predictions.csv')

# Split data for validation (same as in evaluation)
from sklearn.model_selection import train_test_split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Get predictions
y_val_pred = final_model.predict(X_val_split)
y_val_pred_proba = final_model.predict_proba(X_val_split)[:, 1]

# 1. Model Comparison Visualization
print("Creating model comparison visualization...")
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Accuracy comparison
axes[0, 0].bar(model_comparison.index, model_comparison['Accuracy'], color='skyblue', alpha=0.8)
axes[0, 0].set_title('Model Accuracy Comparison', fontweight='bold')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# F1-Score comparison
axes[0, 1].bar(model_comparison.index, model_comparison['F1_Score'], color='lightcoral', alpha=0.8)
axes[0, 1].set_title('Model F1-Score Comparison', fontweight='bold')
axes[0, 1].set_ylabel('F1-Score')
axes[0, 1].tick_params(axis='x', rotation=45)
axes[0, 1].grid(axis='y', alpha=0.3)

# AUC comparison
axes[1, 0].bar(model_comparison.index, model_comparison['AUC'], color='lightgreen', alpha=0.8)
axes[1, 0].set_title('Model AUC Comparison', fontweight='bold')
axes[1, 0].set_ylabel('AUC')
axes[1, 0].tick_params(axis='x', rotation=45)
axes[1, 0].grid(axis='y', alpha=0.3)

# Precision vs Recall scatter plot
axes[1, 1].scatter(model_comparison['Recall'], model_comparison['Precision'], 
                   s=100, alpha=0.7, c=model_comparison['F1_Score'], cmap='viridis')
axes[1, 1].set_xlabel('Recall')
axes[1, 1].set_ylabel('Precision')
axes[1, 1].set_title('Precision vs Recall (colored by F1-Score)', fontweight='bold')
axes[1, 1].grid(alpha=0.3)

# Add model names as annotations
for i, model in enumerate(model_comparison.index):
    axes[1, 1].annotate(model, (model_comparison.iloc[i]['Recall'], model_comparison.iloc[i]['Precision']),
                        xytext=(5, 5), textcoords='offset points', fontsize=8)

plt.tight_layout()
plt.savefig('/home/ubuntu/results_visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Confusion Matrix Heatmap
print("Creating confusion matrix visualization...")
cm = confusion_matrix(y_val_split, y_val_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Rejected', 'Approved'], 
            yticklabels=['Rejected', 'Approved'])
plt.title('Confusion Matrix - Random Forest Model', fontweight='bold', fontsize=16)
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)

# Add percentage annotations
total = cm.sum()
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        percentage = cm[i, j] / total * 100
        plt.text(j + 0.5, i + 0.7, f'({percentage:.1f}%)', 
                ha='center', va='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('/home/ubuntu/results_visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. ROC Curve
print("Creating ROC curve visualization...")
fpr, tpr, _ = roc_curve(y_val_split, y_val_pred_proba)
auc_score = roc_auc_score(y_val_split, y_val_pred_proba)

plt.figure(figsize=(8, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curve', fontweight='bold', fontsize=14)
plt.legend(loc="lower right")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/results_visualizations/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Precision-Recall Curve
print("Creating precision-recall curve visualization...")
precision, recall, _ = precision_recall_curve(y_val_split, y_val_pred_proba)
from sklearn.metrics import average_precision_score
avg_precision = average_precision_score(y_val_split, y_val_pred_proba)

plt.figure(figsize=(8, 8))
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {avg_precision:.3f})')
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curve', fontweight='bold', fontsize=14)
plt.legend(loc="lower left")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/results_visualizations/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Feature Importance Visualization
print("Creating feature importance visualization...")
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'], feature_importance['importance'], color='steelblue', alpha=0.8)
    plt.xlabel('Feature Importance', fontsize=12)
    plt.title('Feature Importance - Random Forest Model', fontweight='bold', fontsize=14)
    plt.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('/home/ubuntu/results_visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()

# 6. Learning Curves
print("Creating learning curves visualization...")
train_sizes, train_scores, val_scores = learning_curve(
    final_model, X_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
plt.xlabel('Training Set Size', fontsize=12)
plt.ylabel('F1-Score', fontsize=12)
plt.title('Learning Curves - Random Forest Model', fontweight='bold', fontsize=14)
plt.legend(loc='best')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/home/ubuntu/results_visualizations/learning_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Prediction Probability Distribution
print("Creating prediction probability distribution...")
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_val_pred_proba[y_val_split == 0], bins=20, alpha=0.7, label='Rejected (Actual)', color='red')
plt.hist(y_val_pred_proba[y_val_split == 1], bins=20, alpha=0.7, label='Approved (Actual)', color='green')
plt.xlabel('Prediction Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Validation Set - Probability Distribution', fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
test_proba = test_predictions['Prediction_Probability']
plt.hist(test_proba, bins=20, alpha=0.7, color='blue', edgecolor='black')
plt.xlabel('Prediction Probability', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Test Set - Probability Distribution', fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/results_visualizations/probability_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Business Impact Visualization
print("Creating business impact visualization...")
# Calculate costs for different thresholds
thresholds = np.linspace(0.1, 0.9, 17)
costs = []
approvals = []
cost_fp = 1000  # Cost of false positive
cost_fn = 100   # Cost of false negative

for threshold in thresholds:
    y_pred_thresh = (y_val_pred_proba >= threshold).astype(int)
    cm_thresh = confusion_matrix(y_val_split, y_pred_thresh)
    tn, fp, fn, tp = cm_thresh.ravel()
    total_cost = fp * cost_fp + fn * cost_fn
    approval_rate = (tp + fp) / len(y_val_split)
    costs.append(total_cost)
    approvals.append(approval_rate)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(thresholds, costs, 'o-', color='red', linewidth=2, markersize=6)
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Total Cost ($)', fontsize=12)
plt.title('Business Cost vs Decision Threshold', fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(thresholds, approvals, 'o-', color='green', linewidth=2, markersize=6)
plt.xlabel('Decision Threshold', fontsize=12)
plt.ylabel('Approval Rate', fontsize=12)
plt.title('Approval Rate vs Decision Threshold', fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/results_visualizations/business_impact.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. Model Performance Metrics Summary
print("Creating performance metrics summary...")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

metrics_data = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'],
    'Score': [
        accuracy_score(y_val_split, y_val_pred),
        precision_score(y_val_split, y_val_pred),
        recall_score(y_val_split, y_val_pred),
        f1_score(y_val_split, y_val_pred),
        roc_auc_score(y_val_split, y_val_pred_proba)
    ]
}

plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
bars = plt.bar(metrics_data['Metric'], metrics_data['Score'], color=colors, alpha=0.8)
plt.ylim(0, 1)
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Metrics Summary', fontweight='bold', fontsize=14)
plt.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, score in zip(bars, metrics_data['Score']):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
             f'{score:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('/home/ubuntu/results_visualizations/performance_summary.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. Test Set Predictions Analysis
print("Creating test set analysis visualization...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Approval distribution
test_approval_counts = test_predictions['Loan_Status'].value_counts()
axes[0, 0].pie(test_approval_counts.values, labels=['Approved', 'Rejected'], autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'], startangle=90)
axes[0, 0].set_title('Test Set - Loan Approval Distribution', fontweight='bold')

# Confidence levels
test_proba = test_predictions['Prediction_Probability']
high_conf = ((test_proba > 0.8) | (test_proba < 0.2)).sum()
medium_conf = (((test_proba >= 0.6) & (test_proba <= 0.8)) | ((test_proba >= 0.2) & (test_proba <= 0.4))).sum()
low_conf = ((test_proba >= 0.4) & (test_proba <= 0.6)).sum()

confidence_data = [high_conf, medium_conf, low_conf]
confidence_labels = ['High\n(>0.8 or <0.2)', 'Medium\n(0.2-0.4 or 0.6-0.8)', 'Low\n(0.4-0.6)']
axes[0, 1].pie(confidence_data, labels=confidence_labels, autopct='%1.1f%%', 
               colors=['darkgreen', 'orange', 'red'], startangle=90)
axes[0, 1].set_title('Test Set - Prediction Confidence Levels', fontweight='bold')

# Probability histogram
axes[1, 0].hist(test_proba, bins=15, alpha=0.7, color='skyblue', edgecolor='black')
axes[1, 0].set_xlabel('Prediction Probability')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('Test Set - Probability Distribution', fontweight='bold')
axes[1, 0].grid(alpha=0.3)

# Approval by probability ranges
prob_ranges = ['0.0-0.2', '0.2-0.4', '0.4-0.6', '0.6-0.8', '0.8-1.0']
range_approvals = []
for i, (low, high) in enumerate([(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]):
    mask = (test_proba >= low) & (test_proba < high) if i < 4 else (test_proba >= low) & (test_proba <= high)
    approval_rate = (test_predictions.loc[mask, 'Loan_Status'] == 'Y').mean() if mask.sum() > 0 else 0
    range_approvals.append(approval_rate)

axes[1, 1].bar(prob_ranges, range_approvals, color='lightblue', alpha=0.8)
axes[1, 1].set_xlabel('Probability Range')
axes[1, 1].set_ylabel('Approval Rate')
axes[1, 1].set_title('Approval Rate by Probability Range', fontweight='bold')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('/home/ubuntu/results_visualizations/test_set_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations created successfully!")
print("Visualizations saved in '/home/ubuntu/results_visualizations/' directory:")
print("1. model_comparison.png - Comparison of all models")
print("2. confusion_matrix.png - Confusion matrix heatmap")
print("3. roc_curve.png - ROC curve analysis")
print("4. precision_recall_curve.png - Precision-recall curve")
print("5. feature_importance.png - Feature importance ranking")
print("6. learning_curves.png - Learning curve analysis")
print("7. probability_distributions.png - Prediction probability distributions")
print("8. business_impact.png - Business cost analysis")
print("9. performance_summary.png - Performance metrics summary")
print("10. test_set_analysis.png - Test set predictions analysis")

