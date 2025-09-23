import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, roc_curve,
                           precision_recall_curve, average_precision_score)
from sklearn.model_selection import learning_curve, validation_curve
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("LOAN ELIGIBILITY PREDICTION - MODEL TESTING & EVALUATION")
print("="*60)

# Load the trained model and data
print("Loading trained model and data...")
final_model = joblib.load('/home/ubuntu/final_loan_model.pkl')
X_train = pd.read_csv('/home/ubuntu/X_train_processed.csv')
y_train = pd.read_csv('/home/ubuntu/y_train.csv').values.ravel()
X_test = pd.read_csv('/home/ubuntu/X_test_processed.csv')

# Load test predictions
test_predictions_df = pd.read_csv('/home/ubuntu/test_predictions.csv')
model_comparison = pd.read_csv('/home/ubuntu/model_comparison_results.csv', index_col=0)

print(f"Model type: {type(final_model).__name__}")
print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Create evaluation directory
import os
os.makedirs('/home/ubuntu/evaluation_results', exist_ok=True)

# Split training data for evaluation (same split as in model development)
from sklearn.model_selection import train_test_split
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Get predictions for evaluation
y_val_pred = final_model.predict(X_val_split)
y_val_pred_proba = final_model.predict_proba(X_val_split)[:, 1]
y_train_pred = final_model.predict(X_train_split)
y_train_pred_proba = final_model.predict_proba(X_train_split)[:, 1]

print("\n" + "="*60)
print("COMPREHENSIVE MODEL EVALUATION")
print("="*60)

# 1. Basic Performance Metrics
print("1. BASIC PERFORMANCE METRICS")
print("-" * 40)

# Training metrics
train_accuracy = accuracy_score(y_train_split, y_train_pred)
train_precision = precision_score(y_train_split, y_train_pred)
train_recall = recall_score(y_train_split, y_train_pred)
train_f1 = f1_score(y_train_split, y_train_pred)
train_auc = roc_auc_score(y_train_split, y_train_pred_proba)

# Validation metrics
val_accuracy = accuracy_score(y_val_split, y_val_pred)
val_precision = precision_score(y_val_split, y_val_pred)
val_recall = recall_score(y_val_split, y_val_pred)
val_f1 = f1_score(y_val_split, y_val_pred)
val_auc = roc_auc_score(y_val_split, y_val_pred_proba)

# Create performance comparison table
performance_metrics = pd.DataFrame({
    'Training': [train_accuracy, train_precision, train_recall, train_f1, train_auc],
    'Validation': [val_accuracy, val_precision, val_recall, val_f1, val_auc]
}, index=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC'])

print("Performance Metrics Comparison:")
print(performance_metrics.round(4))

# Check for overfitting
print(f"\nOverfitting Analysis:")
print(f"Training vs Validation Accuracy Difference: {abs(train_accuracy - val_accuracy):.4f}")
print(f"Training vs Validation F1-Score Difference: {abs(train_f1 - val_f1):.4f}")

if abs(train_accuracy - val_accuracy) > 0.1:
    print("⚠️  Potential overfitting detected (accuracy difference > 0.1)")
elif abs(train_accuracy - val_accuracy) > 0.05:
    print("⚠️  Mild overfitting detected (accuracy difference > 0.05)")
else:
    print("✅ No significant overfitting detected")

# 2. Confusion Matrix Analysis
print(f"\n2. CONFUSION MATRIX ANALYSIS")
print("-" * 40)

cm = confusion_matrix(y_val_split, y_val_pred)
print("Confusion Matrix:")
print(cm)

# Calculate additional metrics from confusion matrix
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)  # Same as recall
npv = tn / (tn + fn)  # Negative Predictive Value
ppv = tp / (tp + fp)  # Positive Predictive Value (same as precision)

print(f"\nDetailed Metrics from Confusion Matrix:")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"True Positives (TP): {tp}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")
print(f"Positive Predictive Value (Precision): {ppv:.4f}")
print(f"Negative Predictive Value: {npv:.4f}")

# 3. ROC Curve and AUC Analysis
print(f"\n3. ROC CURVE AND AUC ANALYSIS")
print("-" * 40)

fpr, tpr, thresholds = roc_curve(y_val_split, y_val_pred_proba)
auc_score = roc_auc_score(y_val_split, y_val_pred_proba)

print(f"AUC Score: {auc_score:.4f}")

# Interpretation of AUC
if auc_score >= 0.9:
    auc_interpretation = "Excellent"
elif auc_score >= 0.8:
    auc_interpretation = "Good"
elif auc_score >= 0.7:
    auc_interpretation = "Fair"
elif auc_score >= 0.6:
    auc_interpretation = "Poor"
else:
    auc_interpretation = "Fail"

print(f"AUC Interpretation: {auc_interpretation}")

# 4. Precision-Recall Analysis
print(f"\n4. PRECISION-RECALL ANALYSIS")
print("-" * 40)

precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_val_split, y_val_pred_proba)
avg_precision = average_precision_score(y_val_split, y_val_pred_proba)

print(f"Average Precision Score: {avg_precision:.4f}")

# Find optimal threshold based on F1-score
f1_scores = []
for threshold in pr_thresholds:
    y_pred_thresh = (y_val_pred_proba >= threshold).astype(int)
    f1_thresh = f1_score(y_val_split, y_pred_thresh)
    f1_scores.append(f1_thresh)

optimal_threshold_idx = np.argmax(f1_scores)
optimal_threshold = pr_thresholds[optimal_threshold_idx]
optimal_f1 = f1_scores[optimal_threshold_idx]

print(f"Optimal Threshold (based on F1): {optimal_threshold:.4f}")
print(f"F1-Score at Optimal Threshold: {optimal_f1:.4f}")

# 5. Feature Importance Analysis
print(f"\n5. FEATURE IMPORTANCE ANALYSIS")
print("-" * 40)

if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("Top 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    
    # Calculate cumulative importance
    feature_importance['cumulative_importance'] = feature_importance['importance'].cumsum()
    features_for_90_percent = len(feature_importance[feature_importance['cumulative_importance'] <= 0.9])
    
    print(f"\nNumber of features needed for 90% importance: {features_for_90_percent}")
    
elif hasattr(final_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': final_model.coef_[0]
    })
    feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    
    print("Top 10 Features by Coefficient Magnitude:")
    print(feature_importance.head(10)[['feature', 'coefficient', 'abs_coefficient']].to_string(index=False))

# 6. Learning Curve Analysis
print(f"\n6. LEARNING CURVE ANALYSIS")
print("-" * 40)

# Generate learning curves
train_sizes, train_scores, val_scores = learning_curve(
    final_model, X_train, y_train, cv=5, n_jobs=-1, 
    train_sizes=np.linspace(0.1, 1.0, 10), scoring='f1'
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

print("Learning Curve Analysis:")
print(f"Final training score: {train_mean[-1]:.4f} (+/- {train_std[-1]:.4f})")
print(f"Final validation score: {val_mean[-1]:.4f} (+/- {val_std[-1]:.4f})")

# Check if more data would help
if val_mean[-1] < val_mean[-2]:
    print("⚠️  Validation score is decreasing - model might benefit from regularization")
elif train_mean[-1] - val_mean[-1] > 0.1:
    print("⚠️  Large gap between training and validation - potential overfitting")
else:
    print("✅ Learning curves look healthy")

# 7. Model Robustness Testing
print(f"\n7. MODEL ROBUSTNESS TESTING")
print("-" * 40)

# Test with different random states
robustness_scores = []
for random_state in [42, 123, 456, 789, 999]:
    X_temp_train, X_temp_val, y_temp_train, y_temp_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
    )
    
    temp_model = type(final_model)(**final_model.get_params())
    temp_model.fit(X_temp_train, y_temp_train)
    temp_pred = temp_model.predict(X_temp_val)
    temp_f1 = f1_score(y_temp_val, temp_pred)
    robustness_scores.append(temp_f1)

robustness_mean = np.mean(robustness_scores)
robustness_std = np.std(robustness_scores)

print(f"Robustness Test Results:")
print(f"F1-Score across different splits: {robustness_mean:.4f} (+/- {robustness_std:.4f})")
print(f"Individual scores: {[f'{score:.4f}' for score in robustness_scores]}")

if robustness_std < 0.05:
    print("✅ Model shows good robustness (low variance across splits)")
elif robustness_std < 0.1:
    print("⚠️  Model shows moderate robustness")
else:
    print("⚠️  Model shows poor robustness (high variance across splits)")

# 8. Business Impact Analysis
print(f"\n8. BUSINESS IMPACT ANALYSIS")
print("-" * 40)

# Assuming business costs
cost_false_positive = 1000  # Cost of approving a bad loan
cost_false_negative = 100   # Cost of rejecting a good loan (opportunity cost)

total_cost = (fp * cost_false_positive) + (fn * cost_false_negative)
total_applications = len(y_val_split)

print(f"Business Impact Analysis (Validation Set):")
print(f"False Positives (Bad loans approved): {fp}")
print(f"False Negatives (Good loans rejected): {fn}")
print(f"Estimated cost of False Positives: ${fp * cost_false_positive:,}")
print(f"Estimated cost of False Negatives: ${fn * cost_false_negative:,}")
print(f"Total estimated cost: ${total_cost:,}")
print(f"Average cost per application: ${total_cost / total_applications:.2f}")

# Compare with baseline (approve all)
baseline_fp = sum(y_val_split == 0)  # All rejected loans would be false positives
baseline_cost = baseline_fp * cost_false_positive
cost_savings = baseline_cost - total_cost
cost_savings_percentage = (cost_savings / baseline_cost) * 100

print(f"\nComparison with 'Approve All' baseline:")
print(f"Baseline cost (approve all): ${baseline_cost:,}")
print(f"Model cost: ${total_cost:,}")
print(f"Cost savings: ${cost_savings:,} ({cost_savings_percentage:.1f}%)")

# 9. Model Comparison Summary
print(f"\n9. MODEL COMPARISON SUMMARY")
print("-" * 40)

print("All Models Performance Comparison:")
print(model_comparison.round(4))

best_models = {
    'Highest Accuracy': model_comparison['Accuracy'].idxmax(),
    'Highest Precision': model_comparison['Precision'].idxmax(),
    'Highest Recall': model_comparison['Recall'].idxmax(),
    'Highest F1-Score': model_comparison['F1_Score'].idxmax(),
    'Highest AUC': model_comparison['AUC'].idxmax()
}

print(f"\nBest performing models by metric:")
for metric, model in best_models.items():
    print(f"{metric}: {model}")

# 10. Test Set Analysis
print(f"\n10. TEST SET PREDICTIONS ANALYSIS")
print("-" * 40)

print("Test Set Prediction Summary:")
print(test_predictions_df['Loan_Status'].value_counts())

approval_rate = (test_predictions_df['Loan_Status'] == 'Y').mean()
print(f"Test set approval rate: {approval_rate:.1%}")

# Confidence analysis
high_confidence = (test_predictions_df['Prediction_Probability'] > 0.8) | (test_predictions_df['Prediction_Probability'] < 0.2)
medium_confidence = (test_predictions_df['Prediction_Probability'] >= 0.6) & (test_predictions_df['Prediction_Probability'] <= 0.8) | \
                   (test_predictions_df['Prediction_Probability'] >= 0.2) & (test_predictions_df['Prediction_Probability'] <= 0.4)
low_confidence = (test_predictions_df['Prediction_Probability'] >= 0.4) & (test_predictions_df['Prediction_Probability'] <= 0.6)

print(f"\nPrediction Confidence Analysis:")
print(f"High confidence predictions: {high_confidence.sum()} ({high_confidence.mean():.1%})")
print(f"Medium confidence predictions: {medium_confidence.sum()} ({medium_confidence.mean():.1%})")
print(f"Low confidence predictions: {low_confidence.sum()} ({low_confidence.mean():.1%})")

# Save evaluation results
evaluation_summary = {
    'Model_Type': type(final_model).__name__,
    'Training_Accuracy': train_accuracy,
    'Validation_Accuracy': val_accuracy,
    'Training_F1': train_f1,
    'Validation_F1': val_f1,
    'AUC_Score': auc_score,
    'Optimal_Threshold': optimal_threshold,
    'Robustness_Mean': robustness_mean,
    'Robustness_Std': robustness_std,
    'Business_Cost': total_cost,
    'Cost_Savings': cost_savings,
    'Test_Approval_Rate': approval_rate
}

evaluation_df = pd.DataFrame([evaluation_summary])
evaluation_df.to_csv('/home/ubuntu/evaluation_results/evaluation_summary.csv', index=False)

# Save detailed metrics
detailed_metrics = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity', 'NPV'],
    'Training': [train_accuracy, train_precision, train_recall, train_f1, train_auc, np.nan, np.nan],
    'Validation': [val_accuracy, val_precision, val_recall, val_f1, val_auc, specificity, npv]
})
detailed_metrics.to_csv('/home/ubuntu/evaluation_results/detailed_metrics.csv', index=False)

print(f"\n" + "="*60)
print("MODEL TESTING AND EVALUATION COMPLETED!")
print("="*60)
print("Evaluation results saved in '/home/ubuntu/evaluation_results/' directory")
print("Key findings:")
print(f"- Final model: {type(final_model).__name__}")
print(f"- Validation F1-Score: {val_f1:.4f}")
print(f"- AUC Score: {auc_score:.4f} ({auc_interpretation})")
print(f"- Business cost savings: ${cost_savings:,} ({cost_savings_percentage:.1f}%)")
print(f"- Model robustness: {robustness_mean:.4f} (+/- {robustness_std:.4f})")

