import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, classification_report, roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
import joblib
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("LOAN ELIGIBILITY PREDICTION - MODEL DEVELOPMENT")
print("="*60)

# Load preprocessed data
print("Loading preprocessed data...")
X_train = pd.read_csv('/home/ubuntu/X_train_processed.csv')
y_train = pd.read_csv('/home/ubuntu/y_train.csv').values.ravel()
X_test = pd.read_csv('/home/ubuntu/X_test_processed.csv')

print(f"Training features shape: {X_train.shape}")
print(f"Training target shape: {y_train.shape}")
print(f"Test features shape: {X_test.shape}")

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Training split shape: {X_train_split.shape}")
print(f"Validation split shape: {X_val_split.shape}")

# Define models to evaluate
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'Support Vector Machine': SVC(random_state=42, probability=True),
    'Naive Bayes': GaussianNB(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Cross-validation setup
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Model evaluation results
results = {}
model_objects = {}

print("\n" + "="*60)
print("MODEL TRAINING AND EVALUATION")
print("="*60)

# Train and evaluate each model
for name, model in models.items():
    print(f"\nTraining {name}...")
    
    # Cross-validation scores
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy')
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Predictions on validation set
    y_pred = model.predict(X_val_split)
    y_pred_proba = model.predict_proba(X_val_split)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_val_split, y_pred)
    precision = precision_score(y_val_split, y_pred)
    recall = recall_score(y_val_split, y_pred)
    f1 = f1_score(y_val_split, y_pred)
    
    if y_pred_proba is not None:
        auc = roc_auc_score(y_val_split, y_pred_proba)
    else:
        auc = None
    
    # Store results
    results[name] = {
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1_Score': f1,
        'AUC': auc
    }
    
    # Store model object
    model_objects[name] = model
    
    print(f"Cross-validation accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
    if auc:
        print(f"AUC: {auc:.4f}")

# Create results DataFrame
results_df = pd.DataFrame(results).T
print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)
print(results_df.round(4))

# Find best model based on F1-score
best_model_name = results_df['F1_Score'].idxmax()
best_model = model_objects[best_model_name]
print(f"\nBest model based on F1-score: {best_model_name}")
print(f"Best F1-score: {results_df.loc[best_model_name, 'F1_Score']:.4f}")

# Hyperparameter tuning for the best model
print(f"\n" + "="*60)
print(f"HYPERPARAMETER TUNING FOR {best_model_name.upper()}")
print("="*60)

# Define parameter grids for different models
param_grids = {
    'Logistic Regression': {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    },
    'Decision Tree': {
        'max_depth': [3, 5, 7, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Random Forest': {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    'Gradient Boosting': {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    },
    'Support Vector Machine': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    },
    'K-Nearest Neighbors': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
}

# Perform hyperparameter tuning for the best model
if best_model_name in param_grids:
    param_grid = param_grids[best_model_name]
    
    # Create a fresh instance of the best model
    if best_model_name == 'Logistic Regression':
        base_model = LogisticRegression(random_state=42, max_iter=1000)
    elif best_model_name == 'Decision Tree':
        base_model = DecisionTreeClassifier(random_state=42)
    elif best_model_name == 'Random Forest':
        base_model = RandomForestClassifier(random_state=42)
    elif best_model_name == 'Gradient Boosting':
        base_model = GradientBoostingClassifier(random_state=42)
    elif best_model_name == 'Support Vector Machine':
        base_model = SVC(random_state=42, probability=True)
    elif best_model_name == 'K-Nearest Neighbors':
        base_model = KNeighborsClassifier()
    else:
        base_model = best_model
    
    # Grid search
    grid_search = GridSearchCV(
        base_model, param_grid, cv=cv, scoring='f1', n_jobs=-1, verbose=1
    )
    
    print("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best cross-validation F1-score: {grid_search.best_score_:.4f}")
    
    # Use the best model
    best_tuned_model = grid_search.best_estimator_
else:
    best_tuned_model = best_model
    print(f"No hyperparameter tuning defined for {best_model_name}")

# Final model evaluation
print(f"\n" + "="*60)
print("FINAL MODEL EVALUATION")
print("="*60)

# Train the final model on full training data
final_model = best_tuned_model
final_model.fit(X_train, y_train)

# Predictions on validation set
y_val_pred = final_model.predict(X_val_split)
y_val_pred_proba = final_model.predict_proba(X_val_split)[:, 1]

# Calculate final metrics
final_accuracy = accuracy_score(y_val_split, y_val_pred)
final_precision = precision_score(y_val_split, y_val_pred)
final_recall = recall_score(y_val_split, y_val_pred)
final_f1 = f1_score(y_val_split, y_val_pred)
final_auc = roc_auc_score(y_val_split, y_val_pred_proba)

print(f"Final Model: {best_model_name}")
print(f"Final Accuracy: {final_accuracy:.4f}")
print(f"Final Precision: {final_precision:.4f}")
print(f"Final Recall: {final_recall:.4f}")
print(f"Final F1-score: {final_f1:.4f}")
print(f"Final AUC: {final_auc:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_val_split, y_val_pred)
print(f"\nConfusion Matrix:")
print(cm)

# Classification Report
print(f"\nClassification Report:")
print(classification_report(y_val_split, y_val_pred))

# Feature Importance (if available)
if hasattr(final_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Feature Importances:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv('/home/ubuntu/feature_importance.csv', index=False)

elif hasattr(final_model, 'coef_'):
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'coefficient': final_model.coef_[0]
    })
    feature_importance['abs_coefficient'] = abs(feature_importance['coefficient'])
    feature_importance = feature_importance.sort_values('abs_coefficient', ascending=False)
    
    print(f"\nTop 10 Feature Coefficients:")
    print(feature_importance.head(10))
    
    # Save feature importance
    feature_importance.to_csv('/home/ubuntu/feature_importance.csv', index=False)

# Save the final model
joblib.dump(final_model, '/home/ubuntu/final_loan_model.pkl')
print(f"\nFinal model saved as 'final_loan_model.pkl'")

# Save model comparison results
results_df.to_csv('/home/ubuntu/model_comparison_results.csv')
print("Model comparison results saved as 'model_comparison_results.csv'")

# Make predictions on test set
print(f"\n" + "="*60)
print("TEST SET PREDICTIONS")
print("="*60)

test_predictions = final_model.predict(X_test)
test_predictions_proba = final_model.predict_proba(X_test)[:, 1]

# Create submission file
test_df_original = pd.read_csv('/home/ubuntu/loan_test.csv')
submission = pd.DataFrame({
    'Loan_ID': test_df_original['Loan_ID'],
    'Loan_Status': ['Y' if pred == 1 else 'N' for pred in test_predictions],
    'Prediction_Probability': test_predictions_proba
})

submission.to_csv('/home/ubuntu/test_predictions.csv', index=False)
print("Test predictions saved as 'test_predictions.csv'")

print(f"\nTest set predictions summary:")
print(f"Total predictions: {len(test_predictions)}")
print(f"Approved: {sum(test_predictions)} ({sum(test_predictions)/len(test_predictions)*100:.1f}%)")
print(f"Rejected: {len(test_predictions) - sum(test_predictions)} ({(len(test_predictions) - sum(test_predictions))/len(test_predictions)*100:.1f}%)")

print(f"\n" + "="*60)
print("MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
print("="*60)

