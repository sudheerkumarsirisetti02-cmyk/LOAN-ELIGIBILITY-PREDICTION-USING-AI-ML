import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

# Load the training data
print("Loading and exploring the loan dataset...")
df = pd.read_csv('/home/ubuntu/loan_train.csv')

print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nDataset info:")
print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nStatistical summary:")
print(df.describe())

# Exploratory Data Analysis
print("\n" + "="*50)
print("EXPLORATORY DATA ANALYSIS")
print("="*50)

# Create visualizations directory
import os
os.makedirs('/home/ubuntu/visualizations', exist_ok=True)

# 1. Target variable distribution
plt.figure(figsize=(8, 6))
target_counts = df['Loan_Status'].value_counts()
plt.pie(target_counts.values, labels=['Approved (Y)', 'Rejected (N)'], autopct='%1.1f%%', startangle=90)
plt.title('Loan Approval Distribution', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ubuntu/visualizations/target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Categorical variables analysis
categorical_cols = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']

fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

for i, col in enumerate(categorical_cols):
    # Create crosstab
    ct = pd.crosstab(df[col], df['Loan_Status'], normalize='index') * 100
    ct.plot(kind='bar', ax=axes[i], color=['#ff7f7f', '#7fbf7f'])
    axes[i].set_title(f'Loan Approval by {col}', fontweight='bold')
    axes[i].set_xlabel(col)
    axes[i].set_ylabel('Percentage')
    axes[i].legend(['Rejected', 'Approved'])
    axes[i].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/home/ubuntu/visualizations/categorical_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Numerical variables analysis
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, col in enumerate(numerical_cols):
    # Box plot for each numerical variable by loan status
    df.boxplot(column=col, by='Loan_Status', ax=axes[i])
    axes[i].set_title(f'{col} by Loan Status')
    axes[i].set_xlabel('Loan Status')
    axes[i].set_ylabel(col)

plt.suptitle('Numerical Variables Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ubuntu/visualizations/numerical_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Correlation matrix
# First, encode categorical variables for correlation analysis
df_encoded = df.copy()
le = LabelEncoder()

for col in categorical_cols + ['Loan_Status']:
    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))

# Handle missing values for correlation
df_encoded['Credit_History'].fillna(df_encoded['Credit_History'].mode()[0], inplace=True)

plt.figure(figsize=(12, 10))
correlation_matrix = df_encoded.drop('Loan_ID', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5, cbar_kws={"shrink": .8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/home/ubuntu/visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Income distribution analysis
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.hist(df['ApplicantIncome'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.title('Applicant Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')

plt.subplot(1, 3, 2)
plt.hist(df['CoapplicantIncome'], bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
plt.title('Coapplicant Income Distribution')
plt.xlabel('Income')
plt.ylabel('Frequency')

plt.subplot(1, 3, 3)
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
plt.hist(df['TotalIncome'], bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.title('Total Income Distribution')
plt.xlabel('Total Income')
plt.ylabel('Frequency')

plt.tight_layout()
plt.savefig('/home/ubuntu/visualizations/income_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Data Preprocessing
print("\n" + "="*50)
print("DATA PREPROCESSING")
print("="*50)

# Load both training and test data for consistent preprocessing
train_df = pd.read_csv('/home/ubuntu/loan_train.csv')
test_df = pd.read_csv('/home/ubuntu/loan_test.csv')

print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")

# Combine for preprocessing
train_df['dataset'] = 'train'
test_df['dataset'] = 'test'
test_df['Loan_Status'] = 'Unknown'  # Placeholder for test set

combined_df = pd.concat([train_df, test_df], ignore_index=True)

# Feature Engineering
print("\nFeature Engineering...")

# 1. Create total income feature
combined_df['TotalIncome'] = combined_df['ApplicantIncome'] + combined_df['CoapplicantIncome']

# 2. Create loan amount to income ratio
combined_df['LoanAmountToIncomeRatio'] = combined_df['LoanAmount'] / (combined_df['TotalIncome'] + 1)  # +1 to avoid division by zero

# 3. Create income per dependent
combined_df['Dependents_num'] = combined_df['Dependents'].replace('3+', '3').astype(int)
combined_df['IncomePerDependent'] = combined_df['TotalIncome'] / (combined_df['Dependents_num'] + 1)

# 4. Create binary features
combined_df['HasCoapplicant'] = (combined_df['CoapplicantIncome'] > 0).astype(int)

# Handle missing values
print("\nHandling missing values...")

# Credit_History: Fill with mode (most common value)
combined_df['Credit_History'].fillna(combined_df['Credit_History'].mode()[0], inplace=True)

# LoanAmount: Fill with median
combined_df['LoanAmount'].fillna(combined_df['LoanAmount'].median(), inplace=True)

# Loan_Amount_Term: Fill with mode
combined_df['Loan_Amount_Term'].fillna(combined_df['Loan_Amount_Term'].mode()[0], inplace=True)

# Gender: Fill with mode
combined_df['Gender'].fillna(combined_df['Gender'].mode()[0], inplace=True)

# Married: Fill with mode
combined_df['Married'].fillna(combined_df['Married'].mode()[0], inplace=True)

# Dependents: Fill with mode
combined_df['Dependents'].fillna(combined_df['Dependents'].mode()[0], inplace=True)

# Self_Employed: Fill with mode
combined_df['Self_Employed'].fillna(combined_df['Self_Employed'].mode()[0], inplace=True)

print("Missing values after imputation:")
print(combined_df.isnull().sum())

# Encode categorical variables
print("\nEncoding categorical variables...")

# Binary encoding for binary categorical variables
binary_cols = ['Gender', 'Married', 'Self_Employed']
for col in binary_cols:
    combined_df[col] = combined_df[col].map({'Male': 1, 'Female': 0, 'Yes': 1, 'No': 0})

# Ordinal encoding for Dependents
combined_df['Dependents'] = combined_df['Dependents'].map({'0': 0, '1': 1, '2': 2, '3+': 3})

# Ordinal encoding for Education
combined_df['Education'] = combined_df['Education'].map({'Not Graduate': 0, 'Graduate': 1})

# One-hot encoding for Property_Area
property_dummies = pd.get_dummies(combined_df['Property_Area'], prefix='Property')
combined_df = pd.concat([combined_df, property_dummies], axis=1)

# Encode target variable
combined_df['Loan_Status'] = combined_df['Loan_Status'].map({'N': 0, 'Y': 1, 'Unknown': -1})

# Drop original categorical columns and unnecessary columns
columns_to_drop = ['Loan_ID', 'Property_Area', 'dataset', 'Dependents_num']
combined_df = combined_df.drop(columns_to_drop, axis=1)

# Split back into train and test
train_processed = combined_df[combined_df['Loan_Status'] != -1].copy()
test_processed = combined_df[combined_df['Loan_Status'] == -1].copy()

# Remove target from test set
X_test = test_processed.drop('Loan_Status', axis=1)

# Separate features and target for training set
X_train = train_processed.drop('Loan_Status', axis=1)
y_train = train_processed['Loan_Status']

print(f"\nProcessed training features shape: {X_train.shape}")
print(f"Processed test features shape: {X_test.shape}")
print(f"Training target shape: {y_train.shape}")

# Feature scaling
print("\nApplying feature scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrames
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Save preprocessed data
X_train_scaled.to_csv('/home/ubuntu/X_train_processed.csv', index=False)
X_test_scaled.to_csv('/home/ubuntu/X_test_processed.csv', index=False)
y_train.to_csv('/home/ubuntu/y_train.csv', index=False)

print("\nPreprocessed data saved successfully!")
print("\nFeature names:")
print(list(X_train.columns))

print("\nPreprocessing completed successfully!")
print("Files saved:")
print("- X_train_processed.csv: Processed training features")
print("- X_test_processed.csv: Processed test features") 
print("- y_train.csv: Training target variable")
print("- Visualizations saved in /home/ubuntu/visualizations/ directory")

