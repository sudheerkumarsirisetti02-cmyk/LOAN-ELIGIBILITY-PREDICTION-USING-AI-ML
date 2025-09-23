import pandas as pd
import numpy as np
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Define the number of samples
n_samples = 614  # Similar to the original dataset size

# Generate synthetic loan data
def generate_loan_data(n_samples):
    data = []
    
    for i in range(n_samples):
        # Generate Loan_ID
        loan_id = f"LP{str(i+1).zfill(6)}"
        
        # Generate Gender (Male: 80%, Female: 20%)
        gender = np.random.choice(['Male', 'Female'], p=[0.8, 0.2])
        
        # Generate Married status (Yes: 65%, No: 35%)
        married = np.random.choice(['Yes', 'No'], p=[0.65, 0.35])
        
        # Generate Dependents (0: 60%, 1: 20%, 2: 15%, 3+: 5%)
        dependents = np.random.choice(['0', '1', '2', '3+'], p=[0.6, 0.2, 0.15, 0.05])
        
        # Generate Education (Graduate: 78%, Not Graduate: 22%)
        education = np.random.choice(['Graduate', 'Not Graduate'], p=[0.78, 0.22])
        
        # Generate Self_Employed (No: 85%, Yes: 15%)
        self_employed = np.random.choice(['No', 'Yes'], p=[0.85, 0.15])
        
        # Generate ApplicantIncome (log-normal distribution)
        applicant_income = int(np.random.lognormal(8.5, 0.8)) * 100
        applicant_income = max(1000, min(applicant_income, 50000))  # Clamp between 1000 and 50000
        
        # Generate CoapplicantIncome (many zeros, some with income)
        if np.random.random() < 0.7:  # 70% chance of no coapplicant income
            coapplicant_income = 0
        else:
            coapplicant_income = int(np.random.lognormal(7.5, 0.9)) * 100
            coapplicant_income = max(0, min(coapplicant_income, 30000))
        
        # Generate LoanAmount (correlated with income)
        total_income = applicant_income + coapplicant_income
        loan_amount = int(np.random.normal(total_income * 0.3, total_income * 0.1))
        loan_amount = max(10, min(loan_amount, 700))  # Clamp between 10 and 700 (in thousands)
        
        # Generate Loan_Amount_Term (mostly 360, some 180, 240, 300)
        loan_term = np.random.choice([360, 180, 240, 300], p=[0.85, 0.08, 0.04, 0.03])
        
        # Generate Credit_History (1.0: 85%, 0.0: 10%, NaN: 5%)
        credit_history_rand = np.random.random()
        if credit_history_rand < 0.85:
            credit_history = 1.0
        elif credit_history_rand < 0.95:
            credit_history = 0.0
        else:
            credit_history = np.nan
        
        # Generate Property_Area (Urban: 40%, Semiurban: 35%, Rural: 25%)
        property_area = np.random.choice(['Urban', 'Semiurban', 'Rural'], p=[0.4, 0.35, 0.25])
        
        # Generate Loan_Status based on realistic factors
        # Higher income, good credit history, lower loan amount relative to income = higher approval chance
        approval_score = 0.5  # Base probability
        
        # Income factor
        if total_income > 8000:
            approval_score += 0.2
        elif total_income > 5000:
            approval_score += 0.1
        elif total_income < 3000:
            approval_score -= 0.2
        
        # Credit history factor
        if credit_history == 1.0:
            approval_score += 0.3
        elif credit_history == 0.0:
            approval_score -= 0.4
        
        # Loan amount to income ratio
        if total_income > 0:
            loan_to_income_ratio = (loan_amount * 1000) / total_income
            if loan_to_income_ratio < 3:
                approval_score += 0.2
            elif loan_to_income_ratio > 6:
                approval_score -= 0.3
        
        # Education factor
        if education == 'Graduate':
            approval_score += 0.1
        
        # Employment factor
        if self_employed == 'No':
            approval_score += 0.05
        
        # Property area factor
        if property_area == 'Urban':
            approval_score += 0.05
        
        # Ensure probability is between 0 and 1
        approval_score = max(0.1, min(approval_score, 0.9))
        
        loan_status = 'Y' if np.random.random() < approval_score else 'N'
        
        data.append([
            loan_id, gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount, loan_term,
            credit_history, property_area, loan_status
        ])
    
    return data

# Generate the data
loan_data = generate_loan_data(n_samples)

# Create DataFrame
columns = [
    'Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    'Credit_History', 'Property_Area', 'Loan_Status'
]

df = pd.DataFrame(loan_data, columns=columns)

# Split into train and test sets (80-20 split)
train_size = int(0.8 * len(df))
train_df = df[:train_size].copy()
test_df = df[train_size:].copy()

# Remove Loan_Status from test set (as it would be in real scenario)
test_df_no_target = test_df.drop('Loan_Status', axis=1)

# Save the datasets
train_df.to_csv('/home/ubuntu/loan_train.csv', index=False)
test_df_no_target.to_csv('/home/ubuntu/loan_test.csv', index=False)
test_df.to_csv('/home/ubuntu/loan_test_with_labels.csv', index=False)  # For evaluation

print("Dataset created successfully!")
print(f"Training set: {len(train_df)} samples")
print(f"Test set: {len(test_df)} samples")
print("\nTraining data preview:")
print(train_df.head())
print("\nDataset info:")
print(train_df.info())
print("\nTarget distribution:")
print(train_df['Loan_Status'].value_counts())

