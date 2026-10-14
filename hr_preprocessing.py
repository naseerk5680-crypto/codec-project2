import pandas as pd

print("1. Loading raw HR dataset...")
df = pd.read_csv('WA_Fn-UseC_-HR-Employee-Attrition.csv')

print("2. Preprocessing and cleaning data...")
# Map the target variable 'Attrition' (Yes/No) to binary numbers (1/0)
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})

# Drop columns that have constant values or don't help prediction
columns_to_drop = ['EmployeeCount', 'Over18', 'StandardHours', 'EmployeeNumber']
df.drop([col for col in columns_to_drop if col in df.columns], axis=1, inplace=True)

# Convert all remaining text columns into numerical format (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

# Save the cleaned dataset
df.to_csv('cleaned_hr_data.csv', index=False)
print("Success! Cleaned data saved as 'cleaned_hr_data.csv'.")