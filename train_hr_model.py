import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

print("1. Loading cleaned HR data...")
df = pd.read_csv('cleaned_hr_data.csv')

# Separate features (X) and target (y)
X = df.drop('Attrition', axis=1)
y = df['Attrition']

# Split into 80% training and 20% testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("2. Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("3. Saving model and test sets...")
with open('hr_model.pkl', 'wb') as f:
    pickle.dump(model, f)

X_test.to_csv('X_test_hr.csv', index=False)
y_test.to_csv('y_test_hr.csv', index=False)
print("Success! Model trained and saved as 'hr_model.pkl'.")