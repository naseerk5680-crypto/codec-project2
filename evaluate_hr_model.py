import pandas as pd
import pickle
from sklearn.metrics import accuracy_score, classification_report

print("1. Loading saved model and test data...")
with open('hr_model.pkl', 'rb') as f:
    model = pickle.load(f)

X_test = pd.read_csv('X_test_hr.csv')
y_test = pd.read_csv('y_test_hr.csv').squeeze()

print("2. Running predictions...")
y_pred = model.predict(X_test)

print("3. Calculating results...")
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
