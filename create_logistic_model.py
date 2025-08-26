# create_logistic_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import pickle

print("--- Starting Model Training Script ---")

# 1. Load the dataset
# We use hospital_A.csv, but you can use any of your feature-complete datasets.
try:
    df = pd.read_csv("data/hospital_A.csv")
    print("Successfully loaded data/hospital_A.csv")
except FileNotFoundError:
    print("Error: Make sure you have a dataset at 'data/hospital_A.csv'")
    exit()

# 2. Prepare the data
# We need to select only the numeric feature columns that the model will use.
# 'NIC', 'gender', 'Name_Encrypted', 'Address_Encrypted' are not features.
feature_columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 
    'SkinThickness', 'Insulin', 'BMI', 
    'DiabetesPedigreeFunction', 'Age'
]
target_column = 'Outcome'

X = df[feature_columns]
y = df[target_column]

# 3. Scale the features
# Logistic Regression performs better when features are on a similar scale.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"Data prepared. Using {len(feature_columns)} features.")

# 4. Train the Logistic Regression model
model = LogisticRegression(random_state=42)
model.fit(X_scaled, y)

print(f"Model trained. Accuracy on training data: {model.score(X_scaled, y):.2f}")

# 5. Save the trained model
model_path = "models/diabetes_model.pkl"
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"SUCCESS: New Logistic Regression model saved to {model_path}")

# Also save the scaler, as it's needed for consistent predictions
scaler_path = "models/scaler.pkl"
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"SUCCESS: Scaler saved to {scaler_path}")