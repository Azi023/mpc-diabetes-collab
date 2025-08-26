# 2_train_logistic_model.py (UPGRADED WITH CONFUSION MATRIX & TEST SET)
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

print("--- Step 2: Training a Model with Proper Evaluation ---")

# 1. Load one of our hospital datasets
try:
    df = pd.read_csv("data/hospital_A.csv")
    print("Successfully loaded 'data/hospital_A.csv' for training.")
except FileNotFoundError:
    print("FATAL ERROR: 'data/hospital_A.csv' not found. Please run '1_create_hospital_data.py' first.")
    exit()

# 2. Define features and target
feature_columns = [
    'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
    'bmi', 'HbA1c_level', 'blood_glucose_level'
]
target_column = 'diabetes'

X = df[feature_columns]
y = df[target_column]

# 3. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Data split: {len(X_train)} training samples, {len(X_test)} testing samples.")

# 4. Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) # Use the same scaler fitted on the training data

# 5. Train the Logistic Regression model
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
print("Model trained successfully.")

# 6. Evaluate the model on the unseen TEST data
print("\n--- Model Evaluation on Test Set ---")
y_pred = model.predict(X_test_scaled)
print(classification_report(y_test, y_pred))

# 7. Generate and display the Confusion Matrix
print("--- Confusion Matrix ---")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Optional: Plot the confusion matrix for visual reports
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix: Logistic Regression')
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("\nConfusion matrix plot saved as 'confusion_matrix.png'")
# plt.show() # Uncomment this line if you want the plot to pop up when you run the script

# 8. Save the final model and the scaler
pickle.dump(model, open("models/diabetes_model.pkl", 'wb'))
pickle.dump(scaler, open("models/scaler.pkl", 'wb'))

print("\nSUCCESS: Final Logistic Regression model and scaler saved to 'models/' folder.")
print("--- Model Training Complete ---")