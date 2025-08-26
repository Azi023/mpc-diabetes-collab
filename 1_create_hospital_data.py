# 1_create_hospital_data.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder

print("--- Step 1: Creating Hospital Datasets from the Source of Truth ---")

# 1. Load the single source of truth dataset
try:
    df = pd.read_csv("data/diabetes_prediction_dataset.csv")
    print("Successfully loaded 'data/diabetes_prediction_dataset.csv'")
except FileNotFoundError:
    print("FATAL ERROR: 'data/diabetes_prediction_dataset.csv' not found. Please download it first.")
    exit()

# 2. Preprocess the data (handle non-numeric columns)
# We will convert 'gender' and 'smoking_history' to numbers so our model can use them.
le_gender = LabelEncoder()
df['gender'] = le_gender.fit_transform(df['gender']) # Female -> 0, Male -> 1, Other -> 2

le_smoking = LabelEncoder()
df['smoking_history'] = le_smoking.fit_transform(df['smoking_history'])

# 3. Create a unique NIC for each patient for our PSI simulation
df.insert(0, 'NIC', range(100000, 100000 + len(df)))
print("Created unique 'NIC' for each patient.")

# 4. Split the data into two hospitals with an overlap
common_patients = df.sample(frac=0.2, random_state=42) # 20% of patients are in both hospitals
remaining_df = df.drop(common_patients.index)

# Create Hospital A with its unique share of patients
hospital_a_unique = remaining_df.sample(frac=0.5, random_state=1)
hospital_a = pd.concat([common_patients, hospital_a_unique])

# Create Hospital B with its unique share of patients
hospital_b_unique = remaining_df.drop(hospital_a_unique.index)
hospital_b = pd.concat([common_patients, hospital_b_unique])

# 5. Save the new, correct hospital files
hospital_a.to_csv("data/hospital_A.csv", index=False)
hospital_b.to_csv("data/hospital_B.csv", index=False)

print(f"Successfully created 'data/hospital_A.csv' with {len(hospital_a)} records.")
print(f"Successfully created 'data/hospital_B.csv' with {len(hospital_b)} records.")
print("--- Data Preparation Complete ---")