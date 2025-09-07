# 1_create_hospital_data.py 
import pandas as pd
import random
from sklearn.preprocessing import LabelEncoder

def generate_sri_lankan_nic():
    """
    Generates a random, syntactically correct Sri Lankan NIC string.
    It will randomly create either the new 12-digit format or the old 9-digit + 'V' format.
    """
    if random.choice([True, False]):
        # Generate a new 12-digit format (e.g., 200112345678)
        return str(random.randint(198000000000, 200599999999))
    else:
        # Generate an old 9-digit format with 'V' (e.g., 851234567V)
        return str(random.randint(700000000, 999999999)) + 'V'

print("--- Step 1: Creating Hospital Datasets with String NICs ---")

# Load the dataset
try:
    df = pd.read_csv("data/diabetes_prediction_dataset.csv")
    print("Successfully loaded 'data/diabetes_prediction_dataset.csv'")
except FileNotFoundError:
    print("FATAL ERROR: 'data/diabetes_prediction_dataset.csv' not found. Please download it first.")
    exit()


# Create a unique STRING NIC for each patient
df.insert(0, 'NIC', [generate_sri_lankan_nic() for _ in range(len(df))])
print("Created unique, realistic string 'NIC' for each patient.")
print(f"Sample NICs: {df['NIC'].head(3).tolist()}")

# Split the data into two hospitals with an overlap
common_patients = df.sample(frac=0.2, random_state=42)
remaining_df = df.drop(common_patients.index)

hospital_a_unique = remaining_df.sample(frac=0.5, random_state=1)
hospital_a = pd.concat([common_patients, hospital_a_unique])

hospital_b_unique = remaining_df.drop(hospital_a_unique.index)
hospital_b = pd.concat([common_patients, hospital_b_unique])

# Save the new, correct hospital files
hospital_a.to_csv("data/hospital_A.csv", index=False)
hospital_b.to_csv("data/hospital_B.csv", index=False)

print(f"Successfully created 'data/hospital_A.csv' with {len(hospital_a)} records.")
print(f"Successfully created 'data/hospital_B.csv' with {len(hospital_b)} records.")
print("--- Data Preparation Complete ---")