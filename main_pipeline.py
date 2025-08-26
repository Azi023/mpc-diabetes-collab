# main_pipeline.py
import pandas as pd
import numpy as np

# Import our custom modules
from src.mpc_psi_module.psi import run_psi
from src.mpc_psi_module.secure_inference import perform_secure_inference_sync, scaler, weights, intercept

print("--- Starting Full Demo Pipeline ---")

# --- STAGE 1: Private Set Intersection (PSI) ---
print("\n[Stage 1: PSI]")
print("Loading data from Hospital A and Hospital B...")

try:
    df_a = pd.read_csv("data/hospital_A.csv")
    df_b = pd.read_csv("data/hospital_B.csv")
    print(f"Hospital A: {len(df_a)} records loaded")
    print(f"Hospital B: {len(df_b)} records loaded")
except FileNotFoundError as e:
    print(f"Error loading data files: {e}")
    print("Please ensure data/hospital_A.csv and data/hospital_B.csv exist")
    exit(1)

# Run the PSI protocol on the NICs
print("Running PSI protocol...")
common_nics = run_psi(df_a['NIC'].tolist(), df_b['NIC'].tolist())
print(f"PSI complete. Found {len(common_nics)} common patients.")

if not common_nics:
    print("No common patients found. Exiting.")
    exit()

# --- STAGE 2: Data Preparation for a Sample Patient ---
print("\n[Stage 2: Data Preparation]")

# Select one patient from the common set for the MPC demo
sample_nic = common_nics[0]
print(f"Selecting sample patient with NIC: {sample_nic} for secure prediction.")

# Get the full record for this patient (we'll assume Hospital A provides it)
patient_record = df_a[df_a['NIC'] == sample_nic]

if patient_record.empty:
    print(f"Error: Patient with NIC {sample_nic} not found in Hospital A data")
    exit(1)

# Define the feature columns that the model expects
feature_columns = [
    'gender', 'age', 'hypertension', 'heart_disease', 'smoking_history',
    'bmi', 'HbA1c_level', 'blood_glucose_level'
]

# Extract and scale the features
try:
    patient_features_unscaled = patient_record[feature_columns]
    patient_features_scaled = scaler.transform(patient_features_unscaled)[0].astype(float).tolist()
    print("Patient feature vector prepared and scaled.")
    print(f"Feature values: {[f'{x:.4f}' for x in patient_features_scaled]}")
except Exception as e:
    print(f"Error preparing patient features: {e}")
    exit(1)

# --- STAGE 3: Secure Multi-Party Computation (MPC) ---
print("\n[Stage 3: MPC]")
print("Starting 3-party secure computation...")

try:
    # Run the MPC pipeline (using synchronous version)
    secure_result = perform_secure_inference_sync(patient_features_scaled)
    print("MPC protocol finished.")
except Exception as e:
    print(f"Error during MPC computation: {e}")
    exit(1)

# --- STAGE 4: Display and Verify Results ---
print("\n[Stage 4: Results]")
print(f"Securely computed score: {secure_result:.6f}")

# For verification, compute the plaintext result
plaintext_score = float(np.dot(weights, patient_features_scaled) + intercept)
print(f"Plaintext score (for verification): {plaintext_score:.6f}")
print("Match:", "YES" if np.isclose(secure_result, plaintext_score, atol=1e-4) else "NO")

# Final prediction
if np.isfinite(secure_result):
    prob = 1.0 / (1.0 + np.exp(-secure_result))
    prediction = "Diabetic" if prob > 0.5 else "Non-diabetic"
    
    print(f"\nPrediction probability: {prob:.4f}")
    print(f"Prediction: {prediction}")
    
    # Additional details
    print(f"\nPatient Details:")
    print(f"- NIC: {sample_nic}")
    print(f"- Risk Score: {secure_result:.6f}")
    print(f"- Confidence: {abs(prob - 0.5) * 2:.1%}")
else:
    print(f"Error: Secure result is not finite: {secure_result}")

print("\n--- Full Demo Pipeline Complete ---")