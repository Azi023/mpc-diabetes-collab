# app.py - The Main Backend Application
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# --- Import all our project modules ---
# Database
from src.db_connection import db_connection # Assumes this is in src folder
# Teammates' modules (hypothetical, based on the functions you asked for)
from src.he_module.encryption import encrypt_patient_data
from src.attack_module.detection import is_attack_detected
# Your modules
from src.mpc_psi_module.psi import run_psi
from src.mpc_psi_module.secure_inference import perform_secure_inference, scaler, weights, intercept, feature_columns

app = Flask(__name__)

# --- API Endpoints ---

@app.route('/patient', methods=['POST'])
def add_patient():
    """Endpoint to add a new patient. Integrates Attack Detection and HE."""
    patient_data = request.json
    
    # 1. Attack Detection
    if is_attack_detected(patient_data):
        return jsonify({"status": "error", "message": "Potential attack detected. Record rejected."}), 400
        
    # 2. HE Encryption
    encrypted_record = encrypt_patient_data(patient_data)
    
    # 3. Save to Database
    # We'll assume two collections, one for each hospital, for the PSI demo
    patients_collection = db_connection.get_collection("hospital_a_patients")
    patients_collection.insert_one(encrypted_record)
    
    return jsonify({"status": "success", "message": f"Patient {patient_data['NIC']} added."}), 201

@app.route('/psi', methods=['POST'])
def trigger_psi():
    """Endpoint to run PSI between the two hospital collections."""
    collection_a = db_connection.get_collection("hospital_a_patients")
    collection_b = db_connection.get_collection("hospital_b_patients")
    
    nics_a = [p['NIC_Hashed'] for p in collection_a.find({}, {'NIC_Hashed': 1})]
    nics_b = [p['NIC_Hashed'] for p in collection_b.find({}, {'NIC_Hashed': 1})]
    
    # Note: PSI should run on hashed or encrypted values
    common_hashes = run_psi(nics_a, nics_b)
    
    return jsonify({"status": "success", "common_patient_count": len(common_hashes), "common_hashes": common_hashes})

@app.route('/predict/<string:nic>', methods=['GET'])
def predict_diabetes(nic):
    """Endpoint to get a secure prediction for a specific patient."""
    # In a real system, this would involve fetching data from both hospitals.
    # For our demo, we'll just fetch from Hospital A.
    collection_a = db_connection.get_collection("hospital_a_patients")
    
    # We'd need to decrypt or use the plaintext features stored for the model
    # This part requires careful design with the HE team.
    # For now, we'll simulate by using a plaintext source.
    df_a = pd.read_csv("data/hospital_A.csv")
    patient_record = df_a[df_a['NIC'] == nic]
    
    if patient_record.empty:
        return jsonify({"status": "error", "message": "Patient not found"}), 404
        
    patient_features_unscaled = patient_record[feature_columns]
    patient_features_scaled = scaler.transform(patient_features_unscaled)[0].astype(float).tolist()
    
    # Run your secure MPC logic
    secure_score = perform_secure_inference(patient_features_scaled)
    
    prob = 1.0 / (1.0 + np.exp(-secure_score))
    prediction = "Diabetic" if prob > 0.5 else "Non-diabetic"
    
    return jsonify({
        "status": "success",
        "nic": nic,
        "secure_score": secure_score,
        "probability": prob,
        "prediction": prediction
    })

if __name__ == '__main__':
    # To run: flask --app app run
    app.run(debug=True, port=5000)