# app.py - The Main Backend Application (Corrected)
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np

# --- Import all our project modules ---
# Database
from src.db_connection import db_connection
# Teammates' modules (hypothetical, based on the functions you asked for)
# NOTE: Make sure the import paths are correct based on your final file structure
from src.he_module.encryption_service import create_encrypted_patient_document
from src.attack_module.attack_detection import is_attack_detected
# Your modules
from src.mpc_psi_module.psi import run_psi
# *** FIX #2: Import the SYNCHRONOUS version of the MPC function ***
from src.mpc_psi_module.secure_inference import perform_secure_inference_sync, scaler, feature_columns, weights, intercept

app = Flask(__name__)

# --- API Endpoints ---

@app.route('/patient', methods=['POST'])
def add_patient():
    """Endpoint to add a new patient. Integrates Attack Detection and HE."""
    patient_data = request.json
    
    if is_attack_detected(patient_data):
        return jsonify({"status": "error", "message": "Potential attack detected. Record rejected."}), 400
        
    encrypted_document = create_encrypted_patient_document(patient_data)
    
    # For the demo, we'll randomly assign to one of two hospital collections
    import random
    collection_name = "hospital_a_patients" if random.choice([True, False]) else "hospital_b_patients"
    patients_collection = db_connection.get_collection(collection_name)
    patients_collection.insert_one(encrypted_document)
    
    return jsonify({"status": "success", "message": f"Patient {patient_data['NIC']} added to {collection_name}."}), 201

@app.route('/psi', methods=['POST'])
def trigger_psi():
    """Endpoint to run PSI between the two hospital collections on their HASHED NICs."""
    collection_a = db_connection.get_collection("hospital_a_patients")
    collection_b = db_connection.get_collection("hospital_b_patients")
    
    # Fetch the already-hashed NICs from the database
    hashes_a = [p['NIC_Hashed'] for p in collection_a.find({}, {'NIC_Hashed': 1})]
    hashes_b = [p['NIC_Hashed'] for p in collection_b.find({}, {'NIC_Hashed': 1})]
    
    # *** FIX #1: Tell run_psi that the inputs are already hashed ***
    common_hashes = run_psi(hashes_a, hashes_b, already_hashed=True)
    
    return jsonify({"status": "success", "common_patient_count": len(common_hashes), "common_hashes": common_hashes})

@app.route('/predict/<string:nic>', methods=['GET'])
def predict_diabetes(nic):
    """Endpoint to get a secure prediction for a specific patient."""
    # For our demo, we simulate fetching the plaintext data for a given NIC
    # In a real system, the hospital would do this internally before sharing.
    df_a = pd.read_csv("data/hospital_A.csv") # Using our simulated plaintext data source
    patient_record = df_a[df_a['NIC'] == nic]
    
    if patient_record.empty:
        return jsonify({"status": "error", "message": "Patient not found"}), 404
        
    patient_features_unscaled = patient_record[feature_columns]
    patient_features_scaled = scaler.transform(patient_features_unscaled)[0].astype(float).tolist()
    
    # *** FIX #2: Call the SYNCHRONOUS version of the MPC function ***
    secure_score = perform_secure_inference_sync(patient_features_scaled)
    
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
    app.run(debug=True, port=5000)