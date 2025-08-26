# src/mpc_psi_module/secure_inference.py (Refactored for Integration)
import pickle, random
from pathlib import Path
import numpy as np
import pandas as pd

# ----- Configuration and Helpers (can be loaded once) -----
M = 3
SCALE_X = 10**6
SCALE_W = 10**6
Q = 2**127 - 1
rng = random.Random(1337)

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

# Load model and scaler
with open(MODELS / "diabetes_model.pkl", "rb") as f: 
    model = pickle.load(f)
with open(MODELS / "scaler.pkl", "rb") as f: 
    scaler = pickle.load(f)

weights = model.coef_[0].astype(float).tolist()
intercept = float(model.intercept_[0])

def to_fixed(x, scale): 
    return int(round(float(x) * scale))

def mod_signed(z, q=Q): 
    z = z % q
    return z - q if z > q // 2 else z

def share_value(v, M=M, q=Q):
    """Share a value using additive secret sharing"""
    # Generate M-1 random shares
    shares = [rng.randrange(q) for _ in range(M - 1)]
    
    # Last share ensures sum equals original value (mod q)
    last_share = (v - sum(shares)) % q
    shares.append(last_share)
    
    return shares

def share_vector(xs, M=M, q=Q):
    """Share each element of a vector"""
    n = len(xs)
    # Create M parties, each gets n shares
    party_shares = [[] for _ in range(M)]
    
    for i, x in enumerate(xs):
        shares = share_value(x, M, q)
        for party in range(M):
            party_shares[party].append(shares[party])
    
    return party_shares

def reconstruct(shares, q=Q): 
    """Reconstruct secret from shares"""
    return sum(shares) % q

def dot_secret_public(x_shares, w_fixed, q=Q):
    """
    Compute dot product where x is secret-shared and w is public
    Returns shares of the dot product result
    """
    M = len(x_shares)
    n = len(w_fixed)
    
    # Each party computes their contribution to the dot product
    output_shares = [0] * M
    
    for party in range(M):
        party_sum = 0
        for j in range(n):
            # Get this party's share of x[j]
            x_share = x_shares[party][j]
            
            # Get public weight w[j]
            w_j = w_fixed[j]
            
            # Compute share of product: x_share * w_j (mod q)
            product = (x_share * w_j) % q
            party_sum = (party_sum + product) % q
        
        output_shares[party] = party_sum
    
    return output_shares

# --- Main MPC Function ---
async def perform_secure_inference(patient_scaled_features):
    """
    Perform secure inference on a patient's scaled features.
    Returns the secure prediction score.
    """
    # Encode the input patient data for MPC
    x_fixed = [to_fixed(x, SCALE_X) for x in patient_scaled_features]
    w_fixed = [to_fixed(w, SCALE_W) for w in weights]
    
    # Perform the MPC Protocol
    x_shares = share_vector(x_fixed, M=M, q=Q)
    dot_shares = dot_secret_public(x_shares, w_fixed, q=Q)
    dot_int = reconstruct(dot_shares, q=Q)
    dot_int_signed = mod_signed(dot_int, q=Q)
    
    # Decode the Result
    linear_score = dot_int_signed / (SCALE_X * SCALE_W)
    secure_total = linear_score + intercept
    
    return secure_total

# Synchronous wrapper for compatibility
def perform_secure_inference_sync(patient_scaled_features):
    """
    Synchronous wrapper for perform_secure_inference.
    Use this if you're not using asyncio/mpyc.
    """
    # Encode the input patient data for MPC
    x_fixed = [to_fixed(x, SCALE_X) for x in patient_scaled_features]
    w_fixed = [to_fixed(w, SCALE_W) for w in weights]
    
    # Perform the MPC Protocol
    x_shares = share_vector(x_fixed, M=M, q=Q)
    dot_shares = dot_secret_public(x_shares, w_fixed, q=Q)
    dot_int = reconstruct(dot_shares, q=Q)
    dot_int_signed = mod_signed(dot_int, q=Q)
    
    # Decode the Result
    linear_score = dot_int_signed / (SCALE_X * SCALE_W)
    secure_total = linear_score + intercept
    
    return secure_total

# This block allows running the file standalone for testing
if __name__ == '__main__':
    # Define the sample patient for standalone testing
    feature_columns = [
        'gender','age','hypertension','heart_disease','smoking_history',
        'bmi','HbA1c_level','blood_glucose_level'
    ]
    
    patient_unscaled = pd.DataFrame([[0,80,0,1,4,25.19,6.6,140]], columns=feature_columns)
    patient_scaled = scaler.transform(patient_unscaled)[0].astype(float).tolist()
    
    # Use synchronous version for standalone testing
    secure_result = perform_secure_inference_sync(patient_scaled)
    
    # Verification for standalone run
    plaintext_score = float(np.dot(weights, patient_scaled) + intercept)
    
    print("\n=== Standalone Test Result ===")
    print(f"Secure linear score + intercept: {secure_result:.6f}")
    print(f"Plaintext linear score        : {plaintext_score:.6f}")
    print("Match:", "YES" if np.isclose(secure_result, plaintext_score, atol=1e-4) else "NO")
    
    prob = 1.0 / (1.0 + np.exp(-secure_result))
    print(f"Prediction probability: {prob:.4f}")
    print("Prediction:", "Diabetic" if prob > 0.5 else "Non-diabetic")