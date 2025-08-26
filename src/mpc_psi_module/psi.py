# src/mpc_psi_module/psi.py
import hashlib
import pandas as pd

# In a real system, this salt would be securely pre-shared between the hospitals.
# For our project, we define it as a constant.
PSI_SALT = "a_very_secret_salt_for_our_rp_demo_123" 

def hash_id(patient_id, salt):
    """Hashes a patient ID with a salt using SHA-256."""
    return hashlib.sha256(str(patient_id).encode() + salt.encode()).hexdigest()

def run_psi(patient_ids_A, patient_ids_B):
    """
    Performs a simple salted-hash based Private Set Intersection.
    
    Args:
        patient_ids_A (list): A list of patient IDs from Hospital A.
        patient_ids_B (list): A list of patient IDs from Hospital B.

    Returns:
        list: A list of the original patient IDs that are in the intersection.
    """
    # Hospital A creates a dictionary mapping its hashed IDs back to the original IDs.
    hashed_set_A = {hash_id(pid, PSI_SALT): pid for pid in patient_ids_A}
    
    # Hospital B does the same.
    hashed_set_B = {hash_id(pid, PSI_SALT): pid for pid in patient_ids_B}

    # The hospitals would exchange ONLY the hashed keys of these dictionaries.
    # They find the intersection of the *hashed* sets.
    intersecting_hashes = set(hashed_set_A.keys()).intersection(set(hashed_set_B.keys()))

    # We can now map the intersecting hashes back to the original IDs.
    intersecting_ids = sorted([hashed_set_A[h] for h in intersecting_hashes])
    return intersecting_ids

# This part allows us to run the file directly for testing.
if __name__ == '__main__':
    print("--- Running PSI Standalone Test ---")
    
    # 1. Load data from the two hospital CSVs
    df_a = pd.read_csv("../../data/hospital_A.csv")
    df_b = pd.read_csv("../../data/hospital_B.csv")

    # 2. Extract the patient ID columns
    ids_a = df_a['NIC'].tolist()
    ids_b = df_b['NIC'].tolist()

    print(f"Hospital A's patient list: {ids_a}")
    print(f"Hospital B's patient list: {ids_b}")

    # --- PROOF STEP 1: Let's see what Hospital A's private dictionary looks like ---
    # We'll just look at the first 5 items for readability.
    PSI_SALT = "a_very_secret_salt_for_our_rp_demo_123" 
    hashed_set_A_proof = {hash_id(pid, PSI_SALT): pid for pid in ids_a}
    print("\n--- Hospital A's Private Data (first 5 items) ---")
    for i, (hashed_id, original_id) in enumerate(hashed_set_A_proof.items()):
        if i >= 5: break
        print(f"  Hashed: {hashed_id[:16]}...  --->  Original NIC: {original_id}")
    # -----------------------------------------------------------------------------
    
    # 3. Run the PSI protocol
    intersection = run_psi(ids_a, ids_b)
    
    print(f"\nProtocol complete. Found intersection: {intersection}")
    print("Note: The hospitals only exchanged hashed, salted IDs, not the original ones.")


    # --- PROOF STEP 2: Let's prove the intersection was on the hashes ---
    hashed_set_B_proof = {hash_id(pid, PSI_SALT): pid for pid in ids_b}
    intersecting_hashes_proof = set(hashed_set_A_proof.keys()).intersection(set(hashed_set_B_proof.keys()))
    print("\n--- Proof of Exchange ---")
    print(f"Total NICs at Hospital A: {len(ids_a)}")
    print(f"Total hashes exchanged by A: {len(hashed_set_A_proof.keys())}")
    print(f"Number of matching HASHES found: {len(intersecting_hashes_proof)}")
    print("This proves the comparison happened on the secure hashes, not the original NICs.")
    # ----------------------------------------------------------------------