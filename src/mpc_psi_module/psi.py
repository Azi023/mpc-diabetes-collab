# src/mpc_psi_module/psi.py
import hashlib
import pandas as pd

# In a real system, this salt would be securely pre-shared between the hospitals.
# For our project, we define it as a constant.
PSI_SALT = "a_very_secret_salt_for_our_rp_demo_123" 

def hash_id(patient_id, salt):
    """Hashes a patient ID with a salt using SHA-256."""
    return hashlib.sha256(str(patient_id).encode() + salt.encode()).hexdigest()

#When you pass it 851234567V, str() does nothing, and it hashes the string. When you pass it an integer, str() converts it to a string first. This is a robust design, and you can confidently tell your panel that your PSI protocol was built to handle alphanumeric identifiers from the start.


def run_psi(ids_A, ids_B, already_hashed=False):
    """
    Performs a simple salted-hash based Private Set Intersection.
    
    Args:
        ids_A (list): A list of patient identifiers from Party A.
        ids_B (list): A list of patient identifiers from Party B.
        already_hashed (bool): If True, treats the inputs as already hashed and skips hashing.
    """
    if already_hashed:
        # If inputs are already hashed, just convert to sets
        hashed_set_A = set(ids_A)
        hashed_set_B = set(ids_B)
        
        # In this mode, we can only return the common hashes, not the original IDs
        return sorted(list(hashed_set_A.intersection(hashed_set_B)))
    else:
        # If inputs are plaintext, perform the full salted-hash protocol
        hashed_map_A = {hash_id(pid, PSI_SALT): pid for pid in ids_A}
        hashed_map_B = {hash_id(pid, PSI_SALT): pid for pid in ids_B}
        
        intersecting_hashes = set(hashed_map_A.keys()).intersection(hashed_map_B.keys())
        
        # Return the original, plaintext IDs from the intersection
        return sorted([hashed_map_A[h] for h in intersecting_hashes if h in hashed_map_A])

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