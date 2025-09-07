# populate_db.py
import pandas as pd
from tqdm import tqdm
from pymongo.errors import BulkWriteError
from pymongo import UpdateOne

from src.db_connection import db_connection
from src.mpc_psi_module.psi import hash_id, PSI_SALT

FEATURE_COLS = [
    "NIC","gender","age","hypertension","heart_disease",
    "bmi","HbA1c_level","blood_glucose_level","diabetes"
]

def load_csv(path):
    # NIC must be string to avoid 12-digit ints
    df = pd.read_csv(path, dtype={"NIC": str})
    df = df[FEATURE_COLS].copy()
    df = df.dropna(subset=["NIC"]).drop_duplicates(subset=["NIC"])
    return df

def upsert_collection(df, collection_name):
    col = db_connection.get_collection(collection_name)
    col.create_index("NIC", unique=True)

    ops = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=collection_name):
        doc = r.to_dict()
        doc["NIC"] = str(doc["NIC"])
        doc["NIC_Hashed"] = hash_id(doc["NIC"], PSI_SALT)
        ops.append(UpdateOne({"NIC": doc["NIC"]}, {"$set": doc}, upsert=True))

        if len(ops) >= 1000:
            _flush(col, ops)
            ops = []

    if ops:
        _flush(col, ops)

def _flush(col, ops):
    try:
        col.bulk_write(ops, ordered=False)
    except BulkWriteError as e:
        # duplicates or races – safe to ignore for our idempotent upserts
        pass

if __name__ == "__main__":
    dfA = load_csv("data/hospital_A.csv")
    dfB = load_csv("data/hospital_B.csv")
    upsert_collection(dfA, "hospital_a_patients")
    upsert_collection(dfB, "hospital_b_patients")
    print("✅ Done. Collections ready for PSI/MPC.")
