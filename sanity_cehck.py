import sys, importlib, pickle
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"

def ok_import(name):
    try:
        importlib.import_module(name)
        print(f"[OK] import {name}")
    except Exception as e:
        print(f"[FAIL] import {name}: {e}")

print("=== Python ===")
print(sys.version)

for lib in ["pandas", "numpy", "sklearn"]:
    ok_import(lib)

print("\n=== Load models ===")
for pkl in ["diabetes_model.pkl", "protection_model_new.pkl"]:
    p = MODELS / pkl
    if p.exists():
        try:
            with open(p, "rb") as f:
                obj = pickle.load(f)
            print(f"[OK] loaded {pkl}: {type(obj)}")
        except Exception as e:
            print(f"[FAIL] load {pkl}: {e}")
    else:
        print(f"[MISS] {pkl}")

print("\n=== Test prediction (if sample CSV present) ===")
csv = None
for name in ["diabetes_prediction_dataset_SAMPLE.csv", "diabetes_prediction_dataset.csv", "diabetes.csv"]:
    p = ROOT / "data" / name
    if p.exists():
        csv = p; break
if not csv:
    print("[SKIP] add a small sample CSV to data/")
    sys.exit(0)

df = pd.read_csv(csv)
for tgt in ["Outcome","label","target","Required Action"]:
    if tgt in df.columns:
        df = df.drop(columns=[tgt])

X = df.select_dtypes(include=["number","bool"]).head(5)
if X.empty:
    print("[FAIL] sample CSV has no numeric features")
    sys.exit(1)

dm = MODELS / "diabetes_model.pkl"
if dm.exists():
    try:
        model = pickle.load(open(dm,"rb"))
        if hasattr(model,"predict_proba"):
            y = model.predict_proba(X)
            print("[OK] predict_proba:", y[:2])
        else:
            y = model.predict(X)
            print("[OK] predict:", y[:2])
    except Exception as e:
        print(f"[FAIL] model prediction: {e}")
else:
    print("[MISS] models/diabetes_model.pkl")
