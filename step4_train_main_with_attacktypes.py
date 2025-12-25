#!/usr/bin/env python3
# step4_train_main_with_attacktypes.py
# Robust Step 4: detects label/attack columns, selects features safely, trains models, saves results.

import os
import sys
import pandas as pd
import numpy as np
import joblib
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)

warnings.filterwarnings("ignore")

# ----- Config / paths -----
PROCESSED_CSV = "data/processed_features.csv"   # change if needed
OUT_DIR = "results"
MODELS_DIR = "models"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(os.path.join(OUT_DIR, "predictions"), exist_ok=True)

# ----- Load dataset -----
if not os.path.exists(PROCESSED_CSV):
    print(f"ERROR: processed features file not found: {PROCESSED_CSV}")
    print("Make sure you ran Step 2 and Step 3 to produce processed_features.csv")
    sys.exit(1)

print("🔍 Loading:", PROCESSED_CSV)
df = pd.read_csv(PROCESSED_CSV)
print(f"Rows: {len(df):,}  Columns: {len(df.columns)}")

# ----- Robust label detection -----
possible_label_names = ['label','target','y','is_phish','phish','class','is_phishing']
label_col = None
for name in possible_label_names:
    if name in df.columns:
        label_col = name
        break

# try to find a 0/1 column if not found
if label_col is None:
    for c in df.columns:
        unique_vals = set(df[c].dropna().unique())
        if unique_vals.issubset({0,1}):
            label_col = c
            break

# fallback: last column
if label_col is None:
    label_col = df.columns[-1]
    print(f"⚠️ Could not auto-detect a label column. Assuming last column: {label_col}")

print(f"Using label column: {label_col}")

# ----- Attack-type detection (optional) -----
possible_attack_names = ['attack_type','attack','type','category']
attack_col = None
for name in possible_attack_names:
    if name in df.columns:
        attack_col = name
        break

if attack_col:
    print(f"Found attack-type column: {attack_col}")
else:
    print("No attack_type column found. Script will infer attack_type later (reporting only).")

# ----- If attack_type missing, create heuristic column (reporting only) -----
def infer_attack_type(url: str):
    u = str(url).lower()
    if any(k in u for k in ["bank", "sbi", "icici", "hdfc"]):
        return "banking_phishing"
    if any(k in u for k in ["pay", "upi", "razorpay", "payment", "card"]):
        return "payment_phishing"
    if any(k in u for k in ["amazon", "flipkart", "shop", "order", "cart"]):
        return "ecommerce_phishing"
    if any(k in u for k in ["facebook", "google", "signin", "login", "account"]):
        return "credential_phishing"
    if any(k in u for k in ["gov", ".gov", "tax", "irs"]):
        return "government_phishing"
    if any(k in u for k in ["edu", ".edu", "student", "portal"]):
        return "education_phishing"
    if any(k in u for k in ["support", "helpdesk", "tech-support", "service"]):
        return "tech_support_scams"
    return "other"

# If attack_col missing and we have a URL column, create inferred attack_type
url_col = None
for candidate in ["url", "domain", "original_url"]:
    if candidate in df.columns:
        url_col = candidate
        break

if not attack_col:
    if url_col:
        df["attack_type"] = df[url_col].fillna("").astype(str).apply(infer_attack_type)
        attack_col = "attack_type"
        print("Inferred attack_type from URL column and created 'attack_type'.")
    else:
        df["attack_type"] = "unknown"
        attack_col = "attack_type"
        print("No URL found; created attack_type='unknown' for all rows.")

# Save attack type counts for reporting
attack_counts = df[attack_col].value_counts().reset_index()
attack_counts.columns = ["attack_type", "count"]
attack_counts.to_csv(os.path.join(OUT_DIR, "attack_type_counts.csv"), index=False)
print("Saved attack type counts ->", os.path.join(OUT_DIR, "attack_type_counts.csv"))
print(attack_counts.head(10).to_string(index=False))

# ----- Feature selection: pick numeric features and avoid text columns -----
exclude = {label_col, attack_col, 'url', 'domain', 'original_url', 'Unnamed: 0'}
numeric_dtypes = (np.integer, np.floating, np.int64, np.float64)
feature_cols = [c for c in df.columns if c not in exclude and np.issubdtype(df[c].dtype, np.number)]

if not feature_cols:
    # fall back to all columns except excluded
    feature_cols = [c for c in df.columns if c not in exclude]
    print("Warning: no purely numeric features detected — falling back to all non-excluded columns.")

print(f"\nSelected {len(feature_cols)} feature columns (sample): {feature_cols[:12]}{'...' if len(feature_cols)>12 else ''}")

X = df[feature_cols].copy()
# Ensure label is integer 0/1 if possible
try:
    y_main = df[label_col].astype(int)
except Exception:
    y_main = df[label_col]

y_attack = df[attack_col] if attack_col in df.columns else df[attack_col]

# ----- Train/test split (stratify if possible) -----
stratify_col = y_main if set(y_main.unique()).issubset({0,1}) else None
if stratify_col is not None:
    X_train, X_test, y_main_train, y_main_test, y_attack_train, y_attack_test = train_test_split(
        X, y_main, y_attack, test_size=0.2, random_state=42, stratify=y_main
    )
else:
    X_train, X_test, y_main_train, y_main_test, y_attack_train, y_attack_test = train_test_split(
        X, y_main, y_attack, test_size=0.2, random_state=42
    )

print(f"Train rows: {len(X_train):,}, Test rows: {len(X_test):,}")

# ----- Scaling for models that need it -----
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

# ----- Train main binary model (Logistic Regression) -----
print("\n🚀 Training main phishing detector (Logistic Regression)...")
main_model = LogisticRegression(max_iter=1000)
main_model.fit(X_train_scaled, y_main_train)
y_main_pred = main_model.predict(X_test_scaled)

# compute probability if available
try:
    y_main_prob = main_model.predict_proba(X_test_scaled)[:,1]
    roc = roc_auc_score(y_main_test, y_main_prob)
except Exception:
    y_main_prob = None
    roc = np.nan

main_metrics = {
    "accuracy": accuracy_score(y_main_test, y_main_pred),
    "precision": precision_score(y_main_test, y_main_pred, zero_division=0),
    "recall": recall_score(y_main_test, y_main_pred, zero_division=0),
    "f1": f1_score(y_main_test, y_main_pred, zero_division=0),
    "roc_auc": float(roc)
}
print("Main model metrics:", main_metrics)

# Save main model and predictions
joblib.dump(main_model, os.path.join(MODELS_DIR, "main_phishing_model.pkl"))
pred_main_df = pd.DataFrame({"y_true": y_main_test.values, "y_pred": y_main_pred})
if y_main_prob is not None:
    pred_main_df["y_prob"] = y_main_prob
pred_main_df.to_csv(os.path.join(OUT_DIR, "predictions", "main_model_predictions.csv"), index=False)
pd.DataFrame([main_metrics]).to_csv(os.path.join(OUT_DIR, "main_model_summary.csv"), index=False)
print("Saved main model, predictions and summary.")

# ----- Train attack-type model (RandomForest multi-class) -----
print("\n🚀 Training attack-type classifier (RandomForest)...")
attack_model = RandomForestClassifier(n_estimators=200, random_state=42)
attack_model.fit(X_train, y_attack_train)
y_attack_pred = attack_model.predict(X_test)

attack_report = classification_report(y_attack_test, y_attack_pred, output_dict=True, zero_division=0)
attack_report_df = pd.DataFrame(attack_report).T
attack_report_df.to_csv(os.path.join(OUT_DIR, "predictions", "attack_type_classification_report.csv"))
pd.DataFrame({"y_true": y_attack_test.values, "y_pred": y_attack_pred}).to_csv(
    os.path.join(OUT_DIR, "predictions", "attack_type_predictions.csv"), index=False
)
joblib.dump(attack_model, os.path.join(MODELS_DIR, "attack_type_model.pkl"))
print("Saved attack-type model, predictions, and report.")

# ----- Final outputs and metadata -----
pd.DataFrame(attack_counts).to_csv(os.path.join(OUT_DIR, "attack_type_counts.csv"), index=False)
summary_meta = {
    "dataset": PROCESSED_CSV,
    "rows": int(len(df)),
    "features_used": len(feature_cols),
    "label_column": label_col,
    "attack_type_column": attack_col
}
with open(os.path.join(MODELS_DIR, "dataset_metadata.txt"), "w") as mf:
    mf.write("Dataset metadata:\n")
    for k,v in summary_meta.items():
        mf.write(f"{k}: {v}\n")
    mf.write("\nNote: attack_type may be heuristically inferred if original column missing.\n")
print("\n🎉 STEP 4 completed. Files saved to 'results/' and 'models/'.")

print("Summary files (some):")
print(" -", os.path.join(OUT_DIR, "main_model_summary.csv"))
print(" -", os.path.join(OUT_DIR, "attack_type_counts.csv"))
print(" -", os.path.join(OUT_DIR, "predictions", "main_model_predictions.csv"))
print(" -", os.path.join(OUT_DIR, "predictions", "attack_type_predictions.csv"))
print(" -", os.path.join(MODELS_DIR, "main_phishing_model.pkl"))
print(" -", os.path.join(MODELS_DIR, "attack_type_model.pkl"))
print(" -", os.path.join(MODELS_DIR, "scaler.pkl"))

# exit successfully
sys.exit(0)

