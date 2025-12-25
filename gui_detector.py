#!/usr/bin/env python3
"""
GUI Phishing Detection & Attack Type Tool (Force attack-type option)
- Paste into gui_detector_force_attack.py
- Ensure models/ has:
    - main_phishing_model.pkl
    - scaler.pkl
    - training_features_list.pkl
Run: python3 gui_detector_force_attack.py
"""

import os
import re
import ipaddress
from urllib.parse import urlparse

import tkinter as tk
from tkinter import messagebox, scrolledtext

import joblib
import tldextract
import pandas as pd

# -----------------------------
# USER CONFIGURATION
# -----------------------------
# If True -> any derived attack_type (not "unknown") forces final result to PHISHING.
# Set to False to rely only on ML + rule scoring (conservative).
FORCE_ATTACK_TYPE_DETECT = True

# -----------------------------
# Safe GUI messagebox wrappers
# -----------------------------
def safe_showerror(title, message):
    try:
        messagebox.showerror(title, message)
    except Exception:
        print(f"[ERROR] {title}: {message}")

def safe_showwarning(title, message):
    try:
        messagebox.showwarning(title, message)
    except Exception:
        print(f"[WARN] {title}: {message}")

# -----------------------------
# Load model artifacts
# -----------------------------
MODELS_DIR = "models"
MODEL_MAIN = os.path.join(MODELS_DIR, "main_phishing_model.pkl")
SCALER_FILE = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURE_LIST_FILE = os.path.join(MODELS_DIR, "training_features_list.pkl")

if not os.path.isdir(MODELS_DIR):
    safe_showerror("Missing Models", f"Models directory '{MODELS_DIR}' not found. Please ensure models are present.")
    raise SystemExit(1)

try:
    main_model = joblib.load(MODEL_MAIN)
    scaler = joblib.load(SCALER_FILE)
    feature_list = joblib.load(FEATURE_LIST_FILE)
except Exception as e:
    safe_showerror("Model Load Error",
                   f"Failed to load model artifacts from '{MODELS_DIR}'.\nException: {e}")
    raise SystemExit(1)

if not hasattr(feature_list, "__iter__"):
    safe_showerror("Feature List Error", "Loaded feature list is not valid.")
    raise SystemExit(1)

# -----------------------------
# FEATURE EXTRACTION (must match training)
# -----------------------------
def extract_features(url: str) -> dict:
    parsed = urlparse(url)
    tld = tldextract.extract(url)

    # Build domain using tldextract when available
    if tld.domain:
        domain = tld.domain + (("." + tld.suffix) if tld.suffix else "")
    else:
        domain = parsed.netloc

    domain_only = (parsed.netloc.split(":")[0] if parsed.netloc else domain)

    # detect IP address in domain
    try:
        ipaddress.ip_address(domain_only)
        has_ip = 1
    except Exception:
        has_ip = 0

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "num_digits": sum(c.isdigit() for c in url),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "has_at": int("@" in url),
        "has_ip": has_ip,
        "https": int(url.lower().startswith("https://")),
        "num_subdirs": parsed.path.count("/") if parsed.path else 0,
        "num_query_params": parsed.query.count("=") if parsed.query else 0,
        "contains_login": int("login" in url.lower()),
        "contains_secure": int("secure" in url.lower()),
        "contains_account": int("account" in url.lower()),
        "contains_update": int("update" in url.lower()),
        "contains_verify": int("verify" in url.lower()),
    }

    return features

# -----------------------------
# Vectorize features (order must match training feature_list)
# -----------------------------
def vectorize(features: dict):
    s = pd.Series(features)
    vec = s.reindex(feature_list, fill_value=0).values
    return vec.reshape(1, -1)

# -----------------------------
# Rule-based attack type categorizer
# -----------------------------
def categorize_attack_type(url: str) -> str:
    u = url.lower()
    if any(k in u for k in ["login", "signin", "sign-in", "password", "log-in"]):
        return "credential_phishing"
    if any(k in u for k in ["bank", "pay", "paypal", "upi", "wallet", "credit", "card"]):
        return "financial_theft"
    if any(k in u for k in ["verify", "update", "reset", "secure", "confirm"]):
        return "account_verification"
    if any(k in u for k in ["win", "free", "bonus", "gift", "prize"]):
        return "prize_scam"
    if any(k in u for k in ["download", ".exe", ".apk", "install"]):
        return "malware_attack"
    return "unknown"

# -----------------------------
# Rule-based suspicion scoring
# -----------------------------
def rule_based_score(url: str):
    score = 0
    reasons = []
    parsed = urlparse(url)
    domain = (parsed.netloc or "").lower()

    if url.count("-") > 3:
        score += 1
        reasons.append("Many hyphens in URL")
    if domain.count("-") > 1 and len(domain) > 15:
        score += 1
        reasons.append("Multiple hyphens in a long domain")

    keywords = ["login", "secure", "verify", "update", "account", "paypal", "bank", "google", "amazon"]
    for k in keywords:
        if k in url.lower() and k not in domain:
            score += 1
            reasons.append(f"Keyword '{k}' found outside primary domain")

    if not url.lower().startswith("https://") and any(k in url.lower() for k in ["login", "account", "secure", "bank", "verify"]):
        score += 1
        reasons.append("HTTP (not HTTPS) used for sensitive keywords")

    if re.match(r"^\d{1,3}(\.\d{1,3}){3}$", parsed.netloc.split(":")[0] or ""):
        score += 2
        reasons.append("Direct IP address in domain")

    if "@" in parsed.netloc:
        score += 2
        reasons.append("'@' found in netloc (possible redirect)")

    impersonation_brands = ["paypal.com", "facebook.com", "google.com", "amazon.com", "microsoft.com"]
    lower_url = url.lower()
    for brand in impersonation_brands:
        # brand substring present but appears before actual netloc -> suspicious
        if brand in lower_url and lower_url.find(brand) < lower_url.find(parsed.netloc.lower()):
            score += 2
            reasons.append(f"Possible brand impersonation: {brand}")

    return score, reasons

# -----------------------------
# Main analyze function
# -----------------------------
def analyze_url_action():
    url = entry_url.get().strip()
    if not url:
        safe_showwarning("Input Required", "Please enter a URL to analyze.")
        return

    output.delete(1.0, tk.END)
    output.insert(tk.END, f"Analyzing URL: {url}\n\n")

    try:
        feats = extract_features(url)
        vec = vectorize(feats)
        scaled = scaler.transform(vec)

        ml_prob = float(main_model.predict_proba(scaled)[0][1])
        ml_pred = int(main_model.predict(scaled)[0])

        attack_type = categorize_attack_type(url)
        rule_score, rule_reasons = rule_based_score(url)

        output.insert(tk.END, f"ML Phishing Probability: {ml_prob:.4f}\n")
        output.insert(tk.END, f"Rule Score: {rule_score}\n")
        output.insert(tk.END, f"Attack Type (derived): {attack_type}\n\n")

        # DECISION
        decision_reasons = []

        if ml_pred == 1 and ml_prob > 0.50:
            decision_reasons.append(f"ML model predicted phishing (P={ml_prob:.2f})")
        elif ml_prob > 0.75:
            decision_reasons.append(f"High ML phishing probability (P={ml_prob:.2f})")

        if rule_score >= 3:
            decision_reasons.append(f"High rule-based suspicion (score={rule_score})")
        elif rule_score >= 1:
            decision_reasons.append(f"Minor rule-based flags (score={rule_score})")

        # Combined decision logic
        is_phish_ml_and_rules = any("High" in d or "predicted phishing" in d for d in decision_reasons) \
                                or (ml_prob > 0.45 and rule_score >= 1)

        # If the user wants to force attack-type detection -> treat any non-unknown attack type as phishing
        attack_type_phishing = (attack_type != "unknown")

        if FORCE_ATTACK_TYPE_DETECT and attack_type_phishing:
            is_phish = True
            decision_reasons.insert(0, f"Derived attack type suggests phishing ({attack_type}) -- forced by config")
        else:
            is_phish = is_phish_ml_and_rules

        attack_label = f" ({attack_type})" if attack_type and attack_type != "unknown" else ""

        if is_phish:
            output.insert(tk.END, f"FINAL RESULT: 🛑 PHISHING WEBSITE DETECTED{attack_label}\n\n", "red")
            output.insert(tk.END, "Decision Factors:\n", "bold")
            for d in decision_reasons:
                output.insert(tk.END, f"- {d}\n")
            if rule_reasons:
                output.insert(tk.END, "\nRule-based Observations:\n", "bold")
                for r in rule_reasons:
                    output.insert(tk.END, f"- {r}\n")
        else:
            output.insert(tk.END, f"FINAL RESULT: ✅ SAFE WEBSITE{attack_label}\n\n", "green")
            if decision_reasons:
                output.insert(tk.END, "Decision Notes:\n", "bold")
                for d in decision_reasons:
                    output.insert(tk.END, f"- {d}\n")
            if rule_reasons:
                output.insert(tk.END, "\nMinor rule observations:\n", "bold")
                for r in rule_reasons:
                    output.insert(tk.END, f"- {r}\n")

    except Exception as e:
        safe_showerror("Analysis Error", f"An error occurred during URL analysis: {e}")
        output.insert(tk.END, f"\nAnalysis Error: {e}\n", "red")

# -----------------------------
# GUI layout
# -----------------------------
root = tk.Tk()
root.title("Phishing Detection & Attack Type GUI (Force Attack-Type Option)")
root.geometry("760x620")
root.minsize(640, 480)

label = tk.Label(root, text="Enter URL to Analyze:", font=("Arial", 14, "bold"))
label.pack(pady=(12, 6))

entry_url = tk.Entry(root, width=100, font=("Arial", 12))
entry_url.pack(padx=10, pady=(0, 8), fill=tk.X)
entry_url.bind("<Return>", lambda event=None: analyze_url_action())

btn = tk.Button(root,
                text="Analyze URL",
                font=("Arial", 12, "bold"),
                bg="#4CAF50", fg="white",
                command=analyze_url_action)
btn.pack(pady=(0, 8))

output = scrolledtext.ScrolledText(root, wrap=tk.WORD, width=100, height=28, font=("Consolas", 10))
output.pack(padx=12, pady=(0, 12), fill=tk.BOTH, expand=True)

output.tag_config("red", foreground="red", font=("Consolas", 11, "bold"))
output.tag_config("green", foreground="green", font=("Consolas", 11, "bold"))
output.tag_config("bold", font=("Consolas", 10, "bold"))

def on_closing():
    try:
        if messagebox.askokcancel("Quit", "Exit the Phishing Detector?"):
            root.destroy()
    except Exception:
        try:
            root.destroy()
        except Exception:
            pass

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

