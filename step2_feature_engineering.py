import pandas as pd
import re
from urllib.parse import urlparse

def extract_features(url):
    try:
        parsed = urlparse(url)
        domain = parsed.netloc
        path = parsed.path

        return {
            "url_length": len(url),
            "domain_length": len(domain),
            "num_digits": sum(c.isdigit() for c in url),
            "num_dots": url.count('.'),
            "num_hyphens": url.count('-'),
            "has_https": 1 if url.startswith("https") else 0,
            "num_subdirs": path.count('/'),
            "num_query_params": url.count('='),
            "contains_login": 1 if "login" in url.lower() else 0,
            "contains_secure": 1 if "secure" in url.lower() else 0,
            "contains_account": 1 if "account" in url.lower() else 0,
            "contains_update": 1 if "update" in url.lower() else 0,
            "contains_verify": 1 if "verify" in url.lower() else 0
        }
    except:
        # If any error occurs, return defaults
        return {
            "url_length": 0,
            "domain_length": 0,
            "num_digits": 0,
            "num_dots": 0,
            "num_hyphens": 0,
            "has_https": 0,
            "num_subdirs": 0,
            "num_query_params": 0,
            "contains_login": 0,
            "contains_secure": 0,
            "contains_account": 0,
            "contains_update": 0,
            "contains_verify": 0
        }

# Load dataset
df = pd.read_csv("data/global_phish_dataset.csv")

# Apply feature extraction
features = df["url"].apply(extract_features).apply(pd.Series)

# Add label
features["label"] = df["label"]

# Save for next step
features.to_csv("processed_features.csv", index=False)

print("✅ Step 2 Completed: processed_features.csv created successfully!")
print("Total rows:", len(features))
print(features.head())
