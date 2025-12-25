# ==========================================
# Phishing Website Detection using ML
# ==========================================

# 1. Import Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

print("\n Libraries imported successfully.\n")
# 2. Load Dataset
DATASET_PATH = "data/phishing_dataset.csv"

df = pd.read_csv(
    DATASET_PATH,
    encoding="utf-8",
    engine="python",
    on_bad_lines="skip"
)

print(df.head())
print(df.shape)


if not os.path.exists(DATASET_PATH):
    print("❌ Dataset not found!")
    print("Expected path:", DATASET_PATH)
    sys.exit(1)



print(" Dataset loaded successfully.")
print("Shape of dataset:", df.shape)
# 3. Data Preprocessing

# Method 1: Drop missing values
df_drop = df.dropna()

# Method 2: Fill missing values with mean
df_fill = df.fillna(df.mean(numeric_only=True))

print("\n Preprocessing done.")
print("After dropna shape:", df_drop.shape)
print("After fillna shape:", df_fill.shape)

# Use fillna version for training
df = df_fill
# 4. Display Dataset Details

print("\n Dataset Head:")
print(df.head())

print("\nDataset Tail:")
print(df.tail())

print("\nDataset Info:")
print(df.info())
# 5. Check Label Column

'''f 'label' not in df.columns:
    print("❌ Label column not found!")
    sys.exit(1)

print("\n Label value counts:")
print(df['label'].value_counts())

# Fix incorrect labels
df['label'] = df['label'].apply(lambda x: 1 if x == 1 else 0)
# Feature & Target split
X = df.drop('label', axis=1)
y = df['Result']'''
label_col = 'Result'

print(df[label_col].value_counts())

df[label_col] = df[label_col].apply(lambda x: 1 if x == 1 else 0)

X = df.drop(label_col, axis=1)
y = df[label_col]
