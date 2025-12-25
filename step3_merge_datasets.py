import pandas as pd
from sklearn.utils import shuffle

print("\nSTEP 3: MERGING DATASETS\n")

# Load datasets
df1 = pd.read_csv("data/phishing_dataset.csv")
df2 = pd.read_csv("data/dataset.csv")

print("Dataset-1 shape:", df1.shape)
print("Dataset-2 shape:", df2.shape)

# Align columns strictly
common_columns = list(set(df1.columns) & set(df2.columns))
df1 = df1[common_columns]
df2 = df2[common_columns]

# Merge datasets
final_df = pd.concat([df1, df2], axis=0)

# Shuffle dataset
final_df = shuffle(final_df, random_state=42)

print("Merged Dataset Shape:", final_df.shape)

# Save merged dataset
final_df.to_csv("data/final_phishing_dataset.csv", index=False)

print("\nMerged dataset saved as final_phishing_dataset.csv")
print("STEP 3 COMPLETED SUCCESSFULLY\n")

